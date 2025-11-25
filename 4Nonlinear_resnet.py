import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import math

"""
### 💻 混合非线性ResNet代码设计思路 (V2 - Bottleneck)

#### 1. 核心问题
V1版本的 `HybridNonLinearBlock` (conv -> nonlinear_conv) 依然会在通道数高的层（如 Layer3, 256x256）导致显存开销巨大。

#### 2. 解决方案 (采纳用户建议)
根据用户的建议，我们采用"将非线性卷积对应的通道数减小"的策略，设计一个"非线性瓶颈"模块 (HybridNonLinearBottleneck)，其结构如下：

1.  `conv1` (线性 `1x1`):  将通道数 `in_planes` (如 256) **降维**到 `planes` (如 64)。
2.  `conv2` (非线性 `3x3`): 在**低维** (64x64) 上执行昂贵的 `NonLinearConvBlock`。
3.  `conv3` (线性 `1x1`):  将通道数 `planes` (如 64) **升维**回 `planes * expansion` (如 256)。

#### 3. 优势
- 显存开销（主要在conv2）的 `C_in * C_out` 复杂度从 `256*256` 骤降到 `64*64`，**极大缓解显存**。
- 允许我们在网络的每一层（`layer2`, `layer3`, `layer4`）都使用非线性计算，而不会OOM。
- 这完全符合用户"减小非线性卷积通道数"和"混合使用"的思想。
"""

# ==============================================================================
# 1. 核心模块：非线性卷积 (EKV Model)
#    【注意】这里必须使用上一版回复中带 for 循环的 forward 方法！
# ==============================================================================

class NonLinearConv2d(nn.Module):
    """
    模拟 EKV 模型的非线性卷积层。
    输入：V_G (电压), 权重：V_th (阈值电压)
    输出：I_k (电流)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding

        self.theta = nn.Parameter(
            torch.empty(out_channels, in_channels, *kernel_size)
        )
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        
        self.alpha = 0.0005625
        self.VD = 0.1
        self.n = 1.5
        self.VT = 0.025
        self.denom = 2 * self.n * self.VT 

        self.v_min = 0.0
        self.v_max = 9.0
        self.theta_min = 1.0
        self.theta_max = 8.0

    def _ekv_f(self, v_in, v_th):
        """ EKV 核心方程 f(V, θ) """
        arg = (v_in - v_th) / self.denom
        arg = torch.clamp(arg, -50, 50) 
        return torch.pow(torch.log(1 + torch.exp(arg)), 2)

    def forward(self, x):
        # 1. 物理约束：钳位权重和输入
        self.theta.data.clamp_(self.theta_min, self.theta_max)
        x_clamped = torch.clamp(x, self.v_min, self.v_max)

        # 2. 展开输入为 Patches
        patches = F.unfold(
            x_clamped, 
            self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        )
        # patches shape: (B, C_in * K * K, L)
        
        B, Cin_K_K, L = patches.shape
        
        # 3. 准备广播 (V_G)
        # (B, C_in*K*K, L) -> (B, L, C_in*K*K)
        v_g = patches.transpose(1, 2)

        # 准备权重 (V_th)
        # (C_out, C_in, K, K) -> (C_out, C_in*K*K)
        v_th_flat = self.theta.view(self.out_channels, -1)

        # 4. 【解决方案】迭代 C_out，用时间换空间 (解决OOM的关键)
        i_k_list = []
        for i in range(self.out_channels):
            # v_g shape:         (B, L, C_in*K*K)
            # v_th_channel shape: (1, 1, C_in*K*K)
            v_th_channel = v_th_flat[i].unsqueeze(0).unsqueeze(0)
            
            term1 = self._ekv_f(v_g, v_th_channel)
            term2 = self._ekv_f(v_g, v_th_channel + self.VD)
            
            # current_patches shape: (B, L, C_in*K*K)
            current_patches = self.alpha * (term1 - term2)

            # 5. 模拟 KCL：电流求和
            # i_k_channel shape: (B, L)
            i_k_channel = current_patches.sum(dim=2)
            i_k_list.append(i_k_channel)

        # 6. 拼接所有输出通道
        # List[ (B, L) ] -> (B, C_out, L)
        i_k = torch.stack(i_k_list, dim=1)
        
        # 7. 转换回图像格式
        out_h = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_w = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        out = i_k.reshape(B, self.out_channels, out_h, out_w)
        
        return out

# ==============================================================================
# 2. 模拟电路封装 (Block) - (保持不变)
# ==============================================================================

class NonLinearConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = NonLinearConv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.v_min = 0.0
        self.v_max = 9.0

    def forward(self, x):
        i_k = self.conv(x)
        v_out = self.bn(i_k)
        v_clamped = torch.clamp(v_out, self.v_min, self.v_max)
        return v_clamped

# ==============================================================================
# 3. 混合 ResNet 架构
# ==============================================================================

class BasicBlock(nn.Module):
    """标准 ResNet BasicBlock - (保持不变)"""
    expansion = 1
    # ... (代码同前，为简洁省略) ...
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 【新模块】 实现您的方案二
class HybridNonLinearBottleneck(nn.Module):
    """
    非线性瓶颈模块 (ResNet-50 风格)
    conv1 (线性 1x1, 降维) -> conv2 (非线性 3x3) -> conv3 (线性 1x1, 升维)
    """
    expansion = 4 # 升维/降维因子

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        
        # 1. 线性 1x1 降维
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 2. 非线性 3x3 (在低维 planes 上计算)
        self.conv2_nonlinear = NonLinearConvBlock(
            planes, planes, kernel_size=3, stride=stride, padding=1
        )
        # 注意：NonLinearConvBlock 内部已经有 BN 和 Clamp，所以这里不需要

        # 3. 线性 1x1 升维
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # Shortcut
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2_nonlinear(out)
        out = self.bn3(self.conv3(out))

        out += identity # 电压相加
        # 同样，末尾没有激活函数
        return out


# 【修改后的主网络】
class HybridResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64

        # 标准 conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 混合架构
        # 我们可以选择在哪一层使用非线性模块
        # 这里演示：layer1用标准，layer2,3,4用非线性瓶颈
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终的 in_planes 是 256 * expansion
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def HybridResNet_Bottleneck():
    # 类似 ResNet-50 的 [3, 4, 6, 3] 结构
    return HybridResNet(HybridNonLinearBottleneck, [3, 4, 6, 3])

# ==============================================================================
# 4. 训练和评估 (保持不变)
# ==============================================================================
# ... (train_one_epoch, evaluate 函数同前，为简洁省略) ...
def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, epoch, total_epochs, warmup_epochs, clip_norm):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Train]")
    
    for i, (inputs, labels) in enumerate(loader_tqdm):
        inputs, labels = inputs.to(device), labels.to(device)

        if epoch < warmup_epochs:
            lr = LEARNING_RATE * (epoch * len(loader) + i) / (warmup_epochs * len(loader))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loader_tqdm.set_postfix(
            loss=running_loss/(i+1), 
            acc=100.*correct/total,
            lr=optimizer.param_groups[0]['lr']
        )
    
    if epoch >= warmup_epochs:
        scheduler.step()

    return running_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device, epoch, total_epochs):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loader_tqdm = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [Test]")
    
    with torch.no_grad():
        for inputs, labels in loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loader_tqdm.set_postfix(
                loss=running_loss/len(loader), 
                acc=100.*correct/total
            )

    return running_loss / len(loader), 100. * correct / total
# ==============================================================================
# 5. 主程序 (修改)
# ==============================================================================

if __name__ == "__main__":
    
    # --- 超参数 ---
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    BATCH_SIZE = 64 # 【建议】鉴于模型更复杂，先从 64 开始尝试
    LEARNING_RATE = 1e-4 
    EPOCHS = 100
    WARMUP_EPOCHS = 10
    CLIP_NORM = 1.0 
    
    # --- 数据加载 (CIFAR-10) ---
    print("Preparing CIFAR-10 dataset...")
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 2.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- 模型、损失、优化器 ---
    print("Building HybridResNet_Bottleneck...")
    
    # 【修改】调用新的 Bottleneck 模型
    model = HybridResNet_Bottleneck().to(DEVICE)
    # print(model) # 取消注释以查看模型结构
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)

    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, trainloader, criterion, optimizer, scaler, scheduler, DEVICE, 
            epoch, EPOCHS, WARMUP_EPOCHS, CLIP_NORM
        )
        test_loss, test_acc = evaluate(
            model, testloader, criterion, DEVICE, epoch, EPOCHS
        )
        
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print("-" * 30)

    print("Training finished.")