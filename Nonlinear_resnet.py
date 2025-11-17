import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# --- 1. EKV 模型与超参数 ---

# EKV 模型物理参数 (来自您的描述)
EKV_ALPHA = 0.0005625
EKV_N = 1.5
EKV_VT = 0.025
EKV_VD = 0.1 # 固定的漏极电压 V_D

# 模拟超参数
R_TIA = 5e4  # 关键超参数! TIA的阻值。需要仔细调参。
INPUT_VOLTAGE_SCALE = 5.0 # 将 (0,1) 的输入图像缩放到 (0, 5V)
INIT_THETA_MEAN = 2.5     # Theta (V_th) 初始化均值，应与输入电压匹配
INIT_THETA_STD = 0.1      # Theta 初始化标准差

# 训练超参数
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
GRAD_CLIP_MAX_NORM = 1.0 # 梯度裁剪，防止爆炸
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 核心模块: 非线性EKV卷积 ---

class NonLinearEKV_Conv2d(nn.Module):
    """
    模拟非线性EKV卷积层。
    输入: V_in (电压)
    输出: I_sum (电流)
    可训练参数: theta (阈值电压)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 可训练参数: 阈值电压 theta
        # 形状: (C_out, C_in, k, k)
        self.theta = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size
        ))
        
        # 关键: 初始化theta到V_in的预期工作点附近
        nn.init.normal_(self.theta, mean=INIT_THETA_MEAN, std=INIT_THETA_STD)
        
        # 注册EKV参数为 buffer (非参数，但随模块移动)
        self.register_buffer('alpha', torch.tensor(EKV_ALPHA))
        self.register_buffer('n', torch.tensor(EKV_N))
        self.register_buffer('VT', torch.tensor(EKV_VT))
        self.register_buffer('VD', torch.tensor(EKV_VD))

    def forward(self, V_in):
        # V_in 形状: (B, C_in, H_in, W_in)
        
        # 1. Unfold: 将 V_in 展开为 patches
        # V_patches 形状: (B, C_in * k * k, L)
        # L 是 patch 数量 (H_out * W_out)
        V_patches = F.unfold(V_in, self.kernel_size, stride=self.stride, padding=self.padding)
        
        # 2. Reshape 以便广播 (Broadcast)
        # V_patches -> (B, 1, L, C_in * k * k)
        V_patches = V_patches.permute(0, 2, 1).unsqueeze(1)
        
        # theta -> (C_out, C_in * k * k)
        theta_flat = self.theta.view(self.out_channels, -1)
        # theta_flat -> (1, C_out, 1, C_in * k * k)
        theta_flat = theta_flat.unsqueeze(0).unsqueeze(2)

        # 3. 应用 EKV 模型 (利用广播)
        # V_diff 形状: (B, C_out, L, C_in * k * k)
        V_diff = V_patches - theta_flat
        
        term1_in = V_diff / (2 * self.n * self.VT)
        term2_in = (V_diff - self.VD) / (2 * self.n * self.VT)

        # F.softplus(x) = ln(1 + exp(x))，用于数值稳定性
        term1 = F.softplus(term1_in).pow(2)
        term2 = F.softplus(term2_in).pow(2)
        
        # I_D_individual: 每个FGT的电流
        # 形状: (B, C_out, L, C_in * k * k)
        I_D_individual = self.alpha * (term1 - term2)

        # 4. 模拟 KCL: 在 patch 维度上求和 (sum over j)
        # I_sum 形状: (B, C_out, L)
        I_sum = I_D_individual.sum(dim=3)

        # 5. Fold: 将输出 reshape 回图像形状
        H_out = (V_in.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (V_in.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # I_out_img 形状: (B, C_out, H_out, W_out)
        I_out_img = I_sum.view(V_in.shape[0], self.out_channels, H_out, W_out)
        
        return I_out_img # 输出是电流

class FixedTIA(nn.Module):
    """
    模拟固定的跨阻放大器 (TIA)。
    输入: I_in (电流)
    输出: V_out (电压)
    """
    def __init__(self, R_TIA):
        super().__init__()
        self.register_buffer('R_TIA', torch.tensor(R_TIA))

    def forward(self, I_in):
        return I_in * self.R_TIA

# --- 3. 核心架构: 电流域 ResNet 块 ---

class NonLinearResBlock(nn.Module):
    """
    电流域求和的非线性 ResNet 块 (方案B)。
    输入: V_in (电压)
    输出: V_out (电压)
    """
    def __init__(self, in_channels, out_channels, stride=1, R_TIA=R_TIA):
        super().__init__()
        
        # F(x) 路径
        self.conv1 = NonLinearEKV_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.tia1 = FixedTIA(R_TIA)
        self.bn1 = nn.BatchNorm2d(out_channels) # 关键的 "工作点控制器"
        
        self.conv2 = NonLinearEKV_Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        # x (Shortcut) 路径
        # 我们总是使用 1x1 卷积，这在物理上更合理 (V_in -> I_x)
        self.shortcut_conv = NonLinearEKV_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        
        # KCL 之后的公共 TIA 和 BN
        self.final_tia = FixedTIA(R_TIA)
        self.final_bn = nn.BatchNorm2d(out_channels) # 输出 V_out

    def forward(self, V_in):
        # F(x) 路径
        I_1 = self.conv1(V_in)
        V_1 = self.bn1(self.tia1(I_1))
        I_F_x = self.conv2(V_1) # F(x) 的输出电流
        
        # x 路径
        I_x = self.shortcut_conv(V_in) # shortcut 的输出电流
        
        # 电流域求和 (KCL)
        I_sum = I_F_x + I_x
        
        # 公共的 TIA 和 BN
        V_out = self.final_bn(self.final_tia(I_sum))
        
        return V_out

# --- 4. 完整模型: EKV-ResNet ---

class EKV_ResNet(nn.Module):
    def __init__(self, block, n_blocks_per_layer, R_TIA=R_TIA):
        super().__init__()
        self.R_TIA = R_TIA
        self.in_channels = 16 # ResNet-20 for CIFAR-10 通常以 16 开始
        
        # Stem (输入层)
        self.conv1 = NonLinearEKV_Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.tia1 = FixedTIA(R_TIA)
        self.bn1 = nn.BatchNorm2d(16)
        
        # 3 个 ResNet stage
        self.layer1 = self._make_layer(block, 16, n_blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(block, 32, n_blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(block, 64, n_blocks_per_layer[2], stride=2)
        
        # Head (数字分类层)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def _make_layer(self, block, out_channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, self.R_TIA))
            self.in_channels = out_channels # 更新下一块的 in_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # x 是 (0, 1) 的图像
        # 1. 缩放到 (0, 5V) 的输入电压
        V_in = x * INPUT_VOLTAGE_SCALE
        
        # 2. Stem
        I_1 = self.conv1(V_in)
        V_1 = self.bn1(self.tia1(I_1))
        
        # 3. ResNet Layers
        V_2 = self.layer1(V_1)
        V_3 = self.layer2(V_2)
        V_4 = self.layer3(V_3)
        
        # 4. Digital Head
        # V_4 是最后一层的输出电压
        out = self.avgpool(V_4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

# --- 5. 数据加载 (CIFAR-10) ---

print("... 正在加载 CIFAR-10 数据 ...")
# CIFAR-10 的均值和标准差
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # 注意: 我们没有使用 (0.5, 0.5, 0.5) 是因为我们的模型 stem
    # 接收的是 (0,1) -> (0, 5V) 的电压。
    # 标准化到 (-1, 1) 可能会导致负电压，这在EKV模型中可能不希望。
    # 让我们重新审视一下：
])

# 让我们使用 (0,1) 的 ToTensor，然后在模型 forward 中缩放
# 这更符合 V_G > 0 的物理直觉
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor() # 转换为 (0, 1)
])
transform_test = transforms.Compose([
    transforms.ToTensor() # 转换为 (0, 1)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# --- 6. 训练循环 ---

# 我们使用一个类似 ResNet-20 的结构 (n=3)
# [3, 3, 3] -> 1 (stem) + 3*3*2 (conv) + 3*3 (sc) = 1 + 18 + 9 = 28 EKV层
# 这会非常非常慢且耗显存。
# 我们从 [2, 2, 2] 开始 (ResNet-14-like)
# [2, 2, 2] -> 1 (stem) + (2*2 + 2*2 + 2*2) (conv) + (2+2+2) (sc) = 1 + 12 + 6 = 19 EKV层
# 让我们从 [1, 1, 1] 开始 (ResNet-8-like) 以便快速验证
# [1, 1, 1] -> 1 + (2+2+2) + (1+1+1) = 10 EKV 层

print(f"... 正在初始化模型 EKV-ResNet([1, 1, 1]) ...")
print(f"训练设备: {DEVICE}")
print(f"关键超参数 R_TIA: {R_TIA}")

model = EKV_ResNet(NonLinearResBlock, n_blocks_per_layer=[1, 1, 1], R_TIA=R_TIA).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 存储历史记录
history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

def train_model(model, trainloader, testloader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, data in enumerate(train_bar):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # --- 显存警告 ---
            # 如果显存爆炸 (OOM)，请减小 BATCH_SIZE
            # 或者使用 [1, 1, 1] 而不是 [2, 2, 2]
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"RuntimeError (可能OOM): {e}")
                print("请尝试减小 BATCH_SIZE")
                return None # 提前终止
            
            # 关键: 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 49:
                train_bar.set_postfix(loss=f"{running_loss / (i+1):.3f}", acc=f"{100 * correct / total:.2f}%")
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- 验证 ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_bar = tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
        with torch.no_grad():
            for data in test_bar:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                test_bar.set_postfix(loss=f"{test_loss / (len(testloader)):.3f}", acc=f"{100 * correct / total:.2f}%")

        test_loss = test_loss / len(testloader)
        test_acc = 100 * correct / total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print("... 训练完成 ...")
    return history

# 运行训练
history = train_model(model, trainloader, testloader, criterion, optimizer, EPOCHS, DEVICE)

# --- 7. 绘图 ---
if history:
    print("... 正在绘制结果 ...")
    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("训练未成功完成，无法绘制图像。")