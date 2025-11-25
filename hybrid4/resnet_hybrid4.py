import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm  # 引入进度条库

# ==========================================
# 1. 全局配置与物理参数
# ==========================================
class Config:
    # 硬件物理参数
    VT = 0.025          # 热电压 (V)
    N_FACTOR = 1.5      # 亚阈值摆幅因子
    VD = 0.5            # 漏极电压
    ALPHA = 2e-6        # 工艺因子
    TIA_GAIN = 200000.0 # 跨阻增益
    
    # 训练配置
    BATCH_SIZE = 128
    NUM_WORKERS = 4     # 适当增加 worker 提高数据加载速度
    
    # 指定 CUDA:1
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    PRETRAIN_PATH = '../best_linear_resnet8.pth'
    
    # 训练轮数
    PRETRAIN_EPOCHS = 15    
    HYBRID_EPOCHS = 50      
    
    # 学习率
    LR_EKV = 0.002          
    LR_LINEAR = 0.005       
    MONITOR_WINDOW = 20     

cfg = Config()
print(f"Using Device: {cfg.DEVICE}")

# ==========================================
# 2. EKV 非线性核心算子 (保持高效向量化)
# ==========================================
class VoltageMapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化映射参数
        self.scale = nn.Parameter(torch.tensor(1.5)) 
        self.bias = nn.Parameter(torch.tensor(3.5)) 

    def forward(self, x):
        x_mapped = x * self.scale + self.bias
        return torch.clamp(x_mapped, 0.0, 9.0)

class NonLinearEKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 阈值参数初始化
        self.theta = nn.Parameter(torch.normal(4.0, 0.8, size=(out_channels, in_channels, kernel_size, kernel_size)))
        
    def forward(self, x):
        N, C, H, W = x.shape
        x_unf = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        # 广播机制: [N, 1, Cin*K*K, L] - [1, Cout, Cin*K*K, 1]
        x_in = x_unf.unsqueeze(1) 
        w_th = self.theta.view(1, self.out_channels, -1, 1)
        
        phi = 2 * cfg.N_FACTOR * cfg.VT
        v_diff = x_in - w_th
        
        # EKV 公式 (Softplus 近似)
        f_term = F.softplus(v_diff / phi).pow(2)
        r_term = F.softplus((v_diff - cfg.VD) / phi).pow(2)
        
        i_syn = cfg.ALPHA * (f_term - r_term)
        out_current = torch.sum(i_syn, dim=2) 
        
        out_voltage = out_current * cfg.TIA_GAIN
        
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        return out_voltage.view(N, self.out_channels, h_out, w_out)

# ==========================================
# 3. 网络架构 (Standard & Hybrid)
# ==========================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class HybridBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(HybridBlock, self).__init__()
        self.mapper = VoltageMapper()
        self.ekv_conv = NonLinearEKVConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        # EKV Part
        v_in = self.mapper(x)
        out = self.ekv_conv(v_in)
        out = self.bn1(out)
        # Linear Part
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, block_type='linear', num_classes=10):
        super(ResNet8, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        
        # Layer 3 切换逻辑
        if block_type == 'hybrid':
            self.layer3 = self._make_layer(HybridBlock, 64, 1, stride=2)
        else:
            self.layer3 = self._make_layer(BasicBlock, 64, 1, stride=2)
            
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==========================================
# 4. 训练引擎 (带进度条)
# ==========================================
def get_dataloader():
    print(">>> Loading CIFAR10 Data...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=cfg.NUM_WORKERS)
    return trainloader, testloader

def train_one_epoch(net, loader, optimizer, criterion, epoch_idx, epochs, is_hybrid=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    # 使用 tqdm 包装 loader
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{epochs}", unit="batch")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if is_hybrid:
            # 物理约束
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if 'theta' in name:
                        param.clamp_(0.1, 9.0)
        
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条后缀
        current_acc = 100.*correct/total
        pbar.set_postfix({"Loss": f"{train_loss/(total/cfg.BATCH_SIZE):.3f}", "Acc": f"{current_acc:.2f}%"})
        
    return train_loss/len(loader), 100.*correct/total

def evaluate(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.*correct/total

# ==========================================
# 5. 主流程
# ==========================================
def main():
    trainloader, testloader = get_dataloader()
    
    # --- Phase 1: 检查权重 ---
    if not os.path.exists(cfg.PRETRAIN_PATH):
        print(f"\n[!] 未找到 {cfg.PRETRAIN_PATH}。开始训练基础模型...")
        net_linear = ResNet8(block_type='linear').to(cfg.DEVICE)
        optimizer = optim.SGD(net_linear.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PRETRAIN_EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(cfg.PRETRAIN_EPOCHS):
            train_one_epoch(net_linear, trainloader, optimizer, criterion, epoch, cfg.PRETRAIN_EPOCHS)
            scheduler.step()
        
        torch.save(net_linear.state_dict(), cfg.PRETRAIN_PATH)
        print(f"   >>> 基础模型已保存至 {cfg.PRETRAIN_PATH}")
        del net_linear
    else:
        print(f"\n[+] 检测到预训练权重: {cfg.PRETRAIN_PATH}")

    # --- Phase 2: 加载并构建混合模型 ---
    print("\n[Phase 2] 构建 Hybrid EKV 模型 (CUDA:1)...")
    net_hybrid = ResNet8(block_type='hybrid').to(cfg.DEVICE)
    
    # 加载权重 (过滤掉 layer3)
    state_dict = torch.load(cfg.PRETRAIN_PATH, map_location=cfg.DEVICE)
    state_dict = {k: v for k, v in state_dict.items() if 'layer3' not in k}
    net_hybrid.load_state_dict(state_dict, strict=False)
    
    # 冻结逻辑
    print("   >>> 冻结参数 (除 Layer3 和 Linear 层外)...")
    for name, param in net_hybrid.named_parameters():
        param.requires_grad = False
        if 'layer3' in name or 'linear' in name:
            param.requires_grad = True

    # --- Phase 3: 混合微调 ---
    ekv_params = [p for n, p in net_hybrid.named_parameters() if 'theta' in n and p.requires_grad]
    other_params = [p for n, p in net_hybrid.named_parameters() if 'theta' not in n and p.requires_grad]
    
    optimizer = optim.SGD([
        {'params': ekv_params, 'lr': cfg.LR_EKV},
        {'params': other_params, 'lr': cfg.LR_LINEAR}
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.HYBRID_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n[Phase 3] 开始混合训练 ({cfg.HYBRID_EPOCHS} Epochs)...")
    acc_history = []
    best_acc = 0.0
    
    for epoch in range(cfg.HYBRID_EPOCHS):
        loss, train_acc = train_one_epoch(net_hybrid, trainloader, optimizer, criterion, epoch, cfg.HYBRID_EPOCHS, is_hybrid=True)
        val_acc = evaluate(net_hybrid, testloader)
        scheduler.step()
        acc_history.append(val_acc)
        
        # 进度条后面打印验证结果
        tqdm.write(f"      Validation Acc: {val_acc:.2f}%")

        # 保存最佳
        if epoch >= (cfg.HYBRID_EPOCHS - cfg.MONITOR_WINDOW):
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net_hybrid.state_dict(), 'resnet8_hybrid_best.pth')

    print(f"\n训练结束。最佳精度: {best_acc:.2f}%")
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(acc_history, label='Hybrid Test Acc')
    plt.axvline(x=cfg.HYBRID_EPOCHS - cfg.MONITOR_WINDOW, color='r', linestyle='--', label='Monitor Window')
    plt.title('Hybrid EKV Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('hybrid_progress.png')
    print("曲线图已保存至 hybrid_progress.png")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断。")