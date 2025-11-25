import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

# ==========================================
# 1. 全局配置 (Global Configuration)
# ==========================================
class Config:
    # --- 物理模型参数 ---
    VT = 0.025          # 热电压 (V) at 300K
    N_FACTOR = 1.5      # 亚阈值摆幅因子
    VD = 0.5            # 漏极电压 (V)
    ALPHA = 2e-6        # 工艺因子 (A/V^2)
    
    # TIA (跨阻放大器) 基础增益。
    # 之前是 200,000 (200kOhm)，现在我们将通过 fan-in 动态调整它
    BASE_TIA = 200000.0 
    
    # --- 训练配置 ---
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    
    # 指定 GPU 设备
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    PRETRAIN_PATH = '../best_linear_resnet8.pth' # 预训练的线性模型路径
    SAVE_PATH = 'resnet8_optimized_double_best.pth'
    
    # 训练超参数
    PRETRAIN_EPOCHS = 15    # 如果没找到预训练权重，先跑多少轮
    HYBRID_EPOCHS = 50      # 混合架构微调轮数
    MONITOR_WINDOW = 20     # 只在最后 20 轮保存最佳模型
    
    # 学习率设置
    LR_EKV = 0.002          # EKV 阈值 (Theta) 的学习率，必须低
    LR_LINEAR = 0.005       # 线性层学习率

cfg = Config()
print(f"Running on device: {cfg.DEVICE}")

# ==========================================
# 2. 优化的 EKV 核心组件 (Physics Layers)
# ==========================================

class AdaptiveVoltageMapper(nn.Module):
    """
    自适应电压映射层
    功能：将上一层的无量纲输出映射到下一层器件的有效工作电压区 (0V - 9V)
    改进：支持自定义初始 Scale 和 Bias，以适应不同层级的数据分布需求
    """
    def __init__(self, init_scale=1.5, init_bias=3.5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, x):
        # 线性变换：y = ax + b
        x_mapped = x * self.scale + self.bias
        # 物理硬截断：保护栅极，防止数值溢出
        return torch.clamp(x_mapped, 0.0, 9.0)

class NormalizedEKVConv2d(nn.Module):
    """
    归一化的 EKV 非线性卷积层
    改进：引入动态 TIA Gain，解决求和导致的数值爆炸问题
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 物理参数 Theta (阈值电压 Vth)
        # 初始化在 4.0V 左右，处于强反型区边缘
        self.theta = nn.Parameter(torch.normal(4.0, 0.8, size=(out_channels, in_channels, kernel_size, kernel_size)))
        
        # --- 关键改进：Fan-in Normalization ---
        # 计算感受野内的输入数量 (Fan-in)
        fan_in = in_channels * kernel_size * kernel_size
        
        # 动态调整 TIA 增益： Gain = Base / sqrt(Fan_in)
        # 这类似于 Xavier Initialization 的思路，保持方差稳定
        # 如果 Fan-in = 576 (64*3*3), sqrt(576)=24, Gain 约变为 8333 Ohm
        self.current_tia_gain = cfg.BASE_TIA / (fan_in ** 0.5)
        
        # 注册为 buffer，这样它会被保存到 state_dict 但不会被 optimizer 更新
        self.register_buffer('tia_gain', torch.tensor(self.current_tia_gain))

    def forward(self, x):
        # x: [N, C, H, W]
        N, C, H, W = x.shape
        
        # 1. Unfold (提取滑动窗口): [N, Cin*K*K, L]
        x_unf = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        # 2. 维度调整以支持广播计算
        # x_in: [N, 1, Cin*K*K, L]
        # theta: [1, Cout, Cin*K*K, 1]
        x_in = x_unf.unsqueeze(1) 
        w_th = self.theta.view(1, self.out_channels, -1, 1)
        
        # 3. EKV 物理公式计算 (Vectorized)
        phi = 2 * cfg.N_FACTOR * cfg.VT
        v_diff = x_in - w_th
        
        # 使用 Softplus 近似 exp，防止溢出
        # I_f = (ln(1 + e^((Vg-Vth)/phi)))^2
        f_term = F.softplus(v_diff / phi).pow(2)
        # I_r = (ln(1 + e^((Vg-Vth-Vd)/phi)))^2
        r_term = F.softplus((v_diff - cfg.VD) / phi).pow(2)
        
        # 突触电流
        i_syn = cfg.ALPHA * (f_term - r_term)
        
        # 4. 电流求和 (Kirchhoff's Current Law)
        out_current = torch.sum(i_syn, dim=2) # [N, Cout, L]
        
        # 5. TIA 转换 (Current -> Voltage)
        # 使用动态计算的增益，防止输出电压达到几百伏
        out_voltage = out_current * self.tia_gain
        
        # 6. Reshape 回图像尺寸
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        return out_voltage.view(N, self.out_channels, h_out, w_out)

# ==========================================
# 3. 网络架构 (Optimized Double EKV Block)
# ==========================================

class BasicBlock(nn.Module):
    """标准线性残差块 (用于 Layer 1, 2)"""
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

class OptimizedDoubleEKVBlock(nn.Module):
    """
    优化的双层全非线性块 (用于 Layer 3)
    特点：针对两层不同的输入分布，使用了不同的 Mapper 初始化策略
    """
    def __init__(self, in_planes, planes, stride=1):
        super(OptimizedDoubleEKVBlock, self).__init__()
        
        # --- Stage 1: 特征提取 ---
        # 输入来自上一层的 ReLU，只有正值。
        # Mapper1: 标准 Bias (3.5V), 标准 Scale (1.5)
        self.mapper1 = AdaptiveVoltageMapper(init_scale=1.5, init_bias=3.5)
        self.ekv1 = NormalizedEKVConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # --- Stage 2: 特征整合 ---
        # 输入来自 BN1，是归一化的高斯分布 (均值0, 方差1)，有正有负。
        # Mapper2 策略：
        # 1. init_scale=0.8 (压缩): 防止数据摆幅过大进入线性区。
        # 2. init_bias=3.0 (降低): 抵消 BN 可能产生的正向偏移，确保更多数据落在指数/平方过渡区。
        self.mapper2 = AdaptiveVoltageMapper(init_scale=0.8, init_bias=3.0) 
        self.ekv2 = NormalizedEKVConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # 残差路径 (保持线性)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        # 第一级非线性卷积
        out = self.mapper1(x)
        out = self.ekv1(out)
        out = self.bn1(out)
        
        # 第二级非线性卷积 (级联)
        out = self.mapper2(out)
        out = self.ekv2(out)
        out = self.bn2(out)
        
        out += identity
        out = F.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, block_type='linear', num_classes=10):
        super(ResNet8, self).__init__()
        self.in_planes = 16
        
        # 头部
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer 1 & 2: 总是线性的 (我们将冻结它们)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        
        # Layer 3: 根据配置决定是否使用 EKV
        if block_type == 'optimized_double':
            print("   [Info] Layer 3 构建为: OptimizedDoubleEKVBlock")
            self.layer3 = self._make_layer(OptimizedDoubleEKVBlock, 64, 1, stride=2)
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
        out = self.layer3(out) # 关键层
        out = F.avg_pool2d(out, out.size()[2:])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==========================================
# 4. 训练引擎 (包含进度条与保存逻辑)
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
    
    # 进度条
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1}/{epochs}", unit="batch")
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if is_hybrid:
            # 1. 梯度截断：防止指数运算导致的梯度爆炸
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            # 2. 物理参数约束：钳位 Theta
            with torch.no_grad():
                for name, param in net.named_parameters():
                    if 'theta' in name:
                        param.clamp_(0.1, 9.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
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
# 5. 主程序流程
# ==========================================
def main():
    trainloader, testloader = get_dataloader()
    
    # --- Phase 1: 检查预训练权重 ---
    if not os.path.exists(cfg.PRETRAIN_PATH):
        print(f"\n[!] 未找到预训练权重 {cfg.PRETRAIN_PATH}。")
        print(f"    开始训练基础线性模型 ({cfg.PRETRAIN_EPOCHS} Epochs) 以获取特征提取器...")
        
        net_linear = ResNet8(block_type='linear').to(cfg.DEVICE)
        optimizer = optim.SGD(net_linear.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.PRETRAIN_EPOCHS)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(cfg.PRETRAIN_EPOCHS):
            train_one_epoch(net_linear, trainloader, optimizer, criterion, epoch, cfg.PRETRAIN_EPOCHS)
            scheduler.step()
        
        torch.save(net_linear.state_dict(), cfg.PRETRAIN_PATH)
        print(f"    >>> 基础模型已保存，准备开始混合微调。\n")
        del net_linear

    # --- Phase 2: 构建 Optimized Double EKV 模型 ---
    print("\n[Phase 2] 构建 Optimized Double EKV Model...")
    net_hybrid = ResNet8(block_type='optimized_double').to(cfg.DEVICE)
    
    # 加载权重 (排除 Layer3)
    print(f"    加载权重: {cfg.PRETRAIN_PATH}")
    state_dict = torch.load(cfg.PRETRAIN_PATH, map_location=cfg.DEVICE)
    state_dict = {k: v for k, v in state_dict.items() if 'layer3' not in k}
    net_hybrid.load_state_dict(state_dict, strict=False)
    
    # 冻结逻辑
    print("    冻结 Layer 1 & 2，仅训练 Layer 3 (EKV) 和 Linear...")
    trainable_params = 0
    for name, param in net_hybrid.named_parameters():
        param.requires_grad = False # 默认冻结
        if 'layer3' in name or 'linear' in name:
            param.requires_grad = True
            trainable_params += param.numel()
    print(f"    可训练参数量: {trainable_params}")

    # --- Phase 3: 混合训练 ---
    # 分组优化器
    ekv_params = [p for n, p in net_hybrid.named_parameters() if 'theta' in n and p.requires_grad]
    other_params = [p for n, p in net_hybrid.named_parameters() if 'theta' not in n and p.requires_grad]
    
    optimizer = optim.SGD([
        {'params': ekv_params, 'lr': cfg.LR_EKV},     # Theta 使用低学习率
        {'params': other_params, 'lr': cfg.LR_LINEAR} # Linear/Mapper 使用常规学习率
    ], momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.HYBRID_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n[Phase 3] 开始微调 ({cfg.HYBRID_EPOCHS} Epochs)...")
    acc_history = []
    best_acc = 0.0
    
    for epoch in range(cfg.HYBRID_EPOCHS):
        loss, train_acc = train_one_epoch(net_hybrid, trainloader, optimizer, criterion, epoch, cfg.HYBRID_EPOCHS, is_hybrid=True)
        val_acc = evaluate(net_hybrid, testloader)
        scheduler.step()
        acc_history.append(val_acc)
        
        tqdm.write(f"      Validation Acc: {val_acc:.2f}%")
        
        # --- 逻辑: 保存最后20轮中的最佳模型 ---
        if epoch >= (cfg.HYBRID_EPOCHS - cfg.MONITOR_WINDOW):
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net_hybrid.state_dict(), cfg.SAVE_PATH)
                tqdm.write(f"      [Save] New Best Model Saved! (Acc: {best_acc:.2f}%)")

    print(f"\n训练结束。最佳精度 (Last {cfg.MONITOR_WINDOW} epochs): {best_acc:.2f}%")
    
    # --- Phase 4: 可视化结果 ---
    plt.figure(figsize=(10, 6))
    plt.plot(acc_history, label='Optimized Double EKV Acc', linewidth=2)
    plt.axvline(x=cfg.HYBRID_EPOCHS - cfg.MONITOR_WINDOW, color='r', linestyle='--', label='Monitor Start')
    plt.axhline(y=best_acc, color='g', linestyle=':', label=f'Best: {best_acc:.2f}%')
    plt.title('Optimized Double EKV Training Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('resnet_hybrid4_2_result.png')
    print("准确率曲线已保存至 resnet_hybrid4_2_result.png")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断。")