import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from tqdm import tqdm

# ==========================================
# 0. 全局路径配置
# ==========================================
class Paths:
    DATA_ROOT = './data'                
    PRETRAIN_MODEL = '../best_linear_resnet8.pth'
    BEST_HYBRID_MODEL = './5-6-1best_hybrid.pth' 
    PLOT_IMG = './5-6-resnet_hybrid_virtual_bn.png' # V7.0 命名

# ==========================================
# 1. 全局物理/训练配置
# ==========================================
class Config:
    # --- 物理参数 ---
    VT = 0.026          
    N_FACTOR = 1.5      
    VD = 0.5            
    
    # [运放饱和] 依然保留，作为最后的物理安全底线
    OPAMP_V_SAT = 6.0   
    
    # --- 泛化增强 ---
    INPUT_NOISE_STD = 0.1     
    WEIGHT_NOISE_STD = 0.05   
    DROPOUT_RATE = 0.1        
    
    # 电流参数
    ALPHA = 10e-6       
    
    # TIA 基础增益
    TIA_GAIN_BASE = 100.0     
    
    # 训练参数
    BATCH_SIZE = 128
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    HYBRID_EPOCHS = 50      
    
    # 学习率配置 (BN 加入后，可以适当提高 LR)
    LR_EKV = 0.005          
    LR_LINEAR = 0.01
    LR_MAPPER = 0.01    
    LR_TIA = 0.01       

cfg = Config()
print(f"Using Device: {cfg.DEVICE}")

# ==========================================
# 2. 核心组件：物理等效层
# ==========================================

class VoltageMapper(nn.Module):
    """
    功能：电平移位器 (Level Shifter)
    改进：仅负责直流偏置 (Bias)，不再负责缩放 (Scale由前面的 TIA/BN 承担)
    """
    def __init__(self, num_channels, target_mean=3.5): 
        super().__init__()
        # 形状 (1, C, 1, 1) 实现通道级控制
        self.bias = nn.Parameter(torch.ones(1, num_channels, 1, 1) * target_mean) 

    def forward(self, x):
        # 仅做加法，模拟 Bias Tee
        x_mapped = x + self.bias
        # 物理供电轨限制 (0V - 8V)
        return torch.clamp(x_mapped, 0.0, 8.0)

class DifferentialEKVConv2d(nn.Module):
    """
    功能：差分 EKV 卷积 + 可学习 TIA + 虚拟 BN
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # [物理权重] Vth 初始化在 2.5V (强反型区)
        self.theta_pos = nn.Parameter(torch.normal(2.5, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
        self.theta_neg = nn.Parameter(torch.normal(2.5, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
        
        # [改进点] 通道级 TIA 增益 (Per-Channel Gain)
        # 形状 (1, C, 1, 1)，每个输出通道拥有独立的硬件放大倍数
        # 动态初始化：根据扇入 (Fan-in) 调整初始增益，保持方差稳定
        scaling_factor = math.sqrt(16 / in_channels)
        init_gain = cfg.TIA_GAIN_BASE * scaling_factor
        # 使用 log 空间参数化保证正值
        self.log_tia_gain = nn.Parameter(torch.ones(1, out_channels, 1, 1) * np.log(init_gain))
        
        # [关键突破] 物理虚拟 BN (Physics-Aware Batch Norm)
        # 这里的 BN 参数 (gamma, beta, mean, var) 在部署时可以数学等价地融合到
        # TIA 增益 (乘以 gamma/sigma) 和 Mapper 偏置 (加 beta - mean*gamma/sigma) 中。
        # 所以它不违反物理约束！
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.phi = 2 * cfg.N_FACTOR * cfg.VT

    def ekv_current(self, v_gs, v_th):
        v_ov_f = (v_gs - v_th) / self.phi
        i_f = F.softplus(v_ov_f).pow(2)
        v_ov_r = (v_gs - v_th - cfg.VD) / self.phi
        i_r = F.softplus(v_ov_r).pow(2)
        current = cfg.ALPHA * (i_f - i_r)
        return current

    def forward(self, x):
        N, C, H, W = x.shape
        x_unf = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_in = x_unf.unsqueeze(1) 
        
        # 权重噪声注入
        if self.training:
            theta_pos_noise = self.theta_pos + torch.randn_like(self.theta_pos) * cfg.WEIGHT_NOISE_STD
            theta_neg_noise = self.theta_neg + torch.randn_like(self.theta_neg) * cfg.WEIGHT_NOISE_STD
        else:
            theta_pos_noise = self.theta_pos
            theta_neg_noise = self.theta_neg

        theta_pos_exp = theta_pos_noise.view(1, self.out_channels, -1, 1)
        theta_neg_exp = theta_neg_noise.view(1, self.out_channels, -1, 1)
        
        # 1. 模拟域：电流计算
        i_pos = self.ekv_current(x_in, theta_pos_exp)
        i_neg = self.ekv_current(x_in, theta_neg_exp)
        
        # 2. 模拟域：KCL 电流汇聚
        sum_i_pos = torch.sum(i_pos, dim=2) 
        sum_i_neg = torch.sum(i_neg, dim=2)
        i_diff = sum_i_pos - sum_i_neg
        
        # 3. 模拟域：TIA 转电压 (通道级增益)
        # 移除 Clamp，给予网络完全自由度，通过后面的 BN/Tanh 来约束
        current_gain = torch.exp(self.log_tia_gain) 
        out_voltage_raw = i_diff * current_gain
        
        # Fold 回图像尺寸
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_voltage_map = out_voltage_raw.view(N, self.out_channels, h_out, w_out)
        
        # 4. 数字域/模拟校准：虚拟 BN
        # 这一步将信号强行拉回标准正态分布，解决饱和与梯度消失
        # 部署时：这一步消失，参数被融合进 TIA 和 Mapper
        out_norm = self.bn(out_voltage_map)
        
        # 5. 模拟域：运放饱和 (Tanh)
        # 限制最终输出幅度，防止下一级被击穿
        out_final = torch.tanh(out_norm / cfg.OPAMP_V_SAT) * cfg.OPAMP_V_SAT
        
        return out_final

# ==========================================
# 4. 网络结构构建
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

class DoubleEKVBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(DoubleEKVBlock, self).__init__()
        
        # Stage 1
        self.mapper1 = VoltageMapper(num_channels=in_planes, target_mean=3.5) 
        self.ekv1 = DifferentialEKVConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.dropout1 = nn.Dropout2d(p=cfg.DROPOUT_RATE)
        
        # Stage 2
        self.mapper2 = VoltageMapper(num_channels=planes, target_mean=3.5)
        self.ekv2 = DifferentialEKVConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(p=cfg.DROPOUT_RATE)

        # Residual Path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
        # [技巧] Zero-Init: 将 EKV2 的 BN 最后一个参数初始化为 0
        # 这使得初始状态下 Residual Block 近似恒等映射 (Identity)
        # 极大地帮助梯度流向浅层
        nn.init.constant_(self.ekv2.bn.weight, 0)

    def forward(self, x):
        identity = self.shortcut(x)
        
        if self.training:
            x = x + torch.randn_like(x) * cfg.INPUT_NOISE_STD
            
        # Stage 1
        out = self.mapper1(x)       
        out = self.ekv1(out)
        out = self.dropout1(out)
        
        # Stage 2
        out = self.mapper2(out)     
        out = self.ekv2(out)
        out = self.dropout2(out)
        
        # Residual Addition
        out += identity
        out = F.relu(out)           
        return out

class ResNet8(nn.Module):
    def __init__(self, block_type='linear'):
        super(ResNet8, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        
        if block_type == 'hybrid':
            self.layer3 = self._make_layer(DoubleEKVBlock, 64, 1, stride=2)
        else:
            self.layer3 = self._make_layer(BasicBlock, 64, 1, stride=2)
            
        self.linear = nn.Linear(64, 10)

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
# 5. 训练与工具函数
# ==========================================
def train(net, loader, optimizer, criterion, epoch):
    net.train()
    train_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{cfg.HYBRID_EPOCHS}', unit='batch')
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)
        optimizer.step()
        
        with torch.no_grad():
            for name, param in net.named_parameters():
                if 'theta' in name:
                    param.clamp_(0.5, 8.0)
                    
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        current_acc = 100. * correct / total
        pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{current_acc:.2f}%'})
        
    return train_loss/len(loader), 100.*correct/total

def test(net, loader):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(cfg.DEVICE), targets.to(cfg.DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.*correct/total

def main():
    if not os.path.exists(Paths.DATA_ROOT):
        os.makedirs(Paths.DATA_ROOT)
        
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=Paths.DATA_ROOT, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root=Paths.DATA_ROOT, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    print("\n>>> Phase 2: Hybrid Training (Replacing Layer3 with Diff EKV)...")
    if not os.path.exists(Paths.PRETRAIN_MODEL):
        raise FileNotFoundError(f"未找到预训练模型: {Paths.PRETRAIN_MODEL}")
        
    net_hybrid = ResNet8('hybrid').to(cfg.DEVICE)
    state = torch.load(Paths.PRETRAIN_MODEL, map_location=cfg.DEVICE)
    state = {k:v for k,v in state.items() if 'layer3' not in k}
    net_hybrid.load_state_dict(state, strict=False)
    
    for name, p in net_hybrid.named_parameters():
        if 'layer3' not in name and 'linear' not in name:
            p.requires_grad = False
    
    ekv_params = [p for n,p in net_hybrid.named_parameters() if 'theta' in n]
    # 物理校准参数 (Mapper, TIA) 以及 BN 参数 都不使用 decay
    calibration_params = [p for n,p in net_hybrid.named_parameters() if 'mapper' in n or 'tia_gain' in n or 'bn' in n]
    other_params = [p for n,p in net_hybrid.named_parameters() 
                    if 'theta' not in n and 'mapper' not in n and 'tia_gain' not in n and 'bn' not in n and p.requires_grad]
    
    optimizer = optim.SGD([
        {'params': ekv_params, 'lr': cfg.LR_EKV, 'weight_decay': 5e-4},
        {'params': calibration_params, 'lr': 0.02, 'weight_decay': 0.0},
        {'params': other_params, 'lr': cfg.LR_LINEAR, 'weight_decay': 5e-4}
    ], momentum=0.9)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.HYBRID_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    train_acc_history = []
    test_acc_history = []
    best_acc = 0.0
    
    for ep in range(cfg.HYBRID_EPOCHS):
        loss, tr_acc = train(net_hybrid, trainloader, optimizer, criterion, ep)
        te_acc = test(net_hybrid, testloader)
        scheduler.step()
        
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(net_hybrid.state_dict(), Paths.BEST_HYBRID_MODEL)
            
        train_acc_history.append(tr_acc)
        test_acc_history.append(te_acc)
        
        print(f"   Epoch {ep} Summary: Train Acc {tr_acc:.2f}% | Test Acc {te_acc:.2f}% (Best: {best_acc:.2f}%)")
        
    plt.figure(figsize=(10, 6))
    plt.plot(train_acc_history, label='Train Accuracy', linestyle='--', alpha=0.7)
    plt.plot(test_acc_history, label='Validation Accuracy', linewidth=2)
    plt.title("Hybrid ResNet (Physics-Aware BN + Zero-Init)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(Paths.PLOT_IMG)
    print(f"Done. Best Validation Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()