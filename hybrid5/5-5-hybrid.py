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
    BEST_HYBRID_MODEL = './4-1-1best_hybrid.pth' 
    PLOT_IMG = './6-4-resnet_hybrid_opamp.png' # 更新图片名

# ==========================================
# 1. 全局物理/训练配置
# ==========================================
class Config:
    # --- 物理参数 ---
    VT = 0.026          
    N_FACTOR = 1.5      
    VD = 0.5            
    
    # [V6.0 新增] 运放饱和电压 (OpAmp Saturation Voltage)
    # 模拟真实硬件中 TIA 的输出摆幅限制。
    # 这不仅防止了信号过载，还引入了类似 Tanh 的非线性，提升特征提取能力。
    OPAMP_V_SAT = 6.0   
    
    # --- 泛化增强 ---
    INPUT_NOISE_STD = 0.1     
    WEIGHT_NOISE_STD = 0.05   
    DROPOUT_RATE = 0.1        
    
    # 电流参数
    ALPHA = 10e-6       
    
    # 基础 TIA 增益
    TIA_GAIN_BASE = 200.0     
    
    # 训练参数
    BATCH_SIZE = 128
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    HYBRID_EPOCHS = 50      
    
    # 学习率配置
    LR_EKV = 0.002          
    LR_LINEAR = 0.005
    LR_MAPPER = 0.02    
    LR_TIA = 0.02       

cfg = Config()
print(f"Using Device: {cfg.DEVICE}")

# ==========================================
# 2. 关键修改 I: 电压映射器 (VoltageMapper)
# 目的：将 TIA 输出的差分电压 (有正有负) 搬移到下一级 Flash 的线性区
# ==========================================
class VoltageMapper(nn.Module):
    def __init__(self, num_channels, target_mean=3.5): 
        # 偏置设定为 3.5V，配合 2.5V 阈值，保证强反型工作
        super().__init__()
        # [V6.0 改进] 形状 (1, C, 1, 1) 实现通道级控制
        # 允许网络独立调整每个通道的直流工作点
        self.bias = nn.Parameter(torch.ones(1, num_channels, 1, 1) * target_mean) 
        # 初始缩放设为 0.5，避免初始信号过大
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1) * 0.5)

    def forward(self, x):
        # 限制 Scale，模拟 VGA (可变增益放大器) 的物理范围
        real_scale = torch.clamp(self.scale.abs(), min=0.01, max=3.0)
        x_mapped = x * real_scale + self.bias
        
        # 物理供电轨限制 (0V - 8V)
        # 任何超出此范围的信号都会被硬件截断
        return torch.clamp(x_mapped, 0.0, 8.0)

# ==========================================
# 3. 关键修改 II: 差分 EKV 卷积 + 运放饱和
# 目的：模拟 Flash 阵列卷积计算 + TIA 读出电路
# ==========================================
class DifferentialEKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # [继承 V5.4] 阈值初始化: 2.5V (强反型区)
        # 降低阈值，确保器件在 3.5V 输入下充分导通，提供强梯度
        self.theta_pos = nn.Parameter(torch.normal(2.5, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
        self.theta_neg = nn.Parameter(torch.normal(2.5, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
        
        # [继承 V5.4] 动态增益初始化 (Power Normalization)
        # 根据输入通道数自动缩放初始增益，实现级间功率匹配
        scaling_factor = math.sqrt(16 / in_channels)
        init_gain = cfg.TIA_GAIN_BASE * scaling_factor
        self.log_tia_gain = nn.Parameter(torch.tensor(np.log(init_gain)))
        
        self.phi = 2 * cfg.N_FACTOR * cfg.VT

    def ekv_current(self, v_gs, v_th):
        # 完整 EKV 模型：包含前向和反向电流分量
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
        
        # 训练时注入权重噪声，增强物理鲁棒性
        if self.training:
            theta_pos_noise = self.theta_pos + torch.randn_like(self.theta_pos) * cfg.WEIGHT_NOISE_STD
            theta_neg_noise = self.theta_neg + torch.randn_like(self.theta_neg) * cfg.WEIGHT_NOISE_STD
        else:
            theta_pos_noise = self.theta_pos
            theta_neg_noise = self.theta_neg

        theta_pos_exp = theta_pos_noise.view(1, self.out_channels, -1, 1)
        theta_neg_exp = theta_neg_noise.view(1, self.out_channels, -1, 1)
        
        # 模拟域电流计算
        i_pos = self.ekv_current(x_in, theta_pos_exp)
        i_neg = self.ekv_current(x_in, theta_neg_exp)
        
        # KCL 电流汇聚
        sum_i_pos = torch.sum(i_pos, dim=2) 
        sum_i_neg = torch.sum(i_neg, dim=2)
        
        i_diff = sum_i_pos - sum_i_neg
        
        # [继承 V5.4] 移除高下限，允许低增益
        current_gain = torch.exp(self.log_tia_gain).clamp(min=0.1, max=5000.0)
        out_voltage_linear = i_diff * current_gain
        
        # [V6.0 核心修正] 运放饱和 (OpAmp Saturation)
        # 物理意义：TIA 输出不可能无限大，会受限于电源轨。
        # 算法意义：引入 Tanh 软饱和，防止数值爆炸，充当非线性激活函数。
        # 公式：V_out = V_sat * tanh(V_linear / V_sat)
        # 这将 ±17V 的信号平滑地压缩到 ±6V，保留了梯度，避免了下一级的硬截断。
        out_voltage = torch.tanh(out_voltage_linear / cfg.OPAMP_V_SAT) * cfg.OPAMP_V_SAT
        
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        return out_voltage.view(N, self.out_channels, h_out, w_out)

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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        if self.training:
            x = x + torch.randn_like(x) * cfg.INPUT_NOISE_STD
            
        out = self.mapper1(x)       
        out = self.ekv1(out)
        out = self.dropout1(out)
        
        out = self.mapper2(out)     
        out = self.ekv2(out)
        out = self.dropout2(out)
        
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
    # [继承 V5.4] 物理校准参数禁用 Weight Decay，防止增益被强行衰减为0
    calibration_params = [p for n,p in net_hybrid.named_parameters() if 'mapper' in n or 'tia_gain' in n]
    other_params = [p for n,p in net_hybrid.named_parameters() 
                    if 'theta' not in n and 'mapper' not in n and 'tia_gain' not in n and p.requires_grad]
    
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
    plt.title("Hybrid ResNet (OpAmp Saturation + Strong Inversion)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(Paths.PLOT_IMG)
    print(f"Done. Best Validation Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()