import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 模型定义复刻 
# (必须与训练代码完全一致，否则无法加载权重)
# ==========================================
class Config:
    VT = 0.025; N_FACTOR = 1.5; VD = 0.5; ALPHA = 2e-6
    BASE_TIA = 500000.0 
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # 指定你要分析的权重文件
    MODEL_PATH = 'resnet8_final_best_4_3.pth' 

cfg = Config()

class AdaptiveVoltageMapper(nn.Module):
    def __init__(self, init_scale=1.5, init_bias=3.5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))
    def forward(self, x):
        return torch.clamp(x * self.scale + self.bias, 0.0, 9.0)

class NormalizedEKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        # 保存维度信息供 forward 使用
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels # [修复关键点] 必须保存 out_channels
        
        self.theta = nn.Parameter(torch.normal(2.5, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
        fan_in = in_channels * kernel_size * kernel_size
        self.current_tia_gain = cfg.BASE_TIA / (fan_in ** 0.5)
        self.register_buffer('tia_gain', torch.tensor(self.current_tia_gain))

    def forward(self, x):
        N, C, H, W = x.shape
        x_unf = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        
        # x_in shape: [N, 1, Cin*K*K, L]
        x_in = x_unf.unsqueeze(1) 
        
        # [修复关键点] view 中只能有一个 -1
        # 我们显式指定 self.out_channels
        # w_th shape: [1, Cout, Cin*K*K, 1]
        w_th = self.theta.view(1, self.out_channels, -1, 1)
        
        phi = 2 * cfg.N_FACTOR * cfg.VT
        v_diff = x_in - w_th
        f_term = F.softplus(v_diff / phi).pow(2)
        r_term = F.softplus((v_diff - cfg.VD) / phi).pow(2)
        out_current = torch.sum(cfg.ALPHA * (f_term - r_term), dim=2) 
        out_voltage = out_current * self.tia_gain
        
        h_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1
        return out_voltage.view(N, -1, h_out, w_out)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class FinalDoubleEKVBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(FinalDoubleEKVBlock, self).__init__()
        self.mapper1 = AdaptiveVoltageMapper(init_scale=1.5, init_bias=2.5)
        self.ekv1 = NormalizedEKVConv2d(in_planes, planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.mapper2 = AdaptiveVoltageMapper(init_scale=2.0, init_bias=4.0) 
        self.ekv2 = NormalizedEKVConv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn1(self.ekv1(self.mapper1(x)))
        out = self.bn2(self.ekv2(self.mapper2(out)))
        out += identity
        out = F.relu(out)
        return out

class ResNet8(nn.Module):
    def __init__(self, block_type='linear', num_classes=10):
        super(ResNet8, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, 2)
        
        if block_type == 'final_double':
            self.layer3 = self._make_layer(FinalDoubleEKVBlock, 64, 1, 2)
        else:
            self.layer3 = self._make_layer(BasicBlock, 64, 1, 2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)); self.in_planes = planes
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
# 2. 真实数据流分析器
# ==========================================
def analyze_real_data():
    print(f">>> 1. 初始化模型并加载权重: {cfg.MODEL_PATH}")
    model = ResNet8(block_type='final_double').to(cfg.DEVICE)
    
    if not os.path.exists(cfg.MODEL_PATH):
        print(f"Error: 找不到权重文件 {cfg.MODEL_PATH}！")
        return

    try:
        state_dict = torch.load(cfg.MODEL_PATH, map_location=cfg.DEVICE)
        model.load_state_dict(state_dict, strict=True)
        print(">>> 权重加载成功 (Strict Mode Passed)。")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("尝试使用 strict=False 加载...")
        try:
            model.load_state_dict(state_dict, strict=False)
            print(">>> 权重加载成功 (Non-Strict Mode)。")
        except Exception as e2:
            print(f"Fatal Error: {e2}")
            return

    model.eval()
    
    print(">>> 2. 准备 CIFAR-10 测试数据")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True)
    
    real_images, _ = next(iter(loader))
    real_images = real_images.to(cfg.DEVICE)
    
    print(">>> 3. 注册 Hooks 截获中间数据")
    activations = {}
    
    def get_data(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    layer3_block = model.layer3[0]
    
    layer3_block.mapper1.register_forward_hook(lambda m, i, o: activations.update({'l3_input': i[0].detach().cpu()}))
    layer3_block.mapper2.register_forward_hook(get_data('stage2_vg'))
    layer3_block.bn2.register_forward_hook(get_data('ekv_path_out'))
    layer3_block.shortcut.register_forward_hook(get_data('shortcut_out'))
    
    print(">>> 4. 运行前向推理...")
    with torch.no_grad():
        _ = model(real_images)
        
    print("\n>>> 5. 分析结果")
    
    ekv_out = activations['ekv_path_out'].numpy().flatten()
    shortcut_out = activations['shortcut_out'].numpy().flatten()
    
    norm_ekv = np.linalg.norm(ekv_out)
    norm_shortcut = np.linalg.norm(shortcut_out)
    ratio = norm_ekv / (norm_shortcut + 1e-6)
    
    print(f"   [Energy] EKV Path Norm:      {norm_ekv:.2f}")
    print(f"   [Energy] Shortcut Path Norm: {norm_shortcut:.2f}")
    print(f"   [Ratio]  EKV / Shortcut:     {ratio:.4f}")
    
    vg2 = activations['stage2_vg'].numpy().flatten()
    theta2 = layer3_block.ekv2.theta.detach().cpu().numpy().flatten()
    
    mean_vg = np.mean(vg2)
    mean_theta = np.mean(theta2)
    
    print(f"   [Op Point] Mean Vg (Input):    {mean_vg:.2f} V")
    print(f"   [Op Point] Mean Theta (Vth):   {mean_theta:.2f} V")
    
    plt.figure(figsize=(20, 12))
    plt.suptitle(f"Diagnosis Result: {cfg.MODEL_PATH}\nRatio={ratio:.2f} (Balanced~1.0)", fontsize=16)
    
    plt.subplot(2, 3, 1)
    plt.hist(shortcut_out, bins=100, alpha=0.5, color='gray', label='Shortcut', density=True)
    plt.hist(ekv_out, bins=100, alpha=0.6, color='blue', label='EKV Path', density=True)
    plt.yscale('log')
    plt.title("1. Branch Output Distribution (Log Scale)")
    plt.legend()
    
    plt.subplot(2, 3, 2)
    l3_in = activations['l3_input'].numpy().flatten()
    plt.hist(l3_in, bins=100, color='green', alpha=0.7)
    plt.title(f"2. Layer 3 Input\nMean: {l3_in.mean():.2f}")
    
    plt.subplot(2, 3, 3)
    plt.hist(vg2, bins=100, color='orange', alpha=0.6, label='Input Vg', density=True)
    plt.hist(theta2, bins=100, color='red', alpha=0.4, label='Learned Theta', density=True)
    plt.axvline(x=mean_vg, color='orange', linestyle='--')
    plt.axvline(x=mean_theta, color='red', linestyle='--')
    plt.title("3. Stage 2: Vg vs Theta Overlap")
    plt.legend()
    
    plt.subplot(2, 3, 4)
    v_sweep = np.linspace(0, 9, 100)
    phi = 2 * cfg.N_FACTOR * cfg.VT
    f_term = np.log(1 + np.exp((v_sweep - mean_theta)/phi))**2
    r_term = np.log(1 + np.exp((v_sweep - mean_theta - cfg.VD)/phi))**2
    i_curve = f_term - r_term
    i_curve = i_curve / np.max(i_curve)
    
    plt.plot(v_sweep, i_curve, 'r-', linewidth=3, label='EKV Curve')
    plt.hist(vg2, bins=50, density=True, color='orange', alpha=0.3, label='Actual Vg')
    plt.title("4. Linearity Check")
    plt.legend()
    
    plt.subplot(2, 3, 5)
    n_dead = np.sum(vg2 < mean_theta - 0.5)
    n_linear = np.sum(vg2 > mean_theta + 3.0)
    n_active = len(vg2) - n_dead - n_linear
    
    plt.pie([n_dead, n_active, n_linear], 
            labels=['Dead', 'Active (Non-Linear)', 'Linear (Resistor)'],
            colors=['gray', 'green', 'red'], autopct='%1.1f%%')
    plt.title("5. Operating Region Stats")
    
    plt.tight_layout()
    save_path = 'diagnose_4_3_result.png'
    plt.savefig(save_path)
    print(f"\n>>> 诊断图片已保存至: {save_path}")

if __name__ == '__main__':
    analyze_real_data()