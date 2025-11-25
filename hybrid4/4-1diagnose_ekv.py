import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 模型定义复刻 (必须与训练代码一致)
# ==========================================
# 为了方便加载，这里简化复刻，实际使用请确保类定义与训练脚本一致
class Config:
    VT = 0.025; N_FACTOR = 1.5; VD = 0.5; ALPHA = 2e-6; TIA_GAIN = 200000.0
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

cfg = Config()

class VoltageMapper(nn.Module):
    def __init__(self, init_bias=3.5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5)) 
        self.bias = nn.Parameter(torch.tensor(init_bias)) 
    def forward(self, x):
        return torch.clamp(x * self.scale + self.bias, 0.0, 9.0)

class NonLinearEKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.in_channels, self.out_channels = in_channels, out_channels
        self.theta = nn.Parameter(torch.normal(4.0, 0.8, size=(out_channels, in_channels, kernel_size, kernel_size)))
    def forward(self, x):
        N, C, H, W = x.shape
        x_unf = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        x_in = x_unf.unsqueeze(1)
        w_th = self.theta.view(1, self.out_channels, -1, 1)
        phi = 2 * cfg.N_FACTOR * cfg.VT
        v_diff = x_in - w_th
        f_term = F.softplus(v_diff / phi).pow(2)
        r_term = F.softplus((v_diff - cfg.VD) / phi).pow(2)
        i_syn = cfg.ALPHA * (f_term - r_term)
        out_current = torch.sum(i_syn, dim=2)
        out_voltage = out_current * cfg.TIA_GAIN
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out_voltage.view(N, self.out_channels, h_out, w_out)

# 简化的 Block 定义用于加载
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x): return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class HybridBlock(nn.Module): # Old Logic
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.mapper = VoltageMapper()
        self.ekv_conv = NonLinearEKVConv2d(in_planes, planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = self.bn1(self.ekv_conv(self.mapper(x)))
        out = self.bn2(self.conv2(out)) + self.shortcut(x)
        return F.relu(out)

class DoubleEKVBlock(nn.Module): # New Logic
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.mapper1 = VoltageMapper(init_bias=3.5)
        self.ekv1 = NonLinearEKVConv2d(in_planes, planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.mapper2 = VoltageMapper(init_bias=3.5)
        self.ekv2 = NonLinearEKVConv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))
    def forward(self, x):
        out = self.bn1(self.ekv1(self.mapper1(x)))
        out = self.bn2(self.ekv2(self.mapper2(out))) + self.shortcut(x)
        return F.relu(out)

class ResNet8(nn.Module):
    def __init__(self, block_type='linear'):
        super().__init__()
        self.in_planes = 16; self.block_type = block_type
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, 2)
        if block_type == 'hybrid': self.layer3 = self._make_layer(HybridBlock, 64, 1, 2)
        elif block_type == 'double': self.layer3 = self._make_layer(DoubleEKVBlock, 64, 1, 2)
        else: self.layer3 = self._make_layer(BasicBlock, 64, 1, 2)
        self.linear = nn.Linear(64, 10)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1); layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride)); self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(F.relu(self.bn1(self.conv1(x))))))
        return self.linear(F.avg_pool2d(out, out.size()[2:]).view(out.size(0), -1))

# ==========================================
# 2. 诊断与绘图工具
# ==========================================
def get_intermediate_data(model, dataloader):
    model.eval()
    data = {}
    hooks = []
    
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, tuple): output = output[0]
            data[name] = output.detach().cpu().numpy().flatten()
        return fn

    # 注册 Hook 到 Layer3 的各个组件
    layer3 = model.layer3[0]
    
    if isinstance(layer3, HybridBlock):
        hooks.append(layer3.mapper.register_forward_hook(hook_fn("Hybrid: Mapper Out (Vg)")))
        hooks.append(layer3.ekv_conv.register_forward_hook(hook_fn("Hybrid: EKV Out (I -> V)")))
        data["Hybrid: Theta"] = layer3.ekv_conv.theta.detach().cpu().numpy().flatten()
        
    elif isinstance(layer3, DoubleEKVBlock):
        hooks.append(layer3.mapper1.register_forward_hook(hook_fn("Double: Mapper1 Out (Vg1)")))
        hooks.append(layer3.ekv1.register_forward_hook(hook_fn("Double: EKV1 Out")))
        hooks.append(layer3.bn1.register_forward_hook(hook_fn("Double: BN1 Out (Norm)")))
        hooks.append(layer3.mapper2.register_forward_hook(hook_fn("Double: Mapper2 Out (Vg2)")))
        hooks.append(layer3.ekv2.register_forward_hook(hook_fn("Double: EKV2 Out")))
        data["Double: Theta1"] = layer3.ekv1.theta.detach().cpu().numpy().flatten()
        data["Double: Theta2"] = layer3.ekv2.theta.detach().cpu().numpy().flatten()

    # 跑一个 Batch
    img, _ = next(iter(dataloader))
    model(img.to(cfg.DEVICE))
    
    for h in hooks: h.remove()
    return data

def plot_diagnosis(data_hybrid, data_double):
    plt.figure(figsize=(20, 12))
    plt.suptitle("Diagnosis: Hybrid vs Double EKV Failure Analysis", fontsize=16)
    
    # 1. 权重分布 (Theta)
    plt.subplot(3, 3, 1)
    plt.hist(data_hybrid["Hybrid: Theta"], bins=50, color='blue', alpha=0.7, label='Hybrid Theta')
    plt.hist(data_double["Double: Theta1"], bins=50, color='green', alpha=0.5, label='Double Theta1')
    plt.axvline(x=0, color='r', linestyle='--'); plt.axvline(x=9, color='r', linestyle='--')
    plt.title("Weights (Theta) Distribution"); plt.legend()
    
    plt.subplot(3, 3, 2)
    plt.hist(data_double["Double: Theta2"], bins=50, color='red', alpha=0.7, label='Double Theta2')
    plt.title("Double EKV Layer 2 Theta"); plt.legend()

    # 2. 第一层输入电压 (Vg)
    plt.subplot(3, 3, 3)
    plt.hist(data_hybrid["Hybrid: Mapper Out (Vg)"], bins=50, color='blue', alpha=0.7, density=True, label='Hybrid Vg')
    plt.hist(data_double["Double: Mapper1 Out (Vg1)"], bins=50, color='green', alpha=0.5, density=True, label='Double Vg1')
    plt.title("Layer 1 Input Voltage (Vg)"); plt.legend()

    # 3. 第一层输出与中间态
    plt.subplot(3, 3, 4)
    plt.hist(data_hybrid["Hybrid: EKV Out (I -> V)"], bins=50, color='blue', alpha=0.7, log=True, label='Hybrid Out')
    plt.hist(data_double["Double: EKV1 Out"], bins=50, color='green', alpha=0.5, log=True, label='Double Out1')
    plt.title("Layer 1 Output (Log Scale)"); plt.legend()

    plt.subplot(3, 3, 5)
    plt.hist(data_double["Double: BN1 Out (Norm)"], bins=50, color='gray', alpha=0.7, label='Double BN1 Out')
    plt.title("Double: Intermediate BN Output (Should be Gaussian)"); plt.legend()
    
    # 4. 关键：第二层输入电压 (Vg2)
    plt.subplot(3, 3, 6)
    plt.hist(data_double["Double: Mapper2 Out (Vg2)"], bins=50, color='orange', alpha=0.7, label='Double Vg2')
    plt.axvline(x=3.0, color='r', linestyle='--', label='Typical Vth')
    plt.title("Double: Layer 2 Input Voltage (CRITICAL)"); plt.legend()
    
    # 5. 第二层输出
    plt.subplot(3, 3, 7)
    plt.hist(data_double["Double: EKV2 Out"], bins=50, color='purple', alpha=0.7, log=True, label='Double Out2')
    plt.title("Double: Layer 2 Output (Log Scale)"); plt.legend()

    plt.tight_layout()
    plt.savefig('4-1-diagnosis_result.png')
    plt.show()

# ==========================================
# 3. 主程序
# ==========================================
def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.49, 0.48, 0.44), (0.20, 0.19, 0.20))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    # 加载 Hybrid
    print("Analyzing Hybrid Model...")
    model_h = ResNet8(block_type='hybrid').to(cfg.DEVICE)
    # 确保路径正确，如果文件名不同请修改
    if os.path.exists('resnet8_hybrid_best.pth'):
        model_h.load_state_dict(torch.load('resnet8_hybrid_best.pth', map_location=cfg.DEVICE))
        data_h = get_intermediate_data(model_h, loader)
    else:
        print("Warning: Hybrid weights not found, creating dummy data.")
        data_h = get_intermediate_data(model_h, loader) # Dummy data

    # 加载 Double
    print("Analyzing Double EKV Model...")
    model_d = ResNet8(block_type='double').to(cfg.DEVICE)
    if os.path.exists('4-1-resnet8_double_ekv_best.pth'):
        model_d.load_state_dict(torch.load('4-1-resnet8_double_ekv_best.pth', map_location=cfg.DEVICE))
        data_d = get_intermediate_data(model_d, loader)
    else:
        print("Error: Double EKV weights not found!")
        return

    plot_diagnosis(data_h, data_d)
    print("Diagnosis plot saved to 'diagnosis_result.png'")

if __name__ == '__main__':
    main()