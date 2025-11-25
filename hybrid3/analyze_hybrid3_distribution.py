# 文件名: analyze_hybrid_resnet_distribution_final.py
# 功能：加载混合模型 → 前向一个batch → 打印关键参数 → 输出【一张大图】包含所有8个关键激活分布

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 添加此行避免内存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ================== 设备 & 数据 ==================
# 切换到空闲GPU（根据nvidia-smi，试cuda:1）
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# 降低batch_size避免OOM
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# ================== 模块定义（与训练脚本100%一致） ==================
class LayerScale(nn.Module):
    def __init__(self, init_value=1.0): super().__init__(); self.scale = nn.Parameter(torch.tensor(init_value))
    def forward(self, x): return x * self.scale

class InputScaling(nn.Module):
    def __init__(self, scale_init=4.5, shift_init=3.8):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.shift = nn.Parameter(torch.tensor(shift_init))
    def forward(self, x): return x * self.scale + self.shift

class EKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride, self.padding = stride, padding
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        k = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.theta = nn.Parameter(torch.empty(out_channels, k))
        nn.init.uniform_(self.theta, 2.8, 4.8)
        self.alpha, self.n, self.VT, self.VD = 5.625e-4, 1.5, 0.026, 0.1
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=padding, stride=stride)
    def log1p_exp(self, x): return torch.log1p(torch.exp(torch.clamp(x, -30, 30)))
    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)                                   # (B, C*k*k, L)
        vg = patches.unsqueeze(1)
        theta = self.theta.view(1, -1, self.theta.shape[-1], 1)
        arg1 = (vg - theta) / (self.n * self.VT)
        arg2 = (vg - theta - self.VD) / (self.n * self.VT)
        I = self.alpha * (self.log1p_exp(arg1)**2 - self.log1p_exp(arg2)**2)
        out = I.sum(dim=2).view(B, -1,
                                (H + 2*self.padding - self.kernel_size[0])//self.stride + 1,
                                (W + 2*self.padding - self.kernel_size[1])//self.stride + 1)
        return out

class LinearBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class EKVBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.prescale = InputScaling(scale_init=4.5, shift_init=3.8)
        self.conv1 = EKVConv2d(in_planes, planes, 3, stride, padding=1)
        self.ls1 = LayerScale(init_value=3.0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = EKVConv2d(planes, planes, 3, 1, padding=1)
        self.ls2 = LayerScale(init_value=3.0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.prescale(x)
        out = self.ls1(self.conv1(out))
        out = self.bn1(out)
        out = self.ls2(self.conv2(out))
        out = self.bn2(out)
        out += identity
        return out

class HybridResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = nn.Sequential(LinearBlock(16, 16))
        self.layer2 = nn.Sequential(LinearBlock(16, 32, stride=2))
        self.layer3 = nn.Sequential(EKVBlock(32, 64, stride=2))
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        x = self.avgpool(l3)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, l2, l3

# ================== 加载模型 ==================
model = HybridResNet8().to(device)
model_path = 'best_hybrid3_ekv_last20.pth'      # ← 修改为你的实际路径
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print(f"Model loaded successfully: {model_path}")

# ================== 打印关键可学习参数 ==================
prescale = model.layer3[0].prescale
ls1 = model.layer3[0].ls1
ls2 = model.layer3[0].ls2
print("\n=== Trained Key Scaling Parameters ===")
print(f"InputScaling  →  scale: {prescale.scale.item():.6f}   shift: {prescale.shift.item():.6f}")
print(f"LayerScale 1  →  γ1: {ls1.scale.item():.6f}")
print(f"LayerScale 2  →  γ2: {ls2.scale.item():.6f}")
print("="*60)

# ================== Hook 收集所有激活 ==================
activations = {}
def hook(name):
    def _hook(m, i, o): activations[name] = o.detach()
    return _hook

model.layer2.register_forward_hook(lambda m,i,o: activations.setdefault('prescale_in', o.detach()))
model.layer3[0].prescale.register_forward_hook(lambda m,i,o: activations.setdefault('prescale_out', o.detach()))
model.layer3[0].conv1.register_forward_hook(hook('conv1_raw'))
model.layer3[0].ls1.register_forward_hook(hook('after_ls1'))
model.layer3[0].bn1.register_forward_hook(hook('after_bn1'))
model.layer3[0].conv2.register_forward_hook(hook('conv2_raw'))
model.layer3[0].ls2.register_forward_hook(hook('after_ls2'))
model.layer3[0].register_forward_hook(hook('ekv_final'))

# ================== 前向一个batch ==================
with torch.no_grad():
    images, _ = next(iter(testloader))
    images = images.to(device)
    _, input_to_l3, ekv_out = model(images)
    activations['input_to_layer3'] = input_to_l3
    activations['ekv_block_final'] = ekv_out
print(f"Forward pass completed, batch_size={images.shape[0]}\n")

# ================== 一张大图绘制所有分布 ==================
fig, axes = plt.subplots(2, 4, figsize=(24, 10), constrained_layout=True)
fig.suptitle('Hybrid ResNet8 (layer3 = Non-linear EKV) Activation Distribution Overview', fontsize=20, fontweight='bold')

plot_list = [
    ('input_to_layer3',   '1. Input to layer3\n(Linear ReLU Output)', 'tab:blue'),
    ('prescale_out',      '2. After InputScaling\n(Mapped to NOR Flash Voltage)', 'tab:orange'),
    ('conv1_raw',         '3. EKVConv1 Raw Current', 'tab:green'),
    ('after_ls1',         f'4. × LayerScale1 (γ1={ls1.scale.item():.3f})', 'tab:red'),
    ('after_bn1',         '5. After BN1', 'tab:purple'),
    ('conv2_raw',         '6. EKVConv2 Raw Current', 'tab:brown'),
    ('after_ls2',         f'7. × LayerScale2 (γ2={ls2.scale.item():.3f})', 'tab:pink'),
    ('ekv_block_final',   '8. EKVBlock Final Output', 'tab:cyan'),
]

for ax, (key, title, color) in zip(axes.flat, plot_list):
    data = activations[key].cpu().numpy().flatten()
    ax.hist(data, bins=120, density=True, color=color, alpha=0.75, edgecolor='black', linewidth=0.3)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 关键统计数字直接写在子图上（全部英文）
    stats = f"min: {data.min():.4f}\nmax: {data.max():.4f}\nmean: {data.mean():.4f}\nstd: {data.std():.4f}"
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.savefig('EKV_layer3_activation_distribution_ONE_PICTURE.png', dpi=300, bbox_inches='tight')
plt.close()
print("=== A complete large figure has been saved: EKV_layer3_activation_distribution_ONE_PICTURE.png ===")
print("   2×4 layout, all key node distributions at a glance, strongly recommended to zoom in for details!")