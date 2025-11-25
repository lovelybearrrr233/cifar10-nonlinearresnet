# -*- coding: utf-8 -*-
"""
Double EKV Layer3 逐层分布可视化工具
功能：分析 ResNet8 中 DoubleEKVBlock 的内部数据流，包括激活分布和 Theta 参数分布
传入的模型路径为model_path
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

#设置路径



# ================== 1. 模型定义 (保持与训练代码完全一致) ==================
class Config:
    VT = 0.025          
    N_FACTOR = 1.5      
    VD = 0.5            
    ALPHA = 2e-6        
    TIA_GAIN = 200000.0 
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

cfg = Config()

class VoltageMapper(nn.Module):
    def __init__(self, init_bias=3.5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.5)) 
        self.bias = nn.Parameter(torch.tensor(init_bias)) 

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
        self.mapper1 = VoltageMapper(init_bias=3.5)
        self.ekv1 = NonLinearEKVConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        # Stage 2
        self.mapper2 = VoltageMapper(init_bias=3.5) 
        self.ekv2 = NonLinearEKVConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        # Residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.mapper1(x)       
        out = self.ekv1(out)       
        out = self.bn1(out)         
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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        if block_type == 'hybrid':
            self.layer3 = self._make_layer(DoubleEKVBlock, 64, 1, stride=2)
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

# ================== 2. 初始化环境与加载模型 ==================

print(f"Using device: {cfg.DEVICE}")

# 2.1 准备数据
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 2.2 加载模型
model = ResNet8(block_type='hybrid').to(cfg.DEVICE)
model_path = '4-1-resnet8_double_ekv_best.pth'  # <--- 请确保这里是你的权重文件路径

if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(state_dict)
else:
    print(f"Warning: {model_path} not found. Using random initialization for demonstration.")

model.eval()

# ================== 3. Hook 注册 (关键步骤) ==================
activations = {}

# 使用闭包函数避免 lambda 变量捕获问题
def save_output(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册 Layer2 输出 (即 Layer3 输入)
model.layer2.register_forward_hook(save_output('0_Input_to_Layer3'))

# 注册 Layer3 内部所有子模块输出
l3 = model.layer3[0] # 获取 DoubleEKVBlock 实例
l3.mapper1.register_forward_hook(save_output('1_Mapper1_Output'))
l3.ekv1.register_forward_hook(save_output('2_EKV1_Output_Voltage'))
l3.bn1.register_forward_hook(save_output('3_BN1_Output'))
l3.mapper2.register_forward_hook(save_output('4_Mapper2_Output'))
l3.ekv2.register_forward_hook(save_output('5_EKV2_Output_Voltage'))
l3.bn2.register_forward_hook(save_output('6_BN2_Output'))
# 注册整个 Layer3 的最终输出
l3.register_forward_hook(save_output('7_Layer3_Final_Output'))

# ================== 4. 执行推理 ==================
print("Running inference on one batch...")
with torch.no_grad():
    images, _ = next(iter(testloader))
    images = images.to(cfg.DEVICE)
    model(images)

# ================== 5. 可视化绘图 ==================
# 我们有 8 个激活分布 + 2 个 Theta 参数分布 = 10 张图
# 布局: 2 行 5 列
fig, axes = plt.subplots(2, 5, figsize=(25, 10), constrained_layout=True)
fig.suptitle('Double EKV Block Internal Data Flow & Parameter Distribution', fontsize=20, fontweight='bold')

# 准备绘图数据配置: (Key/Source, Title, Color)
# 注意：Theta 不是通过 Hook 获取的，而是直接从模型参数获取
theta1_data = l3.ekv1.theta.detach().cpu().numpy().flatten()
theta2_data = l3.ekv2.theta.detach().cpu().numpy().flatten()

# 定义每张图的内容
# 格式: (数据来源类型, Key或Data, Title, Color)
# 类型 0: Activations 字典 key
# 类型 1: 直接的 Numpy 数据 (用于 Theta)
plot_configs = [
    # --- Row 1: Stage 1 Flow ---
    (0, '0_Input_to_Layer3',     '1. Layer2 Output (Input to L3)\n(Linear ReLU features)', 'tab:blue'),
    (0, '1_Mapper1_Output',      '2. Mapper1 Output (Vg)\n(Mapped to 0~9V)', 'tab:orange'),
    (1, theta1_data,             '3. EKV1 Theta Distribution\n(Learned Thresholds)', 'tab:gray'),
    (0, '2_EKV1_Output_Voltage', '4. EKV1 Output (Voltage)\n(Non-linear current * Gain)', 'tab:green'),
    (0, '3_BN1_Output',          '5. BN1 Output\n(Normalized Features)', 'tab:purple'),

    # --- Row 2: Stage 2 Flow ---
    (0, '4_Mapper2_Output',      '6. Mapper2 Output (Vg)\n(Remapped from BN1)', 'tab:orange'),
    (1, theta2_data,             '7. EKV2 Theta Distribution\n(Learned Thresholds)', 'tab:gray'),
    (0, '5_EKV2_Output_Voltage', '8. EKV2 Output (Voltage)\n(Stage 2 Response)', 'tab:brown'),
    (0, '6_BN2_Output',          '9. BN2 Output\n(Normalized Features)', 'tab:pink'),
    (0, '7_Layer3_Final_Output', '10. Layer3 Final Output\n(After Residual & ReLU)', 'tab:cyan'),
]

for ax, (d_type, source, title, color) in zip(axes.flat, plot_configs):
    # 获取数据
    if d_type == 0:
        data = activations[source].cpu().numpy().flatten()
    else:
        data = source # 已经是 numpy
        
    # 绘制直方图
    ax.hist(data, bins=100, density=False, weights=np.ones_like(data)/len(data), color=color, alpha=0.75, edgecolor='black', linewidth=0.3)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 统计信息
    stats = f"Min: {data.min():.2f}\nMax: {data.max():.2f}\nMean: {data.mean():.2f}\nStd: {data.std():.2f}"
    ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

# 保存图片
save_name = '4-1Double_EKV_Flow_Analysis.png'
plt.savefig(save_name, dpi=300)
print(f"\nVisualization saved to: {save_name}")
print("-" * 50)
print("Analysis Guide:")
print("1. Compare [Mapper Output] vs [Theta Distribution]. Overlap area indicates active neurons.")
print("2. Check [EKV Output]. High values indicate strong feature matching (resonance).")
print("3. Check [BN Output]. Should be roughly centered around 0 with std ~1.")