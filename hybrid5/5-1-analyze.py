# -*- coding: utf-8 -*-
"""
差分EKV混合模型逐层分布可视化工具
功能：分析 ResNet8 中 DoubleEKVBlock 的内部数据流，包括激活分布和差分Theta参数分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ================== 1. 模型定义 (与训练代码完全一致) ==================

class Config:
    VT = 0.026          
    N_FACTOR = 1.5      
    VD = 0.2            
    ALPHA = 10e-6       
    TIA_GAIN = 2000.0   
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

cfg = Config()

class VoltageMapper(nn.Module):
    def __init__(self, num_channels, target_mean=4.0):
        super().__init__()
        self.bias = nn.Parameter(torch.ones(1, num_channels, 1, 1) * target_mean) 
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))

    def forward(self, x):
        real_scale = torch.clamp(self.scale.abs(), min=0.01, max=3.0)
        x_mapped = x * real_scale + self.bias
        return torch.clamp(x_mapped, 0.0, 8.0)

class DifferentialEKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.theta_pos = nn.Parameter(torch.normal(4.0, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
        self.theta_neg = nn.Parameter(torch.normal(4.0, 0.5, size=(out_channels, in_channels, kernel_size, kernel_size)))
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

        theta_pos_exp = self.theta_pos.view(1, self.out_channels, -1, 1)
        theta_neg_exp = self.theta_neg.view(1, self.out_channels, -1, 1)
        
        i_pos = self.ekv_current(x_in, theta_pos_exp)
        i_neg = self.ekv_current(x_in, theta_neg_exp)
        
        sum_i_pos = torch.sum(i_pos, dim=2) 
        sum_i_neg = torch.sum(i_neg, dim=2)
        
        i_diff = sum_i_pos - sum_i_neg
        out_voltage = i_diff * cfg.TIA_GAIN
        
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
        self.mapper1 = VoltageMapper(num_channels=in_planes, target_mean=4.0)
        self.ekv1 = DifferentialEKVConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        
        self.mapper2 = VoltageMapper(num_channels=planes, target_mean=4.0)
        self.ekv2 = DifferentialEKVConv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        # Stage 1
        out = self.mapper1(x)       
        out = self.ekv1(out)
        out = self.dropout1(out)
        
        # Stage 2
        out = self.mapper2(out)     
        out = self.ekv2(out)
        out = self.dropout2(out)
        
        out += identity
        out = F.relu(out)           
        return out

class ResNet8(nn.Module):
    def __init__(self, block_type='hybrid'):
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

# ================== 2. 初始化环境与加载模型 ==================

print(f"Using device: {cfg.DEVICE}")

# 准备数据
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# 加载训练好的混合模型
model = ResNet8(block_type='hybrid').to(cfg.DEVICE)
model_path = './5-1-best_hybrid.pth'  # 你的训练模型路径

if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=cfg.DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded successfully!")
else:
    print(f"Warning: {model_path} not found. Using random initialization for demonstration.")

model.eval()

# ================== 3. Hook 注册 ==================
activations = {}

def save_output(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# 注册 Layer2 输出 (即 Layer3 输入)
model.layer2.register_forward_hook(save_output('0_Input_to_Layer3'))

# 注册 Layer3 内部所有子模块输出
l3 = model.layer3[0]  # 获取 DoubleEKVBlock 实例
l3.mapper1.register_forward_hook(save_output('1_Mapper1_Output'))
l3.ekv1.register_forward_hook(save_output('2_EKV1_Output_Voltage'))
l3.dropout1.register_forward_hook(save_output('3_Dropout1_Output'))
l3.mapper2.register_forward_hook(save_output('4_Mapper2_Output'))
l3.ekv2.register_forward_hook(save_output('5_EKV2_Output_Voltage'))
l3.dropout2.register_forward_hook(save_output('6_Dropout2_Output'))
# 注册整个 Layer3 的最终输出
l3.register_forward_hook(save_output('7_Layer3_Final_Output'))

# ================== 4. 执行推理 ==================
print("Running inference on one batch...")
with torch.no_grad():
    images, _ = next(iter(testloader))
    images = images.to(cfg.DEVICE)
    model(images)

# ================== 5. 可视化绘图 ==================
# 创建更大的画布来容纳所有图
fig, axes = plt.subplots(3, 5, figsize=(25, 15), constrained_layout=True)
fig.suptitle('Differential EKV Hybrid Model - Internal Data Flow & Parameter Analysis', 
             fontsize=20, fontweight='bold')

# 准备绘图数据配置
# 获取差分EKV的Theta参数
theta1_pos = l3.ekv1.theta_pos.detach().cpu().numpy().flatten()
theta1_neg = l3.ekv1.theta_neg.detach().cpu().numpy().flatten()
theta2_pos = l3.ekv2.theta_pos.detach().cpu().numpy().flatten()
theta2_neg = l3.ekv2.theta_neg.detach().cpu().numpy().flatten()

# 定义每张图的内容
plot_configs = [
    # --- Row 1: Stage 1 Flow ---
    (0, '0_Input_to_Layer3',     '1. Layer2 Output\n(Input to EKV Block)', 'tab:blue'),
    (0, '1_Mapper1_Output',      '2. Mapper1 Output\n(Channel-wise Mapping)', 'tab:orange'),
    (1, theta1_pos,              '3. EKV1 Theta_Pos\n(Positive Thresholds)', 'tab:red'),
    (1, theta1_neg,              '4. EKV1 Theta_Neg\n(Negative Thresholds)', 'tab:green'),
    (0, '2_EKV1_Output_Voltage', '5. EKV1 Output\n(Differential Voltage)', 'tab:purple'),

    # --- Row 2: Stage 2 Flow ---
    (0, '3_Dropout1_Output',     '6. Dropout1 Output\n(After Regularization)', 'tab:brown'),
    (0, '4_Mapper2_Output',      '7. Mapper2 Output\n(Stage 2 Remapping)', 'tab:orange'),
    (1, theta2_pos,              '8. EKV2 Theta_Pos\n(Stage 2 Positive)', 'tab:red'),
    (1, theta2_neg,              '9. EKV2 Theta_Neg\n(Stage 2 Negative)', 'tab:green'),
    (0, '5_EKV2_Output_Voltage', '10. EKV2 Output\n(Final EKV Response)', 'tab:purple'),

    # --- Row 3: Final Processing ---
    (0, '6_Dropout2_Output',     '11. Dropout2 Output\n(Final Regularization)', 'tab:brown'),
    (0, '7_Layer3_Final_Output', '12. Layer3 Final\n(After Residual & ReLU)', 'tab:cyan'),
    (2, [theta1_pos, theta1_neg],'13. EKV1 Theta Diff\n(Pos vs Neg Distribution)', ['red', 'green']),
    (2, [theta2_pos, theta2_neg],'14. EKV2 Theta Diff\n(Pos vs Neg Distribution)', ['red', 'green']),
    (3, l3.mapper1,              '15. Mapper Params\n(Scale & Bias Distributions)', ['blue', 'orange']),
]

for ax, (d_type, source, title, color) in zip(axes.flat, plot_configs):
    ax.clear()
    
    if d_type == 0:  # 激活值
        data = activations[source].cpu().numpy().flatten()
        counts, bins, _ = ax.hist(data, bins=100, density=False, 
                                 weights=np.ones_like(data)/len(data),
                                 color=color, alpha=0.75, edgecolor='black', linewidth=0.3)
        
    elif d_type == 1:  # 单个参数分布
        data = source
        counts, bins, _ = ax.hist(data, bins=100, density=False,
                                 weights=np.ones_like(data)/len(data),
                                 color=color, alpha=0.75, edgecolor='black', linewidth=0.3)
        
    elif d_type == 2:  # 对比分布 (Theta Pos vs Neg)
        data_pos, data_neg = source
        ax.hist(data_pos, bins=100, density=False, weights=np.ones_like(data_pos)/len(data_pos),
                color=color[0], alpha=0.6, label='Theta_Pos', edgecolor='black', linewidth=0.3)
        ax.hist(data_neg, bins=100, density=False, weights=np.ones_like(data_neg)/len(data_neg),
                color=color[1], alpha=0.6, label='Theta_Neg', edgecolor='black', linewidth=0.3)
        ax.legend(fontsize=8)
        
    elif d_type == 3:  # Mapper参数分布
        mapper = source
        scale_data = mapper.scale.detach().cpu().numpy().flatten()
        bias_data = mapper.bias.detach().cpu().numpy().flatten()
        
        ax.hist(scale_data, bins=50, density=False, weights=np.ones_like(scale_data)/len(scale_data),
                color=color[0], alpha=0.6, label='Scale', edgecolor='black', linewidth=0.3)
        ax.hist(bias_data, bins=50, density=False, weights=np.ones_like(bias_data)/len(bias_data),
                color=color[1], alpha=0.6, label='Bias', edgecolor='black', linewidth=0.3)
        ax.legend(fontsize=8)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 统计信息 (只对单数据分布显示)
    if d_type in [0, 1]:
        stats = f"Min: {data.min():.2f}\nMax: {data.max():.2f}\nMean: {data.mean():.2f}\nStd: {data.std():.2f}"
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=8, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

# 隐藏多余的子图
for i in range(len(plot_configs), 15):
    axes.flat[i].set_visible(False)

# 保存图片
save_name = '5-1-Differential_EKV_Hybrid_Analysis.png'
plt.savefig(save_name, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {save_name}")

# ================== 6. 额外分析：差分效果验证 ==================
print("\n" + "="*60)
print("差分EKV关键指标分析:")
print("="*60)

# 分析EKV1的差分输出
ekv1_output = activations['2_EKV1_Output_Voltage'].cpu().numpy()
print(f"EKV1输出统计:")
print(f"  - 正值比例: {np.mean(ekv1_output > 0)*100:.2f}%")
print(f"  - 负值比例: {np.mean(ekv1_output < 0)*100:.2f}%")
print(f"  - 零值比例: {np.mean(ekv1_output == 0)*100:.2f}%")

# 分析Theta参数对称性
theta1_diff_mean = np.mean(theta1_pos - theta1_neg)
theta2_diff_mean = np.mean(theta2_pos - theta2_neg)
print(f"\nTheta参数对称性:")
print(f"  - EKV1 Theta_Pos vs Theta_Neg 均值差: {theta1_diff_mean:.3f}V")
print(f"  - EKV2 Theta_Pos vs Theta_Neg 均值差: {theta2_diff_mean:.3f}V")

# 分析Mapper参数
print(f"\nMapper参数范围:")
print(f"  - Mapper1 Scale: [{l3.mapper1.scale.min().item():.3f}, {l3.mapper1.scale.max().item():.3f}]")
print(f"  - Mapper1 Bias:  [{l3.mapper1.bias.min().item():.3f}, {l3.mapper1.bias.max().item():.3f}]")
print(f"  - Mapper2 Scale: [{l3.mapper2.scale.min().item():.3f}, {l3.mapper2.scale.max().item():.3f}]")
print(f"  - Mapper2 Bias:  [{l3.mapper2.bias.min().item():.3f}, {l3.mapper2.bias.max().item():.3f}]")

print("\n分析完成！")