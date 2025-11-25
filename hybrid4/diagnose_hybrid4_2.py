import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. 复刻模型定义 (必须与 resnet_hybrid4_2.py 完全一致)
# ==========================================
class Config:
    VT = 0.025; N_FACTOR = 1.5; VD = 0.5; ALPHA = 2e-6
    BASE_TIA = 200000.0 # 注意：这里复刻你训练时的配置
    DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.in_channels, self.out_channels = in_channels, out_channels
        self.theta = nn.Parameter(torch.normal(4.0, 0.8, size=(out_channels, in_channels, kernel_size, kernel_size)))
        fan_in = in_channels * kernel_size * kernel_size
        self.current_tia_gain = cfg.BASE_TIA / (fan_in ** 0.5)
        self.register_buffer('tia_gain', torch.tensor(self.current_tia_gain))

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
        out_voltage = out_current * self.tia_gain
        h_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out_voltage.view(N, self.out_channels, h_out, w_out)

class OptimizedDoubleEKVBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.mapper1 = AdaptiveVoltageMapper(init_scale=1.5, init_bias=3.5)
        self.ekv1 = NormalizedEKVConv2d(in_planes, planes, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.mapper2 = AdaptiveVoltageMapper(init_scale=0.8, init_bias=3.0) 
        self.ekv2 = NormalizedEKVConv2d(planes, planes, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.bn1(self.ekv1(self.mapper1(x)))
        out = self.bn2(self.ekv2(self.mapper2(out)))
        return F.relu(out + identity)

class ResNet8(nn.Module):
    # 简化的 ResNet8 容器，用于加载权重
    def __init__(self):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 1, 1)
        self.layer2 = self._make_layer(32, 1, 2)
        self.layer3 = self._make_layer_ekv(64, 1, 2) # 重点
        self.linear = nn.Linear(64, 10)

    def _make_layer(self, planes, num_blocks, stride): # 标准层
        return nn.Sequential(nn.Conv2d(self.in_planes, planes, 3, stride, 1, bias=False), nn.BatchNorm2d(planes),
                             nn.Conv2d(planes, planes, 3, 1, 1, bias=False), nn.BatchNorm2d(planes))
        self.in_planes = planes # (Bug fix logic for simple structure)
    
    def _make_layer_ekv(self, planes, num_blocks, stride):
        # 这里手动模拟 ResNet 的构建逻辑，确保 layer3 名字对齐
        self.in_planes = 32 # Force input planes
        return nn.Sequential(OptimizedDoubleEKVBlock(32, planes, stride))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # 这里我们不跑完整模型，我们只跑 layer3 的诊断
        pass

# 重新定义一个简单的 wrapper 来只运行 Layer 3
class DiagnosisWrapper(nn.Module):
    def __init__(self, trained_layer3):
        super().__init__()
        self.layer3 = trained_layer3
    def forward(self, x):
        return self.layer3(x)

# ==========================================
# 2. 诊断逻辑
# ==========================================
def diagnose():
    # 1. 加载模型
    print(">>> Loading Model...")
    try:
        # 我们需要完整的 ResNet 类定义来加载权重，但为了简化，我们手动重构 Layer3
        # 最好的办法是实例化一个完整的 ResNet8，然后加载权重
        full_model = ResNet8() 
        # HACK: 动态替换 layer1/2 以匹配 keys
        # 但为了避免麻烦，我们假设 layer3 的 keys 是独立的。
        
        # 让我们直接加载权重字典，手动提取 Layer3
        state_dict = torch.load('resnet8_optimized_double_best.pth', map_location=cfg.DEVICE)
        
        # 实例化 Layer 3
        block = OptimizedDoubleEKVBlock(32, 64, stride=2).to(cfg.DEVICE)
        
        # 手动提取 layer3 的权重并加载
        layer3_dict = {}
        for k, v in state_dict.items():
            if 'layer3.0.' in k:
                new_key = k.replace('layer3.0.', '')
                layer3_dict[new_key] = v
        
        block.load_state_dict(layer3_dict)
        print(">>> Layer 3 Weights Loaded Successfully.")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    # 2. 准备数据 (Layer 3 的输入通常是 Batch x 32 x 16 x 16)
    # 我们用随机数据模拟上一层的输出，或者用真实数据跑前几层
    # 为了最真实的诊断，我们需要真实数据的分布。
    # 假设 Layer 2 输出是 ReLU 后的，分布在 [0, +inf)
    dummy_input = torch.abs(torch.randn(128, 32, 16, 16)).to(cfg.DEVICE) 
    
    # 3. 注册 Hooks (前向和反向)
    activations = {}
    gradients = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy().flatten()
        return hook

    def get_gradient(name):
        def hook(model, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().cpu().numpy().flatten()
        return hook

    # 注册钩子
    block.mapper1.register_forward_hook(get_activation('1_Mapper1_Out_Vg'))
    block.ekv1.register_forward_hook(get_activation('2_EKV1_Out_V'))
    block.bn1.register_forward_hook(get_activation('3_BN1_Out'))
    block.mapper2.register_forward_hook(get_activation('4_Mapper2_Out_Vg'))
    block.ekv2.register_forward_hook(get_activation('5_EKV2_Out_V'))
    
    # 注册反向传播钩子 (检查梯度)
    block.ekv1.register_full_backward_hook(get_gradient('Grad_EKV1_Out')) # EKV1 输出处的梯度
    block.ekv2.register_full_backward_hook(get_gradient('Grad_EKV2_Out')) # EKV2 输出处的梯度

    # 4. 执行前向和反向传播
    block.train() # 启用 BN 的统计更新，虽然我们不更新权重
    output = block(dummy_input)
    loss = output.mean() # 伪造一个 loss
    loss.backward()

    # 5. 绘图分析
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"Double EKV V2 Diagnosis (Acc: 69.58%)\nCheck for: Dead Zones & Vanishing Gradients", fontsize=16)

    # --- 第一行：前向传播 Layer 1 ---
    plt.subplot(3, 3, 1)
    plt.hist(activations['1_Mapper1_Out_Vg'], bins=50, color='blue', alpha=0.7)
    plt.axvline(x=4.0, color='r', linestyle='--', label='Init Theta')
    plt.title("Stage 1 Input Vg (Should cover Theta)")
    plt.legend()

    plt.subplot(3, 3, 2)
    plt.hist(activations['2_EKV1_Out_V'], bins=50, color='blue', alpha=0.7, log=True)
    plt.title("Stage 1 Output Voltage (Is it too small?)")

    plt.subplot(3, 3, 3)
    plt.hist(activations['3_BN1_Out'], bins=50, color='gray', alpha=0.7)
    plt.title("Stage 1 BN Output (Normalized)")

    # --- 第二行：前向传播 Layer 2 (关键故障点) ---
    plt.subplot(3, 3, 4)
    plt.hist(activations['4_Mapper2_Out_Vg'], bins=50, color='orange', alpha=0.7)
    plt.axvline(x=4.0, color='r', linestyle='--', label='Init Theta')
    plt.title("Stage 2 Input Vg (CRITICAL CHECK)")
    plt.legend()

    plt.subplot(3, 3, 5)
    plt.hist(activations['5_EKV2_Out_V'], bins=50, color='orange', alpha=0.7, log=True)
    plt.title("Stage 2 Output Voltage")

    # --- 第三行：反向传播 (梯度检查) ---
    plt.subplot(3, 3, 7)
    g1 = gradients.get('Grad_EKV1_Out', np.array([0]))
    plt.hist(g1, bins=50, color='green', alpha=0.7, log=True)
    plt.title(f"Gradient at EKV1 Output\nRMS: {np.sqrt(np.mean(g1**2)):.2e}")

    plt.subplot(3, 3, 8)
    g2 = gradients.get('Grad_EKV2_Out', np.array([0]))
    plt.hist(g2, bins=50, color='red', alpha=0.7, log=True)
    plt.title(f"Gradient at EKV2 Output\nRMS: {np.sqrt(np.mean(g2**2)):.2e}")

    # Theta 分布
    plt.subplot(3, 3, 9)
    theta1 = block.ekv1.theta.detach().cpu().numpy().flatten()
    theta2 = block.ekv2.theta.detach().cpu().numpy().flatten()
    plt.hist(theta1, bins=50, alpha=0.5, label='Theta 1')
    plt.hist(theta2, bins=50, alpha=0.5, label='Theta 2')
    plt.title("Learned Theta Distributions")
    plt.legend()

    plt.tight_layout()
    plt.savefig('diagnosis_v3_result.png')
    print("Diagnosis saved to diagnosis_v3_result.png")

if __name__ == '__main__':
    diagnose()