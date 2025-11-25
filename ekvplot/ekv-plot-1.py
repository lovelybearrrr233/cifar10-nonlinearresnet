import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 配置参数
class Config:
    VT = 0.025          # 热电压
    N_FACTOR = 1.5      # 亚阈值摆幅因子
    VD = 0.5            # 漏源电压
    ALPHA = 1e-6        # 电流增益因子
    TIA_GAIN = 2000.0   # 跨阻放大器增益

# 初始化配置
cfg = Config()
phi = 2 * cfg.N_FACTOR * cfg.VT  # 2nVT（EKV模型核心参数）

# 输入电压范围：0~10V，采样400点（求导需更密集采样保证精度）
Vg = np.linspace(0, 10, 400, dtype=np.float32)
# 权重（阈值电压）：2~9V的8个等差序列
thetas = np.linspace(2, 9, 8)

# 定义EKV电流计算函数（支持torch张量，便于自动求导）
def ekv_current_torch(Vg, theta, cfg, phi):
    v_diff = Vg - theta
    f_term = F.softplus(v_diff / phi).pow(2)
    r_term = F.softplus((v_diff - cfg.VD) / phi).pow(2)
    return cfg.ALPHA * (f_term - r_term)

# 预存储电流和梯度结果
current_dict = {}
gradient_dict = {}

# 遍历每个阈值电压，计算电流和梯度
for theta in thetas:
    # 转换为torch张量并开启梯度追踪
    Vg_tensor = torch.tensor(Vg, requires_grad=True)
    theta_tensor = torch.tensor(theta, dtype=torch.float32)
    
    # 计算电流
    I = ekv_current_torch(Vg_tensor, theta_tensor, cfg, phi)
    current_dict[theta] = I.detach().numpy()
    
    # 计算梯度（dI/dVg）：对Vg求导
    I.sum().backward()  # 求和后求导，避免逐元素求导的维度问题
    dI_dVg = Vg_tensor.grad.detach().numpy()
    gradient_dict[theta] = dI_dVg

# 绘制双张子图（转移特性 + 梯度曲线）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

# 子图1：转移特性曲线（I-Vg）
for theta in thetas:
    ax1.plot(Vg, current_dict[theta], label=f'V_th = {theta:.1f}V')
ax1.set_xlabel('Gate Voltage (V)', fontsize=12)
ax1.set_ylabel('Drain Current (A)', fontsize=12)
ax1.set_title('EKV Model Transfer Characteristics (I-Vg)', fontsize=14, fontweight='bold')
ax1.grid(alpha=0.3)
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(0, 10)
ax1.set_ylim(bottom=0)  # 电流非负

# 子图2：梯度曲线（dI/dVg - Vg）
for theta in thetas:
    ax2.plot(Vg, gradient_dict[theta], label=f'V_th = {theta:.1f}V')
ax2.set_xlabel('Gate Voltage (V)', fontsize=12)
ax2.set_ylabel('dI/dVg (A/V)', fontsize=12)
ax2.set_title('Gradient of Transfer Characteristics (dI/dVg)', fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xlim(0, 10)

# 保存图片（可选）
plt.savefig('ekv_transfer_and_gradient.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印关键信息（可选）
print("=== 梯度计算完成 ===")
print(f"输入电压范围：0~10V（采样{len(Vg)}点）")
print(f"阈值电压列表：{[round(t,1) for t in thetas]}V")