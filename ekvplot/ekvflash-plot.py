import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters (超参数)
alpha = 0.0005625  # Current gain factor
n = 1.5            # Subthreshold swing factor
VT = 0.025         # Thermal voltage (V)
V_D = 0.1          # Drain-source voltage (fixed)
theta_list = np.linspace(1, 8, 8)  # Threshold voltage: 8 uniform values from 1~8V
V_G = np.linspace(0, 10, 200)      # Gate voltage: 200 sampling points from 0~10V


# Define EKV model to calculate drain current
def ekv_current(V_g, theta, alpha, n, VT, V_D):
    term1 = np.log(1 + np.exp((V_g - theta) / (2 * n * VT))) ** 2
    term2 = np.log(1 + np.exp((V_g - theta - V_D) / (2 * n * VT))) ** 2
    return alpha * (term1 - term2)


# Calculate and print I_D when V_G=8V for each threshold voltage
print("Drain current (I_D) at V_G=8V for different threshold voltages (V_th):")
print("-" * 60)
print(f"{'V_th (V)':<12} {'I_D (A)':<20} {'I_D (uA)':<20}")  # 补充微安单位更易读
print("-" * 60)
for theta in theta_list:
    I_D_at_8V = ekv_current(8, theta, alpha, n, VT, V_D)
    print(f"{theta:<12.1f} {I_D_at_8V:<20.6e} {I_D_at_8V*1e6:<20.2f}")
print("-" * 60)


# Plot transfer characteristic curves (全英文标注)
plt.figure(figsize=(10, 6))
for theta in theta_list:
    I_D = ekv_current(V_G, theta, alpha, n, VT, V_D)
    plt.plot(V_G, I_D, label=f'$V_{{th}}$ = {theta:.1f}V')

# Chart settings (English labels)
plt.xlabel('Gate Voltage $V_G$ (V)', fontsize=12)
plt.ylabel('Drain Current $I_D$ (A)', fontsize=12)
plt.title('EKV Model Transistor Transfer Characteristic Curves', fontsize=14)
plt.grid(alpha=0.3)
plt.legend(loc='upper left', fontsize=10)
plt.xlim(0, 10)
plt.tight_layout()  # 自动调整布局防止标签被截断
plt.savefig('ekv_transfer_characteristics.png', dpi=300)
plt.show()