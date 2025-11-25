import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm 
from torch.cuda.amp import autocast
import sys # (新增)

# =================================================================
# !! 修复: 自动将子目录添加到 Python 路径
# =================================================================
# 1. 获取当前脚本 (evaluate.py) 所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 构建子目录 (LeNet5-MNIST-Pytorch) 的路径
#    (根据您的训练日志)
model_dir = os.path.join(current_dir, "LeNet5-MNIST-Pytorch")
# 3. 将子目录添加到 sys.path
sys.path.append(model_dir)

try:
    # =================================================================
    # !! 修复: 从正确的文件名 (NonLinear_ResNet_CIFAR10) 导入
    # !! (根据您的训练日志)
    # =================================================================
    from NonLinear_ResNet_CIFAR10 import NonLinearConv2d, NonLinearBasicBlock, NonLinearResNet, NonLinearResNet15, accuracy
except ModuleNotFoundError:
    print(f"错误: 无法从 '{model_dir}/NonLinear_ResNet_CIFAR10.py' 导入模型。")
    print("请确保 'NonLinear_ResNet_CIFAR10.py' 文件存在于 'LeNet5-MNIST-Pytorch' 子目录中。")
    exit()
except ImportError as e:
    print(f"导入时发生错误: {e}")
    print("请确保 NonLinear_ResNet_CIFAR10.py 脚本没有语法错误，并且所有依赖都已安装。")
    exit()
# =================================================================

print("=== 开始评估 ===")

# --- 设置 ---
BATCH_SIZE = 32 # (保持与训练时一致)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# =================================================================
# !! 确认: 使用您指定的 'NonLinear_ResNet_best.pt'
# =================================================================
MODEL_FILE_TO_LOAD = os.path.join(model_dir, 'NonLinear_ResNet_best.pt')

if not os.path.exists(MODEL_FILE_TO_LOAD):
    print(f"错误：找不到模型文件 '{MODEL_FILE_TO_LOAD}'。")
    print("请确保您运行过训练脚本，并且它至少保存过一次最佳模型。")
    exit()

# --- 加载数据 (仅测试集) ---
print(f"--- 正在加载 CIFAR-10 测试集 (Batch Size: {BATCH_SIZE}) ---")
voltage_transform = transforms.Lambda(lambda x: x * 9.0) 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    voltage_transform
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform_test)
TestLoader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, 
                                         shuffle=False, num_workers=2)
print("数据集加载完毕。")

# --- 加载模型 ---
print(f"--- 正在加载模型: {MODEL_FILE_TO_LOAD} ---")
model = NonLinearResNet15()
model.load_state_dict(torch.load(MODEL_FILE_TO_LOAD))
model.to(device)
model.eval() # 设为评估模式
print("模型加载完毕。")

# --- 运行评估 ---
print("--- 正在测试集上运行评估... ---")
test_acc_list = []
with torch.no_grad(): # (评估时不需要梯度)
    # (修复: 换回旧的、可工作的 AMP 语法)
    with autocast():
        for inputs, labels in tqdm(TestLoader, desc="评估中"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_acc_list.append(accuracy(outputs, labels).cpu())

avg_test_acc = np.mean(test_acc_list)
print("\n" + "="*30)
print(f"评估完成。")
print(f"模型 '{MODEL_FILE_TO_LOAD}' 的最终测试集准确率: {avg_test_acc*100:.2f}%")
print("="*30)