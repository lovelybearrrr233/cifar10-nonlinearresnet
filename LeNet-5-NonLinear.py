"""
================================================================================
代码撰写思路 (Code Logic Explanation)
================================================================================

这是 LeNet-5-NonLinear.py 的 v2 "torchvision" 修复版 v2。

1.  **目标**: 
    修复用户遇到的 "NameError: name 'np' is not defined"

2.  **错误分析**:
    * 该错误发生在 `if __name__ == "__main__":` 块中的 
      `np.mean(epochAccuracy)` 这一行。
    * 原因是当切换到 `torchvision` 数据加载器时，
      我(AI)错误地删除了顶层的 `import numpy as np`，
      因为我以为它不再被需要。

3.  **修改点 (已修复)**:
    * **在脚本顶部的 import 部分，重新添加 `import numpy as np`。**

4.  **其他**: 
    脚本的其余部分（`torchvision` 加载、`CrossEntropyLoss`、
    `.reshape()` 修复等）与上一版本 (v2-torchvision-FIXED) 保持一致。

================================================================================
"""

import os
from matplotlib.pyplot import imshow
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt

# (添加了 torchvision 导入)
import torchvision
import torchvision.transforms as transforms

# ================================================
# !! 修复：重新导入 numpy
import numpy as np
# ================================================

# (自定义数据加载器已全部移除)

# ----------------------------------------------------------------------------
# 2. 核心模块：非线性模拟卷积层 (保持不变)
# ----------------------------------------------------------------------------
class NonLinearConv2d(nn.Module):
    """
    模拟 NOR Flash 非线性卷积的自定义PyTorch模块。
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 alpha=0.0005625, 
                 R_TIA=0.1,  
                 n=1.5, 
                 VT=0.025, 
                 V_D=0.1,    
                 V_min=0.0, 
                 V_max=9.0
                 ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.V_min = V_min
        self.V_max = V_max
        
        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('R_TIA', torch.tensor(R_TIA))
        self.register_buffer('n', torch.tensor(n))
        self.register_buffer('VT', torch.tensor(VT))
        self.register_buffer('V_D', torch.tensor(V_D))
        self.register_buffer('denom', torch.tensor(2 * n * VT))

        self.theta = nn.Parameter(
            torch.empty(in_channels * kernel_size * kernel_size, out_channels)
        )
        nn.init.uniform_(self.theta, V_min, V_max)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (B, C_in, H_in, W_in)
        patches = F.unfold(x, self.kernel_size, 
                           stride=self.stride, 
                           padding=self.padding)
        V_j = patches.permute(0, 2, 1).unsqueeze(-1)
        theta_b = self.theta.unsqueeze(0).unsqueeze(0)

        # EKV (核心)
        term1_in = (V_j - theta_b) / self.denom
        term2_in = (V_j - theta_b - self.V_D) / self.denom
        
        term1 = F.softplus(term1_in)**2
        term2 = F.softplus(term2_in)**2
        
        current_contributions = self.alpha * (term1 - term2)
        I_k = torch.sum(current_contributions, dim=2)
        V_out_flat = I_k * self.R_TIA

        H_out = (x.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (x.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # .permute() 导致输出张量非连续
        V_out = V_out_flat.permute(0, 2, 1).view(x.shape[0], self.out_channels, H_out, W_out)

        V_normalized = self.bn(V_out)
        return V_normalized

    def clip_theta(self):
        self.theta.data.clamp_(self.V_min, self.V_max)


# (MyDataset 类已删除)

# ----------------------------------------------------------------------------
# 4. 修改后的 LeNet-5 网络结构 (已修复 .reshape)
# ----------------------------------------------------------------------------
class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.V_min = 0.0
        self.V_max = 9.0
        
        self.C1 = NonLinearConv2d(1, 6, 5, stride=1, padding=2, 
                                 V_min=self.V_min, V_max=self.V_max)
        self.S2 = NonLinearConv2d(6, 6, 2, stride=2, padding=0,
                                 V_min=self.V_min, V_max=self.V_max)
        self.C3 = NonLinearConv2d(6, 16, 5, stride=1, padding=0,
                                 V_min=self.V_min, V_max=self.V_max)
        self.S4 = NonLinearConv2d(16, 16, 2, stride=2, padding=0,
                                 V_min=self.V_min, V_max=self.V_max)
        
        self.clippable_layers = [self.C1, self.S2, self.C3, self.S4]

        # 全连接层 (保留不变)
        self.layer2 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), # C5
            nn.ReLU(),
            nn.Linear(120, 84),         # F6
            nn.ReLU(),
            nn.Linear(84,10), # Output
        )
    
    def forward(self, x):
        x = self.C1(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.S4(x) # (B, 16, 5, 5)
        
        # (修复：使用 .reshape() 处理非连续张量)
        x = x.reshape(-1, 16 * 5 * 5) 
        
        x = self.layer2(x) #
        return x

    def clip_all_theta(self):
        """在训练循环中调用，钳位所有非线性层的 theta 参数"""
        for layer in self.clippable_layers:
            layer.clip_theta()

# ----------------------------------------------------------------------------
# 5. 准确率函数 (适用于 torchvision)
# ----------------------------------------------------------------------------
def accuracy(output , label):
    # 适用于 torchvision (non-one-hot) 的版本
    pred = torch.max(output, 1)[1] # 获取预测的索引
    rightNum = torch.sum(pred.eq(label)) # 与 (B) 维的真实标签比较
    return rightNum / len(label)


# ----------------------------------------------------------------------------
# 6. 主训练循环 (Main)
# ----------------------------------------------------------------------------
if __name__ == "__main__":    
    
    # 超参数
    LEARNINGRATE = 0.005
    epochNums = 30
    SaveModelEveryNEpoch = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型实例化
    model = LeNet_5()
    
    # 参数管理器
    try:
        from imports.ParametersManager import ParametersManager
        parManager = ParametersManager(device)
    except ImportError:
        print("警告: 无法从 'imports' 导入 ParametersManager。")
        print("请确保 'imports/ParametersManager.py' 文件存在。")
        print("为使脚本可运行，将使用一个 'dummy' 管理器。")
        class DummyManager:
            def __init__(self, device): self.EpochDone = 0; self.TrainACC = []; self.TestACC = []
            def loadFromFile(self, path): print("DummyManager: 无法加载")
            def setModelParameters(self, model): pass
            def oneEpochDone(self, lr, train, test, loss): 
                self.EpochDone += 1; self.TrainACC.append(train); self.TestACC.append(test)
            def loadModelParameters(self, model): pass
            def saveToFile(self, path): print(f"DummyManager: 假装保存到 {path}")
            def show(self): print("DummyManager: 训练完成。")
        parManager = DummyManager(device)

    # 加载预训练模型
    if os.path.exists("./model.pt"):
        try:
            parManager.loadFromFile('./model.pt')
            parManager.setModelParameters(model)
        except (RuntimeError, KeyError) as e: 
            print(f"!!! 加载模型失败: {e}")
            print("!!! 架构不匹配或文件损坏。请删除 'model.pt' 文件并重新开始训练。")
            parManager = type(parManager)(device) 
            print("--- 已重置 parManager，将从头开始训练 ---")
    else:
        print('=== 未找到预训练模型，将从头开始训练 ===')

    model.to(device)
    
    # 损失函数 (使用 CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNINGRATE, momentum=0.9)
    
    # --- torchvision 数据加载器 ---
    print("\n正在加载 MNIST 数据集 (使用 torchvision)...")
    transform = transforms.Compose([
        transforms.ToTensor(), # [0, 1]
        transforms.Lambda(lambda x: x * 9.0) # [0, 9V]
    ])
    
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    TrainLoader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
    
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    TestLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)
    print("数据集加载完毕。")
    # ---

    print('len(TrainLoader):{}'.format(len(TrainLoader)))

    TrainACC = []
    TestACC = []
    GlobalLoss = []
    
    for epoch in range(epochNums):
        print("===开始本轮的Epoch {} == 总计是Epoch {}===".format(epoch, parManager.EpochDone))
        
        epochAccuracy = []
        epochLoss = []
        
        model.train() 
        for batch_id, (inputs,label) in enumerate(TrainLoader):
            
            inputs, label = inputs.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(inputs) 
            loss = criterion(output,label)
            loss.backward() 
            optimizer.step() 
            
            # !! 关键步骤：钳位(Clamping) !!
            model.clip_all_theta()
            
            epochAccuracy.append(accuracy(output,label).cpu())
            epochLoss.append(loss.item()) 
            
            if batch_id > 0 and batch_id % (len(TrainLoader) // 10) == 0: 
                # (np.mean 现在可以正常工作)
                print("    当前运行到[{}/{}], 目前Epoch准确率为：{:.2f}%，Loss：{:.6f}".format(batch_id,len(TrainLoader), np.mean(epochAccuracy) * 100, loss))
        
        TrainACC.append(np.mean(epochAccuracy)) 
        GlobalLoss.append(np.mean(epochLoss))
        
        # ==========验证集测试============
        localTestACC = []
        model.eval()
        with torch.no_grad():
            for inputs, label in TestLoader:
                inputs, label = inputs.to(device), label.to(device)
                output = model(inputs)
                localTestACC.append(accuracy(output,label).cpu())
        
        TestACC.append(np.mean(localTestACC))
        print("当前Epoch结束，训练集准确率为：{:3f}%，测试集准确率为：{:3f}%".format(TrainACC[-1] * 100, TestACC[-1] * 100))
        
        parManager.oneEpochDone(LEARNINGRATE,TrainACC[-1],TestACC[-1],GlobalLoss[-1])
        
        if epoch == epochNums - 1 or (epoch + 1) % SaveModelEveryNEpoch == 0:
            parManager.loadModelParameters(model)
            parManager.saveToFile('./model.pt')
            
    parManager.show()
    
    # 绘图
    plt.figure(figsize=(10,7))
    plt.plot(range(parManager.EpochDone),parManager.TrainACC,marker='*' ,color='r',label='Train')
    plt.plot(range(parManager.EpochDone),parManager.TestACC,marker='*' ,color='b',label='Test')

    plt.xlabel('Epochs')
    plt.ylabel('ACC')
    plt.legend()
    plt.title("LeNet-5 (Non-Linear Conv) on MNIST")

    plt.savefig('Train_NonLinear.jpg')
    plt.show()