import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# V4: 导入 AMP
from torch.cuda.amp import autocast, GradScaler

# --- 1. EKV 模型与超参数 ---

# EKV 模型物理参数
EKV_ALPHA = 0.0005625
EKV_N = 1.5
EKV_VT = 0.025
EKV_VD = 0.1 

# --- V3/V4 超参数 (保持 V3 的物理参数) ---
R_TIA = 1e4               # TIA 增益
INPUT_VOLTAGE_SCALE = 9.0 # 物理范围 (0, 9V)
INIT_THETA_MEAN = 4.5     # 匹配 (0, 9V) 的中点
INIT_THETA_STD = 0.5      
BN_INIT_BIAS = 4.5        # 引导 BN 输出均值为 4.5V
BN_INIT_WEIGHT = 2.0      # 引导 BN 输出标准差为 2.0V

# --- 训练超参数 ---
BATCH_SIZE = 32           # 根据您的要求
EPOCHS = 30
LEARNING_RATE = 3e-4      
GRAD_CLIP_MAX_NORM = 1.0  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 核心模块 (与 V3 相同) ---

class NonLinearEKV_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.theta = nn.Parameter(torch.Tensor(
            out_channels, in_channels, kernel_size, kernel_size
        ))
        nn.init.normal_(self.theta, mean=INIT_THETA_MEAN, std=INIT_THETA_STD)
        self.register_buffer('alpha', torch.tensor(EKV_ALPHA))
        self.register_buffer('n', torch.tensor(EKV_N))
        self.register_buffer('VT', torch.tensor(EKV_VT))
        self.register_buffer('VD', torch.tensor(EKV_VD))

    def forward(self, V_in):
        V_patches = F.unfold(V_in, self.kernel_size, stride=self.stride, padding=self.padding)
        V_patches = V_patches.permute(0, 2, 1).unsqueeze(1)
        theta_flat = self.theta.view(self.out_channels, -1)
        theta_flat = theta_flat.unsqueeze(0).unsqueeze(2)
        V_diff = V_patches - theta_flat
        term1_in = V_diff / (2 * self.n * self.VT)
        term2_in = (V_diff - self.VD) / (2 * self.n * self.VT)
        term1 = F.softplus(term1_in).pow(2)
        term2 = F.softplus(term2_in).pow(2)
        I_D_individual = self.alpha * (term1 - term2)
        I_sum = I_D_individual.sum(dim=3)
        H_out = (V_in.shape[2] + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (V_in.shape[3] + 2 * self.padding - self.kernel_size) // self.stride + 1
        I_out_img = I_sum.view(V_in.shape[0], self.out_channels, H_out, W_out)
        return I_out_img 

class FixedTIA(nn.Module):
    def __init__(self, R_TIA):
        super().__init__()
        self.register_buffer('R_TIA', torch.tensor(R_TIA))
    def forward(self, I_in):
        return I_in * self.R_TIA

class NonLinearResBlock(nn.Module):
    """
    V3 逻辑: 采用 torch.clamp(0.0, 9.0)
    """
    def __init__(self, in_channels, out_channels, stride=1, R_TIA=R_TIA):
        super().__init__()
        self.conv1 = NonLinearEKV_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.tia1 = FixedTIA(R_TIA)
        self.bn1 = nn.BatchNorm2d(out_channels) 
        self.conv2 = NonLinearEKV_Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut_conv = NonLinearEKV_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.final_tia = FixedTIA(R_TIA)
        self.final_bn = nn.BatchNorm2d(out_channels) 

    def forward(self, V_in):
        I_1 = self.conv1(V_in)
        V_1_unclamped = self.bn1(self.tia1(I_1))
        V_1 = torch.clamp(V_1_unclamped, 0.0, INPUT_VOLTAGE_SCALE)
        
        I_F_x = self.conv2(V_1) 
        I_x = self.shortcut_conv(V_in)
        I_sum = I_F_x + I_x
        
        V_sum_unclamped = self.final_bn(self.final_tia(I_sum))
        V_out = torch.clamp(V_sum_unclamped, 0.0, INPUT_VOLTAGE_SCALE)
        
        return V_out

# --- 4. 完整模型: EKV-ResNet (V4 瘦网络版) ---

class EKV_ResNet(nn.Module):
    def __init__(self, block, n_blocks_per_layer, R_TIA=R_TIA):
        super().__init__()
        self.R_TIA = R_TIA
        
        # V4 修改: 通道数减半
        self.in_channels = 8
        
        # Stem (输入层)
        # V4 修改: 3 -> 8 (原为 3 -> 16)
        self.conv1 = NonLinearEKV_Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.tia1 = FixedTIA(R_TIA)
        self.bn1 = nn.BatchNorm2d(8)
        
        # 3 个 ResNet stage
        # V4 修改: 通道数 16, 32, 64 -> 8, 16, 32
        self.layer1 = self._make_layer(block, 8,  n_blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(block, 16, n_blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(block, 32, n_blocks_per_layer[2], stride=2)
        
        # Head (数字分类层)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # V4 修改: 64 -> 32
        self.fc = nn.Linear(32, 10)
        
        # V3: 应用自定义初始化
        self.apply(self._initialize_weights)

    def _make_layer(self, block, out_channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, self.R_TIA))
            self.in_channels = out_channels 
        return nn.Sequential(*layers)

    def _initialize_weights(self, m):
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, BN_INIT_WEIGHT) # gamma 设为 2.0
            nn.init.constant_(m.bias, BN_INIT_BIAS)     # beta 设为 4.5
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        V_in = x * INPUT_VOLTAGE_SCALE
        
        I_1 = self.conv1(V_in)
        V_1_unclamped = self.bn1(self.tia1(I_1))
        V_1 = torch.clamp(V_1_unclamped, 0.0, INPUT_VOLTAGE_SCALE)
        
        V_2 = self.layer1(V_1)
        V_3 = self.layer2(V_2)
        V_4 = self.layer3(V_3)
        
        out = self.avgpool(V_4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

# --- 5. 数据加载 (CIFAR-10) ---

print("... 正在加载 CIFAR-10 数据 ...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor() 
])
transform_test = transforms.Compose([
    transforms.ToTensor() 
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# --- 6. 训练循环 (V4: 引入 AMP) ---

# V4: 初始化 GradScaler
scaler = GradScaler()

# 我们仍然从 [1, 1, 1] (ResNet-8-like) 开始
print(f"... 正在初始化模型 EKV-ResNet([1, 1, 1]) ...")
print(f"训练设备: {DEVICE}")
print(f"V4 策略: 瘦网络 (8, 16, 32) + AMP + Clamp(0, 9V)")
print(f"V4 超参数 BATCH_SIZE: {BATCH_SIZE}")

model = EKV_ResNet(NonLinearResBlock, n_blocks_per_layer=[1, 1, 1], R_TIA=R_TIA).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 存储历史记录
history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

def train_model(model, trainloader, testloader, criterion, optimizer, scaler, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for i, data in enumerate(train_bar):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            
            # V4: 开启 autocast
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # V4: 使用 scaler
            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                return None
            
            # V4: unscale 梯度以便裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            
            # V4: scaler.step()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 50 == 49:
                train_bar.set_postfix(loss=f"{running_loss / (i+1):.3f}", acc=f"{100 * correct / total:.2f}%")
        
        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # --- 验证 ---
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        test_bar = tqdm(testloader, desc=f"Epoch {epoch+1}/{epochs} [Test]")
        with torch.no_grad():
            for data in test_bar:
                inputs, labels = data[0].to(device), data[1].to(device)
                # 验证时也可以用 autocast，但 no_grad() 已经很省显存了
                # with autocast():
                outputs = model(inputs) 
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                test_bar.set_postfix(loss=f"{test_loss / (len(testloader)):.3f}", acc=f"{100 * correct / total:.2f}%")

        test_loss = test_loss / len(testloader)
        test_acc = 100 * correct / total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    print("... 训练完成 ...")
    return history

# 运行训练
history = train_model(model, trainloader, testloader, criterion, optimizer, scaler, EPOCHS, DEVICE)

# --- 7. 绘图 ---
if history:
    print("... 正在绘制结果 ...")
    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("训练未成功完成，无法绘制图像。")