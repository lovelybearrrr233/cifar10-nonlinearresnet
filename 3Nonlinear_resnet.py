import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torch.cuda.amp import autocast, GradScaler

# --- 1. EKV 模型与超参数 ---
EKV_ALPHA = 0.0005625
EKV_N = 1.5
EKV_VT = 0.025
EKV_VD = 0.1 

# --- V6 超参数 ---
R_TIA = 1e4               
INPUT_VOLTAGE_SCALE = 9.0 # (0, 9V) 物理范围
INIT_THETA_MEAN = 4.5     
INIT_THETA_STD = 1.5      # V6 修改: 增大标准差 (原为 0.5)，增加多样性
BN_INIT_BIAS = 4.5        # 模拟 BN 的 beta
BN_INIT_WEIGHT = 2.0      # 模拟 BN 的 gamma

# --- 训练超参数 ---
BATCH_SIZE = 32           
EPOCHS = 30
LEARNING_RATE = 3e-4      
GRAD_CLIP_MAX_NORM = 1.0  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. 核心模块: (V6 Theta 钳位) ---

class NonLinearEKV_Conv2d(nn.Module):
    """
    V6: 
    1. Kaiming-to-Threshold 初始化 (STD=1.5)
    2. 在 forward 中 clamp(theta) 
    """
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
        
        self.init_kaiming_to_threshold()
        
        self.register_buffer('alpha', torch.tensor(EKV_ALPHA))
        self.register_buffer('n', torch.tensor(EKV_N))
        self.register_buffer('VT', torch.tensor(EKV_VT))
        self.register_buffer('VD', torch.tensor(EKV_VD))

    def init_kaiming_to_threshold(self):
        with torch.no_grad():
            W_kaiming = torch.empty_like(self.theta)
            nn.init.kaiming_normal_(W_kaiming, a=0, mode='fan_in', nonlinearity='relu')
            W_std = W_kaiming.std()
            if W_std == 0: W_std = 1.0
                
            # V6: 使用新的 INIT_THETA_STD
            self.theta.data = INIT_THETA_MEAN - (W_kaiming / W_std) * INIT_THETA_STD
            self.theta.data = torch.clamp(self.theta.data, 0.1, 8.9) # 钳位在 (0,9V) 内部

    def forward(self, V_in):
        # V6 关键: 将 theta 钳位到物理范围 (0, 9V)
        theta_clamped = torch.clamp(self.theta, 0.0, INPUT_VOLTAGE_SCALE)
        
        V_patches = F.unfold(V_in, self.kernel_size, stride=self.stride, padding=self.padding)
        V_patches = V_patches.permute(0, 2, 1).unsqueeze(1)
        
        # V6: 使用钳位后的 theta
        theta_flat = theta_clamped.view(self.out_channels, -1)
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
    """ (与 V5 相同, 依赖 V3 钳位逻辑) """
    def __init__(self, in_channels, out_channels, stride=1, R_TIA=R_TIA):
        super().__init__()
        self.conv1 = NonLinearEKV_Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.tia1 = FixedTIA(R_TIA)
        self.bn1 = nn.BatchNorm2d(out_channels) # 模拟 BN
        self.conv2 = NonLinearEKV_Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut_conv = NonLinearEKV_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.final_tia = FixedTIA(R_TIA)
        self.final_bn = nn.BatchNorm2d(out_channels) # 模拟 BN

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

# --- 4. 完整模型: EKV-ResNet (V6 混合网络版) ---

class EKV_ResNet(nn.Module):
    def __init__(self, block, n_blocks_per_layer, R_TIA=R_TIA):
        super().__init__()
        self.R_TIA = R_TIA
        
        # V6: "瘦网络" 通道数
        digital_channels = 8
        analog_channels = [8, 16, 32]
        
        self.in_channels = digital_channels # EKV 层的 in_channels
        
        # V6: 数字 Stem
        self.stem_conv = nn.Conv2d(3, digital_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(digital_channels)
        self.stem_relu = nn.ReLU(inplace=True)
        
        # V6: 模拟 Backbone
        self.layer1 = self._make_layer(block, analog_channels[0], n_blocks_per_layer[0], stride=1)
        self.layer2 = self._make_layer(block, analog_channels[1], n_blocks_per_layer[1], stride=2)
        self.layer3 = self._make_layer(block, analog_channels[2], n_blocks_per_layer[2], stride=2)
        
        # 数字 Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(analog_channels[2], 10)
        
        # V6: 应用自定义初始化
        self.apply(self._initialize_weights)

    def _make_layer(self, block, out_channels, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, self.R_TIA))
            self.in_channels = out_channels 
        return nn.Sequential(*layers)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # V6: 数字 Stem 的标准 Kaiming 初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            # V6: 区分数字 BN 和模拟 BN
            if m == self.stem_bn:
                # 数字 BN: 标准初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            else:
                # 模拟 BN (在 ResBlock 内部): 引导初始化
                nn.init.constant_(m.weight, BN_INIT_WEIGHT) # gamma
                nn.init.constant_(m.bias, BN_INIT_BIAS)     # beta
        elif isinstance(m, nn.Linear):
            # 数字 Head 的标准 Kaiming 初始化
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
        # NonLinearEKV_Conv2d 的 theta 已在其 __init__ 中被 V6 策略初始化

    def forward(self, x):
        # x 是 CIFAR-10 归一化后的 (mean 0, std 1)
        
        # 1. V6: 数字 Stem
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        
        # 2. V6: 数字-模拟 "驱动器"
        #    将 (mean 0, std 1) 映射到 (0, 9V) 的 V_G 范围
        V_in_analog = torch.clamp(x * BN_INIT_WEIGHT + BN_INIT_BIAS, 0.0, INPUT_VOLTAGE_SCALE)
        
        # 3. 模拟 Backbone
        V_2 = self.layer1(V_in_analog)
        V_3 = self.layer2(V_2)
        V_4 = self.layer3(V_3)
        
        # 4. 数字 Head
        out = self.avgpool(V_4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

# --- 5. 数据加载 (V6: 必须使用 Normalize) ---

print("... 正在加载 CIFAR-10 数据 ...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # V6: 必须为数字 Stem 添加标准归一化
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # V6: 必须为数字 Stem 添加标准归一化
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

# --- 6. 训练循环 (与 V5 相同) ---

scaler = GradScaler()
print(f"... 正在初始化模型 EKV-ResNet([1, 1, 1]) ...")
print(f"训练设备: {DEVICE}")
print(f"V6 策略: 混合数字 Stem + 模拟 Backbone + Theta Clamp + Theta STD={INIT_THETA_STD}")
print(f"V6 超参数 BATCH_SIZE: {BATCH_SIZE}, Start LR: {LEARNING_RATE}")

model = EKV_ResNet(NonLinearResBlock, n_blocks_per_layer=[1, 1, 1], R_TIA=R_TIA).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': [],
    'lr': []
}

def train_model(model, trainloader, testloader, criterion, optimizer, scaler, scheduler, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        current_lr = scheduler.get_last_lr()[0]
        history['lr'].append(current_lr)
        
        train_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train] LR={current_lr:.1E}")
        for i, data in enumerate(train_bar):
            # V6: inputs 现在是归一化后的数据
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            try:
                scaler.scale(loss).backward()
            except RuntimeError as e:
                print(f"RuntimeError: {e}")
                return None
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
            
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
        
        scheduler.step()
    
    print("... 训练完成 ...")
    return history

# 运行训练
history = train_model(model, trainloader, testloader, criterion, optimizer, scaler, scheduler, EPOCHS, DEVICE)

# --- 7. 绘图 (与 V5 相同) ---
if history:
    print("... 正在绘制结果 ...")
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("训练未成功完成，无法绘制图像。")