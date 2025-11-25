# 文件名: hybrid_resnet_ekv_hybrid3_final.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================== 数据 ==================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

# ================== 辅助模块 ==================
class LayerScale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value))

    def forward(self, x):
        return x * self.scale

class InputScaling(nn.Module):
    def __init__(self, scale_init=4.5, shift_init=3.8):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.shift = nn.Parameter(torch.tensor(shift_init))

    def forward(self, x):
        return x * self.scale + self.shift

# ================== EKVConv2d ==================
class EKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        k = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.theta = nn.Parameter(torch.empty(out_channels, k))
        nn.init.uniform_(self.theta, 2.8, 4.8)          # 窄窗口初始化
        
        self.alpha = 5.625e-4
        self.n = 1.5
        self.VT = 0.026
        self.VD = 0.1
        
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=padding, stride=stride)

    def log1p_exp(self, x):
        return torch.log1p(torch.exp(torch.clamp(x, -30, 30)))   # 唯一数值保护

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)                              # (B, C*k*k, L)
        L = patches.shape[-1]
        
        vg = patches.unsqueeze(1)                             # (B,1,C*k*k,L)
        theta = self.theta.view(1, -1, self.theta.shape[-1], 1)
        
        arg1 = (vg - theta) / (self.n * self.VT)
        arg2 = (vg - theta - self.VD) / (self.n * self.VT)
        
        I = self.alpha * (self.log1p_exp(arg1)**2 - self.log1p_exp(arg2)**2)
        out = I.sum(dim=2).view(B, -1, 
                                (H + 2*self.padding - self.kernel_size[0])//self.stride + 1,
                                (W + 2*self.padding - self.kernel_size[1])//self.stride + 1)
        return out

# ================== 线性 Block（冻结） ==================
class LinearBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# ================== 非线性 EKV Block（电压相加，硬件可接受） ==================
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
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.prescale(x)
        out = self.ls1(self.conv1(out))
        out = self.bn1(out)
        
        out = self.ls2(self.conv2(out))
        out = self.bn2(out)
        
        out += identity
        return out

# ================== 混合 ResNet8 ==================
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
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ================== 主程序 ==================
model = HybridResNet8().to(device)

# 加载预训练线性权重
pretrained_path = 'best_linear_resnet8.pth'
if not os.path.exists(pretrained_path):
    print("未找到 best_linear_resnet8.pth，请先训练纯线性版本！")
    sys.exit()

pretrained = torch.load(pretrained_path, map_location=device)
model_dict = model.state_dict()
pretrained = {k: v for k, v in pretrained.items() if 'layer3' not in k and k in model_dict}
model_dict.update(pretrained)
model.load_state_dict(model_dict)

# 冻结线性层
for name, p in model.named_parameters():
    if 'layer3' not in name and 'fc' not in name:
        p.requires_grad = False
    else:
        print(f"训练参数: {name}")

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.003, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

train_accs, test_accs = [], []
best_acc = 0.0
best_state = None
monitor_start_epoch = 80  # 最后20轮开始监控

print("开始训练 Hybrid3 EKV ResNet8（最后20轮选最佳）...")
for epoch in range(100):
    model.train()
    correct = total = 0
    for x, y in tqdm(trainloader, desc=f'Epoch {epoch:03d}', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    train_acc = 100. * correct / total
    train_accs.append(train_acc)
    
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    test_acc = 100. * correct / total
    test_accs.append(test_acc)
    
    if epoch >= monitor_start_epoch and test_acc > best_acc:
        best_acc = test_acc
        best_state = model.state_dict()
        print(f" *** 新最佳 (最后20轮): {test_acc:.3f}% ***")
    
    print(f"Epoch {epoch:03d} | Train {train_acc:.2f}% | Test {test_acc:.3f}% | Best(last20) {best_acc:.3f}%")
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'final_hybrid3_ekv.pth')
if best_state is not None:
    torch.save(best_state, 'best_hybrid3_ekv_last20.pth')
    print(f"\n=== 训练完成！最后20轮最佳模型已保存为: best_hybrid3_ekv_last20.pth (精度: {best_acc:.3f}%) ===")
else:
    torch.save(model.state_dict(), 'best_hybrid3_ekv_last20.pth')
    print("\n=== 训练完成！(未触发监控窗口，使用最后模型保存) ===")

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(train_accs, label='Train Acc')
plt.plot(test_accs, label='Test Acc')
plt.axvline(80, color='gray', linestyle='--', label='Monitor Start (Epoch 80)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Hybrid3 EKV-ResNet8 on CIFAR-10')
plt.legend()
plt.grid(True)
plt.savefig('hybrid3_ekv_accuracy_curve.png', dpi=200, bbox_inches='tight')
plt.show()
print("准确率曲线已保存为 hybrid3_ekv_accuracy_curve.png")