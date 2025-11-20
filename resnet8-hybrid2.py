# 文件名: hybrid_resnet_ekv_v2_optimized.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# ================== 数据预处理保持标准 ==================
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

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, pin_memory=True)

# ================== 关键：可训练输入缩放层 ==================
class InputScaling(nn.Module):
    def __init__(self, scale_init=5.0, shift_init=4.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_init))   # γ 控制工作区
        self.shift = nn.Parameter(torch.tensor(shift_init))   # β 控制中心点

    def forward(self, x):
        return x * self.scale + self.shift

# ================== 优化后的 EKV 卷积 ==================
class EKVConv2d(nn.Module):
    """
    优化点：
    1. theta 窄窗口初始化 [2.5, 4.5]V，让初始过阈值电压适中
    2. 加入 per-layer 可训练 gain（等效TIA增益）
    3. 彻底去掉 clamp
    4. 使用数值稳定的 log1p_exp
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        k = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.theta = nn.Parameter(torch.empty(out_channels, k))
        nn.init.uniform_(self.theta, 2.5, 4.5)   # 窄窗口！关键
        
        self.gain = nn.Parameter(torch.tensor(50.0))   # 可训练输出放大，初始50倍左右
        
        # EKV 物理常数（固定）
        self.alpha = 5.625e-4
        self.n = 1.5
        self.VT = 0.026
        self.VD = 0.1
        
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=padding, stride=stride)

    def log1p_exp(self, x):
        return torch.log1p(torch.exp(torch.clamp(x, -30, 30)))

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)  # (B, C*kH*kW, L)
        L = patches.shape[-1]
        
        vg = patches.unsqueeze(1)  # (B,1,C*kH*kW,L)
        theta = self.theta.view(1, -1, self.theta.shape[-1], 1)  # (1,O,C*kH*kW,1)
        
        arg1 = (vg - theta) / (self.n * self.VT)
        arg2 = (vg - theta - self.VD) / (self.n * self.VT)
        
        I = self.alpha * (self.log1p_exp(arg1)**2 - self.log1p_exp(arg2)**2)
        I_sum = I.sum(dim=2)  # (B,O,L)
        
        out = I_sum * self.gain
        
        h_out = (H + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        w_out = (W + 2*self.padding - self.kernel_size[1]) // self.stride + 1
        out = out.view(B, -1, h_out, w_out)
        return out

# ================== 非线性残差块 ==================
class EKVBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.scale1 = InputScaling(scale_init=4.0, shift_init=3.5)
        self.conv1 = EKVConv2d(in_planes, planes, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.scale2 = InputScaling(scale_init=3.0, shift_init=3.0)
        self.conv2 = EKVConv2d(planes, planes, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride=stride) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_planes, planes, 1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.scale1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        out = self.scale2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        return out

# ================== 标准线性 Block（冻结） ==================
class LinearBlock(nn.Module):
    expansion = 1
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

# ================== 混合 ResNet ==================
class HybridResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = nn.Sequential(LinearBlock(16, 16))
        self.layer2 = nn.Sequential(LinearBlock(16, 32, stride=2))
        self.layer3 = nn.Sequential(EKVBlock(32, 64, stride=2))   # 只有最后一大层是非线性的
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ================== 主程序 ==================
model = HybridResNet8().to(device)

# 加载预训练线性模型
pretrained = torch.load('best_linear_resnet8.pth', map_location=device)
model_dict = model.state_dict()
pretrained = {k: v for k, v in pretrained.items() if k in model_dict and 'layer3' not in k}
model_dict.update(pretrained)
model.load_state_dict(model_dict)

# 冻结前面的线性层
for name, p in model.named_parameters():
    if 'layer3' not in name and 'fc' not in name:
        p.requires_grad = False
    else:
        print(f"训练参数: {name}")

optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train_epoch():
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for x, y in tqdm(trainloader, desc='Train', leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_sum += loss.item() * y.size(0)
        _, pred = out.max(1)
        total += y.size(0)
        correct += pred.eq(y).sum().item()
    return loss_sum/total, 100.*correct/total

def test_epoch():
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(testloader, desc='Test', leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            total += y.size(0)
            correct += pred.eq(y).sum().item()
    return 100.*correct/total

best_acc = 0
for epoch in range(100):
    train_loss, train_acc = train_epoch()
    test_acc = test_epoch()
    scheduler.step()
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_hybrid_ekv.pth')
    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best: {best_acc:.2f}%")

print(f"最终最佳精度: {best_acc:.3f}%")