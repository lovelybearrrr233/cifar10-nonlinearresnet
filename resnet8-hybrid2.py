# 文件名: 2_train_hybrid_nonlinear_plt_tqdm.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import sys
from tqdm import tqdm

# 1. 数据加载 (不变)
print("正在加载数据...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 2. 标准Block (不变)
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

# 3. EKV非线性卷积 (加clamp I>0防负)
class EKVNonLinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EKVNonLinearConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = 0.05625  # x100放大
        self.R = 0.1  # 固定，不训
        self.n = 1.5
        self.VT = 0.025
        self.VD = 0.1
        k_size_sq = kernel_size * kernel_size
        self.theta = nn.Parameter(torch.empty(out_channels, in_channels * k_size_sq))
        nn.init.uniform_(self.theta, 1.0, 8.0)
        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=padding)
        self.scale = nn.Parameter(torch.tensor(1.0))  # 可学放大，模拟额外放大

    def log1p_exp_stable(self, arg):
        max_arg = 20.0
        min_arg = -50.0
        arg = torch.clamp(arg, min=min_arg, max=max_arg)
        return torch.where(arg > max_arg, arg, torch.log1p(torch.exp(arg)))

    def _ekv_f(self, arg):
        return torch.pow(self.log1p_exp_stable(arg), 2)

    def forward(self, x):
        B = x.size(0)
        self.theta.data.clamp_(1.0, 8.0)
        vg = self.unfold(x)
        vg = vg.unsqueeze(1)
        vth_all = self.theta.view(1, self.out_channels, -1, 1)
        try:
            arg1 = (vg - vth_all) / (2 * self.n * self.VT)
            arg2 = (vg - vth_all - self.VD) / (2 * self.n * self.VT)
            term1 = self._ekv_f(arg1)
            term2 = self._ekv_f(arg2)
            currents = self.alpha * (term1 - term2)
            currents = torch.clamp(currents, min=0.0)  # 防负电流
            I_k_all = torch.sum(currents, dim=2)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("\n!!! OOM 错误: 显存不足。请按我们之前讨论的，实现'方案B' (循环C_out) 来解决。 !!!\n")
                raise e
            else:
                raise e
        V_out = I_k_all * self.R * self.scale  # 先放大，再BN
        H_out = (x.shape[2] + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (x.shape[3] + 2*self.padding - self.kernel_size) // self.stride + 1
        V_out = V_out.view(B, self.out_channels, H_out, W_out)
        return V_out

# 4. 非线性Block (normalize输入到0-9V，无ReLU)
class NonLinearBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(NonLinearBasicBlock, self).__init__()
        self.conv1 = EKVNonLinearConv(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = EKVNonLinearConv(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        # Normalize to 0-9V (min-max scale, per-batch)
        x_min = x.min(dim=(1,2,3), keepdim=True)[0]
        x_max = x.max(dim=(1,2,3), keepdim=True)[0]
        x = 9.0 * (x - x_min) / (x_max - x_min + 1e-8)  # 防除零
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        # Normalize conv2输入
        out_min = out.min(dim=(1,2,3), keepdim=True)[0]
        out_max = out.max(dim=(1,2,3), keepdim=True)[0]
        out = 9.0 * (out - out_min) / (out_max - out_min + 1e-8)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = torch.clamp(out, min=0.0, max=9.0)  # 最终确保
        return out

# 5. 混合ResNet (不变)
class HybridResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        self.layer3 = self.non_linear_layer(64, 1, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def non_linear_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(NonLinearBasicBlock(self.in_planes, planes, strd))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 核心：加载 (加载linear和shortcut)
model = HybridResNet().to(device)
pretrained_path = 'best_linear_resnet8.pth'
print(f"正在从 {pretrained_path} 加载预训练权重...")
try:
    pretrained_dict = torch.load(pretrained_path)
except FileNotFoundError:
    print(f"!! 错误: 未找到 '{pretrained_path}'。!!")
    print("!! 请先运行第一个脚本 (1_train_linear_resnet8_plt_tqdm.py) 来生成该文件。!!")
    sys.exit()
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and ('layer3' not in k or 'shortcut' in k)}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
print("权重加载完毕，包括linear层。")

# 渐进冻结：先全训，后冻
def set_freeze(epoch, freeze_after=30):
    if epoch >= freeze_after:
        print(f"Epoch {epoch}: 冻结前层 (conv1, layer1, layer2)")
        for name, param in model.named_parameters():
            if 'layer3' not in name and 'linear' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    else:
        print(f"Epoch {epoch}: 全参数训练 (渐进稳定)")
        for param in model.parameters():
            param.requires_grad = True

# 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 全参数初始
NUM_EPOCHS = 100
MONITOR_LAST_N_EPOCHS = 30
START_MONITORING_EPOCH = NUM_EPOCHS - MONITOR_LAST_N_EPOCHS
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
start_epoch = 0
epoch_list = []
train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []
best_acc_in_window = 0.0
best_model_state = None

# 训练循环 (加梯度监控)
def train(epoch):
    set_freeze(epoch)  # 渐进冻结
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    grad_norms = []  # 监控梯度
    progress_bar = tqdm(trainloader, desc=f'Epoch {epoch:03d} [Train]', unit='batch')
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}, skipping backward.")
            continue
        loss.backward()
        # 梯度监控
        grad_norm = nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=10.0  # 调大到10
        )
        grad_norms.append(grad_norm.item())
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix(
            Loss=f'{(train_loss / (batch_idx + 1)):.3f}',
            Acc=f'{(100. * correct / total):.2f}%',
            Grad_Norm=f'{grad_norm:.3f}'
        )
    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total
    print(f'Epoch {epoch} Grad Norm min: {min(grad_norms):.3f}, max: {max(grad_norms):.3f}, mean: {sum(grad_norms)/len(grad_norms):.3f}')
    return avg_loss, avg_acc

# test不变
def test(epoch):
    global best_acc_in_window, best_model_state
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc=f'Epoch {epoch:03d} [Test] ', unit='batch', leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar.set_postfix(
                Loss=f'{(test_loss / (batch_idx + 1)):.3f}',
                Acc=f'{(100. * correct / total):.2f}%'
            )
    avg_loss = test_loss / len(testloader)
    avg_acc = 100. * correct / total
    print(f'Epoch: {epoch} [Test] Loss: {avg_loss:.3f} | Acc: {avg_acc:.3f}%')
    if epoch >= START_MONITORING_EPOCH:
        if avg_acc > best_acc_in_window:
            print(f'*** 新高准确率 (在最后 {MONITOR_LAST_N_EPOCHS} 中): {avg_acc:.3f}% (优于 {best_acc_in_window:.3f}%) ***')
            best_acc_in_window = avg_acc
            best_model_state = model.state_dict()
    return avg_loss, avg_acc

print(f"开始训练混合非线性 ResNet-8，共 {NUM_EPOCHS} 轮...")
print(f"将在第 {START_MONITORING_EPOCH} 轮开始监控最佳模型。")
for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()
    epoch_list.append(epoch)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)

# 保存和绘图 (不变)
print(f"训练完成。在最后 {MONITOR_LAST_N_EPOCHS} 轮中的最佳准确率: {best_acc_in_window:.3f}%")
print("正在保存模型...")
torch.save(model.state_dict(), 'last_hybrid_resnet8.pth')
print("最后一个 epoch 模型已保存到 'last_hybrid_resnet8.pth'")
if best_model_state:
    torch.save(best_model_state, 'best_hybrid_resnet8.pth')
    print("最佳准确率模型 (来自监控窗口) 已保存到 'best_hybrid_resnet8.pth'")
else:
    print("警告: 未能在监控窗口中记录最佳模型。")
    print("将保存最后一个 epoch 的模型为 'best_hybrid_resnet8.pth'")
    torch.save(model.state_dict(), 'best_hybrid_resnet8.pth')

print("正在绘制训练曲线...")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_list, train_loss_history, label='Train Loss (Hybrid)')
plt.plot(epoch_list, test_loss_history, label='Test Loss (Hybrid)')
plt.axvline(x=START_MONITORING_EPOCH, color='grey', linestyle='--', label=f'Start Monitoring (Epoch {START_MONITORING_EPOCH})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs (Hybrid)')
plt.subplot(1, 2, 2)
plt.plot(epoch_list, train_acc_history, label='Train Accuracy (Hybrid)')
plt.plot(epoch_list, test_acc_history, label='Test Accuracy (Hybrid)')
plt.axvline(x=START_MONITORING_EPOCH, color='grey', linestyle='--', label=f'Start Monitoring (Epoch {START_MONITORING_EPOCH})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs. Epochs (Hybrid)')
plt.tight_layout()
plt.savefig('hybrid_resnet8_curves.png')
plt.show()
print("图像已保存为 'hybrid_resnet8_curves.png'")