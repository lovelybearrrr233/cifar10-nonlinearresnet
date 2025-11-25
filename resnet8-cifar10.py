# 文件名: 1_train_linear_resnet8_plt_window.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. 数据加载 (CIFAR-10)
print("正在加载数据...")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

# 2. 模型定义 (标准 ResNet-8)
# (这部分代码与之前完全相同，此处省略以便简洁)
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
        out += self.shortcut(x) # 电压相加
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
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

def resnet8():
    return ResNet(BasicBlock, [1, 1, 1])

model = resnet8().to(device)

# 3. 训练设置
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

# --- 修改点：定义训练和监控轮数 ---
NUM_EPOCHS = 200 # 总训练轮数
MONITOR_LAST_N_EPOCHS = 30 # 只在最后 30 轮中寻找最佳
START_MONITORING_EPOCH = NUM_EPOCHS - MONITOR_LAST_N_EPOCHS

# 调整学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
start_epoch = 0
# --- 修改结束 ---

# --- 用于绘图的历史记录列表 ---
epoch_list = []
train_acc_history = []
test_acc_history = []
train_loss_history = []
test_loss_history = []
# -----------------------------

# --- 用于跟踪最后N轮中的最佳模型 ---
best_acc_in_window = 0.0
best_model_state = None
# -------------------------

# 4. 训练与验证循环
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = train_loss / len(trainloader)
    avg_acc = 100. * correct / total
    print(f'Epoch: {epoch} [Train] Loss: {avg_loss:.3f} | Acc: {avg_acc:.3f}%')
    return avg_loss, avg_acc

def test(epoch):
    # 声明修改全局变量
    global best_acc_in_window, best_model_state 
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = test_loss / len(testloader)
    avg_acc = 100. * correct / total
    print(f'Epoch: {epoch} [Test]  Loss: {avg_loss:.3f} | Acc: {avg_acc:.3f}%')

    # --- 修改点 ---
    # 只有在达到监控窗口后，才开始跟踪最佳模型
    if epoch >= START_MONITORING_EPOCH:
        if avg_acc > best_acc_in_window:
            print(f'*** 新高准确率 (在最后 {MONITOR_LAST_N_EPOCHS} 轮中): {avg_acc:.3f}% (优于 {best_acc_in_window:.3f}%) ***')
            best_acc_in_window = avg_acc
            best_model_state = model.state_dict() # 存入内存，而不是硬盘
    # --- 修改结束 ---
        
    return avg_loss, avg_acc

print(f"开始训练标准 ResNet-8，共 {NUM_EPOCHS} 轮...")
print(f"将在第 {START_MONITORING_EPOCH} 轮开始监控最佳模型。")

for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    scheduler.step()
    
    # 记录数据用于绘图
    epoch_list.append(epoch)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    test_loss_history.append(test_loss)
    test_acc_history.append(test_acc)


print(f"训练完成。在最后 {MONITOR_LAST_N_EPOCHS} 轮中的最佳准确率: {best_acc_in_window:.3f}%")

# --- 训练结束后保存模型 ---
print("正在保存模型...")
# 1. 保存最后一个 epoch 的模型
torch.save(model.state_dict(), 'last_linear_resnet8.pth')
print("最后一个 epoch 模型已保存到 'last_linear_resnet8.pth'")

# 2. 保存最佳模型
if best_model_state:
    torch.save(best_model_state, 'best_linear_resnet8.pth')
    print("最佳准确率模型 (来自监控窗口) 已保存到 'best_linear_resnet8.pth'")
else:
    # 这种情况可能发生于MONITOR_LAST_N_EPOCHS=0，或者模型在监控窗口内准确率为0
    print("警告: 未能在监控窗口中记录最佳模型。")
    print("将保存最后一个 epoch 的模型为 'best_linear_resnet8.pth'")
    torch.save(model.state_dict(), 'best_linear_resnet8.pth')
# --- 保存结束 ---


# --- 训练结束后开始绘图 ---
print("正在绘制训练曲线...")
plt.figure(figsize=(12, 5))

# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(epoch_list, train_loss_history, label='Train Loss')
plt.plot(epoch_list, test_loss_history, label='Test Loss')
# 添加一个垂直线标记监控窗口的开始
plt.axvline(x=START_MONITORING_EPOCH, color='grey', linestyle='--', label=f'Start Monitoring (Epoch {START_MONITORING_EPOCH})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs (Linear)')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epoch_list, train_acc_history, label='Train Accuracy')
plt.plot(epoch_list, test_acc_history, label='Test Accuracy')
# 添加一个垂直线标记监控窗口的开始
plt.axvline(x=START_MONITORING_EPOCH, color='grey', linestyle='--', label=f'Start Monitoring (Epoch {START_MONITORING_EPOCH})')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs. Epochs (Linear)')

plt.tight_layout()
plt.savefig('linear_resnet8_curves.png') # 保存图像
plt.show() # 显示图像
print("图像已保存为 'linear_resnet8_curves.png'")
# ---------------------------