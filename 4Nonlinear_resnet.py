# 设计思路：
# 本代码基于ResNet-20架构，针对CIFAR-10数据集实现了一个混合卷积网络。
# 为了解决全非线性卷积可能导致的梯度爆炸或消失问题，采用混合策略：初始层使用线性卷积（稳定处理原始输入），后续层使用非线性卷积（模拟浮栅晶体管的非线性转移特性）。
# 非线性卷积基于EKV模型，直接融入非线性激活，减少硬件部署中的ADC/DAC开销。
# 网络结构：conv1为线性，其后layer1为线性残差块，layer2和layer3为非线性残差块。
# 训练设置：使用SGD优化器，低初始学习率以避免爆炸，梯度裁剪，混合精度可选（未启用）。
# BatchNorm层用于归一化激活值到电压范围，残差连接缓解梯度问题。
# 池化使用平均池化，替换MaxPool以适应非线性域。
# 内存优化：batch_size=32，若爆显存可降低；非线性层仅用于部分块。
# 训练epoch设置为50，适用于快速测试；实际可根据收敛调整。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as nn_utils  # 用于梯度裁剪

# 自定义非线性卷积层，基于EKV模型模拟浮栅晶体管非线性转移特性
class NonLinearConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,
                 alpha=0.0005625, R=0.1, n=1.5, VT=0.025, V_D=0.1):
        super(NonLinearConv2d, self).__init__()
        self.in_channels = in_channels  # 输入通道数
        self.out_channels = out_channels  # 输出通道数
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)  # 内核大小
        self.stride = stride  # 步幅
        self.padding = padding  # 填充
        self.alpha = alpha  # 电流增益因子
        self.R = R  # 跨阻放大器阻值
        self.n = n  # 亚阈值摆幅因子
        self.VT = VT  # 热电压
        self.V_D = V_D  # 漏极电压（固定）
        
        # theta作为可训练参数，形状：(out_channels, in_channels, kH, kW)，初始化在4-6V范围
        self.theta = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size) * 2 + 4)

    def forward(self, input_voltage):  # 输入电压张量：batch x in_ch x H x W，已缩放到电压范围
        # 展开输入为im2col格式：(batch, outH*outW, in_ch * kH * kW)
        unfolded = F.unfold(input_voltage, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        B, C_kH_kW, outH_outW = unfolded.shape
        # 重塑为：B, outH_outW, in_ch, kH_kW
        unfolded = unfolded.view(B, self.in_channels, self.kernel_size[0]*self.kernel_size[1], outH_outW).permute(0, 3, 1, 2)
        
        # 展平theta为：out_ch, in_ch * kH_kW
        theta_flat = self.theta.view(self.out_channels, -1)
        
        # 计算每个位置的输出电流I
        I = torch.zeros(B, outH_outW, self.out_channels, device=input_voltage.device)
        for k in range(self.out_channels):  # 逐输出通道计算
            theta_k = theta_flat[k].view(1, 1, self.in_channels, self.kernel_size[0]*self.kernel_size[1])
            term1 = torch.log(1 + torch.exp((unfolded - theta_k) / (2 * self.n * self.VT))) ** 2
            term2 = torch.log(1 + torch.exp((unfolded - theta_k - self.V_D) / (2 * self.n * self.VT))) ** 2
            I_k = (term1 - term2).sum(dim=(2, 3))  # 沿输入通道和内核位置求和
            I[:, :, k] = I_k
        
        # 应用alpha和跨阻放大得到输出电压
        output_voltage = self.alpha * I * self.R
        
        # 重塑回2D特征图：首先permute为B, out_ch, outH_outW
        output_voltage = output_voltage.permute(0, 2, 1)
        
        # 计算输出高度和宽度
        out_H = (input_voltage.shape[2] + 2 * self.padding - self.kernel_size[0]) // self.stride + 1
        out_W = (input_voltage.shape[3] + 2 * self.padding - self.kernel_size[1]) // self.stride + 1
        
        # 重塑为B, out_ch, out_H, out_W
        output_voltage = output_voltage.view(B, self.out_channels, out_H, out_W)
        
        # 钳位到[0,9]V范围，避免信息丢失但防止溢出
        output_voltage = torch.clamp(output_voltage, 0, 9)
        return output_voltage

# 残差块，支持线性或非线性卷积
class BasicBlock(nn.Module):
    expansion = 1  # 通道扩展因子

    def __init__(self, in_planes, planes, stride=1, use_nonlinear=False):
        super(BasicBlock, self).__init__()
        ConvLayer = NonLinearConv2d if use_nonlinear else nn.Conv2d  # 根据标志选择卷积类型
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # BatchNorm归一化
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()  # 捷径连接
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        # 如果是线性卷积，则添加ReLU；非线性已内置激活
        if isinstance(self.conv1, nn.Conv2d):
            out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # 残差加法（假设电压或电流等价缩放）
        if isinstance(self.conv2, nn.Conv2d):
            out = F.relu(out)
        return out

# 混合ResNet-20网络，用于CIFAR-10
class HybridResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(HybridResNet, self).__init__()
        self.in_planes = 16  # 初始通道数

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 第一层线性卷积
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 3, stride=1, use_nonlinear=False)  # layer1: 线性残差块
        self.layer2 = self._make_layer(32, 3, stride=2, use_nonlinear=True)   # layer2: 非线性残差块
        self.layer3 = self._make_layer(64, 3, stride=2, use_nonlinear=True)   # layer3: 非线性残差块
        self.avgpool = nn.AvgPool2d(8)  # 平均池化
        self.fc = nn.Linear(64, num_classes)  # 全连接分类层

    def _make_layer(self, planes, num_blocks, stride, use_nonlinear):
        strides = [stride] + [1] * (num_blocks - 1)  # 步幅列表
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, use_nonlinear))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 初始卷积 + ReLU
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # 展平
        out = self.fc(out)
        return out

# 训练函数
def train():
    # 数据增强和归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),     # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # CIFAR-10均值/方差归一化
    ])
    # 若需缩放到电压范围，可添加：transforms.Lambda(lambda x: x * 10)  # [0,1] -> [0,10]V

    # 加载CIFAR-10训练集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)  # 数据加载器，batch=32（若爆显存降低）

    net = HybridResNet()  # 实例化网络
    net.cuda()  # 移到GPU
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)  # SGD优化器，低初始LR
    scheduler = CosineAnnealingLR(optimizer, T_max=50)  # 余弦退火调度器，T_max=总epoch

    # 训练循环
    for epoch in range(50):  # 50个epoch
        net.train()  # 训练模式
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()  # 移到GPU
            optimizer.zero_grad()  # 清零梯度
            outputs = net(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            nn_utils.clip_grad_norm_(net.parameters(), 1.0)  # 梯度裁剪，防止爆炸
            optimizer.step()  # 更新参数
        scheduler.step()  # 更新学习率
        # 此处可添加验证循环，监控准确率

# 运行训练
train()  # 取消注释以运行