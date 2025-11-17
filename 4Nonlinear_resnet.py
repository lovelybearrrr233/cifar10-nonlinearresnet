import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import gc

# 设置设备
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 设置内存优化环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class MemoryEfficientNonLinearConv2d(nn.Module):
    """
    内存高效的非线性卷积层
    使用通道分组和逐点计算避免大张量
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 alpha=0.0005625, n=1.5, VT=0.025, R=0.1, voltage_range=(0, 10), groups=8):
        super(MemoryEfficientNonLinearConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.voltage_range = voltage_range
        self.groups = groups
        
        # EKV模型参数
        self.alpha = alpha
        self.n = n
        self.VT = VT
        self.R = R
        self.VD = 0.1
        
        # 可训练参数：阈值电压theta
        self.theta = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1 + 4.0)
        
        # 跨阻放大器增益
        self.transimpedance_gain = R
        
        # BatchNorm归一化
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # 初始化权重
        self._initialize_weights()
        
        # 预计算常量
        self.register_buffer('two_n_VT', torch.tensor(2 * n * VT))
    
    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.theta, a=0, mode='fan_in', nonlinearity='relu')
    
    def efficient_ekv_model(self, V, theta):
        """
        高效EKV模型计算
        避免创建大中间张量
        """
        # 确保电压在合理范围内
        V = torch.clamp(V, self.voltage_range[0], self.voltage_range[1])
        
        # 计算差值
        diff = V - theta
        
        # 使用更高效的近似计算
        # 避免同时计算所有位置，使用逐元素计算
        term1 = diff / self.two_n_VT
        term2 = (diff - self.VD) / self.two_n_VT
        
        # 使用分段计算避免大张量
        # 对于正区域使用近似，负区域使用零
        log_term1 = torch.zeros_like(term1)
        log_term2 = torch.zeros_like(term2)
        
        # 只对正区域计算
        pos_mask1 = term1 > -20  # 扩大计算范围以确保精度
        pos_mask2 = term2 > -20
        
        if pos_mask1.any():
            log_term1[pos_mask1] = torch.log1p(torch.exp(torch.clamp(term1[pos_mask1], -20, 20)))
        if pos_mask2.any():
            log_term2[pos_mask2] = torch.log1p(torch.exp(torch.clamp(term2[pos_mask2], -20, 20)))
        
        # 计算电流
        current = self.alpha * (log_term1**2 - log_term2**2)
        
        return current
    
    def forward_group(self, x, group_idx):
        """
        处理单个组的前向传播
        """
        batch_size, _, height, width = x.shape
        
        # 计算每个组处理的输出通道数
        channels_per_group = self.out_channels // self.groups
        start_ch = group_idx * channels_per_group
        end_ch = (group_idx + 1) * channels_per_group
        
        # 获取当前组的theta
        theta_group = self.theta[start_ch:end_ch]
        
        # 使用unfold获取所有局部区域
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.view(batch_size, self.in_channels, self.kernel_size * self.kernel_size, -1)
        x_unfold = x_unfold.permute(0, 3, 1, 2)  # [batch, spatial, in_ch, k*k]
        
        # 计算输出尺寸
        output_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        spatial_size = output_height * output_width
        
        # 初始化当前组的输出
        group_output = torch.zeros(batch_size, channels_per_group, output_height, output_width, device=x.device)
        
        # 对每个空间位置和输出通道单独计算
        for spatial_idx in range(spatial_size):
            # 获取当前空间位置的输入块
            x_patch = x_unfold[:, spatial_idx, :, :]  # [batch, in_ch, k*k]
            
            # 扩展以匹配当前组的输出通道
            x_patch_expanded = x_patch.unsqueeze(1).expand(-1, channels_per_group, -1, -1)  # [batch, group_ch, in_ch, k*k]
            
            # 应用EKV模型
            theta_flat = theta_group.view(channels_per_group, self.in_channels, -1)  # [group_ch, in_ch, k*k]
            theta_expanded = theta_flat.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, group_ch, in_ch, k*k]
            
            current_patch = self.efficient_ekv_model(x_patch_expanded, theta_expanded)
            
            # 在输入通道和卷积核维度上求和
            current_patch = current_patch.sum(dim=(2, 3))  # [batch, group_ch]
            
            # 存储结果
            h_idx = spatial_idx // output_width
            w_idx = spatial_idx % output_width
            group_output[:, :, h_idx, w_idx] = current_patch
        
        return group_output
    
    def forward(self, x):
        """
        分组处理的前向传播，大幅减少内存使用
        """
        # 初始化输出
        output = torch.zeros(x.size(0), self.out_channels, 
                           (x.size(2) + 2 * self.padding - self.kernel_size) // self.stride + 1,
                           (x.size(3) + 2 * self.padding - self.kernel_size) // self.stride + 1,
                           device=x.device)
        
        # 分组处理
        for i in range(self.groups):
            start_ch = i * (self.out_channels // self.groups)
            end_ch = (i + 1) * (self.out_channels // self.groups)
            
            # 处理当前组
            group_output = self.forward_group(x, i)
            
            # 将结果放入输出张量
            output[:, start_ch:end_ch, :, :] = group_output
            
            # 及时释放内存
            if i < self.groups - 1:
                torch.cuda.empty_cache()
        
        # 通过跨阻放大器转换为电压
        voltage_output = output * self.transimpedance_gain
        
        # BatchNorm归一化
        voltage_output = self.batch_norm(voltage_output)
        
        # 钳位到安全电压范围
        voltage_output = torch.clamp(voltage_output, self.voltage_range[0], self.voltage_range[1])
        
        return voltage_output

class CompactResidualBlock(nn.Module):
    """
    紧凑型残差块，大幅减少内存使用
    """
    def __init__(self, in_channels, out_channels, stride=1, use_nonlinear=True):
        super(CompactResidualBlock, self).__init__()
        
        self.use_nonlinear = use_nonlinear
        
        if use_nonlinear:
            self.conv1 = MemoryEfficientNonLinearConv2d(in_channels, out_channels, kernel_size=3, 
                                                      stride=stride, padding=1, groups=4)
            self.conv2 = MemoryEfficientNonLinearConv2d(out_channels, out_channels, kernel_size=3, 
                                                      stride=1, padding=1, groups=4)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                 stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                 stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if use_nonlinear:
                self.shortcut = MemoryEfficientNonLinearConv2d(in_channels, out_channels, 
                                                             kernel_size=1, stride=stride, padding=0, groups=2)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                            stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
    
    def forward(self, x):
        if self.use_nonlinear:
            out = self.conv1(x)
            out = self.conv2(out)
            out += self.shortcut(x)
            return out
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
            return out

class UltraCompactResNet(nn.Module):
    """
    超紧凑型ResNet，大幅减少内存使用
    """
    def __init__(self, num_classes=10, use_nonlinear=True):
        super(UltraCompactResNet, self).__init__()
        
        self.use_nonlinear = use_nonlinear
        
        # 第一层使用传统卷积
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 大幅减少通道数
        self.bn1 = nn.BatchNorm2d(16)
        
        # 紧凑的残差层
        self.layer1 = self._make_layer(16, 16, 1, stride=1, use_nonlinear=use_nonlinear)
        self.layer2 = self._make_layer(16, 32, 1, stride=2, use_nonlinear=use_nonlinear)
        self.layer3 = self._make_layer(32, 64, 1, stride=2, use_nonlinear=use_nonlinear)
        
        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)  # 减少全连接层输入维度
        
        # 初始化权重
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_nonlinear):
        layers = []
        layers.append(CompactResidualBlock(in_channels, out_channels, stride, use_nonlinear))
        for _ in range(1, num_blocks):
            layers.append(CompactResidualBlock(out_channels, out_channels, 1, use_nonlinear))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 第一层使用传统卷积+ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        
        # 残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 分类
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def memory_cleanup():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()

def train_compact_model(model, train_loader, test_loader, epochs=50, lr=0.01):
    """
    紧凑模型的训练函数
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 训练阶段
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # 计算准确率
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
            
            # 每几个batch清理一次内存
            if batch_idx % 20 == 0:
                memory_cleanup()
        
        # 更新学习率
        scheduler.step()
        
        # 评估阶段
        test_acc = evaluate_model(model, test_loader)
        test_accuracies.append(test_acc)
        train_losses.append(running_loss / len(train_loader))
        
        print(f'Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Test Acc = {test_acc:.2f}%')
        
        # 每个epoch后清理内存
        memory_cleanup()
    
    return train_losses, test_accuracies

def evaluate_model(model, test_loader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total

def plot_results(train_losses, test_accuracies, model_name):
    """绘制训练结果"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title(f'{model_name} - Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    # 数据预处理
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

    # 加载CIFAR-10数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # 使用batch_size=32
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("Dataset loaded successfully!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # 清理内存
    memory_cleanup()

    # 实验1: 传统卷积（基准）
    print("\n=== Training Conventional UltraCompact ResNet ===")
    conventional_model = UltraCompactResNet(use_nonlinear=False).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in conventional_model.parameters())
    print(f"Conventional model parameters: {total_params:,}")
    
    # 只训练5个epoch作为演示
    conv_losses, conv_accuracies = train_compact_model(
        conventional_model, train_loader, test_loader, epochs=5, lr=0.1)

    # 清理内存
    memory_cleanup()
    del conventional_model
    memory_cleanup()

    # 实验2: 非线性卷积
    print("\n=== Training NonLinear UltraCompact ResNet ===")
    nonlinear_model = UltraCompactResNet(use_nonlinear=True).to(device)
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in nonlinear_model.parameters())
    print(f"Nonlinear model parameters: {total_params:,}")
    
    # 只训练5个epoch作为演示
    nonlin_losses, nonlin_accuracies = train_compact_model(
        nonlinear_model, train_loader, test_loader, epochs=5, lr=0.01)

    # 绘制结果比较
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(conv_losses, label='Conventional ResNet')
    plt.plot(nonlin_losses, label='NonLinear ResNet')
    plt.title('Training Loss Comparison (5 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(conv_accuracies, label='Conventional ResNet')
    plt.plot(nonlin_accuracies, label='NonLinear ResNet')
    plt.title('Test Accuracy Comparison (5 Epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ultra_compact_comparison_5_epochs.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 最终准确率
    final_nonlin_acc = evaluate_model(nonlinear_model, test_loader)
    
    print(f"\n=== Final Results (After 5 Epochs) ===")
    print(f"NonLinear UltraCompact ResNet Test Accuracy: {final_nonlin_acc:.2f}%")

    # 保存模型
    torch.save(nonlinear_model.state_dict(), 'ultra_compact_nonlinear_resnet.pth')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()