# File name: analyze_layer2_distribution.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data loading
print("Loading data...")
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model definition (same as training)
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

class EKVNonLinearConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(EKVNonLinearConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.alpha = 0.0005625
        self.R = 0.1
        self.n = 1.5
        self.VT = 0.025
        self.VD = 0.1
        k_size_sq = kernel_size * kernel_size
        self.theta = nn.Parameter(torch.empty(out_channels, in_channels * k_size_sq))
        nn.init.uniform_(self.theta, 1.0, 8.0)
        self.unfold = nn.Unfold(kernel_size, stride=stride, padding=padding)

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
        arg1 = (vg - vth_all) / (2 * self.n * self.VT)
        arg2 = (vg - vth_all - self.VD) / (2 * self.n * self.VT)
        term1 = self._ekv_f(arg1)
        term2 = self._ekv_f(arg2)
        currents = self.alpha * (term1 - term2)
        I_k_all = torch.sum(currents, dim=2)
        V_out = I_k_all * self.R
        H_out = (x.shape[2] + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (x.shape[3] + 2*self.padding - self.kernel_size) // self.stride + 1
        V_out = V_out.view(B, self.out_channels, H_out, W_out)
        return V_out

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
        # Record values before clamping
        pre_clamp_values = x.detach().cpu()
        x = torch.clamp(x, min=0.0, max=10.0)
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.clamp(out, min=0.0, max=10.0)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = torch.clamp(out, min=0.0, max=10.0)
        return out, pre_clamp_values

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
        
        # For storing intermediate results
        self.layer2_output = None
        self.pre_clamp_values = None

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
        
        # Save layer2 output
        self.layer2_output = out.detach().cpu()
        
        # Pass to layer3 and get pre-clamp values
        out, self.pre_clamp_values = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def analyze_distribution():
    """Analyze data distribution after layer2"""
    print("Loading trained model...")
    model = HybridResNet().to(device)
    
    # Load trained model
    model_path = 'best_hybrid_resnet8.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        print("Please make sure best_hybrid_resnet8.pth exists")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Collect layer2 output data
    print("Collecting layer2 output data...")
    all_layer2_outputs = []
    all_pre_clamp_values = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(testloader, desc="Processing test set")):
            inputs = inputs.to(device)
            _ = model(inputs)
            
            # Collect layer2 output
            if model.layer2_output is not None:
                all_layer2_outputs.append(model.layer2_output)
            
            # Collect pre-clamp values
            if model.pre_clamp_values is not None:
                all_pre_clamp_values.append(model.pre_clamp_values)
            
            # Process only part of data to avoid memory issues
            if batch_idx >= 10:  # Process 10 batches, ~1000 images
                break
    
    # Combine all data
    if all_layer2_outputs:
        layer2_data = torch.cat(all_layer2_outputs, dim=0)
        layer2_data = layer2_data.flatten().numpy()
        print(f"Layer2 Output Statistics:")
        print(f"  Data points: {len(layer2_data)}")
        print(f"  Min: {layer2_data.min():.6f}")
        print(f"  Max: {layer2_data.max():.6f}")
        print(f"  Mean: {layer2_data.mean():.6f}")
        print(f"  Std: {layer2_data.std():.6f}")
    else:
        print("Error: Failed to collect layer2 output data")
        return
    
    if all_pre_clamp_values:
        pre_clamp_data = torch.cat(all_pre_clamp_values, dim=0)
        pre_clamp_data = pre_clamp_data.flatten().numpy()
        print(f"\nPre-clamp Data Statistics:")
        print(f"  Data points: {len(pre_clamp_data)}")
        print(f"  Min: {pre_clamp_data.min():.6f}")
        print(f"  Max: {pre_clamp_data.max():.6f}")
        print(f"  Mean: {pre_clamp_data.mean():.6f}")
        print(f"  Std: {pre_clamp_data.std():.6f}")
        
        # Calculate information loss
        below_zero = np.sum(pre_clamp_data < 0)
        above_ten = np.sum(pre_clamp_data > 10)
        total_points = len(pre_clamp_data)
        
        loss_below_zero = below_zero / total_points * 100
        loss_above_ten = above_ten / total_points * 100
        total_loss = loss_below_zero + loss_above_ten
        
        print(f"\n=== Information Loss Analysis ===")
        print(f"Data points below 0: {below_zero}/{total_points} ({loss_below_zero:.2f}%)")
        print(f"Data points above 10: {above_ten}/{total_points} ({loss_above_ten:.2f}%)")
        print(f"Total information loss: {total_loss:.2f}%")
    else:
        print("Error: Failed to collect pre-clamp data")
        return
    
    # Plot distributions
    print("\nPlotting data distributions...")
    plt.figure(figsize=(15, 10))
    
    # 1. Layer2 output distribution
    plt.subplot(2, 3, 1)
    plt.hist(layer2_data, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Layer2 Output Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Pre-clamp data distribution
    plt.subplot(2, 3, 2)
    plt.hist(pre_clamp_data, bins=100, alpha=0.7, color='red', edgecolor='black')
    plt.axvline(x=0, color='green', linestyle='--', label='Clamp Lower Bound (0V)')
    plt.axvline(x=10, color='orange', linestyle='--', label='Clamp Upper Bound (10V)')
    plt.title('Layer3 Input Pre-clamp Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Pre-clamp vs post-clamp comparison
    plt.subplot(2, 3, 3)
    # Post-clamp data (actual data entering layer3)
    clamped_data = np.clip(pre_clamp_data, 0, 10)
    
    plt.hist(pre_clamp_data, bins=100, alpha=0.5, color='red', label='Pre-clamp', edgecolor='black')
    plt.hist(clamped_data, bins=100, alpha=0.5, color='blue', label='Post-clamp', edgecolor='black')
    plt.axvline(x=0, color='green', linestyle='--', label='Clamp Boundaries')
    plt.axvline(x=10, color='green', linestyle='--')
    plt.title('Pre-clamp vs Post-clamp Comparison')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Violin plot - Layer2 output
    plt.subplot(2, 3, 4)
    sns.violinplot(y=layer2_data, color='lightblue')
    plt.title('Layer2 Output Violin Plot')
    plt.ylabel('Value')
    
    # 5. Violin plot - Pre-clamp data
    plt.subplot(2, 3, 5)
    sns.violinplot(y=pre_clamp_data, color='lightcoral')
    plt.axhline(y=0, color='green', linestyle='--', alpha=0.7)
    plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7)
    plt.title('Pre-clamp Data Violin Plot')
    plt.ylabel('Value')
    
    # 6. Information loss visualization
    plt.subplot(2, 3, 6)
    categories = ['Valid Range', '< 0', '> 10']
    values = [100 - total_loss, loss_below_zero, loss_above_ten]
    colors = ['green', 'red', 'orange']
    
    plt.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    plt.title('Data Distribution Ratio')
    plt.ylabel('Percentage (%)')
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('layer2_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed statistics
    print("\nSaving detailed statistics...")
    stats = {
        'layer2_stats': {
            'min': float(layer2_data.min()),
            'max': float(layer2_data.max()),
            'mean': float(layer2_data.mean()),
            'std': float(layer2_data.std()),
            'data_points': len(layer2_data)
        },
        'pre_clamp_stats': {
            'min': float(pre_clamp_data.min()),
            'max': float(pre_clamp_data.max()),
            'mean': float(pre_clamp_data.mean()),
            'std': float(pre_clamp_data.std()),
            'data_points': len(pre_clamp_data)
        },
        'information_loss': {
            'below_zero_percent': float(loss_below_zero),
            'above_ten_percent': float(loss_above_ten),
            'total_loss_percent': float(total_loss),
            'below_zero_count': int(below_zero),
            'above_ten_count': int(above_ten)
        }
    }
    
    # Print recommendations
    print("\n=== Analysis Recommendations ===")
    if total_loss < 1:
        print("✓ Small information loss (<1%), clamping has negligible impact")
    elif total_loss < 5:
        print("⚠ Moderate information loss (1-5%), consider adjusting voltage range or adding normalization")
    elif total_loss < 10:
        print("⚠ Significant information loss (5-10%), recommended to optimize voltage range")
    else:
        print("✗ Severe information loss (>10%), strongly recommend redesigning voltage range or network architecture")
    
    if pre_clamp_data.min() < -1 or pre_clamp_data.max() > 11:
        print("⚠ Wide data distribution range, consider adding BatchNorm or adjusting network parameters")
    
    print(f"\nAnalysis complete! Results saved to 'layer2_distribution_analysis.png'")

if __name__ == "__main__":
    analyze_distribution()