# File name: analyze_layer2_detailed.py
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

# Instrumented BasicBlock to track intermediate results
class InstrumentedBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(InstrumentedBasicBlock, self).__init__()
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
        
        # For storing intermediate results
        self.input_data = None
        self.after_conv1 = None
        self.after_bn1 = None
        self.after_relu1 = None
        self.after_conv2 = None
        self.after_bn2 = None
        self.after_shortcut = None
        self.output_data = None

    def forward(self, x):
        # Store input
        self.input_data = x.detach().cpu()
        
        # First convolution
        out = self.conv1(x)
        self.after_conv1 = out.detach().cpu()
        
        # BatchNorm and ReLU
        out = self.bn1(out)
        self.after_bn1 = out.detach().cpu()
        
        out = F.relu(out)
        self.after_relu1 = out.detach().cpu()
        
        # Second convolution
        out = self.conv2(out)
        self.after_conv2 = out.detach().cpu()
        
        # Second BatchNorm
        out = self.bn2(out)
        self.after_bn2 = out.detach().cpu()
        
        # Shortcut connection
        identity = self.shortcut(x)
        self.after_shortcut = identity.detach().cpu()
        
        # Add and final ReLU
        out += identity
        out = F.relu(out)
        self.output_data = out.detach().cpu()
        
        return out

class InstrumentedHybridResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(InstrumentedHybridResNet, self).__init__()
        self.in_planes = 16
        
        # First layer (conv1 + bn1 + relu)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Layer1 with instrumented blocks
        self.layer1 = self._make_layer(InstrumentedBasicBlock, 16, 1, stride=1)
        
        # Layer2 with instrumented blocks
        self.layer2 = self._make_layer(InstrumentedBasicBlock, 32, 1, stride=2)
        
        # For storing intermediate results from conv1 and bn1
        self.after_conv1 = None
        self.after_bn1 = None
        self.after_relu1 = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # First convolution
        out = self.conv1(x)
        self.after_conv1 = out.detach().cpu()
        
        # First BatchNorm and ReLU
        out = self.bn1(out)
        self.after_bn1 = out.detach().cpu()
        
        out = F.relu(out)
        self.after_relu1 = out.detach().cpu()
        
        # Layer1
        out = self.layer1(out)
        
        # Layer2
        out = self.layer2(out)
        
        return out

def analyze_layer2_detailed():
    """Detailed analysis of layer2 intermediate results"""
    print("Loading trained model...")
    model = InstrumentedHybridResNet().to(device)
    
    # Load trained model
    model_path = 'best_hybrid_resnet8.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        return
    
    # Load state dict, ignoring keys that don't match
    pretrained_dict = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    
    # Filter out incompatible keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                      if k in model_dict and model_dict[k].shape == v.shape}
    
    # Load the filtered weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model.eval()
    print("Model loaded successfully!")
    
    # Collect data from one batch
    print("Collecting intermediate results from one batch...")
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx >= 1:  # Only process one batch
                break
                
            inputs = inputs.to(device)
            _ = model(inputs)
            
            # Collect all intermediate results
            intermediate_data = {}
            
            # From conv1, bn1, relu
            intermediate_data['input_to_network'] = inputs.detach().cpu().flatten().numpy()
            intermediate_data['after_conv1'] = model.after_conv1.flatten().numpy()
            intermediate_data['after_bn1'] = model.after_bn1.flatten().numpy()
            intermediate_data['after_relu1'] = model.after_relu1.flatten().numpy()
            
            # From layer1 (if it's instrumented)
            if hasattr(model.layer1[0], 'input_data'):
                intermediate_data['input_to_layer1'] = model.layer1[0].input_data.flatten().numpy()
                intermediate_data['after_layer1_conv1'] = model.layer1[0].after_conv1.flatten().numpy()
                intermediate_data['after_layer1_bn1'] = model.layer1[0].after_bn1.flatten().numpy()
                intermediate_data['after_layer1_relu1'] = model.layer1[0].after_relu1.flatten().numpy()
                intermediate_data['after_layer1_conv2'] = model.layer1[0].after_conv2.flatten().numpy()
                intermediate_data['after_layer1_bn2'] = model.layer1[0].after_bn2.flatten().numpy()
                intermediate_data['after_layer1_shortcut'] = model.layer1[0].after_shortcut.flatten().numpy()
                intermediate_data['output_of_layer1'] = model.layer1[0].output_data.flatten().numpy()
            
            # From layer2
            if hasattr(model.layer2[0], 'input_data'):
                intermediate_data['input_to_layer2'] = model.layer2[0].input_data.flatten().numpy()
                intermediate_data['after_layer2_conv1'] = model.layer2[0].after_conv1.flatten().numpy()
                intermediate_data['after_layer2_bn1'] = model.layer2[0].after_bn1.flatten().numpy()
                intermediate_data['after_layer2_relu1'] = model.layer2[0].after_relu1.flatten().numpy()
                intermediate_data['after_layer2_conv2'] = model.layer2[0].after_conv2.flatten().numpy()
                intermediate_data['after_layer2_bn2'] = model.layer2[0].after_bn2.flatten().numpy()
                intermediate_data['after_layer2_shortcut'] = model.layer2[0].after_shortcut.flatten().numpy()
                intermediate_data['output_of_layer2'] = model.layer2[0].output_data.flatten().numpy()
            
            break
    
    # Print statistics for each step
    print("\n=== Detailed Statistics ===")
    for step_name, data in intermediate_data.items():
        if len(data) > 0:
            print(f"\n{step_name}:")
            print(f"  Min: {data.min():.6f}")
            print(f"  Max: {data.max():.6f}")
            print(f"  Mean: {data.mean():.6f}")
            print(f"  Std: {data.std():.6f}")
            print(f"  Zero percentage: {(np.sum(data == 0) / len(data) * 100):.2f}%")
            print(f"  Negative percentage: {(np.sum(data < 0) / len(data) * 100):.2f}%")
    
    # Plot distributions
    print("\nPlotting detailed distributions...")
    
    # Create subplots
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    # Plot key steps
    key_steps = [
        'input_to_network',
        'after_conv1', 
        'after_bn1',
        'after_relu1',
        'input_to_layer2',
        'after_layer2_conv1',
        'after_layer2_bn1',
        'after_layer2_relu1',
        'after_layer2_conv2',
        'after_layer2_bn2',
        'after_layer2_shortcut',
        'output_of_layer2'
    ]
    
    for i, step_name in enumerate(key_steps):
        if i >= len(axes) or step_name not in intermediate_data:
            continue
            
        data = intermediate_data[step_name]
        ax = axes[i]
        
        # Plot histogram
        ax.hist(data, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(f'{step_name}', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Min: {data.min():.3f}\nMax: {data.max():.3f}\nMean: {data.mean():.3f}\nStd: {data.std():.3f}\nZero%: {(np.sum(data == 0) / len(data) * 100):.1f}%'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)
    
    # Hide unused subplots
    for i in range(len(key_steps), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('layer2_detailed_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create focused plot on layer2 steps
    print("\nCreating focused layer2 analysis...")
    layer2_steps = [
        'input_to_layer2',
        'after_layer2_conv1',
        'after_layer2_bn1', 
        'after_layer2_relu1',
        'after_layer2_conv2',
        'after_layer2_bn2',
        'after_layer2_shortcut',
        'output_of_layer2'
    ]
    
    plt.figure(figsize=(15, 10))
    for i, step_name in enumerate(layer2_steps):
        if step_name not in intermediate_data:
            continue
            
        plt.subplot(2, 4, i+1)
        data = intermediate_data[step_name]
        
        # Plot with log scale for y-axis to better see distribution
        plt.hist(data, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.yscale('log')  # Log scale to see distribution details
        plt.title(f'{step_name}', fontsize=10)
        plt.xlabel('Value')
        plt.ylabel('Frequency (log)')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at zero
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('layer2_focused_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze the root cause
    print("\n=== Root Cause Analysis ===")
    
    if 'output_of_layer2' in intermediate_data:
        layer2_output = intermediate_data['output_of_layer2']
        zero_percentage = (np.sum(layer2_output == 0) / len(layer2_output) * 100)
        print(f"Layer2 output has {zero_percentage:.2f}% zeros")
        
        # Check ReLU effects
        if 'after_layer2_bn1' in intermediate_data and 'after_layer2_relu1' in intermediate_data:
            before_relu = intermediate_data['after_layer2_bn1']
            after_relu = intermediate_data['after_layer2_relu1']
            
            negative_before_relu = np.sum(before_relu < 0)
            zeros_after_relu = np.sum(after_relu == 0)
            
            print(f"Before ReLU: {negative_before_relu/len(before_relu)*100:.2f}% negative values")
            print(f"After ReLU: {zeros_after_relu/len(after_relu)*100:.2f}% zeros (ReLU effect)")
        
        # Check BatchNorm effects
        if 'after_layer2_conv1' in intermediate_data and 'after_layer2_bn1' in intermediate_data:
            before_bn = intermediate_data['after_layer2_conv1']
            after_bn = intermediate_data['after_layer2_bn1']
            
            print(f"Conv1 output range: [{before_bn.min():.3f}, {before_bn.max():.3f}]")
            print(f"After BN1 range: [{after_bn.min():.3f}, {after_bn.max():.3f}]")
            
            # Check if BN is causing saturation
            bn_std = after_bn.std()
            print(f"BN1 output std: {bn_std:.6f}")
            if bn_std < 0.1:
                print("WARNING: BN1 output has very low variance, might be causing saturation")
    
    # Recommendations
    print("\n=== Recommendations ===")
    print("1. Check if BatchNorm parameters (weight/bias) are causing saturation")
    print("2. Consider adjusting learning rate or BatchNorm momentum")
    print("3. Verify if the network is properly trained (check training curves)")
    print("4. Consider using different activation functions (LeakyReLU, PReLU)")
    print("5. Check if gradient flow is proper during training")
    
    print(f"\nDetailed analysis complete! Results saved to:")
    print("- layer2_detailed_distributions.png")
    print("- layer2_focused_analysis.png")

if __name__ == "__main__":
    analyze_layer2_detailed()