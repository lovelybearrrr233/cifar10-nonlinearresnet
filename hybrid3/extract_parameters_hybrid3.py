# File name: extract_scaling_parameters.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model classes exactly as in your training code
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

class EKVConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        k = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.theta = nn.Parameter(torch.empty(out_channels, k))
        nn.init.uniform_(self.theta, 2.8, 4.8)
        
        self.alpha = 5.625e-4
        self.n = 1.5
        self.VT = 0.026
        self.VD = 0.1
        
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, padding=padding, stride=stride)

    def log1p_exp(self, x):
        return torch.log1p(torch.exp(torch.clamp(x, -30, 30)))

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.unfold(x)
        L = patches.shape[-1]
        
        vg = patches.unsqueeze(1)
        theta = self.theta.view(1, -1, self.theta.shape[-1], 1)
        
        arg1 = (vg - theta) / (self.n * self.VT)
        arg2 = (vg - theta - self.VD) / (self.n * self.VT)
        
        I = self.alpha * (self.log1p_exp(arg1)**2 - self.log1p_exp(arg2)**2)
        out = I.sum(dim=2).view(B, -1, 
                                (H + 2*self.padding - self.kernel_size[0])//self.stride + 1,
                                (W + 2*self.padding - self.kernel_size[1])//self.stride + 1)
        return out

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

def extract_and_analyze_scaling_parameters():
    """提取和分析模型中的缩放参数"""
    print("Loading trained model...")
    model = HybridResNet8().to(device)
    
    # Load trained model
    model_path = 'best_hybrid3_ekv_last20.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        print("Please make sure the model file exists in the current directory")
        return
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Extract scaling parameters
    print("\n" + "="*80)
    print("EXTRACTING SCALING PARAMETERS")
    print("="*80)
    
    scaling_params = {}
    
    # Get the EKVBlock from layer3
    ekv_block = model.layer3[0]
    
    # Extract InputScaling parameters
    if hasattr(ekv_block, 'prescale'):
        prescale = ekv_block.prescale
        scaling_params['InputScaling'] = {
            'scale': prescale.scale.item(),
            'shift': prescale.shift.item(),
            'type': 'InputScaling',
            'description': 'Maps linear output to EKV input range'
        }
    
    # Extract LayerScale parameters
    if hasattr(ekv_block, 'ls1'):
        ls1 = ekv_block.ls1
        scaling_params['LayerScale1'] = {
            'scale': ls1.scale.item(),
            'shift': 'N/A',  # LayerScale only has scale
            'type': 'LayerScale',
            'description': 'Scaling after first EKV convolution'
        }
    
    if hasattr(ekv_block, 'ls2'):
        ls2 = ekv_block.ls2
        scaling_params['LayerScale2'] = {
            'scale': ls2.scale.item(),
            'shift': 'N/A',
            'type': 'LayerScale', 
            'description': 'Scaling after second EKV convolution'
        }
    
    # Extract EKV theta parameters (threshold voltages)
    if hasattr(ekv_block, 'conv1'):
        conv1_theta = ekv_block.conv1.theta
        scaling_params['EKV_Conv1_Theta'] = {
            'mean': conv1_theta.mean().item(),
            'std': conv1_theta.std().item(),
            'min': conv1_theta.min().item(),
            'max': conv1_theta.max().item(),
            'type': 'EKV_Threshold',
            'description': 'Threshold voltages for first EKV convolution'
        }
    
    if hasattr(ekv_block, 'conv2'):
        conv2_theta = ekv_block.conv2.theta
        scaling_params['EKV_Conv2_Theta'] = {
            'mean': conv2_theta.mean().item(),
            'std': conv2_theta.std().item(),
            'min': conv2_theta.min().item(),
            'max': conv2_theta.max().item(),
            'type': 'EKV_Threshold',
            'description': 'Threshold voltages for second EKV convolution'
        }
    
    # Print parameter summary
    print("\nScaling Parameter Summary:")
    print("-" * 100)
    print(f"{'Parameter':<20} | {'Scale':<10} | {'Shift':<10} | {'Type':<15} | Description")
    print("-" * 100)
    
    for param_name, param_info in scaling_params.items():
        scale_val = f"{param_info['scale']:.6f}" if isinstance(param_info['scale'], float) else param_info['scale']
        shift_val = f"{param_info['shift']:.6f}" if isinstance(param_info.get('shift'), float) else param_info.get('shift', 'N/A')
        
        print(f"{param_name:<20} | {scale_val:<10} | {shift_val:<10} | {param_info['type']:<15} | {param_info['description']}")
    
    # Create detailed analysis
    print("\n" + "="*80)
    print("DETAILED PARAMETER ANALYSIS")
    print("="*80)
    
    # Analyze InputScaling effect
    if 'InputScaling' in scaling_params:
        input_scaling = scaling_params['InputScaling']
        scale = input_scaling['scale']
        shift = input_scaling['shift']
        
        print(f"\nInputScaling Analysis:")
        print(f"  Transformation: y = {scale:.6f} * x + {shift:.6f}")
        print(f"  Expected input range mapping:")
        print(f"    If input in [0, 1] → output in [{shift:.3f}, {scale + shift:.3f}]")
        print(f"    If input in [-1, 1] → output in [{-scale + shift:.3f}, {scale + shift:.3f}]")
        
        # Check if output range is reasonable for EKV
        min_output = -scale + shift if scale > 0 else scale + shift
        max_output = scale + shift if scale > 0 else -scale + shift
        
        print(f"  Output range analysis:")
        print(f"    Theoretical range: [{min_output:.3f}, {max_output:.3f}]")
        
        if min_output >= 0 and max_output <= 10:
            print(f"    ✓ Within EKV safe range [0, 10]")
        else:
            print(f"    ⚠ Outside EKV safe range [0, 10]")
            
        if 2 <= min_output and max_output <= 8:
            print(f"    ✓ Within optimal EKV working range [2, 8]")
        else:
            print(f"    ⚠ Outside optimal EKV working range [2, 8]")
    
    # Analyze LayerScale effects
    layer_scales = {k: v for k, v in scaling_params.items() if v['type'] == 'LayerScale'}
    for scale_name, scale_info in layer_scales.items():
        scale_val = scale_info['scale']
        print(f"\n{scale_name} Analysis:")
        print(f"  Scaling factor: {scale_val:.6f}")
        
        if scale_val > 1.0:
            print(f"    Amplifying signal (scale > 1)")
        elif scale_val < 1.0:
            print(f"    Attenuating signal (scale < 1)")
        else:
            print(f"    No scaling (scale = 1)")
        
        if abs(scale_val - 1.0) > 5.0:
            print(f"    ⚠ Large scaling factor - might cause instability")
    
    # Analyze EKV threshold parameters
    ekv_params = {k: v for k, v in scaling_params.items() if v['type'] == 'EKV_Threshold'}
    for ekv_name, ekv_info in ekv_params.items():
        mean_theta = ekv_info['mean']
        std_theta = ekv_info['std']
        
        print(f"\n{ekv_name} Analysis:")
        print(f"  Mean threshold: {mean_theta:.6f}V")
        print(f"  Std of thresholds: {std_theta:.6f}V")
        print(f"  Range: [{ekv_info['min']:.6f}V, {ekv_info['max']:.6f}V]")
        
        # Check if thresholds are in reasonable range
        if 2.0 <= mean_theta <= 6.0:
            print(f"    ✓ Mean threshold in good EKV range")
        else:
            print(f"    ⚠ Mean threshold outside typical EKV range")
        
        if std_theta > 1.0:
            print(f"    ⚠ High threshold variance - might cause inconsistent behavior")
        elif std_theta < 0.1:
            print(f"    ⚠ Very low threshold variance - limited diversity")
    
    # Create visualizations
    print("\nCreating parameter visualizations...")
    
    # 1. Scaling parameters bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot scaling factors
    scaling_factors = {}
    for param_name, param_info in scaling_params.items():
        if 'scale' in param_info and isinstance(param_info['scale'], float):
            scaling_factors[param_name] = param_info['scale']
    
    if scaling_factors:
        plt.subplot(2, 2, 1)
        names = list(scaling_factors.keys())
        values = list(scaling_factors.values())
        
        bars = plt.bar(names, values, color='skyblue', alpha=0.7, edgecolor='black')
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Neutral (scale=1)')
        plt.ylabel('Scaling Factor')
        plt.title('Scaling Parameters')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    # 2. EKV threshold distributions
    ekv_thresholds = {}
    for param_name, param_info in scaling_params.items():
        if param_info['type'] == 'EKV_Threshold':
            ekv_thresholds[param_name] = param_info
    
    if ekv_thresholds:
        plt.subplot(2, 2, 2)
        names = list(ekv_thresholds.keys())
        means = [info['mean'] for info in ekv_thresholds.values()]
        stds = [info['std'] for info in ekv_thresholds.values()]
        
        plt.errorbar(names, means, yerr=stds, fmt='o', capsize=5, 
                    markersize=8, linewidth=2, label='Mean ± Std')
        plt.axhline(y=3.0, color='green', linestyle='--', alpha=0.7, label='Typical EKV range')
        plt.axhline(y=5.0, color='green', linestyle='--', alpha=0.7)
        plt.ylabel('Threshold Voltage (V)')
        plt.title('EKV Threshold Voltages')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 3. InputScaling transformation visualization
    if 'InputScaling' in scaling_params:
        plt.subplot(2, 2, 3)
        input_scaling = scaling_params['InputScaling']
        scale = input_scaling['scale']
        shift = input_scaling['shift']
        
        # Plot transformation
        x = np.linspace(-2, 2, 100)
        y = scale * x + shift
        
        plt.plot(x, y, 'b-', linewidth=2, label=f'y = {scale:.3f}x + {shift:.3f}')
        plt.axhline(y=0, color='red', linestyle=':', alpha=0.5, label='Zero line')
        plt.axhline(y=10, color='orange', linestyle=':', alpha=0.5, label='Upper limit')
        plt.fill_between(x, 2, 8, alpha=0.2, color='green', label='Optimal EKV range')
        
        plt.xlabel('Input (Linear Layer Output)')
        plt.ylabel('Output (EKV Input)')
        plt.title('InputScaling Transformation')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Parameter relationships
    plt.subplot(2, 2, 4)
    
    # Compare different scaling parameters
    param_data = []
    for param_name, param_info in scaling_params.items():
        if 'scale' in param_info and isinstance(param_info['scale'], float):
            param_data.append({
                'name': param_name,
                'scale': param_info['scale'],
                'type': param_info['type']
            })
    
    if param_data:
        df = pd.DataFrame(param_data)
        colors = {'InputScaling': 'red', 'LayerScale': 'blue', 'EKV_Threshold': 'green'}
        
        for param_type in df['type'].unique():
            subset = df[df['type'] == param_type]
            plt.scatter(subset['name'], subset['scale'], 
                       label=param_type, s=100, alpha=0.7)
        
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Neutral')
        plt.ylabel('Scaling Factor')
        plt.title('Parameter Scale Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('scaling_parameters_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save parameters to CSV for further analysis
    print("\nSaving parameters to CSV...")
    param_data = []
    for param_name, param_info in scaling_params.items():
        row = {'Parameter': param_name, 'Type': param_info['type']}
        
        # Add all numeric values
        for key, value in param_info.items():
            if key not in ['type', 'description'] and isinstance(value, (int, float)):
                row[key] = value
        
        param_data.append(row)
    
    if param_data:
        df = pd.DataFrame(param_data)
        df.to_csv('scaling_parameters.csv', index=False)
        print("Parameters saved to 'scaling_parameters.csv'")
    
    # Final recommendations
    print("\n" + "="*80)
    print("PARAMETER OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    # Check for potential issues and provide recommendations
    recommendations = []
    
    # InputScaling recommendations
    if 'InputScaling' in scaling_params:
        scale = scaling_params['InputScaling']['scale']
        shift = scaling_params['InputScaling']['shift']
        
        if abs(scale) > 10:
            recommendations.append("Consider reducing InputScaling scale factor (currently too large)")
        elif abs(scale) < 1:
            recommendations.append("Consider increasing InputScaling scale factor (currently too small)")
        
        if shift < 0:
            recommendations.append("InputScaling shift is negative - might push data below 0V")
        elif shift > 8:
            recommendations.append("InputScaling shift is too high - might push data near upper limit")
    
    # LayerScale recommendations
    layer_scales = [v for k, v in scaling_params.items() if v['type'] == 'LayerScale']
    for i, ls in enumerate(layer_scales):
        scale = ls['scale']
        if abs(scale - 1.0) > 5.0:
            recommendations.append(f"LayerScale{i+1} has extreme scaling ({scale:.3f}) - consider normalizing")
    
    # EKV threshold recommendations
    ekv_thresholds = [v for k, v in scaling_params.items() if v['type'] == 'EKV_Threshold']
    for i, ekv in enumerate(ekv_thresholds):
        mean_theta = ekv['mean']
        if mean_theta < 2.0 or mean_theta > 6.0:
            recommendations.append(f"EKV convolution {i+1} thresholds outside optimal range ({mean_theta:.3f}V)")
    
    if recommendations:
        print("Recommendations for improvement:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("All parameters appear to be in reasonable ranges.")
    
    print(f"\nAnalysis complete!")
    print(f"Generated files:")
    print(f"- scaling_parameters_analysis.png")
    print(f"- scaling_parameters.csv")

if __name__ == "__main__":
    extract_and_analyze_scaling_parameters()