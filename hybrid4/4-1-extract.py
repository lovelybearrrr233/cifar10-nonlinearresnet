import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def extract_layer3_bn_parameters(model_path):
    """
    从训练好的模型中提取layer3中所有BN层的参数
    """
    # 加载模型状态字典
    print(f"加载模型: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 提取layer3中所有BN层参数
    bn_params = {}
    
    for param_name, param_value in state_dict.items():
        if 'layer3' in param_name and 'bn' in param_name:
            bn_params[param_name] = {
                'value': param_value.numpy(),
                'shape': param_value.shape,
                'requires_grad': True  # BN层参数在训练时通常是可训练的
            }
            print(f"找到BN层参数: {param_name}")
            print(f"  形状: {param_value.shape}")
            print(f"  数值范围: [{param_value.min():.6f}, {param_value.max():.6f}]")
            print("-" * 50)
    
    return bn_params

def extract_all_parameters(model_path):
    """
    提取模型中所有相关参数：VoltageMapper和layer3的BN层
    """
    # 加载模型状态字典
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 分类提取参数
    voltage_mapper_params = {}
    bn_params = {}
    ekv_params = {}
    other_params = {}
    
    for param_name, param_value in state_dict.items():
        param_info = {
            'value': param_value.numpy(),
            'shape': param_value.shape,
            'requires_grad': True
        }
        
        if 'mapper' in param_name:
            voltage_mapper_params[param_name] = param_info
        elif 'layer3' in param_name and 'bn' in param_name:
            bn_params[param_name] = param_info
        elif 'theta' in param_name:
            ekv_params[param_name] = param_info
        elif 'layer3' in param_name or 'linear' in param_name:
            other_params[param_name] = param_info
    
    return {
        'voltage_mapper': voltage_mapper_params,
        'bn_layer3': bn_params,
        'ekv_theta': ekv_params,
        'other_trainable': other_params
    }

def analyze_bn_parameters(bn_params):
    """
    详细分析BN层参数
    """
    print("\n" + "="*60)
    print("Layer3 BN层参数详细分析")
    print("="*60)
    
    for param_name, param_info in bn_params.items():
        values = param_info['value']
        print(f"\n参数: {param_name}")
        print(f"  形状: {param_info['shape']}")
        print(f"  均值: {values.mean():.6f}")
        print(f"  标准差: {values.std():.6f}")
        print(f"  范围: [{values.min():.6f}, {values.max():.6f}]")
        
        # 对于weight和bias，显示更多统计信息
        if 'weight' in param_name:
            print(f"  BN权重 - 正值比例: {(values > 0).mean():.2%}")
        elif 'bias' in param_name:
            print(f"  BN偏置 - 零值附近比例: {(np.abs(values) < 0.1).mean():.2%}")
    
    return bn_params

def visualize_bn_parameters(bn_params):
    """
    可视化BN层参数分布
    """
    if not bn_params:
        print("未找到BN层参数")
        return
    
    # 创建子图
    n_params = len(bn_params)
    fig, axes = plt.subplots(2, n_params, figsize=(5*n_params, 8))
    if n_params == 1:
        axes = axes.reshape(2, 1)
    
    for idx, (param_name, param_info) in enumerate(bn_params.items()):
        values = param_info['value'].flatten()
        
        # 上排：直方图
        axes[0, idx].hist(values, bins=50, alpha=0.7, edgecolor='black')
        axes[0, idx].set_title(f'{param_name}\nShape: {param_info["shape"]}')
        axes[0, idx].set_xlabel('Value')
        axes[0, idx].set_ylabel('Frequency')
        axes[0, idx].grid(True, alpha=0.3)
        
        # 下排：箱线图
        axes[1, idx].boxplot(values)
        axes[1, idx].set_title(f'{param_name} - Boxplot')
        axes[1, idx].set_ylabel('Value')
        axes[1, idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('layer3_bn_parameters.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_parameters_to_files(all_params, output_dir="./model_parameters"):
    """
    保存所有参数到文件
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每种类型的参数
    for param_type, params_dict in all_params.items():
        type_dir = os.path.join(output_dir, param_type)
        os.makedirs(type_dir, exist_ok=True)
        
        for param_name, param_info in params_dict.items():
            # 保存为npy文件
            filename = param_name.replace('.', '_') + '.npy'
            filepath = os.path.join(type_dir, filename)
            np.save(filepath, param_info['value'])
            
            # 保存统计信息
            stats = {
                'shape': param_info['shape'],
                'mean': float(param_info['value'].mean()),
                'std': float(param_info['value'].std()),
                'min': float(param_info['value'].min()),
                'max': float(param_info['value'].max())
            }
            
            # 保存统计信息为txt
            stats_file = os.path.join(type_dir, filename.replace('.npy', '_stats.txt'))
            with open(stats_file, 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
    
    print(f"\n所有参数已保存到目录: {output_dir}")

def print_parameter_summary(all_params):
    """
    打印参数总结
    """
    print("\n" + "="*80)
    print("模型可训练参数总结")
    print("="*80)
    
    total_params = 0
    for param_type, params_dict in all_params.items():
        type_count = len(params_dict)
        type_params = sum(np.prod(param_info['shape']) for param_info in params_dict.values())
        total_params += type_params
        
        print(f"\n{param_type.upper()}:")
        print(f"  参数数量: {type_count}")
        print(f"  总参数量: {type_params}")
        for param_name, param_info in params_dict.items():
            param_count = np.prod(param_info['shape'])
            print(f"    - {param_name}: {param_info['shape']} ({param_count} 个参数)")
    
    print(f"\n总计可训练参数: {total_params}")

# 使用示例
if __name__ == "__main__":
    # 替换为您的实际模型路径
    model_path = "4-1-resnet8_double_ekv_best.pth"
    
    try:
        print("开始提取模型参数...")
        
        # 方法1: 提取所有相关参数
        print("\n方法1: 提取所有相关参数")
        all_params = extract_all_parameters(model_path)
        
        # 方法2: 专门分析BN层参数
        print("\n方法2: 分析Layer3 BN层参数")
        bn_params = analyze_bn_parameters(all_params['bn_layer3'])
        
        # 方法3: 可视化BN层参数
        print("\n方法3: 可视化BN层参数")
        visualize_bn_parameters(all_params['bn_layer3'])
        
        # 方法4: 保存所有参数到文件
        print("\n方法4: 保存参数到文件")
        save_parameters_to_files(all_params)
        
        # 方法5: 打印参数总结
        print_parameter_summary(all_params)
        
        # 特别显示VoltageMapper参数
        print("\n" + "="*60)
        print("VoltageMapper参数:")
        print("="*60)
        for param_name, param_info in all_params['voltage_mapper'].items():
            values = param_info['value']
            print(f"{param_name}: {values}")
        
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {model_path}")
        print("请确保模型文件存在，或修改model_path变量为正确的路径")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()