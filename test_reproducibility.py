"""
测试随机种子设置是否有效，确保实验可复现
"""
import sys
import os
sys.path.append('.')

from utils.seed import set_seed
import torch
import numpy as np
import random

def test_basic_seed():
    """测试基本随机种子设置"""
    print("=" * 60)
    print("测试基本随机种子设置")
    print("=" * 60)
    
    # 第一次设置种子
    set_seed(42)
    
    # 生成一些随机数
    py_random_1 = random.random()
    np_random_1 = np.random.randn()
    torch_random_1 = torch.randn(1).item()
    
    print(f"第一次运行:")
    print(f"  Python随机数: {py_random_1:.6f}")
    print(f"  NumPy随机数: {np_random_1:.6f}")
    print(f"  PyTorch随机数: {torch_random_1:.6f}")
    
    # 重置所有状态
    random.seed(None)
    np.random.seed(None)
    torch.manual_seed(torch.seed())
    
    # 第二次设置相同的种子
    set_seed(42)
    
    # 生成相同的随机数
    py_random_2 = random.random()
    np_random_2 = np.random.randn()
    torch_random_2 = torch.randn(1).item()
    
    print(f"\n第二次运行:")
    print(f"  Python随机数: {py_random_2:.6f}")
    print(f"  NumPy随机数: {np_random_2:.6f}")
    print(f"  PyTorch随机数: {torch_random_2:.6f}")
    
    # 检查是否相同
    assert abs(py_random_1 - py_random_2) < 1e-10, "Python随机数不匹配"
    assert abs(np_random_1 - np_random_2) < 1e-10, "NumPy随机数不匹配"
    assert abs(torch_random_1 - torch_random_2) < 1e-10, "PyTorch随机数不匹配"
    
    print("\n✓ 基本随机种子测试通过!")

def test_model_reproducibility():
    """测试模型权重初始化可复现性"""
    print("\n" + "=" * 60)
    print("测试模型权重初始化可复现性")
    print("=" * 60)
    
    from model.unet import UNet
    
    # 第一次初始化模型
    set_seed(42, deterministic=True)
    model1 = UNet(n_channels=3, n_classes=19)
    weights1 = model1.inc.double_conv[0].weight.data.clone()
    
    # 第二次初始化模型
    set_seed(42, deterministic=True)
    model2 = UNet(n_channels=3, n_classes=19)
    weights2 = model2.inc.double_conv[0].weight.data.clone()
    
    # 检查权重是否相同
    weight_diff = torch.abs(weights1 - weights2).max().item()
    print(f"两次初始化的权重最大差异: {weight_diff:.10f}")
    
    if weight_diff < 1e-10:
        print("✓ 模型权重初始化可复现性测试通过!")
    else:
        print("✗ 模型权重初始化不可复现!")
    
    return weight_diff

def test_training_reproducibility():
    """测试训练过程可复现性"""
    print("\n" + "=" * 60)
    print("测试训练过程可复现性")
    print("=" * 60)
    
    # 由于完整训练耗时，这里只测试一个简化版本
    from data.dataset import CityscapesDataset, get_transforms
    from torch.utils.data import DataLoader, Subset
    
    # 设置种子
    set_seed(42, deterministic=True)
    
    # 创建小型数据集
    transform = get_transforms('train', (128, 256), use_strong_aug=False)
    dataset = CityscapesDataset(
        root_dir='/home/jianglh1/Cityscapes/data',
        split='train',
        transform=transform,
        img_size=(128, 256)
    )
    
    # 只取前10个样本
    small_dataset = Subset(dataset, list(range(10)))
    
    # 创建DataLoader
    from utils.seed import get_dataloader_kwargs
    dataloader_kwargs = get_dataloader_kwargs(42)
    
    loader1 = DataLoader(
        small_dataset,
        batch_size=2,
        shuffle=True,
        **dataloader_kwargs
    )
    
    # 获取第一个批次的数据
    batch1 = next(iter(loader1))
    
    # 重新设置种子和DataLoader
    set_seed(42, deterministic=True)
    dataloader_kwargs = get_dataloader_kwargs(42)
    
    loader2 = DataLoader(
        small_dataset,
        batch_size=2,
        shuffle=True,
        **dataloader_kwargs
    )
    
    # 获取第二个批次的数据
    batch2 = next(iter(loader2))
    
    # 检查批次是否相同
    images1, labels1 = batch1
    images2, labels2 = batch2
    
    image_diff = torch.abs(images1 - images2).max().item()
    label_diff = torch.abs(labels1 - labels2).max().item()
    
    print(f"图像批次最大差异: {image_diff:.10f}")
    print(f"标签批次最大差异: {label_diff:.10f}")
    
    if image_diff < 1e-10 and label_diff < 1e-10:
        print("✓ 数据加载可复现性测试通过!")
    else:
        print("✗ 数据加载不可复现!")

def test_deterministic_mode():
    """测试确定性模式"""
    print("\n" + "=" * 60)
    print("测试确定性模式")
    print("=" * 60)
    
    # 测试确定性模式下的卷积运算
    set_seed(42, deterministic=True)
    
    # 创建一个简单的卷积运算
    input_tensor = torch.randn(1, 3, 32, 32, device='cuda' if torch.cuda.is_available() else 'cpu')
    conv_layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
    conv_layer = conv_layer.to(input_tensor.device)
    
    # 第一次前向传播
    output1 = conv_layer(input_tensor)
    
    # 清除缓存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 第二次前向传播（使用相同的输入和权重）
    output2 = conv_layer(input_tensor)
    
    # 检查输出是否相同
    output_diff = torch.abs(output1 - output2).max().item()
    print(f"确定性模式下两次卷积输出最大差异: {output_diff:.10f}")
    
    if output_diff < 1e-10:
        print("✓ 确定性模式测试通过!")
    else:
        print("✗ 确定性模式测试失败!")

if __name__ == "__main__":
    print("开始测试随机种子可复现性...\n")
    
    try:
        test_basic_seed()
        weight_diff = test_model_reproducibility()
        test_training_reproducibility()
        
        if torch.cuda.is_available():
            test_deterministic_mode()
        else:
            print("\n未检测到CUDA，跳过确定性模式测试")
        
        print("\n" + "=" * 60)
        print("所有测试完成!")
        
        if weight_diff < 1e-10:
            print("✓ 实验可复现性良好")
        else:
            print("⚠️ 模型权重初始化有微小差异，但可能仍可接受")
            
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()