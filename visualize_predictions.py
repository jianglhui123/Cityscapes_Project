#!/usr/bin/env python3
"""
Cityscapes语义分割模型预测可视化脚本
可以独立运行，可视化已训练模型的预测效果
"""
import sys
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """设置环境，导入必要的模块"""
    try:
        from data.dataset import CityscapesDataset, get_transforms
        from data.classes import TRAINID_TO_COLOR
        from model.unet import get_model
        print("✓ 环境设置完成")
        return CityscapesDataset, get_transforms, TRAINID_TO_COLOR, get_model
    except ImportError as e:
        print(f"✗ 导入模块失败: {e}")
        print("请确保在项目根目录运行此脚本")
        sys.exit(1)

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ 配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        return None

def load_model(model_path, model_name, num_classes, device):
    """加载训练好的模型"""
    try:
        # 获取模型
        if model_name == 'unet':
            from model.unet import UNet
            model = UNet(n_channels=3, n_classes=num_classes)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ 加载模型状态字典")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print(f"✓ 加载状态字典")
        else:
            # 假设整个文件就是模型权重
            model.load_state_dict(checkpoint)
            print(f"✓ 直接加载模型权重")
        
        model.to(device)
        model.eval()
        
        # 打印模型信息
        if 'miou' in checkpoint:
            print(f"  模型mIoU: {checkpoint['miou']:.4f}")
        if 'epoch' in checkpoint:
            print(f"  训练epoch数: {checkpoint['epoch']}")
        
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return None

def create_dataloader(config, split='val', num_samples=4):
    """创建数据加载器"""
    try:
        CityscapesDataset, get_transforms, _, _ = setup_environment()
        
        data_config = config['data']
        
        # 获取图像尺寸
        if isinstance(data_config['img_size'], list):
            img_size = tuple(data_config['img_size'])
        else:
            img_size = data_config['img_size']
        
        # 创建数据集
        transform = get_transforms(split, img_size, use_strong_aug=False)
        dataset = CityscapesDataset(
            root_dir=data_config['root_dir'],
            split=split,
            transform=transform,
            img_size=img_size
        )
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=num_samples,
            shuffle=True,
            num_workers=0,  # 简化，避免多进程问题
            pin_memory=True
        )
        
        print(f"✓ 创建{split}集数据加载器，样本数: {len(dataset)}")
        return dataloader, dataset
    except Exception as e:
        print(f"✗ 数据加载器创建失败: {e}")
        return None, None

def visualize_single_prediction(model, image, label, device, class_colors):
    """可视化单个预测结果"""
    model.eval()
    
    with torch.no_grad():
        # 添加batch维度并移动到设备
        image_batch = image.unsqueeze(0).to(device)
        outputs = model(image_batch)
        pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    
    # 准备图像
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    # 准备标签
    label_np = label.cpu().numpy()
    
    # 创建彩色标签
    label_color = np.zeros((label_np.shape[0], label_np.shape[1], 3), dtype=np.uint8)
    pred_color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    
    for train_id, color in class_colors.items():
        if train_id != 255:  # 忽略ignore类
            label_color[label_np == train_id] = color
            pred_color[pred == train_id] = color
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title('输入图像', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(label_color)
    axes[1].set_title('真实标签', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(pred_color)
    axes[2].set_title('模型预测', fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig

def visualize_batch_predictions(model, dataloader, device, num_samples, class_colors, save_path=None):
    """可视化批次预测结果"""
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    # 转换到CPU numpy
    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    
    # 限制样本数量
    num_samples = min(num_samples, len(images))
    
    # 创建图形
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    # 如果只有一个样本，确保axes是二维数组
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 原始图像（反标准化）
        img = images_np[i].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # 真实标签颜色
        label_color = np.zeros((labels_np[i].shape[0], labels_np[i].shape[1], 3), dtype=np.uint8)
        pred_color = np.zeros((preds_np[i].shape[0], preds_np[i].shape[1], 3), dtype=np.uint8)
        
        for train_id, color in class_colors.items():
            if train_id != 255:  # 忽略ignore类
                label_color[labels_np[i] == train_id] = color
                pred_color[preds_np[i] == train_id] = color
        
        # 显示
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'样本 {i+1}: 输入图像', fontsize=12)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(label_color)
        axes[i, 1].set_title(f'样本 {i+1}: 真实标签', fontsize=12)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title(f'样本 {i+1}: 模型预测', fontsize=12)
        axes[i, 2].axis('off')
    
    plt.suptitle('Cityscapes语义分割预测结果', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 预测结果已保存到: {save_path}")
    else:
        plt.show()
    
    return fig

def calculate_metrics(preds, labels, num_classes=19, ignore_index=255):
    """计算评估指标"""
    preds_flat = preds.flatten()
    labels_flat = labels.flatten()
    
    # 忽略特定类别
    valid_mask = labels_flat != ignore_index
    preds_valid = preds_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]
    
    # 计算像素准确率
    pixel_acc = np.mean(preds_valid == labels_valid)
    
    # 计算每个类别的IoU
    ious = []
    for cls in range(num_classes):
        pred_inds = preds_valid == cls
        target_inds = labels_valid == cls
        
        intersection = np.sum(pred_inds & target_inds)
        union = np.sum(pred_inds | target_inds)
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    # 计算平均IoU，忽略nan
    ious = np.array(ious)
    valid_ious = ious[~np.isnan(ious)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else 0
    
    return pixel_acc, miou

def evaluate_model(model, dataloader, device, num_classes=19, num_batches=10):
    """评估模型性能"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("评估模型性能...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # 计算指标
    pixel_acc, miou = calculate_metrics(all_preds, all_labels, num_classes=num_classes)
    
    return pixel_acc, miou

def main():
    parser = argparse.ArgumentParser(description='Cityscapes语义分割模型预测可视化')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--model', type=str, default='./results/weights/best_model.pth',
                       help='模型权重路径 (默认: ./results/weights/best_model.pth)')
    parser.add_argument('--output', type=str, default='./results/plots',
                       help='输出目录 (默认: ./results/plots)')
    parser.add_argument('--samples', type=int, default=4,
                       help='可视化样本数量 (默认: 4)')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val'],
                       help='使用哪个数据集分割 (默认: val)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--evaluate', action='store_true',
                       help='是否评估模型性能')
    parser.add_argument('--single', action='store_true',
                       help='是否只可视化单个样本')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cityscapes语义分割模型预测可视化")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # 检查文件路径
    if not os.path.exists(args.config):
        print(f"✗ 配置文件不存在: {args.config}")
        return
    
    if not os.path.exists(args.model):
        print(f"✗ 模型文件不存在: {args.model}")
        # 尝试其他可能的路径
        alternative_paths = [
            './results/weights/final_model.pth',
            './results/weights/checkpoint_epoch_20.pth',
            './results/weights/checkpoint_epoch_10.pth'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                args.model = alt_path
                print(f"使用替代模型: {args.model}")
                break
        else:
            print("请指定正确的模型路径")
            return
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置
    config = load_config(args.config)
    if config is None:
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    
    model = load_model(args.model, model_name, num_classes, device)
    if model is None:
        return
    
    # 加载类别颜色映射
    _, _, TRAINID_TO_COLOR, _ = setup_environment()
    
    # 创建数据加载器
    dataloader, dataset = create_dataloader(config, args.split, args.samples)
    if dataloader is None:
        return
    
    # 评估模型性能（可选）
    if args.evaluate:
        print("\n" + "-" * 40)
        print("评估模型性能")
        print("-" * 40)
        
        # 创建专门用于评估的数据加载器
        eval_dataloader, _ = create_dataloader(config, args.split, num_samples=8)
        if eval_dataloader:
            pixel_acc, miou = evaluate_model(
                model, eval_dataloader, device, 
                num_classes=num_classes,
                num_batches=10
            )
            print(f"像素准确率: {pixel_acc:.4f}")
            print(f"mIoU: {miou:.4f}")
    
    # 可视化预测结果
    print("\n" + "-" * 40)
    print("生成预测可视化")
    print("-" * 40)
    
    if args.single:
        # 可视化单个样本
        images, labels = next(iter(dataloader))
        image = images[0]
        label = labels[0]
        
        save_path = os.path.join(args.output, 'single_prediction.png')
        fig = visualize_single_prediction(
            model, image, label, device, TRAINID_TO_COLOR
        )
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 单样本预测结果已保存到: {save_path}")
    else:
        # 可视化批次样本
        timestamp = torch.randint(1000, 9999, (1,)).item()
        save_path = os.path.join(args.output, f'batch_predictions_{args.seed}.png')
        
        fig = visualize_batch_predictions(
            model, dataloader, device, 
            args.samples, TRAINID_TO_COLOR,
            save_path=save_path
        )
    
    print("\n" + "=" * 60)
    print("可视化完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()