import yaml
import torch
from torch.utils.data import DataLoader
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import CityscapesDataset, get_transforms
from model.unet import get_model
from train.trainer import Trainer
from utils.visualize import plot_training_history
import argparse

def load_config(config_path='config/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Cityscapes语义分割训练')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    save_config = config['save']
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and training_config['device'] == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 准备数据
    print("加载数据集...")
    
    # 获取是否使用强数据增强的配置
    use_strong_aug = data_config.get('use_strong_aug', False)
    
    train_transform = get_transforms(
        'train', 
        tuple(data_config['img_size']), 
        use_strong_aug=use_strong_aug
    )
    val_transform = get_transforms(
        'val', 
        tuple(data_config['img_size']), 
        use_strong_aug=False  # 验证集不使用强增强
    )
    
    train_dataset = CityscapesDataset(
        root_dir=data_config['root_dir'],
        split='train',
        transform=train_transform,
        img_size=tuple(data_config['img_size'])
    )
    
    val_dataset = CityscapesDataset(
        root_dir=data_config['root_dir'],
        split='val',
        transform=val_transform,
        img_size=tuple(data_config['img_size'])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    
    # 创建模型
    print(f"创建模型: {model_config['name']}")
    model = get_model(model_config['name'], model_config['num_classes'])
    
    # 如果有检查点，恢复训练
    if args.resume:
        print(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['miou']
        print(f"恢复 epoch {start_epoch}, 最佳mIoU: {best_miou:.4f}")
    else:
        start_epoch = 0
        best_miou = 0
    
    # 创建训练器配置
    trainer_config = {
        'num_epochs': training_config['num_epochs'],
        'learning_rate': training_config['learning_rate'],
        'weight_decay': training_config['weight_decay'],
        'batch_size': data_config['batch_size'],
        'save_dir': save_config['dir'],
        
        # 添加新的配置项
        'optimizer': training_config.get('optimizer', 'adamw'),
        'scheduler': training_config.get('scheduler', 'plateau'),
        'use_amp': training_config.get('use_amp', True),
        
        # 传递scheduler_config（如果有）
        'scheduler_config': training_config.get('scheduler_config', {})
    }
    
    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    
    # 训练
    train_losses, val_losses, val_mious = trainer.train()
    
    # 绘制训练历史
    plot_training_history(
        train_losses, 
        val_losses, 
        val_mious,
        save_path=os.path.join(save_config['dir'], 'plots', 'training_history.png')
    )
    
    print(f"训练完成！最佳mIoU: {trainer.best_miou:.4f}")
    print(f"模型保存在: {save_config['dir']}")

if __name__ == "__main__":
    main()