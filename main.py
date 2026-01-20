import yaml
import torch
from torch.utils.data import DataLoader
import sys
import os
import argparse

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.seed import set_seed, get_dataloader_kwargs
from data.dataset import CityscapesDataset, get_transforms
from model.unet import get_model
from train.trainer import Trainer
from utils.visualize import plot_training_history
from utils.logger import setup_logger

def load_config(config_path='config/config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def backup_config(config, save_dir):
    """备份配置文件到结果目录"""
    backup_path = os.path.join(save_dir, 'config_backup.yaml')
    with open(backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return backup_path

def main():
    parser = argparse.ArgumentParser(description='Cityscapes语义分割训练')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子（在加载配置后，创建模型和数据之前）
    seed = args.seed
    set_seed(seed)
    
    # 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    save_config = config['save']
    

    # 获取DataLoader的随机种子参数
    dataloader_kwargs = get_dataloader_kwargs(seed)
    
    # 设置日志记录器，使用配置文件中的保存目录
    log_dir = os.path.join(save_config['dir'], 'logs')
    logger = setup_logger(log_dir=log_dir, name='Cityscapes_Train').get_logger()

    logger.info("=" * 60)
    logger.info("Cityscapes语义分割项目启动")
    logger.info("=" * 60)
    logger.info(f"加载配置文件: {args.config}")
    logger.info(f"日志保存目录: {log_dir}")

# 备份配置文件
    config_backup_path = backup_config(config, save_config['dir'])
    logger.info(f"配置文件备份到: {config_backup_path}")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and training_config['device'] == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 准备数据
    logger.info("加载数据集...")
    
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
    
    # 使用带有随机种子控制的DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        **dataloader_kwargs  # 添加随机种子控制
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True,
        **dataloader_kwargs  # 添加随机种子控制
    )
    
    logger.info(f"训练集: {len(train_dataset)} 样本")
    logger.info(f"验证集: {len(val_dataset)} 样本")
    logger.info(f"批次大小: {data_config['batch_size']}")
    logger.info(f"工作线程数: {data_config['num_workers']}")
    logger.info(f"使用强数据增强: {use_strong_aug}")
    
    # 创建模型
    logger.info(f"创建模型: {model_config['name']}")
    model = get_model(model_config['name'], model_config['num_classes'])
    
    # 如果有检查点，恢复训练
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        best_miou = checkpoint['miou']
        logger.info(f"恢复 epoch {start_epoch}, 最佳mIoU: {best_miou:.4f}")
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
        'seed': seed,  # 传递随机种子
        
        # 优化器配置项
        'optimizer': training_config.get('optimizer', 'adamw'),
        'scheduler': training_config.get('scheduler', 'plateau'),
        'use_amp': training_config.get('use_amp', True),
        
        # SGD特定参数（如果使用SGD）
        'momentum': training_config.get('momentum', 0.9),
        'nesterov': training_config.get('nesterov', True),

        # 传递scheduler_config（如果有）
        'scheduler_config': training_config.get('scheduler_config', {})
    }
    
    # 记录训练配置
    logger.info("训练配置:")
    logger.info(f"  Epoch数: {trainer_config['num_epochs']}")
    logger.info(f"  学习率: {trainer_config['learning_rate']}")
    logger.info(f"  权重衰减: {trainer_config['weight_decay']}")
    logger.info(f"  优化器: {trainer_config['optimizer']}")
    logger.info(f"  调度器: {trainer_config['scheduler']}")
    logger.info(f"  混合精度: {trainer_config['use_amp']}")

    # 创建训练器
    trainer = Trainer(model, train_loader, val_loader, trainer_config)
    
    # 训练
    logger.info("开始训练...")
    train_losses, val_losses, val_mious = trainer.train()
    
    # 绘制训练历史
    plot_training_history(
        train_losses, 
        val_losses, 
        val_mious,
        save_path=os.path.join(save_config['dir'], 'plots', 'training_history.png')
    )
    
    logger.info(f"训练完成！最佳mIoU: {trainer.best_miou:.4f}")
    logger.info(f"模型保存在: {save_config['dir']}")
    logger.info(f"使用的随机种子: {seed}")

if __name__ == "__main__":
    main()