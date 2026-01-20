import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm

from utils.metrics import calculate_iou, calculate_pixel_accuracy
from .optimizer import create_optimizer, create_scheduler
from torch.cuda.amp import autocast, GradScaler

from utils.logger import get_logger

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 设置随机种子（如果提供了）
        if 'seed' in config:
            import random
            import numpy as np
            import torch
            
            seed = config['seed']
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            print(f"Trainer内部随机种子: {seed}")

        # 设置日志记录器
        self.logger = get_logger()

        # 确保配置参数是正确类型
        self._sanitize_config()
        
        # 记录配置信息
        self.logger.info("=" * 60)
        self.logger.info("训练器初始化")
        self.logger.info("=" * 60)

        # 设备设置
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"使用设备: {self.device}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print(f"使用设备: {self.device}")
        
        self.model = self.model.to(self.device)     # 将整个神经网络模型（包括所有层、参数、缓冲区）移动到指定的计算设备上。

        # 记录模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"模型参数总数: {total_params:,}")
        self.logger.info(f"可训练参数: {trainable_params:,}") 

        # 损失函数 - 忽略255（ignore类）
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # 混合精度训练设置
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            self.logger.info("启用混合精度训练 (AMP)")
        else:
            self.scaler = None
        
        # 创建优化器（使用新的函数）
        self.optimizer = create_optimizer(model, config)
        
        # 创建调度器（需要知道每个epoch的步数）
        self.scheduler_type = config.get('scheduler', 'plateau')
        if self.scheduler_type == 'onecycle':
            # OneCycleLR需要在训练开始前知道steps_per_epoch
            self.steps_per_epoch = len(train_loader)
            self.scheduler = create_scheduler(
                self.optimizer, 
                config, 
                steps_per_epoch=self.steps_per_epoch
            )
        else:
            self.scheduler = create_scheduler(self.optimizer, config)
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.best_miou = 0
        
        # 创建结果目录
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], 'weights'), exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], 'plots'), exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], 'logs'), exist_ok=True)

        self.logger.info("训练器初始化完成")
        self.logger.info("-" * 60)
    
    def _sanitize_config(self):
        """清理配置参数，确保类型正确"""
        # 确保必要的键存在
        required_keys = ['learning_rate', 'weight_decay', 'batch_size', 'num_epochs']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"配置中缺少必要的键: {key}")
        
        # 转换类型
        self.config['learning_rate'] = float(self.config['learning_rate'])
        self.config['weight_decay'] = float(self.config['weight_decay'])
        self.config['batch_size'] = int(self.config['batch_size'])
        self.config['num_epochs'] = int(self.config['num_epochs'])
        
        # 添加默认值（如果不存在）
        self.config.setdefault('save_dir', './results')
    
    def train_epoch(self):
        """训练一个epoch（支持混合精度）"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='训练')
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            
            # 确保标签为long类型
            if labels.dtype != torch.long:
                labels = labels.long()
            labels = labels.to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            if self.use_amp and self.scaler is not None:
                # 混合精度训练
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # 缩放损失并反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 普通训练
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # 更新学习率（对于OneCycleLR，每一步都需要更新）
            if self.scheduler_type == 'onecycle':
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证"""
        self.model.eval()   # 将神经网络切换到测试模式
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc='验证')
            for images, labels in progress_bar:
                images = images.to(self.device)
                
                # 确保标签为long类型
                if labels.dtype != torch.long:
                    labels = labels.long()
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # 获取预测
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # 计算指标
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        miou = calculate_iou(all_preds, all_labels, num_classes=19, ignore_index=255)
        pixel_acc = calculate_pixel_accuracy(all_preds, all_labels, ignore_index=255)
        
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, miou, pixel_acc
    
    def train(self):
        """完整训练过程"""
        self.logger.info("=" * 60)
        self.logger.info("开始训练")
        self.logger.info("=" * 60)
        self.logger.info(f"训练epoch数: {self.config['num_epochs']}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"批次大小: {self.config['batch_size']}")
        self.logger.info(f"学习率: {self.config['learning_rate']}")
        self.logger.info(f"权重衰减: {self.config['weight_decay']}")
        
        start_time = time.time()

        for epoch in range(self.config['num_epochs']):
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            self.logger.info(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, miou, pixel_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_mious.append(miou)
            
            # 更新学习率（对于非OneCycleLR调度器）
            if self.scheduler_type != 'onecycle':
                if self.scheduler_type == 'plateau':
                    self.scheduler.step(miou)  # 基于验证指标更新
                else:
                    self.scheduler.step()  # 基于epoch更新
            
            # 记录结果
            self.logger.info(f"训练损失: {train_loss:.6f}")
            self.logger.info(f"验证损失: {val_loss:.6f}")
            self.logger.info(f"验证mIoU: {miou:.6f}")
            self.logger.info(f"像素准确率: {pixel_acc:.6f}")
            
            # 保存最佳模型
            if miou > self.best_miou:
                self.best_miou = miou
                save_path = os.path.join(self.config['save_dir'], 'weights', 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'miou': miou,
                    'config': self.config
                }, save_path)
                self.logger.info(f"✨ 保存最佳模型，mIoU: {miou:.6f} ✨")
                self.logger.info(f"保存路径: {save_path}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                save_path = os.path.join(self.config['save_dir'], 'weights', f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'miou': miou,
                    'config': self.config
                }, save_path)
                self.logger.info(f"💾 保存检查点: {save_path}")
        
        # 计算总训练时间
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("训练完成!")
        self.logger.info(f"总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
        self.logger.info(f"最佳mIoU: {self.best_miou:.6f}")
        self.logger.info(f"{'='*60}")
        
        # 保存最终模型
        final_path = os.path.join(self.config['save_dir'], 'weights', 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_miou': self.best_miou
        }, final_path)
        self.logger.info(f"保存最终模型到: {final_path}")
        
        return self.train_losses, self.val_losses, self.val_mious