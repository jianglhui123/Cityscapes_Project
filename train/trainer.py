import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from tqdm import tqdm

from utils.metrics import calculate_iou, calculate_pixel_accuracy

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 确保配置参数是正确类型
        self._sanitize_config()
        
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
        
        # 损失函数 - 忽略255（ignore类）
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        
        # 优化器（确保参数是浮点数）
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=float(self.config['learning_rate']),
            weight_decay=float(self.config['weight_decay'])
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_mious = []
        self.best_miou = 0
        
        # 创建结果目录
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], 'weights'), exist_ok=True)
        os.makedirs(os.path.join(config['save_dir'], 'plots'), exist_ok=True)
    
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
        """训练一个epoch"""
        self.model.train()  # 将神经网络切换到训练模式，影响模型中Dropout层和BatchNorm层
        total_loss = 0
        progress_bar = tqdm(self.train_loader, desc='训练')     # 进度条
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            # images = images.to(self.device, non_blocking=True)

            # 确保标签为long类型
            if labels.dtype != torch.long:
                labels = labels.long()
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
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
        print(f"开始训练，共 {self.config['num_epochs']} 个epoch")
        print(f"设备: {self.device}")
        print(f"批次大小: {self.config['batch_size']}")
        print(f"学习率: {self.config['learning_rate']}")
        print(f"权重衰减: {self.config['weight_decay']}")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.config['num_epochs']}")
            print(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, miou, pixel_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_mious.append(miou)
            
            # 更新学习率
            self.scheduler.step(miou)
            
            # 打印结果
            print(f"\n结果:")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证mIoU: {miou:.4f}")
            print(f"像素准确率: {pixel_acc:.4f}")
            
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
                print(f"保存最佳模型到 {save_path}，mIoU: {miou:.4f}")
            
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
                print(f"保存检查点到 {save_path}")
        
        print(f"\n{'='*50}")
        print(f"训练完成！最佳mIoU: {self.best_miou:.4f}")
        print(f"{'='*50}")
        
        # 保存最终模型
        final_path = os.path.join(self.config['save_dir'], 'weights', 'final_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_miou': self.best_miou
        }, final_path)
        print(f"保存最终模型到 {final_path}")
        
        return self.train_losses, self.val_losses, self.val_mious