import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def plot_training_history(train_losses, val_losses, val_mious, save_path=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', marker='o')
    axes[0].plot(val_losses, label='验证损失', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练和验证损失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # mIoU曲线
    axes[1].plot(val_mious, label='验证mIoU', color='green', marker='^')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('mIoU')
    axes[1].set_title('验证mIoU')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def visualize_predictions(model, dataloader, device, num_samples=4):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批数据
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
    
    # 转换为CPU numpy
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    
    # 从classes.py导入颜色映射
    from data.classes import TRAINID_TO_COLOR
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(min(num_samples, len(images))):
        # 原始图像（反标准化）
        img = images[i].transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # 真实标签颜色
        label_color = np.zeros((labels[i].shape[0], labels[i].shape[1], 3), dtype=np.uint8)
        for train_id, color in TRAINID_TO_COLOR.items():
            label_color[labels[i] == train_id] = color
        
        # 预测标签颜色
        pred_color = np.zeros((preds[i].shape[0], preds[i].shape[1], 3), dtype=np.uint8)
        for train_id, color in TRAINID_TO_COLOR.items():
            pred_color[preds[i] == train_id] = color
        
        # 显示
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('原始图像')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(label_color)
        axes[i, 1].set_title('真实标签')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(pred_color)
        axes[i, 2].set_title('预测标签')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()