import numpy as np
import torch

def calculate_iou(pred, target, num_classes=19, ignore_index=255):
    """
    计算mIoU
    """
    ious = []
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # 忽略特定类别
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        
        if union == 0:
            ious.append(float('nan'))  # 避免除以0
        else:
            ious.append(float(intersection) / float(union))
    
    # 计算平均IoU，忽略nan
    ious = np.array(ious)
    valid_ious = ious[~np.isnan(ious)]
    miou = np.mean(valid_ious) if len(valid_ious) > 0 else 0
    
    return miou

def calculate_pixel_accuracy(pred, target, ignore_index=255):
    """
    计算像素准确率
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    
    # 忽略特定类别
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    
    correct = (pred == target).sum()
    total = len(pred)
    
    return correct / total if total > 0 else 0

def calculate_class_accuracy(pred, target, num_classes=19, ignore_index=255):
    """
    计算每个类别的准确率
    """
    class_acc = []
    
    for cls in range(num_classes):
        mask = target == cls
        if mask.sum() > 0:
            acc = (pred[mask] == cls).sum() / mask.sum()
            class_acc.append(acc)
        else:
            class_acc.append(float('nan'))
    
    return np.array(class_acc)