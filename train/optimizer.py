"""
优化器和学习率调度器配置
"""
import torch
import torch.optim as optim
import math
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
def create_optimizer(model, config):
    """
    创建优化器
    
    Args:
        model: 神经网络模型
        config: 配置字典，包含优化器相关参数
    
    Returns:
        optimizer: 优化器
    """
    # 获取优化器类型，默认为AdamW
    optimizer_type = config.get('optimizer', 'adamw')
    learning_rate = float(config.get('learning_rate', 0.001))
    weight_decay = float(config.get('weight_decay', 1e-4))
    
    if optimizer_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        momentum = float(config.get('momentum', 0.9))
        nesterov = bool(config.get('nesterov', True))
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
        print(f"创建SGD优化器: lr={learning_rate}, momentum={momentum}, "
              f"weight_decay={weight_decay}, nesterov={nesterov}")
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    print(f"创建优化器: {optimizer_type}, 学习率: {learning_rate}, 权重衰减: {weight_decay}")
    return optimizer

def create_scheduler(optimizer, config, steps_per_epoch=None):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置字典
        steps_per_epoch: 每个epoch的步数（对于OneCycleLR需要）
    
    Returns:
        scheduler: 学习率调度器
    """
    scheduler_type = config.get('scheduler', 'plateau')
    scheduler_config = config.get('scheduler_config', {})
    
    print(f"创建调度器，类型: {scheduler_type}")
    
    if scheduler_type == 'onecycle':
        # OneCycleLR调度器
        max_lr = float(config.get('learning_rate', 0.001))
        epochs = int(config.get('num_epochs', 20))
        scheduler_config = config.get('scheduler_config', {})
        
        if steps_per_epoch is None:
            print("警告: OneCycleLR需要steps_per_epoch参数，使用默认值100")
            steps_per_epoch = 100
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=scheduler_config.get('pct_start', 0.1),
            div_factor=scheduler_config.get('div_factor', 10),
            final_div_factor=scheduler_config.get('final_div_factor', 100)
        )
        print(f"创建OneCycleLR调度器: max_lr={max_lr}, epochs={epochs}, steps_per_epoch={steps_per_epoch}")
        
    elif scheduler_type == 'cosine':
        # 余弦退火调度器，可选预热
        epochs = int(config.get('num_epochs', 20))
        eta_min = scheduler_config.get('eta_min', float(config.get('learning_rate', 0.001)) * 0.01)
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        if warmup_epochs > 0:
            # 带预热的余弦退火
            print(f"创建带预热的余弦退火调度器: T_max={epochs}, "
                  f"eta_min={eta_min}, warmup_epochs={warmup_epochs}")
            
            # 定义调度器函数
            def lr_lambda(epoch):
                # 预热阶段
                if epoch < warmup_epochs:
                    return float(epoch) / float(max(1, warmup_epochs))
                # 余弦退火阶段
                progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                return eta_min + 0.5 * (1 - eta_min) * (1 + math.cos(math.pi * progress))
            
            scheduler = LambdaLR(optimizer, lr_lambda)
        else:
            # 标准余弦退火
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=eta_min
            )
            print(f"创建CosineAnnealingLR调度器: T_max={epochs}, eta_min={eta_min}")
        
    else:
        # ReduceLROnPlateau调度器（默认）
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        print("创建ReduceLROnPlateau调度器")
    
    return scheduler