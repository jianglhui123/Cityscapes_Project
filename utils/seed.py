"""
随机种子设置，确保实验可复现
"""
import random
import numpy as np
import torch
import os

def set_seed(seed=42, deterministic=True):
    """
    设置所有随机种子以确保可复现性
    
    Args:
        seed: 随机种子值
        deterministic: 是否使用确定性算法（可能降低性能但提高可复现性）
    """
    # Python内置随机
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    
    # 设置CuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True  # 可以提高性能但降低可复现性
    
    # 设置环境变量（可选）
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子设置为: {seed}")
    print(f"确定性模式: {deterministic}")
    print(f"CuDNN确定性: {torch.backends.cudnn.deterministic}")
    print(f"CuDNN基准: {torch.backends.cudnn.benchmark}")

def seed_worker(worker_id):
    """
    为DataLoader的工作进程设置随机种子
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloader_kwargs(seed=42):
    """
    获取用于DataLoader的随机种子相关参数
    
    Returns:
        dict: 包含generator和worker_init_fn的字典
    """
    # 创建随机数生成器
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return {
        'generator': generator,
        'worker_init_fn': seed_worker
    }