"""
Cityscapes语义分割项目

基于Cityscapes数据集实现室外场景语义分割深度学习模型。
"""

__version__ = "0.1.0"
__author__ = "你的名字"

# 可选：导入常用的模块
try:
    from data import CityscapesDataset
    from model import UNet
    from train import Trainer
    from utils import calculate_iou, plot_training_history
except ImportError:
    pass  # 在安装过程中可能会失败