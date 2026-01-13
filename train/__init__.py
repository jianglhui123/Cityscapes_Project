"""
训练相关模块
"""
from .trainer import Trainer
from .optimizer import create_optimizer, create_scheduler

__version__ = "1.0.0"
__all__ = [
    "Trainer",
    "create_optimizer",
    "create_scheduler"
]