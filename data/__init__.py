"""
Cityscapes数据集相关模块
"""
from .dataset import CityscapesDataset, get_transforms
from .prepare import CityscapesPreparer
from .classes import CITYSCAPES_CLASSES, ID_TO_TRAINID, TRAINID_TO_COLOR

__version__ = "1.0.0"
__all__ = [
    "CityscapesDataset",
    "get_transforms",
    "CityscapesPreparer",
    "CITYSCAPES_CLASSES",
    "ID_TO_TRAINID",
    "TRAINID_TO_COLOR"
]