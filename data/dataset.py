import os
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 注意：我们将在下面添加导入语句
# 为了测试，先直接定义类

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, img_size=(512, 1024)):
        """
        Args:
            root_dir: 数据根目录
            split: 'train', 'val', 'test'
            transform: 数据增强
            img_size: 输出图像尺寸 (H, W)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        
        # 收集所有图像路径
        self.images = []
        self.labels = []
        
        img_dir = os.path.join(root_dir, 'leftImg8bit', split)
        label_dir = os.path.join(root_dir, 'gtFine', split)
        
        cities = sorted(os.listdir(img_dir))
        
        for city in cities:
            city_img_dir = os.path.join(img_dir, city)
            city_label_dir = os.path.join(label_dir, city)
            
            if not os.path.exists(city_label_dir):      # 这里可能有点多余
                continue
                
            img_files = sorted([f for f in os.listdir(city_img_dir) 
                              if f.endswith('_leftImg8bit.png')])
            
            for img_file in img_files:
                base_name = img_file.replace('_leftImg8bit.png', '')
                label_file = f'{base_name}_gtFine_labelIds.png'
                label_path = os.path.join(city_label_dir, label_file)
                
                if os.path.exists(label_path):
                    self.images.append(os.path.join(city_img_dir, img_file))
                    self.labels.append(label_path)
        
        print(f"加载 {split} 集: {len(self.images)} 个样本")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 读取图像
        img_path = self.images[idx]
        label_path = self.labels[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 读取标签并映射到19类
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label = self.map_labels(label)
        
        # 调整大小
        if self.img_size:
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.img_size[1], self.img_size[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']
        else:
            # 转换为tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            label = torch.from_numpy(label).long()
        
        return image, label
    
    def map_labels(self, label):
        """将34类标签映射到19类"""
        # 直接使用ID_TO_TRAINID映射     # ？classes.py里不是有了吗
        ID_TO_TRAINID = {
            0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255,
            10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 29: 255, 30: 16, 31: 17, 32: 18, 33: 255, -1: 255
        }
        
        label_copy = label.copy()
        for k, v in ID_TO_TRAINID.items():
            label_copy[label == k] = v
        return label_copy

def get_transforms(split='train', img_size=(512, 1024), use_strong_aug=False):
    """获取数据增强转换"""
    if split == 'train':
        if use_strong_aug:
            # 更强的数据增强
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            # 原来的数据增强
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.HueSaturationValue(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    else:
        # 验证集使用简单的转换
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])