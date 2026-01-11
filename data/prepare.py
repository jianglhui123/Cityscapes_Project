import os
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class CityscapesPreparer:
    def __init__(self, data_root='/home/jianglh1/Cityscapes/data'):
        self.data_root = data_root
        self.train_dir = os.path.join(data_root, 'leftImg8bit/train')
        self.val_dir = os.path.join(data_root, 'leftImg8bit/val')
        self.label_dir = os.path.join(data_root, 'gtFine')
        
    def check_data_integrity(self):
        """检查数据完整性"""
        print("检查训练集...")
        train_cities = os.listdir(self.train_dir)
        print(f"训练城市数量: {len(train_cities)}")
        
        for city in train_cities[:2]:  # 只检查前两个城市
            city_path = os.path.join(self.train_dir, city)
            images = [f for f in os.listdir(city_path) if f.endswith('.png')]
            print(f"{city}: {len(images)} 张图片")
            
            # 检查对应的标签文件
            label_city_path = os.path.join(self.label_dir, 'train', city)
            if os.path.exists(label_city_path):
                labels = [f for f in os.listdir(label_city_path) if '_labelIds.png' in f]
                print(f"{city}: {len(labels)} 个标签")
                
                if len(images) == len(labels):
                    print("✓ 图像-标签匹配")
                else:
                    print("✗ 图像-标签不匹配")
        
        print("\n检查验证集...")
        val_cities = os.listdir(self.val_dir)
        print(f"验证城市数量: {len(val_cities)}")
        
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        stats = {
            'train': {'images': 0, 'cities': []},
            'val': {'images': 0, 'cities': []},
            'test': {'images': 0, 'cities': []}
        }
        
        for split in ['train', 'val']:
            split_dir = os.path.join(self.data_root, 'leftImg8bit', split)
            cities = os.listdir(split_dir)
            stats[split]['cities'] = cities
            
            image_count = 0
            for city in cities:
                city_path = os.path.join(split_dir, city)
                images = [f for f in os.listdir(city_path) if f.endswith('.png')]
                image_count += len(images)
            
            stats[split]['images'] = image_count
        
        return stats
    
    def visualize_sample(self, city='aachen', split='train', sample_idx=0):
        """可视化样本"""
        # 获取图像路径
        img_dir = os.path.join(self.data_root, 'leftImg8bit', split, city)
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        
        if sample_idx >= len(img_files):
            sample_idx = 0
            
        img_path = os.path.join(img_dir, img_files[sample_idx])
        
        # 获取标签路径
        label_dir = os.path.join(self.label_dir, split, city)
        base_name = img_files[sample_idx].replace('_leftImg8bit.png', '')
        label_path = os.path.join(label_dir, f'{base_name}_gtFine_labelIds.png')
        
        # 读取图像和标签
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img)
        axes[0].set_title(f'原始图像: {os.path.basename(img_path)}')
        axes[0].axis('off')
        
        axes[1].imshow(label, cmap='tab20')
        axes[1].set_title('语义标签')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"图像尺寸: {img.shape}")
        print(f"标签尺寸: {label.shape}")
        print(f"标签类别数: {len(np.unique(label))}")
        
        return img, label

if __name__ == "__main__":
    preparer = CityscapesPreparer()
    preparer.check_data_integrity()
    stats = preparer.get_dataset_stats()
    print(f"\n数据集统计:")
    print(f"训练集: {stats['train']['images']} 张图片")
    print(f"验证集: {stats['val']['images']} 张图片")
    
    # 可视化一个样本
    img, label = preparer.visualize_sample(city='aachen', split='train', sample_idx=0)