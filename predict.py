#!/usr/bin/env python3
"""
Cityscapes语义分割模型预测脚本
用于对单张图片或批量图片进行语义分割
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
import time
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import sys

sys.path.append('.')

from utils.seed import set_seed
from data.classes import ID_TO_TRAINID, TRAINID_TO_COLOR
from model.unet import UNet

class CityscapesPredictor:
    def __init__(self, model_path, device='cuda', img_size=(512, 1024)):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径
            device: 设备 ('cuda' 或 'cpu')
            img_size: 输入图像尺寸 (高度, 宽度)
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.img_size = img_size
        
        print(f"使用设备: {self.device}")
        print(f"输入尺寸: {img_size}")
        
        # 加载模型
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # 预处理参数（与训练时一致）
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        # 类别数量
        self.num_classes = 19
        
    def load_model(self, model_path):
        """加载训练好的模型"""
        print(f"加载模型: {model_path}")
        
        # 创建模型结构
        model = UNet(n_channels=3, n_classes=self.num_classes)
        
        # 加载权重
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("从checkpoint加载模型权重")
                if 'miou' in checkpoint:
                    print(f"模型mIoU: {checkpoint['miou']:.4f}")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("从state_dict加载模型权重")
            else:
                model.load_state_dict(checkpoint)
                print("直接加载模型权重")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = model.to(self.device)
        print(f"模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def preprocess_image(self, image):
        """
        预处理图像，与训练时保持一致
        
        Args:
            image: 输入图像 (numpy array, BGR或RGB)
        
        Returns:
            tensor: 预处理后的图像tensor
            original_size: 原始图像尺寸
        """
        # 确保是RGB格式
        if len(image.shape) == 2:  # 灰度图
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:
            # 假设是BGR（OpenCV默认），转为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_size = image.shape[:2]  # (H, W)
        
        # 调整大小
        if self.img_size:
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]), 
                             interpolation=cv2.INTER_LINEAR)
        
        # 归一化 [0, 255] -> [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 标准化（与训练时一致）
        image = (image - self.mean) / self.std
        
        # 转换维度: HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        
        # 转换为tensor并添加batch维度
        image = torch.from_numpy(image).float()
        image = image.unsqueeze(0)  # 添加batch维度
        
        return image, original_size
    
    def predict(self, image_tensor):
        """
        进行预测
        
        Args:
            image_tensor: 预处理后的图像tensor [1, 3, H, W]
        
        Returns:
            pred: 预测的类别ID图 [H, W]
            prob: 预测的概率图 [H, W, num_classes]
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            
            # 前向传播
            outputs = self.model(image_tensor)
            
            # 获取预测类别
            pred = torch.argmax(outputs, dim=1)
            
            # 获取softmax概率
            probs = F.softmax(outputs, dim=1)
            
            # 移回CPU
            pred = pred.squeeze(0).cpu().numpy()  # [H, W]
            probs = probs.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
        
        return pred, probs
    
    def postprocess_prediction(self, pred, original_size):
        """
        后处理预测结果
        
        Args:
            pred: 预测的类别ID图
            original_size: 原始图像尺寸
        
        Returns:
            pred_resized: 调整到原始尺寸的预测图
            color_pred: 彩色化的预测图
        """
        # 调整到原始尺寸
        if pred.shape != original_size:
            pred_resized = cv2.resize(pred.astype(np.uint8), 
                                     (original_size[1], original_size[0]),
                                     interpolation=cv2.INTER_NEAREST)
        else:
            pred_resized = pred
        
        # 转换为彩色图像
        color_pred = self.label_to_color(pred_resized)
        
        return pred_resized, color_pred
    
    def label_to_color(self, label):
        """将标签ID转换为彩色图像"""
        h, w = label.shape
        color_label = np.zeros((h, w, 3), dtype=np.uint8)
        
        for train_id, color in TRAINID_TO_COLOR.items():
            color_label[label == train_id] = color
        
        return color_label
    
    def process_single_image(self, image_path, output_dir=None, save_all=False):
        """
        处理单张图片
        
        Args:
            image_path: 输入图片路径
            output_dir: 输出目录
            save_all: 是否保存所有中间结果
        
        Returns:
            dict: 包含预测结果
        """
        print(f"\n处理图片: {image_path}")
        
        # 读取图片
        if not os.path.exists(image_path):
            print(f"图片不存在: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        # 预处理
        start_time = time.time()
        image_tensor, original_size = self.preprocess_image(image)
        preprocess_time = time.time() - start_time
        
        # 预测
        start_time = time.time()
        pred, probs = self.predict(image_tensor)
        predict_time = time.time() - start_time
        
        # 后处理
        start_time = time.time()
        pred_resized, color_pred = self.postprocess_prediction(pred, original_size)
        postprocess_time = time.time() - start_time
        
        # 将原始图像转换为RGB用于显示
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 统计信息
        stats = self.get_prediction_stats(pred_resized)
        
        # 打印结果
        print(f"预处理时间: {preprocess_time:.3f}s")
        print(f"预测时间: {predict_time:.3f}s")
        print(f"后处理时间: {postprocess_time:.3f}s")
        print(f"总时间: {preprocess_time+predict_time+postprocess_time:.3f}s")
        print(f"预测统计: {stats}")
        
        # 保存结果
        result = {
            'original': original_rgb,
            'prediction': pred_resized,
            'color_prediction': color_pred,
            'probabilities': probs,
            'stats': stats,
            'original_size': original_size,
            'prediction_size': pred.shape
        }
        
        if output_dir:
            self.save_results(image_path, result, output_dir, save_all)
        
        return result
    
    def get_prediction_stats(self, pred):
        """获取预测统计信息"""
        unique, counts = np.unique(pred, return_counts=True)
        stats = {}
        
        # 统计每个类别的像素数量
        for cls, count in zip(unique, counts):
            if cls != 255:  # 忽略ignore类
                stats[int(cls)] = int(count)
        
        return stats
    
    def save_results(self, image_path, result, output_dir, save_all=False):
        """保存结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取文件名
        filename = Path(image_path).stem
        original = result['original']
        color_pred = result['color_prediction']
        
        # 1. 保存彩色预测图
        pred_path = os.path.join(output_dir, f"{filename}_prediction.png")
        cv2.imwrite(pred_path, cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR))
        print(f"保存预测图: {pred_path}")
        
        # 2. 保存叠加图
        overlay = self.create_overlay(original, color_pred)
        overlay_path = os.path.join(output_dir, f"{filename}_overlay.png")
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # 3. 保存并排对比图
        comparison = self.create_comparison(original, color_pred)
        comparison_path = os.path.join(output_dir, f"{filename}_comparison.png")
        cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
        
        # 4. 保存原始图像（如果输出目录不同）
        original_path = os.path.join(output_dir, f"{filename}_original.png")
        cv2.imwrite(original_path, cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        
        # 5. 保存类别统计
        stats_path = os.path.join(output_dir, f"{filename}_stats.txt")
        with open(stats_path, 'w') as f:
            f.write(f"图像: {image_path}\n")
            f.write(f"原始尺寸: {result['original_size'][1]}x{result['original_size'][0]}\n")
            f.write(f"预测尺寸: {result['prediction_size'][1]}x{result['prediction_size'][0]}\n")
            f.write("\n类别像素统计:\n")
            for cls, count in result['stats'].items():
                percentage = (count / np.prod(result['original_size'])) * 100
                f.write(f"类别 {cls}: {count} 像素 ({percentage:.2f}%)\n")
        
        # 6. 保存概率图（如果save_all为True）
        if save_all:
            # 保存每个类别的概率图
            probs_dir = os.path.join(output_dir, "probabilities")
            os.makedirs(probs_dir, exist_ok=True)
            
            probs = result['probabilities']
            for cls in range(probs.shape[2]):
                prob_map = (probs[:, :, cls] * 255).astype(np.uint8)
                prob_path = os.path.join(probs_dir, f"{filename}_prob_class_{cls}.png")
                cv2.imwrite(prob_path, prob_map)
    
    def create_overlay(self, original, color_pred, alpha=0.5):
        """创建原始图像和预测的叠加图"""
        # 确保尺寸相同
        if original.shape != color_pred.shape:
            color_pred_resized = cv2.resize(color_pred, 
                                          (original.shape[1], original.shape[0]))
        else:
            color_pred_resized = color_pred
        
        # 创建叠加
        overlay = cv2.addWeighted(original, 1 - alpha, 
                                color_pred_resized, alpha, 0)
        return overlay
    
    def create_comparison(self, original, color_pred):
        """创建并排对比图"""
        # 调整预测图尺寸与原始图一致
        if original.shape != color_pred.shape:
            color_pred_resized = cv2.resize(color_pred, 
                                          (original.shape[1], original.shape[0]))
        else:
            color_pred_resized = color_pred
        
        # 创建并排图像
        h, w = original.shape[:2]
        comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
        comparison[:, :w, :] = original
        comparison[:, w:, :] = color_pred_resized
        
        # 添加分隔线
        comparison[:, w-2:w+2, :] = [255, 255, 255]
        
        return comparison
    
    def visualize_result(self, result, show=True, save_path=None):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 原始图像
        axes[0, 0].imshow(result['original'])
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 预测图
        axes[0, 1].imshow(result['color_prediction'])
        axes[0, 1].set_title('语义分割结果')
        axes[0, 1].axis('off')
        
        # 叠加图
        overlay = self.create_overlay(result['original'], result['color_prediction'])
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('叠加图 (α=0.5)')
        axes[1, 0].axis('off')
        
        # 类别统计（文本）
        axes[1, 1].axis('off')
        stats_text = "类别统计:\n\n"
        total_pixels = np.prod(result['original_size'])
        
        for cls, count in result['stats'].items():
            percentage = (count / total_pixels) * 100
            stats_text += f"类别 {cls}: {count} 像素 ({percentage:.2f}%)\n"
        
        axes[1, 1].text(0, 0.5, stats_text, fontsize=12, 
                       verticalalignment='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"保存可视化结果: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def process_batch(self, input_paths, output_dir, save_all=False):
        """
        批量处理图片
        
        Args:
            input_paths: 输入图片路径列表或目录
            output_dir: 输出目录
            save_all: 是否保存所有中间结果
        """
        # 处理输入路径
        if isinstance(input_paths, str):
            if os.path.isdir(input_paths):
                # 如果是目录，获取所有图片
                image_paths = glob.glob(os.path.join(input_paths, "*.jpg")) + \
                             glob.glob(os.path.join(input_paths, "*.png")) + \
                             glob.glob(os.path.join(input_paths, "*.jpeg"))
            else:
                # 如果是通配符模式
                image_paths = glob.glob(input_paths)
        else:
            image_paths = input_paths
        
        print(f"找到 {len(image_paths)} 张图片")
        
        if not image_paths:
            print("没有找到图片")
            return
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 批量处理
        total_time = 0
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n处理第 {i+1}/{len(image_paths)} 张图片")
            
            try:
                result = self.process_single_image(image_path, output_dir, save_all)
                if result:
                    results.append(result)
                    # 简单的可视化（每5张显示一张）
                    if (i + 1) % 5 == 0:
                        self.visualize_result(result, show=False, 
                                            save_path=os.path.join(output_dir, 
                                                                 f"visualization_{i+1}.png"))
            except Exception as e:
                print(f"处理图片失败 {image_path}: {e}")
        
        print(f"\n批量处理完成! 共处理 {len(results)} 张图片")
        
        # 生成汇总报告
        if results:
            self.generate_summary_report(results, output_dir)
        
        return results
    
    def generate_summary_report(self, results, output_dir):
        """生成批量处理汇总报告"""
        report_path = os.path.join(output_dir, "summary_report.txt")
        
        total_images = len(results)
        total_pixels = sum(np.prod(r['original_size']) for r in results)
        
        # 统计所有图片的类别分布
        class_counts = {}
        for result in results:
            for cls, count in result['stats'].items():
                if cls not in class_counts:
                    class_counts[cls] = 0
                class_counts[cls] += count
        
        with open(report_path, 'w') as f:
            f.write("批量语义分割汇总报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"处理图片总数: {total_images}\n")
            f.write(f"总像素数: {total_pixels:,}\n\n")
            
            f.write("类别像素统计:\n")
            for cls in sorted(class_counts.keys()):
                percentage = (class_counts[cls] / total_pixels) * 100
                f.write(f"类别 {cls}: {class_counts[cls]:,} 像素 ({percentage:.2f}%)\n")
            
            f.write("\n处理时间统计:\n")
            # 这里可以添加时间统计，但需要修改process_single_image返回时间信息
        
        print(f"生成汇总报告: {report_path}")
        
        # 创建类别分布图
        self.plot_class_distribution(class_counts, total_pixels, output_dir)
    
    def plot_class_distribution(self, class_counts, total_pixels, output_dir):
        """绘制类别分布图"""
        classes = sorted(class_counts.keys())
        counts = [class_counts[cls] for cls in classes]
        percentages = [(count / total_pixels) * 100 for count in counts]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 柱状图
        axes[0].bar(classes, counts)
        axes[0].set_xlabel('类别ID')
        axes[0].set_ylabel('像素数量')
        axes[0].set_title('类别像素数量分布')
        axes[0].grid(True, alpha=0.3)
        
        # 饼图
        if len(classes) <= 10:  # 类别太多时饼图不好看
            axes[1].pie(counts, labels=[str(cls) for cls in classes], 
                       autopct='%1.1f%%', startangle=90)
            axes[1].set_title('类别像素比例分布')
        else:
            axes[1].text(0.5, 0.5, "类别太多，不显示饼图", 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('类别分布')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "class_distribution.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"保存类别分布图: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Cityscapes语义分割预测')
    parser.add_argument('--model', type=str, required=True,
                       help='模型权重路径 (e.g., ./results/weights/best_model.pth)')
    parser.add_argument('--input', type=str, required=True,
                       help='输入图片路径、目录或通配符模式')
    parser.add_argument('--output', type=str, default='./predictions',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='运行设备')
    parser.add_argument('--img_size', type=str, default='512,1024',
                       help='输入图像尺寸 "height,width"')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式')
    parser.add_argument('--save_all', action='store_true',
                       help='保存所有中间结果（概率图等）')
    parser.add_argument('--visualize', action='store_true',
                       help='可视化结果')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 解析图像尺寸
    img_size = tuple(map(int, args.img_size.split(',')))
    
    # 创建预测器
    predictor = CityscapesPredictor(args.model, args.device, img_size)
    
    # 处理图片
    if args.batch:
        # 批量处理模式
        results = predictor.process_batch(args.input, args.output, args.save_all)
        
        if args.visualize and results:
            # 可视化最后一张图片
            predictor.visualize_result(results[-1], save_path=os.path.join(args.output, "final_visualization.png"))
    else:
        # 单张图片模式
        result = predictor.process_single_image(args.input, args.output, args.save_all)
        
        if args.visualize and result:
            predictor.visualize_result(result, save_path=os.path.join(args.output, "visualization.png"))


if __name__ == "__main__":
    main()