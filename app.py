#!/usr/bin/env python3
"""
简单的Web界面用于语义分割演示
需要安装: pip install flask pillow
"""

from flask import Flask, render_template, request, send_file, jsonify
import torch
import numpy as np
import cv2
import os
import io
from PIL import Image
import base64
import json
import time

from model.unet import UNet
from data.classes import TRAINID_TO_COLOR, CITYSCAPES_CLASSES, ID_TO_NAME
app = Flask(__name__)

# 全局变量
model = None
device = None

def init_model(model_path='./results_seed42_lr0.005_onecycle/weights/best_model.pth'):
    """初始化模型"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = UNet(n_channels=3, n_classes=19)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        print(f"模型加载完成: {model_path}")
    else:
        print(f"模型文件不存在: {model_path}")
        model = None
    
    return model

def predict_image(image_array):
    """预测单张图片"""
    if model is None:
        return None
    
    # 预处理
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # 调整大小
    img_size = (512, 1024)
    image = cv2.resize(image_array, (img_size[1], img_size[0]))
    
    # 归一化和标准化
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    
    # 转换维度
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float().unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
    
    # 转换为彩色
    color_pred = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for train_id, color in TRAINID_TO_COLOR.items():
        color_pred[pred == train_id] = color
    
    # 调整回原始尺寸
    original_size = image_array.shape[:2]
    color_pred = cv2.resize(color_pred, (original_size[1], original_size[0]))
    
    return color_pred

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    try:
        # 读取图片
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 转换为OpenCV格式
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # 确保是RGB
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
        
        # BGR转RGB（如果OpenCV读取）
        if image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        
        # 预测
        start_time = time.time()
        prediction = predict_image(image_array)
        inference_time = time.time() - start_time
        
        if prediction is None:
            return jsonify({'error': '模型未初始化'})
        
        # 转换为base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR))
        prediction_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # 原始图像也转为base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
        original_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # 创建叠加图
        alpha = 0.5
        overlay = cv2.addWeighted(image_array, 1 - alpha, prediction, alpha, 0)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        overlay_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # 统计信息
        unique, counts = np.unique(prediction.reshape(-1, 3), axis=0, return_counts=True)
        stats = []
        for color, count in zip(unique, counts):
            # 找到对应的类别ID
            for cls, cls_color in TRAINID_TO_COLOR.items():
                if np.array_equal(color, cls_color):
                    percentage = (count / np.prod(prediction.shape[:2])) * 100
                    # 获取类别名称
                    class_name = CITYSCAPES_CLASSES[cls] if cls < len(CITYSCAPES_CLASSES) else f'Class {cls}'
                    stats.append({
                        'class': int(cls),
                        'class_name': class_name,  # 添加类别名称
                        'pixels': int(count),
                        'percentage': float(percentage),
                        'color': [int(c) for c in color]
                    })
                    break
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'prediction': prediction_b64,
            'overlay': overlay_b64,
            'inference_time': inference_time,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# @app.route('/demo')
# def demo():
#     """演示页面"""
#     # 使用示例图片
#     demo_images = [
#         {'name': '城市街道1', 'path': 'demo/city1.jpg'},
#         {'name': '城市街道2', 'path': 'demo/city2.jpg'},
#         {'name': '高速公路', 'path': 'demo/highway.jpg'},
#     ]
#     return render_template('demo.html', demo_images=demo_images)

@app.route('/demo')
def demo():
    """演示页面 - 动态加载示例图片"""
    try:
        # 创建演示目录（如果不存在）
        demo_dir = 'static/demo'
        os.makedirs(demo_dir, exist_ok=True)
        
        # 获取演示目录中的所有图片
        demo_images = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        
        for filename in os.listdir(demo_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # 获取相对路径用于Web访问
                rel_path = f'demo/{filename}'
                # 获取文件创建时间
                file_path = os.path.join(demo_dir, filename)
                stat = os.stat(file_path)
                created_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_ctime))
                
                demo_images.append({
                    'name': filename,
                    'path': rel_path,
                    'created': created_time,
                    'size': os.path.getsize(file_path) // 1024  # KB
                })
        
        # 如果没有演示图片，使用默认占位图
        if not demo_images:
            print("未找到演示图片，使用默认示例")
            # 创建一些默认演示图片（从Cityscapes数据集中）
            self.create_default_demo_images()
            # 重新扫描
            demo_images = []
            for filename in os.listdir(demo_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    rel_path = f'demo/{filename}'
                    demo_images.append({
                        'name': filename,
                        'path': rel_path,
                        'created': '2024-01-01 00:00:00',
                        'size': 500
                    })
        
        return render_template('demo.html', demo_images=demo_images)
    except Exception as e:
        print(f"演示页面错误: {e}")
        return render_template('demo.html', demo_images=[], error=str(e))

def create_default_demo_images(self):
    """创建默认演示图片（从Cityscapes验证集复制）"""
    import shutil
    from pathlib import Path
    
    demo_dir = 'static/demo'
    os.makedirs(demo_dir, exist_ok=True)
    
    # Cityscapes数据集路径
    cityscapes_dir = '/home/jianglh1/Cityscapes/data'
    val_images_dir = os.path.join(cityscapes_dir, 'leftImg8bit/val')
    
    if os.path.exists(val_images_dir):
        print("从Cityscapes验证集复制演示图片...")
        
        # 获取验证集中的一些图片
        all_images = []
        for city in os.listdir(val_images_dir):
            city_dir = os.path.join(val_images_dir, city)
            if os.path.isdir(city_dir):
                images = [os.path.join(city_dir, f) for f in os.listdir(city_dir) 
                         if f.endswith('.png')]
                all_images.extend(images)
        
        # 随机选择3张图片作为演示
        import random
        random.seed(42)
        selected_images = random.sample(all_images, min(3, len(all_images)))
        
        for i, src_path in enumerate(selected_images, 1):
            dst_path = os.path.join(demo_dir, f'demo_{i}.png')
            try:
                shutil.copy2(src_path, dst_path)
                print(f"复制: {src_path} -> {dst_path}")
                
                # 如果需要，可以转换为JPG以减小文件大小
                if dst_path.endswith('.png'):
                    img = Image.open(dst_path)
                    jpg_path = dst_path.replace('.png', '.jpg')
                    img.convert('RGB').save(jpg_path, 'JPEG', quality=90)
                    os.remove(dst_path)  # 删除PNG版本
                    dst_path = jpg_path
                
            except Exception as e:
                print(f"复制失败 {src_path}: {e}")
    else:
        print("Cityscapes数据集不存在，创建空白演示图片")
        # 创建空白占位图
        for i in range(1, 4):
            img_path = os.path.join(demo_dir, f'demo_{i}.png')
            img = Image.new('RGB', (512, 256), color=(73, 109, 137))
            
            # 添加文字
            from PIL import ImageDraw, ImageFont
            try:
                draw = ImageDraw.Draw(img)
                # 尝试使用默认字体
                try:
                    font = ImageFont.truetype("arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                text = f"Demo Image {i}"
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                position = ((512 - text_width) // 2, (256 - text_height) // 2)
                draw.text(position, text, font=font, fill=(255, 255, 255))
            except:
                pass
            
            img.save(img_path)
            print(f"创建占位图: {img_path}")

@app.route('/upload_demo', methods=['POST'])
def upload_demo():
    """上传图片到演示图库"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '没有文件'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 检查文件类型
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        filename = file.filename
        if '.' not in filename or filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'success': False, 'error': '不支持的文件格式'})
        
        # 创建安全的文件名
        from werkzeug.utils import secure_filename
        safe_filename = secure_filename(filename)
        
        # 确保文件名唯一
        demo_dir = 'static/demo'
        os.makedirs(demo_dir, exist_ok=True)
        
        # 如果文件已存在，添加时间戳
        base_name, ext = os.path.splitext(safe_filename)
        counter = 1
        final_filename = safe_filename
        
        while os.path.exists(os.path.join(demo_dir, final_filename)):
            final_filename = f"{base_name}_{counter}{ext}"
            counter += 1
        
        # 保存文件
        file_path = os.path.join(demo_dir, final_filename)
        file.save(file_path)
        
        print(f"演示图片上传成功: {final_filename}")
        
        return jsonify({
            'success': True,
            'filename': final_filename,
            'message': '上传成功'
        })
        
    except Exception as e:
        print(f"上传失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/demo_image/<filename>')
def demo_image(filename):
    """提供演示图片"""
    demo_dir = 'static/demo'
    file_path = os.path.join(demo_dir, filename)
    
    if os.path.exists(file_path):
        return send_file(file_path)
    else:
        return "图片不存在", 404

@app.route('/delete_demo/<filename>', methods=['DELETE'])
def delete_demo(filename):
    """删除演示图片"""
    try:
        demo_dir = 'static/demo'
        file_path = os.path.join(demo_dir, filename)
        
        if os.path.exists(file_path):
            os.remove(file_path)
            return jsonify({'success': True, 'message': '删除成功'})
        else:
            return jsonify({'success': False, 'error': '文件不存在'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/model_info')
def model_info():
    """获取模型信息"""
    if model is None:
        return jsonify({'error': '模型未初始化'})
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return jsonify({
        'device': str(device),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_loaded': model is not None
    })

def initialize_demo_images():
    """初始化演示图片"""
    demo_dir = 'static/demo'
    os.makedirs(demo_dir, exist_ok=True)
    
    # 检查是否有演示图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    has_images = False
    
    for filename in os.listdir(demo_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            has_images = True
            break
    
    if not has_images:
        print("未找到演示图片，正在创建...")
        create_default_demo_images()

def create_default_demo_images():
    """创建默认演示图片"""
    import shutil
    from PIL import Image, ImageDraw, ImageFont
    
    demo_dir = 'static/demo'
    
    # 首先尝试从Cityscapes数据集复制
    cityscapes_dir = '/home/jianglh1/Cityscapes/data'
    val_images_dir = os.path.join(cityscapes_dir, 'leftImg8bit/val')
    
    if os.path.exists(val_images_dir):
        print("从Cityscapes验证集复制演示图片...")
        
        try:
            # 获取所有验证集图片
            all_images = []
            for city in os.listdir(val_images_dir):
                city_dir = os.path.join(val_images_dir, city)
                if os.path.isdir(city_dir):
                    images = [os.path.join(city_dir, f) for f in os.listdir(city_dir) 
                             if f.endswith('_leftImg8bit.png')]
                    all_images.extend(images)
            
            # 随机选择3张
            import random
            random.seed(42)
            selected_images = random.sample(all_images, min(3, len(all_images)))
            
            for i, src_path in enumerate(selected_images, 1):
                dst_name = f"city_demo_{i}.jpg"
                dst_path = os.path.join(demo_dir, dst_name)
                
                # 读取并保存为JPG
                img = Image.open(src_path)
                img = img.convert('RGB')
                
                # 调整大小以减小文件
                img.thumbnail((800, 600))
                
                # 保存
                img.save(dst_path, 'JPEG', quality=85)
                print(f"创建演示图片: {dst_name}")
                
        except Exception as e:
            print(f"从Cityscapes复制失败: {e}")
            create_placeholder_images()
    else:
        print("Cityscapes数据集不存在，创建占位图片")
        create_placeholder_images()

def create_placeholder_images():
    """创建占位图片"""
    demo_dir = 'static/demo'
    
    for i in range(1, 4):
        img_path = os.path.join(demo_dir, f'demo_{i}.jpg')
        
        # 创建带有示例文本的图片
        img = Image.new('RGB', (640, 360), color=(73, 109, 137))
        draw = ImageDraw.Draw(img)
        
        try:
            # 尝试加载字体
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                # 如果arial不存在，使用默认字体
                font = ImageFont.load_default()
            
            text = f"演示图片 {i}"
            draw.text((200, 150), text, fill=(255, 255, 255), font=font)
            
        except Exception as e:
            print(f"添加文字失败: {e}")
        
        img.save(img_path, 'JPEG', quality=90)
        print(f"创建占位图片: demo_{i}.jpg")

# if __name__ == '__main__':
#     # 初始化模型
#     init_model()
    
#     # 创建模板目录
#     os.makedirs('templates', exist_ok=True)
    
#     # 创建静态目录用于保存上传文件
#     os.makedirs('static/uploads', exist_ok=True)
    
#     # 运行Flask应用
#     app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    # 初始化模型
    init_model()
    
    # 初始化演示图片
    initialize_demo_images()
    
    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/demo', exist_ok=True)
    os.makedirs('static/uploads', exist_ok=True)
    
    # 运行Flask应用
    print("服务器启动中...")
    print("访问地址: http://localhost:5000")
    print("演示图库: http://localhost:5000/demo")
    
    app.run(host='0.0.0.0', port=5000, debug=True)