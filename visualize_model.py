import sys
sys.path.append('.')
import torch
from model.unet import UNet
from torchinfo import summary
import torch.nn as nn

def print_model_summary():
    """打印模型摘要"""
    print("=" * 60)
    print("U-Net 模型结构摘要")
    print("=" * 60)
    
    # 创建模型
    model = UNet(n_channels=3, n_classes=19)
    
    # 打印模型层
    print("\n模型层结构:")
    print("-" * 40)
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"{name:20} | 参数数量: {num_params:>10,}")
    
    # 打印总参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    return model

def export_model_architecture(model, save_path='model_architecture.txt'):
    """导出模型架构到文本文件"""
    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("U-Net 语义分割模型详细架构\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("模型总览:\n")
        f.write("-" * 40 + "\n")
        f.write(str(model) + "\n\n")
        
        f.write("详细层结构:\n")
        f.write("-" * 40 + "\n")
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                f.write(f"{name}:\n")
                f.write(f"  类型: {module.__class__.__name__}\n")
                if hasattr(module, 'weight') and module.weight is not None:
                    f.write(f"  权重形状: {list(module.weight.shape)}\n")
                if hasattr(module, 'bias') and module.bias is not None:
                    f.write(f"  偏置形状: {list(module.bias.shape)}\n")
                f.write("\n")
        
        # 计算各层输出形状（假设输入为[1, 3, 512, 1024]）
        f.write("\n前向传播各层输出形状:\n")
        f.write("-" * 40 + "\n")
        
        # 手动计算各层输出
        x = torch.randn(1, 3, 512, 1024)
        layers_info = []
        
        # 编码器
        x1 = model.inc(x)
        layers_info.append(("Input", x.shape))
        layers_info.append(("Inc (DoubleConv)", x1.shape))
        
        x2 = model.down1(x1)
        layers_info.append(("Down1", x2.shape))
        
        x3 = model.down2(x2)
        layers_info.append(("Down2", x3.shape))
        
        x4 = model.down3(x3)
        layers_info.append(("Down3", x4.shape))
        
        x5 = model.down4(x4)
        layers_info.append(("Down4 (瓶颈层)", x5.shape))
        
        # 解码器
        x = model.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = model.conv1(x)
        layers_info.append(("Up1 + Conv1", x.shape))
        
        x = model.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = model.conv2(x)
        layers_info.append(("Up2 + Conv2", x.shape))
        
        x = model.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = model.conv3(x)
        layers_info.append(("Up3 + Conv3", x.shape))
        
        x = model.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = model.conv4(x)
        layers_info.append(("Up4 + Conv4", x.shape))
        
        output = model.outc(x)
        layers_info.append(("Output", output.shape))
        
        for layer_name, shape in layers_info:
            f.write(f"{layer_name:20} | 输出形状: {list(shape)}\n")
    
    print(f"\n模型架构已保存到: {save_path}")

def visualize_with_torchviz(model, save_path='model_graph.png'):
    """使用torchviz可视化模型计算图"""
    try:
        from torchviz import make_dot
        import graphviz
        
        # 创建示例输入
        x = torch.randn(1, 3, 256, 512)  # 使用小一点的尺寸
        y = model(x)
        
        # 生成计算图
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        # 保存为PDF或PNG
        dot.format = 'png'
        dot.render(save_path.replace('.png', ''), cleanup=True)
        
        print(f"模型计算图已保存为: {save_path}")
        print("注意: 对于U-Net，计算图可能非常大且复杂")
        
    except ImportError:
        print("torchviz未安装，无法生成计算图")
        print("安装命令: pip install torchviz graphviz")
        print("同时需要系统安装graphviz: sudo apt-get install graphviz")

def create_ascii_model_diagram():
    """创建ASCII模型图"""
    diagram = """
    ===================================================================
                              U-Net 模型架构图
    ===================================================================

                            输入图像 (3×512×1024)
                                   │
                                   ▼
                            ┌─────────────┐
                            │   DoubleConv │
                            │    64通道    │
                            └─────────────┘
                                   │
    ┌──────────────────────────────┼──────────────────────────────┐
    │                              ▼                              │
    │                     ┌─────────────────┐                     │
    │                     │   MaxPool2d(2)  │                     │
    │                     │   DoubleConv    │←────────────────────┘
    │                     │    128通道      │                     上采样+跳跃连接
    │                     └─────────────────┘
    │                              │
    │                              ▼
    │                     ┌─────────────────┐
    │                     │   MaxPool2d(2)  │
    │                     │   DoubleConv    │←────────────────────┐
    │                     │    256通道      │                      │
    │                     └─────────────────┘                      │
    │                              │                               │
    │                              ▼                               │
    │                     ┌─────────────────┐                      │
    │                     │   MaxPool2d(2)  │                      │
    │                     │   DoubleConv    │←─────────────────────┤
    │                     │    512通道      │                      │
    │                     └─────────────────┘                      │
    │                              │                               │
    │                              ▼                               │
    │                     ┌─────────────────┐                      │
    │                     │   MaxPool2d(2)  │                      │
    │                     │   DoubleConv    │─────┐                │
    │                     │   1024通道      │     │                │
    │                     └─────────────────┘     │                │
    │                              │              │                │
    │                              ▼              │                │
    │                     ┌─────────────────┐     │                │
    │                     │    ConvTrans    │     │                │
    │                     │    上采样2×     │     │                │
    │                     └─────────────────┘     │                │
    │                              │              │                │
    │                ┌─────────────┼──────────────┘                │
    │                │             ▼                                │
    │                │    ┌─────────────────┐                      │
    │                │    │   拼接+DoubleConv│                      │
    │                │    │    512通道      │                      │
    │                │    └─────────────────┘                      │
    │                │             │                                │
    │                │             ▼                                │
    │                │    ┌─────────────────┐                      │
    │                │    │    ConvTrans    │                      │
    │                │    │    上采样2×     │                      │
    │                │    └─────────────────┘                      │
    │                │             │                                │
    │                │    ┌────────┼─────────┐                      │
    │                │    │        ▼         │                      │
    │                │    │┌─────────────────┐│                     │
    │                │    ││   拼接+DoubleConv││                     │
    │                │    ││    256通道      ││                     │
    │                │    │└─────────────────┘│                     │
    │                │    │        │          │                     │
    │                │    │        ▼          │                     │
    │                │    │┌─────────────────┐│                     │
    │                │    ││    ConvTrans    ││                     │
    │                │    ││    上采样2×     ││                     │
    │                │    │└─────────────────┘│                     │
    │                │    │        │          │                     │
    │                │    │┌───────┼─────────┘│                     │
    │                │    ││       ▼          │                     │
    │                │    ││┌─────────────────┐│                    │
    │                │    │││   拼接+DoubleConv││                    │
    │                │    │││    128通道      ││                    │
    │                │    ││└─────────────────┘│                    │
    │                │    ││       │           │                    │
    │                │    ││       ▼           │                    │
    │                │    ││┌─────────────────┐│                    │
    │                └─────┼││    ConvTrans    ││                    │
    │                      │││    上采样2×     ││                    │
    │                      ││└─────────────────┘│                    │
    │                      ││       │           │                    │
    │                      ││┌──────┼───────────┘                    │
    │                      │││      ▼                                │
    │                      │││┌─────────────────┐                   │
    │                      ││││   拼接+DoubleConv│                   │
    │                      ││││    64通道       │                   │
    │                      │││└─────────────────┘                   │
    │                      │││      │                                │
    │                      │││      ▼                                │
    │                      │││┌─────────────────┐                   │
    │                      ││││     1×1卷积      │                   │
    │                      ││││    19通道输出    │                   │
    │                      │││└─────────────────┘                   │
    │                      │││      │                                │
    │                      │││      ▼                                │
    └──────────────────────┼┼┼─────────                           │
                           │││        │                            │
                           │││        ▼                            │
                           │││ 语义分割输出 (19×512×1024)           │
                           │││                                      │
                           └└└──────────────────────────────────────┘

    图例:
    ───► 前向传播
    ◄─── 跳跃连接（特征拼接）
    □   卷积层
    ○   池化层/上采样层
    ║   特征图传输
    """
    
    with open('model_architecture_ascii.txt', 'w') as f:
        f.write(diagram)
    
    print(diagram)
    print("\nASCII模型图已保存到: model_architecture_ascii.txt")

def create_model_plot_matplotlib():
    """使用matplotlib创建模型结构示意图"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # 设置背景
        ax.set_facecolor('#f5f5f5')
        
        # 定义颜色
        colors = {
            'input': '#FF6B6B',
            'conv': '#4ECDC4',
            'pool': '#45B7D1',
            'up': '#96CEB4',
            'concat': '#FFEAA7',
            'output': '#DDA0DD'
        }
        
        # 定义位置
        x_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y_positions = [0, -2, -4, -6, -8, -6, -4, -2, 0]
        layer_names = [
            '输入\n3×512×1024',
            'Conv64\n+ReLU',
            'Conv128\n+ReLU',
            'Conv256\n+ReLU',
            'Conv512\n+ReLU',
            'UpConv512\n+Concat',
            'UpConv256\n+Concat',
            'UpConv128\n+Concat',
            'UpConv64\n+Concat'
        ]
        
        # 绘制U-Net的U形结构
        for i, (x, y, name) in enumerate(zip(x_positions, y_positions, layer_names)):
            # 选择颜色
            if i == 0:
                color = colors['input']
            elif i == len(x_positions) - 1:
                color = colors['output']
            elif i < 5:  # 编码器
                color = colors['conv']
            else:  # 解码器
                color = colors['up']
            
            # 绘制矩形
            rect = patches.Rectangle(
                (x-0.4, y-0.4), 0.8, 0.8,
                linewidth=2, edgecolor='black',
                facecolor=color, alpha=0.8
            )
            ax.add_patch(rect)
            
            # 添加文本
            ax.text(x, y, name, ha='center', va='center', fontsize=9, fontweight='bold')
            
            # 连接线
            if i < len(x_positions) - 1:
                # 向下连接
                if i < 4:
                    ax.plot([x, x_positions[i+1]], [y-0.4, y_positions[i+1]+0.4], 
                           'k-', linewidth=1.5, alpha=0.7)
                
                # 跳跃连接（从编码器到解码器）
                if i < 4:
                    target_idx = 8 - i  # 对称位置
                    ax.plot([x_positions[i]+0.4, x_positions[target_idx]-0.4],
                           [y_positions[i], y_positions[target_idx]],
                           'k--', linewidth=1, alpha=0.5)
        
        # 添加瓶颈层
        bottleneck_x, bottleneck_y = 5, -10
        rect_bottleneck = patches.Rectangle(
            (bottleneck_x-0.4, bottleneck_y-0.4), 0.8, 0.8,
            linewidth=2, edgecolor='black',
            facecolor=colors['conv'], alpha=0.8
        )
        ax.add_patch(rect_bottleneck)
        ax.text(bottleneck_x, bottleneck_y, 'Conv1024\n瓶颈层', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 连接瓶颈层
        ax.plot([x_positions[4], bottleneck_x], [y_positions[4]-0.4, bottleneck_y+0.4], 
               'k-', linewidth=1.5, alpha=0.7)
        ax.plot([bottleneck_x, x_positions[5]], [bottleneck_y-0.4, y_positions[5]+0.4], 
               'k-', linewidth=1.5, alpha=0.7)
        
        # 添加输出层
        output_x, output_y = 10, 0
        rect_output = patches.Rectangle(
            (output_x-0.4, output_y-0.4), 0.8, 0.8,
            linewidth=2, edgecolor='black',
            facecolor=colors['output'], alpha=0.8
        )
        ax.add_patch(rect_output)
        ax.text(output_x, output_y, '输出\n19×512×1024', 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 连接最后层到输出
        ax.plot([x_positions[-1]+0.4, output_x-0.4], [y_positions[-1], output_y], 
               'k-', linewidth=1.5, alpha=0.7)
        
        # 设置图形属性
        ax.set_xlim(0, 11)
        ax.set_ylim(-11, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('U-Net 语义分割模型架构图', fontsize=16, fontweight='bold', pad=20)
        
        # 添加图例
        legend_elements = [
            patches.Patch(facecolor=colors['input'], edgecolor='black', label='输入层'),
            patches.Patch(facecolor=colors['conv'], edgecolor='black', label='卷积层'),
            patches.Patch(facecolor=colors['up'], edgecolor='black', label='上采样层'),
            patches.Patch(facecolor=colors['output'], edgecolor='black', label='输出层'),
        ]
        ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                 ncol=4, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('model_architecture_simple.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("模型结构图已保存为: model_architecture_simple.png")
        
    except Exception as e:
        print(f"创建matplotlib图时出错: {e}")

def main():
    """主函数"""
    print("正在生成U-Net模型结构图...")
    
    # 1. 打印模型摘要
    model = print_model_summary()
    
    # 2. 导出详细架构
    export_model_architecture(model)
    
    # 3. 创建ASCII图
    create_ascii_model_diagram()
    
    # 4. 尝试使用torchviz（可选）
    visualize_with_torchviz(model)
    
    # 5. 创建简单的matplotlib图
    create_model_plot_matplotlib()
    
    print("\n" + "=" * 60)
    print("模型结构图生成完成！")
    print("生成的文件:")
    print("  1. model_architecture.txt - 详细文本架构")
    print("  2. model_architecture_ascii.txt - ASCII图示")
    print("  3. model_architecture_simple.png - 简单示意图")
    print("  4. model_graph.png - 计算图（如果安装了torchviz）")
    print("=" * 60)

if __name__ == "__main__":
    main()