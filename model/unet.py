import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),     # 2D卷积层，3*3卷积核，边缘填充1像素，保持空间尺寸不变
            nn.BatchNorm2d(out_channels),       # 批量归一化
            nn.ReLU(inplace=True),  
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),    # 双卷积：两个连续的卷积层可以在不增加感受野的情况下增加模型深度
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=19):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),    # 下采样：最大池化
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )
        
        # 解码器
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)   # 上采样: 通过转置卷积恢复空间尺寸
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        # Dropout防止过拟合
        self.dropout = nn.Dropout2d(0.3)    # 丢弃概率为0.3，30%的特征通道在训练时会被随机丢弃
    
    def forward(self, x):
        # 编码器
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)
        
        # 解码器
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)   # 使用跳跃连接（skip connections）将编码器特征与解码器特征拼接
        x = self.conv1(x)
        x = self.dropout(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        x = self.dropout(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        
        # 上采样到原始大小（如果输入被下采样了）
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], 
                                 mode='bilinear', align_corners=True)
        
        return logits

def get_model(model_name='unet', num_classes=19):
    """获取模型"""
    if model_name == 'unet':
        return UNet(n_channels=3, n_classes=num_classes)
    else:
        raise ValueError(f"未知模型: {model_name}")