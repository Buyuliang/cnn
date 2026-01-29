# model.py
"""
U-Net 网络模型，用于数字图像分类识别
基于 U-Net 编码器 + 分类头
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """U-Net 中的双卷积块"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetClassifier(nn.Module):
    """
    U-Net 分类模型
    使用 U-Net 的编码器部分提取特征，然后进行分类
    用于识别图片中的单个数字（0-9）
    """
    def __init__(self, in_channels=1, num_classes=10):
        super(UNetClassifier, self).__init__()
        
        # 编码器（下采样路径）
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        x1 = self.pool(enc1)
        
        enc2 = self.enc2(x1)
        x2 = self.pool(enc2)
        
        enc3 = self.enc3(x2)
        x3 = self.pool(enc3)
        
        enc4 = self.enc4(x3)
        x4 = self.pool(enc4)
        
        # 瓶颈层
        bottleneck = self.bottleneck(x4)
        
        # 全局平均池化
        features = self.global_pool(bottleneck)
        features = features.view(features.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        
        return output


# 为了兼容性，保留原来的 UNet 类（用于分割任务）
class UNet(nn.Module):
    """
    U-Net 网络模型（分割版本）
    用于图像分割，识别图片中的数字区域
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        # 编码器（下采样路径）
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器（上采样路径）
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # 输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.activation = nn.Sigmoid()
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        x1 = self.pool(enc1)
        
        enc2 = self.enc2(x1)
        x2 = self.pool(enc2)
        
        enc3 = self.enc3(x2)
        x3 = self.pool(enc3)
        
        enc4 = self.enc4(x3)
        x4 = self.pool(enc4)
        
        # 瓶颈层
        bottleneck = self.bottleneck(x4)
        
        # 解码器路径（带跳跃连接）
        up4 = self.up4(bottleneck)
        # 处理尺寸不匹配的情况
        if up4.size() != enc4.size():
            up4 = F.interpolate(up4, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = self.dec4(torch.cat([enc4, up4], dim=1))
        
        up3 = self.up3(dec4)
        if up3.size() != enc3.size():
            up3 = F.interpolate(up3, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([enc3, up3], dim=1))
        
        up2 = self.up2(dec3)
        if up2.size() != enc2.size():
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([enc2, up2], dim=1))
        
        up1 = self.up1(dec2)
        if up1.size() != enc1.size():
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([enc1, up1], dim=1))
        
        # 输出
        output = self.final_conv(dec1)
        output = self.activation(output)
        
        return output
