# dataset.py
"""
数据集类，用于加载数字图像和对应的数字标签
每个图像包含单个数字（0-9）
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


class DigitalDataset(Dataset):
    """
    数字识别数据集
    每个图像包含单个数字，标签为对应的数字（0-9）
    """
    def __init__(self, data_dir=None, image_size=256, num_samples=None):
        """
        Args:
            data_dir: 数据目录路径，包含 images/ 子文件夹和对应的 .txt 标签文件
            image_size: 图像尺寸
            num_samples: 如果提供，将生成随机数字图像（用于训练）
        """
        self.image_size = image_size
        
        if data_dir and os.path.exists(data_dir):
            # 从文件夹加载数据
            self.images_dir = os.path.join(data_dir, 'images')
            self.image_files = sorted([f for f in os.listdir(self.images_dir) 
                                      if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.mode = 'load'
        else:
            # 生成随机数据模式
            self.num_samples = num_samples if num_samples else 200
            self.mode = 'generate'
    
    def __len__(self):
        if self.mode == 'load':
            return len(self.image_files)
        else:
            return self.num_samples
    
    def _load_label(self, image_path):
        """从对应的文本文件加载标签"""
        # 标签文件与图像文件在同一目录，文件名相同但扩展名为.txt
        base_name = os.path.splitext(image_path)[0]
        label_path = base_name + '.txt'
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                content = f.read().strip()
                label = int(content)
        else:
            # 如果标签文件不存在，尝试从文件名提取
            filename = os.path.basename(image_path)
            # 尝试从文件名中提取数字（格式：train_0001_5.png 或 val_0001_3.png）
            try:
                # 先尝试从文件名末尾提取数字
                name_without_ext = os.path.splitext(filename)[0]
                parts = name_without_ext.split('_')
                # 如果最后一部分是数字，使用它
                if parts[-1].isdigit():
                    label = int(parts[-1])
                else:
                    # 否则尝试从文件名中查找数字模式
                    import re
                    numbers = re.findall(r'\d+', name_without_ext)
                    if numbers:
                        # 使用最后一个数字（通常是标签）
                        label = int(numbers[-1]) % 10  # 确保在 0-9 范围内
                    else:
                        label = 0  # 默认标签
            except:
                label = 0  # 默认标签
        
        return label
    
    def _generate_single_digit_image(self):
        """生成包含单个数字的图像和标签"""
        # 创建背景图像（随机噪声或纯色）
        img = np.random.rand(self.image_size, self.image_size) * 0.3
        
        # 随机选择数字 0-9
        digit = np.random.randint(0, 10)
        
        # 随机位置和大小
        size = np.random.randint(60, 120)
        x = np.random.randint(20, self.image_size - size - 20)
        y = np.random.randint(20, self.image_size - size - 20)
        
        # 创建数字图像
        digit_img = self._create_digit_shape(digit, size)
        
        # 将数字放置到图像上
        h, w = digit_img.shape
        x_end = min(x + h, self.image_size)
        y_end = min(y + w, self.image_size)
        h_actual = x_end - x
        w_actual = y_end - y
        
        img[x:x_end, y:y_end] = np.maximum(
            img[x:x_end, y:y_end],
            digit_img[:h_actual, :w_actual]
        )
        
        # 添加一些噪声
        img = img + np.random.randn(self.image_size, self.image_size) * 0.1
        img = np.clip(img, 0, 1)
        
        return img.astype(np.float32), digit
    
    def _create_digit_shape(self, digit, size):
        """创建真实的数字图像"""
        # 创建一个 PIL 图像
        img = Image.new('L', (size, size), color=0)
        draw = ImageDraw.Draw(img)
        
        # 尝试加载字体，如果失败则使用默认字体
        try:
            # 尝试使用系统字体
            font_size = int(size * 0.7)
            try:
                # Linux 系统字体路径
                font_paths = [
                    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
                    '/System/Library/Fonts/Helvetica.ttc',  # macOS
                    'C:/Windows/Fonts/arial.ttf',  # Windows
                ]
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                
                if font is None:
                    # 使用默认字体
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 计算文本位置（居中）
        text = str(digit)
        # 获取文本边界框
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 居中绘制数字
        x = (size - text_width) // 2 - bbox[0]
        y = (size - text_height) // 2 - bbox[1]
        
        # 绘制白色数字
        draw.text((x, y), text, fill=255, font=font)
        
        # 转换为 numpy 数组并归一化
        canvas = np.array(img, dtype=np.float32) / 255.0
        
        return canvas
    
    def __getitem__(self, idx):
        if self.mode == 'load':
            # 从文件加载
            img_path = os.path.join(self.images_dir, self.image_files[idx])
            
            img = Image.open(img_path).convert('L')
            
            # 调整尺寸
            img = img.resize((self.image_size, self.image_size))
            
            # 转换为 numpy 数组并归一化
            img = np.array(img, dtype=np.float32) / 255.0
            
            # 加载标签
            label = self._load_label(img_path)
        else:
            # 生成随机数据
            img, label = self._generate_single_digit_image()
        
        # 转换为 tensor 并添加通道维度
        img = torch.tensor(img).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(label, dtype=torch.long)  # 分类标签
        
        return img, label
