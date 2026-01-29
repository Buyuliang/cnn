# generate_data.py
"""
生成数字识别训练数据集
生成包含单个数字的图像和对应的文本标签文件
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import DigitalDataset
from PIL import Image


def generate_dataset(output_dir="data", num_samples=500, image_size=256, split_ratio=0.8):
    """
    生成数字识别数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 生成样本数量
        image_size: 图像尺寸
        split_ratio: 训练集比例
    """
    # 创建目录结构
    train_images_dir = os.path.join(output_dir, "train", "images")
    val_images_dir = os.path.join(output_dir, "val", "images")
    
    for dir_path in [train_images_dir, val_images_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 创建数据集
    dataset = DigitalDataset(image_size=image_size, num_samples=num_samples)
    
    # 划分训练集和验证集
    train_size = int(num_samples * split_ratio)
    
    print(f"开始生成数据集...")
    print(f"总样本数: {num_samples}")
    print(f"训练集: {train_size}, 验证集: {num_samples - train_size}")
    
    # 生成训练集
    for i in range(train_size):
        img, label = dataset[i]
        img_np = img.squeeze().numpy()
        
        # 保存图像
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        image_filename = f"train_{i:04d}.png"
        image_path = os.path.join(train_images_dir, image_filename)
        img_pil.save(image_path)
        
        # 保存标签文件（与图像同名，扩展名为.txt）
        label_filename = f"train_{i:04d}.txt"
        label_path = os.path.join(train_images_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(str(label.item()))
        
        if (i + 1) % 50 == 0:
            print(f"已生成训练样本: {i + 1}/{train_size}")
    
    # 生成验证集
    for i in range(train_size, num_samples):
        img, label = dataset[i]
        img_np = img.squeeze().numpy()
        
        # 保存图像
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        val_idx = i - train_size
        image_filename = f"val_{val_idx:04d}.png"
        image_path = os.path.join(val_images_dir, image_filename)
        img_pil.save(image_path)
        
        # 保存标签文件
        label_filename = f"val_{val_idx:04d}.txt"
        label_path = os.path.join(val_images_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(str(label.item()))
        
        if (val_idx + 1) % 50 == 0:
            print(f"已生成验证样本: {val_idx + 1}/{num_samples - train_size}")
    
    print(f"\n数据集生成完成！")
    print(f"训练集保存在: {os.path.join(output_dir, 'train', 'images')}")
    print(f"验证集保存在: {os.path.join(output_dir, 'val', 'images')}")
    print(f"每个图像都有对应的 .txt 标签文件（包含数字 0-9）")
    
    # 生成一些可视化样本
    visualize_samples(dataset, output_dir, num_samples=10)


def visualize_samples(dataset, output_dir, num_samples=10):
    """可视化一些样本"""
    vis_dir = os.path.join(output_dir, "samples")
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"\n生成可视化样本...")
    for i in range(min(num_samples, len(dataset))):
        img, label = dataset[i]
        img_np = img.squeeze().numpy()
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_np, cmap='gray')
        plt.title(f"数字: {label.item()}", fontsize=16)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"sample_{i+1}_digit_{label.item()}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"可视化样本保存在: {vis_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='生成数字识别训练数据集')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=500, help='生成样本数量')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集比例')
    
    args = parser.parse_args()
    
    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_size=args.image_size,
        split_ratio=args.split_ratio
    )
