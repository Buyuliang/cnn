# generate_data.py
"""
生成随机几何图像及对应二值标签，并保存到 output/ 文件夹
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset import GeometricShapesDataset

# ------------------------------
# 配置
# ------------------------------
output_dir = "output/data_samples"
os.makedirs(output_dir, exist_ok=True)
num_samples = 10       # 生成 10 个示例
image_size = 64        # 图像尺寸 64x64

# ------------------------------
# 数据集
# ------------------------------
dataset = GeometricShapesDataset(num_samples=num_samples, image_size=image_size)

# ------------------------------
# 保存图像和标签
# ------------------------------
for i in range(num_samples):
    img, mask = dataset[i]
    img = img.squeeze().numpy()
    mask = mask.squeeze().numpy()
    
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.title("Input Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.title("Label Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_{i+1}.png"))
    plt.close()

print(f"已生成 {num_samples} 个数据样本，保存在 {output_dir} 文件夹")
