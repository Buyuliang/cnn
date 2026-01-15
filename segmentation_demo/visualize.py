# visualize.py - 批量推理指定文件夹图片
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from model import CNN2DSegmentation

# --------------------------
# 参数设置
# --------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = "output"
data_samples_dir = os.path.join(output_dir, "data_samples")
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, "cnn2d_model.pth")

# --------------------------
# 加载模型
# --------------------------
model = CNN2DSegmentation().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print(f"已加载模型：{model_path}")

# --------------------------
# 批量推理 data_samples 文件夹
# --------------------------
if os.path.exists(data_samples_dir) and len(os.listdir(data_samples_dir)) > 0:
    files = sorted([f for f in os.listdir(data_samples_dir) if f.endswith(".png")])
    for i, f in enumerate(files):
        img_path = os.path.join(data_samples_dir, f)

        # 读取图片并灰度化
        img = Image.open(img_path).convert("L")  # 转灰度
        img = np.array(img, dtype=np.float32) / 255.0  # 归一化 0~1

        # 转 tensor，shape -> (1,1,H,W)
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

        # 推理
        with torch.no_grad():
            pred = model(img_tensor).cpu().squeeze().numpy()

        # 可视化对比图
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.title("Input")
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        plt.subplot(1,2,2)
        plt.title("Prediction")
        plt.imshow(pred > 0.5, cmap='gray')  # 二值化
        plt.axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"inference_{i+1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"{img_path} 推理完成，结果保存到 {save_path}")
else:
    print(f"目录不存在或没有 PNG 图片: {data_samples_dir}")
