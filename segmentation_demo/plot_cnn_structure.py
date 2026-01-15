# plot_cnn_structure.py
"""
画 CNN2DSegmentation 的示意图（Netron 风格）
- 标出 Conv1/Conv2/Conv3
- 标出输入输出尺寸
- 标出激活函数
- 保存图像到 output/cnn_structure.png
"""

import matplotlib.pyplot as plt

# ----------------------------
# 配置图形
# ----------------------------
fig, ax = plt.subplots(figsize=(6, 5))
ax.axis('off')

# 定义每一层信息
layers = [
    {"name": "Input", "size": "(1,64,64)", "color": "#FFD700"},
    {"name": "Conv1", "size": "(16,64,64)", "color": "#87CEFA"},
    {"name": "ReLU",  "size": "(16,64,64)", "color": "#98FB98"},
    {"name": "Conv2", "size": "(32,64,64)", "color": "#87CEFA"},
    {"name": "ReLU",  "size": "(32,64,64)", "color": "#98FB98"},
    {"name": "Conv3", "size": "(1,64,64)",  "color": "#87CEFA"},
    {"name": "Sigmoid", "size": "(1,64,64)", "color": "#FFB6C1"},
]

# ----------------------------
# 绘制每一层
# ----------------------------
x = 0.5
y_start = 0.9
y_step = 0.12

for i, layer in enumerate(layers):
    y = y_start - i * y_step
    rect = plt.Rectangle((x-0.2, y-0.04), 0.4, 0.08, facecolor=layer["color"], edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y, f"{layer['name']}\n{layer['size']}", ha='center', va='center', fontsize=10)

# ----------------------------
# 画箭头
# ----------------------------
for i in range(len(layers)-1):
    y0 = y_start - i*y_step - 0.04
    y1 = y_start - (i+1)*y_step + 0.04
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

# ----------------------------
# 保存图像
# ----------------------------
plt.title("CNN2DSegmentation Structure (Netron Style)", fontsize=12)
plt.savefig("output/cnn_structure.png", dpi=200, bbox_inches='tight')
plt.show()
print("CNN2DSegmentation 结构图已保存到 output/cnn_structure.png")
