# plot_training_flow.py
"""
CNN2DSegmentation 训练流程 + 矩阵计算示意图
- 展示每层卷积计算 (a × b = c)
- 展示激活函数
- 展示损失计算和反向传播
- 输出 PNG: output/training_flow.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

# ----------------------------
# 层信息
# ----------------------------
layers = [
    {"name": "Input", "shape": "(4,1,64,64)", "color": "#FFD700"},
    {"name": "Conv1", "shape": "(4,16,64,64)", "weight": "(16,1,3,3)", "color": "#87CEFA"},
    {"name": "ReLU",  "shape": "(4,16,64,64)", "color": "#98FB98"},
    {"name": "Conv2", "shape": "(4,32,64,64)", "weight": "(32,16,3,3)", "color": "#87CEFA"},
    {"name": "ReLU",  "shape": "(4,32,64,64)", "color": "#98FB98"},
    {"name": "Conv3", "shape": "(4,1,64,64)",  "weight": "(1,32,1,1)", "color": "#87CEFA"},
    {"name": "Sigmoid", "shape": "(4,1,64,64)", "color": "#FFB6C1"},
    {"name": "Loss", "shape": "scalar", "color": "#FFA07A"},
    {"name": "Backprop", "shape": "gradients", "color": "#D3D3D3"},
]

# ----------------------------
# 绘制
# ----------------------------
fig, ax = plt.subplots(figsize=(6,9))
ax.axis('off')

x = 0.5
y_start = 0.95
y_step = 0.1

for i, layer in enumerate(layers):
    y = y_start - i*y_step
    rect = plt.Rectangle((x-0.25, y-0.04), 0.5, 0.08, facecolor=layer["color"], edgecolor='black')
    ax.add_patch(rect)
    
    if "weight" in layer:
        text = f"{layer['name']}\n{layer['shape']}\n× {layer['weight']}"
    else:
        text = f"{layer['name']}\n{layer['shape']}"
    ax.text(x, y, text, ha='center', va='center', fontsize=10)

# 画箭头（前向传播）
for i in range(len(layers)-2):  # Loss 和 Backprop 分开
    y0 = y_start - i*y_step - 0.04
    y1 = y_start - (i+1)*y_step + 0.04
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8))

# 画箭头（反向传播）
y_loss = y_start - (len(layers)-2)*y_step + 0.02
y_back = y_start - (len(layers)-1)*y_step - 0.02
ax.annotate('', xy=(x, y_start-0.04), xytext=(x, y_back),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8, linestyle='--'))

plt.title("CNN2DSegmentation 训练流程 + 矩阵计算示意图", fontsize=12)
plt.savefig("output/training_flow.png", dpi=200, bbox_inches='tight')
plt.show()
print("训练流程图已保存到 output/training_flow.png")
