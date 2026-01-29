# 数字识别项目

基于 U-Net 的数字识别系统，用于识别图片中的单个数字（0-9）。

## 项目结构

```
digital_recognition/
├── model.py          # U-Net 分类模型
├── dataset.py        # 数据集类
├── generate_data.py  # 数据生成脚本
├── train.py         # 模型训练脚本
├── predict.py       # 模型预测脚本
└── README.md        # 项目说明文档
```

## 数据格式

现在数据集的格式是：

```
data/
├── train/
│   └── images/
│       ├── train_0000.png    # 图片
│       ├── train_0000.txt    # 标签文件（包含数字，如 "5"）
│       ├── train_0001.png
│       ├── train_0001.txt
│       └── ...
└── val/
    └── images/
        ├── val_0000.png
        ├── val_0000.txt
        └── ...
```

**注意**：每个图片文件都有对应的 `.txt` 标签文件，标签文件中只包含一个数字（0-9），表示图片中的数字。

## 使用方法

### 1. 生成数据集

生成训练和验证数据集：

```bash
python generate_data.py --output_dir data --num_samples 500 --image_size 256
```

参数说明：
- `--output_dir`: 输出目录（默认：`data`）
- `--num_samples`: 生成样本数量（默认：`500`）
- `--image_size`: 图像尺寸（默认：`256`）
- `--split_ratio`: 训练集比例（默认：`0.8`）

### 2. 训练模型

训练数字识别模型：

```bash
python train.py --data_dir data --batch_size 16 --epochs 50
```

参数说明：
- `--data_dir`: 数据目录路径（默认：`data`）
- `--batch_size`: 批次大小（默认：`16`）
- `--epochs`: 训练轮数（默认：`50`）
- `--lr`: 学习率（默认：`0.001`）
- `--image_size`: 图像尺寸（默认：`256`）
- `--save_dir`: 模型保存目录（默认：`checkpoints`）
- `--num_classes`: 分类类别数（默认：`10`，即 0-9）

### 3. 预测单张图片

对单张图片进行数字识别：

```bash
python predict.py --model_path checkpoints/best_model.pth --image_path path/to/image.png
```

参数说明：
- `--model_path`: 模型文件路径（必需）
- `--image_path`: 单张图片路径
- `--output_dir`: 输出目录（默认：`predictions`）
- `--image_size`: 图像尺寸（默认：`256`）
- `--num_classes`: 分类类别数（默认：`10`）

### 4. 批量预测

对目录中的所有图片进行批量预测：

```bash
python predict.py --model_path checkpoints/best_model.pth --image_dir data/val/images/
```

参数说明：
- `--image_dir`: 图片目录路径（批量预测）
- 其他参数与单张图片预测相同

## 模型说明

本项目使用基于 U-Net 编码器的分类模型（`UNetClassifier`）：
- 输入：单通道灰度图像（256x256）
- 输出：10 个类别的分类结果（0-9）
- 损失函数：交叉熵损失（CrossEntropyLoss）
- 优化器：Adam

## 输出说明

### 训练输出

训练过程中会显示：
- 训练损失和准确率
- 验证损失和准确率
- 每个类别的识别准确率（每 5 个 epoch）

模型会保存在 `checkpoints/` 目录：
- `best_model.pth`: 验证准确率最高的模型
- `final_model.pth`: 最终训练完成的模型
- `checkpoint_epoch_*.pth`: 定期保存的检查点

### 预测输出

预测结果会保存在 `predictions/` 目录：
- `*_prediction.png`: 包含原始图像和概率分布的可视化结果

预测会显示：
- 预测的数字类别
- 置信度（概率）
- 所有 10 个类别的概率分布

## 依赖要求

主要依赖包：
- `torch` - PyTorch 深度学习框架
- `PIL` (Pillow) - 图像处理
- `numpy` - 数值计算
- `matplotlib` - 可视化
- `tqdm` - 进度条显示

可选依赖：
- `opencv-python` - 图像处理（用于某些高级功能）
- `scipy` - 科学计算（用于某些替代方法）

## 注意事项

1. 每个图片只包含一个数字（0-9）
2. 标签文件必须与图片文件同名，扩展名为 `.txt`
3. 标签文件中只包含一个数字，没有其他字符
4. 建议使用 GPU 进行训练以加快速度