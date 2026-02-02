# ONNX 图与 model.py 的对应关系

## 1. 导出流程

- **导出入口**：`train.py` 中的 `export_onnx(model, onnx_path, image_size, device)`
- **被导出模型**：`model.py` 里的 **`UNetClassifier`**（不是分割用的 `UNet`）
- **导出方式**：`torch.onnx.export(model, dummy_input, ...)` 会执行一次 `model(dummy_input)`，按 **forward 的实际执行顺序** 记录计算图，再转成 ONNX 节点。

因此：**ONNX 图 = UNetClassifier.forward() 的计算图的一种等价表示**。

---

## 2. 输入/输出对应

| ONNX 图        | model.py 对应 |
|----------------|----------------|
| 输入名 `input` | `forward(self, x)` 的 `x`，形状 `(batch, 1, H, W)`，如 (1, 1, 256, 256) |
| 输出名 `output`| `forward` 的返回值，即 `self.classifier(features)`，形状 `(batch, 10)`，10 类 logits |

导出时用了 `input_names=['input']`、`output_names=['output']`，所以 ONNX 里会看到这两个名字。

---

## 3. 计算图与 model.py 结构的对应（按 forward 顺序）

`UNetClassifier.forward()` 里每一步，都会在 ONNX 里变成若干算子（节点），大致对应如下：

| model.py 代码 / 模块 | ONNX 中的体现 |
|----------------------|----------------|
| `enc1 = self.enc1(x)` | **DoubleConv**：Conv → BatchNormalization → Relu → Conv → BatchNormalization → Relu（可能被融合或拆成多个节点） |
| `x1 = self.pool(enc1)` | **MaxPool** (kernel=2, stride=2) |
| `enc2 = self.enc2(x1)` | 同上，128 通道的 DoubleConv |
| `x2 = self.pool(enc2)` | MaxPool |
| `enc3 = self.enc3(x2)` | 256 通道 DoubleConv |
| `x3 = self.pool(enc3)` | MaxPool |
| `enc4 = self.enc4(x3)` | 512 通道 DoubleConv |
| `x4 = self.pool(enc4)` | MaxPool |
| `bottleneck = self.bottleneck(x4)` | 1024 通道 DoubleConv |
| `features = self.global_pool(bottleneck)` | **GlobalAveragePool** (AdaptiveAvgPool2d(1)) |
| `features.view(..., -1)` | **Reshape**，把 (B,1024,1,1) 压成 (B, 1024) |
| `self.classifier(features)` | Dropout(0.5) → **MatMul + Add** (Linear 1024→512) → Relu → Dropout(0.3) → **MatMul + Add** (Linear 512→10) |

推理时 Dropout 通常被关掉或变成恒等，所以 ONNX 里可能看不到或变成 Identity。

---

## 4. PyTorch 算子 → ONNX 算子（常见）

| model.py 中的层 | ONNX 算子名（典型） |
|-----------------|----------------------|
| nn.Conv2d       | Conv |
| nn.BatchNorm2d  | BatchNormalization |
| nn.ReLU         | Relu |
| nn.MaxPool2d    | MaxPool |
| nn.AdaptiveAvgPool2d(1) | GlobalAveragePool |
| view / flatten  | Reshape |
| nn.Linear       | Gemm 或 MatMul + Add |
| nn.Dropout      | Dropout（推理时常被优化掉） |

ONNX 图里的“节点”就是这些算子；边的方向就是张量流动顺序，和 forward 里的变量依赖一致。

---

## 5. 如何自己对照 ONNX 图与 model.py

1. **看输入**：找到名为 `input` 的输入，对应 `forward(x)` 的 `x`。
2. **看输出**：找到名为 `output` 的节点，对应最后的 logits。
3. **顺藤摸瓜**：从 `input` 出发沿边往后走，顺序会大致是：  
   enc1(Conv+BN+Relu×2) → pool → enc2 → pool → enc3 → pool → enc4 → pool → bottleneck → GlobalAveragePool → Reshape → Linear → … → **output**。
4. **用 Netron**：打开 `.onnx` 文件，可点击每个节点看 name、input/output、属性（kernel_size、strides 等），与上面表格里的层一一对照。

---

## 6. 小结

- **ONNX 图** = 对 **`model.py` 中 `UNetClassifier.forward()`** 的静态、与框架无关的描述。
- **对应方式**：按 forward 的执行顺序，每一层/每一步会变成 ONNX 里的一串节点；输入/输出名由 `export_onnx` 里的 `input_names`/`output_names` 决定。
- 要验证是否一致：用同一份权重，在 PyTorch 里跑一次 `model(x)`，在 ONNX Runtime 里用同一 `x` 跑一次，对比输出是否接近（一般完全一致或误差极小）。
