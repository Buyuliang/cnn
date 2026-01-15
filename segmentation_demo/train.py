# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import GeometricShapesDataset
from model import CNN2DSegmentation
import os

# --------------------------
# 配置
# --------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

train_dataset = GeometricShapesDataset(num_samples=300)
val_dataset = GeometricShapesDataset(num_samples=50)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

model = CNN2DSegmentation().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 15

# --------------------------
# 日志文件
# --------------------------
log_file = os.path.join(output_dir, "training_log.txt")
with open(log_file, "w") as f:
    f.write("Epoch | Train Loss | Val Loss | Pixel Accuracy\n")

# --------------------------
# 训练循环
# --------------------------
for epoch in range(1, num_epochs+1):
    model.train()
    train_loss = 0.0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)
            
            preds = (outputs > 0.5).float()
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
    val_loss /= len(val_loader.dataset)
    pixel_accuracy = correct_pixels / total_pixels

    log_line = f"{epoch}/{num_epochs} | {train_loss:.4f} | {val_loss:.4f} | {pixel_accuracy:.2f}"
    print(log_line)
    with open(log_file, "a") as f:
        f.write(log_line + "\n")

# --------------------------
# 导出模型
# --------------------------
model_path = os.path.join(output_dir, "cnn2d_model.pth")
torch.save(model.state_dict(), model_path)
print(f"训练完成，模型已保存至 {model_path}")
