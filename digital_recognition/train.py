# train.py
"""
U-Net 分类模型训练脚本
用于训练数字识别模型（分类任务：识别 0-9）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import numpy as np
from model import UNetClassifier
from dataset import DigitalDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="训练中")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 更新进度条
        accuracy = 100 * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = 100 * correct / total
    return avg_loss, avg_accuracy


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="验证中")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            total_loss += loss.item()
            num_batches += 1
            
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = 100 * correct / total
    return avg_loss, avg_accuracy


def export_onnx(model, onnx_path, image_size=256, device=None):
    """将模型导出为 ONNX 格式"""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    dummy_input = torch.randn(1, 1, image_size, image_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        opset_version=12,
    )
    print(f"ONNX 模型已保存: {onnx_path}")


def calculate_class_accuracy(model, dataloader, device, num_classes=10):
    """计算每个类别的准确率"""
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    class_accuracies = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[i] = acc
        else:
            class_accuracies[i] = 0.0
    
    return class_accuracies


def main():
    parser = argparse.ArgumentParser(description='训练 U-Net 数字识别分类模型')
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数（0-9）')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据集
    train_data_dir = os.path.join(args.data_dir, 'train')
    val_data_dir = os.path.join(args.data_dir, 'val')
    
    print("加载训练集...")
    train_dataset = DigitalDataset(data_dir=train_data_dir, image_size=args.image_size)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("加载验证集...")
    val_dataset = DigitalDataset(data_dir=val_data_dir, image_size=args.image_size)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = UNetClassifier(in_channels=1, num_classes=args.num_classes).to(device)
    
    # 损失函数（交叉熵损失用于分类）
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume and os.path.exists(args.resume):
        print(f"从 {args.resume} 恢复训练...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"从 epoch {start_epoch} 继续训练")
    
    # 训练循环
    print("\n开始训练...")
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"\n训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
            }, best_model_path)
            print(f"保存最佳模型到 {best_model_path} (验证准确率: {val_acc:.2f}%)")
            best_onnx_path = os.path.join(args.save_dir, 'best_model.onnx')
            export_onnx(model, best_onnx_path, args.image_size, device)

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
            }, checkpoint_path)
            print(f"保存检查点到 {checkpoint_path}")
        
        # 每5个epoch打印每个类别的准确率
        if (epoch + 1) % 5 == 0:
            print("\n各类别准确率:")
            class_accs = calculate_class_accuracy(model, val_loader, device, args.num_classes)
            for digit, acc in class_accs.items():
                print(f"  数字 {digit}: {acc:.2f}%")
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'train_acc': train_accuracies[-1],
        'val_loss': val_losses[-1],
        'val_acc': val_accuracies[-1],
        'best_val_acc': best_val_acc,
    }, final_model_path)
    print(f"\n训练完成！最终模型保存到 {final_model_path}")
    final_onnx_path = os.path.join(args.save_dir, 'final_model.onnx')
    export_onnx(model, final_onnx_path, args.image_size, device)

    # 打印训练历史
    print("\n训练历史:")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最终训练损失: {train_losses[-1]:.4f}, 训练准确率: {train_accuracies[-1]:.2f}%")
    print(f"最终验证损失: {val_losses[-1]:.4f}, 验证准确率: {val_accuracies[-1]:.2f}%")


if __name__ == "__main__":
    main()
