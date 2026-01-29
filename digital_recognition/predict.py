# predict.py
"""
U-Net 分类模型预测脚本
用于对单张图片或多张图片进行数字识别预测（分类任务：识别 0-9）
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
import os
from model import UNetClassifier
import matplotlib.pyplot as plt


def load_model(model_path, device, num_classes=10):
    """加载训练好的分类模型"""
    model = UNetClassifier(in_channels=1, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=256):
    """预处理输入图像"""
    # 读取图像
    img = Image.open(image_path).convert('L')
    original_size = img.size
    
    # 调整尺寸
    img = img.resize((image_size, image_size))
    
    # 转换为 numpy 数组并归一化
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # 转换为 tensor 并添加批次和通道维度
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return img_tensor, original_size, img_array


def predict_single_image(model, image_path, device, image_size=256, save_output=True, output_dir='predictions'):
    """对单张图片进行预测"""
    print(f"处理图片: {image_path}")
    
    # 预处理
    img_tensor, original_size, img_array = preprocess_image(image_path, image_size)
    img_tensor = img_tensor.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # 获取所有类别的概率
    all_probs = probabilities[0].cpu().numpy()
    
    print(f"预测结果: 数字 {predicted_class}")
    print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"所有类别概率:")
    for i, prob in enumerate(all_probs):
        print(f"  数字 {i}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 可视化结果
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建可视化图像
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 原始图像 (use English in plot to avoid CJK font warning)
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title(f'Original Image\nPredicted: digit {predicted_class} (conf: {confidence*100:.2f}%)',
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # 概率分布条形图
        axes[1].bar(range(10), all_probs * 100, color='steelblue', alpha=0.7)
        axes[1].bar(predicted_class, all_probs[predicted_class] * 100,
                   color='red', alpha=0.9, label=f'Predicted: {predicted_class}')
        axes[1].set_xlabel('Digit class', fontsize=12)
        axes[1].set_ylabel('Probability (%)', fontsize=12)
        axes[1].set_title('Classification probability', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(10))
        axes[1].set_ylim([0, 100])
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].legend()
        
        # 在条形图上标注概率值
        for i, prob in enumerate(all_probs):
            if prob > 0.01:  # 只显示大于1%的概率
                axes[1].text(i, prob * 100 + 2, f'{prob*100:.1f}%', 
                           ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f'{base_name}_prediction.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"结果已保存到: {output_path}")
    
    return {
        'predicted_digit': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probs,
        'image_path': image_path
    }


def predict_batch(model, image_dir, device, image_size=256, output_dir='predictions'):
    """批量预测图片"""
    # 支持的图像格式
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            result = predict_single_image(
                model, image_path, device, image_size, 
                save_output=True, output_dir=output_dir
            )
            result['image_file'] = image_file
            results.append(result)
            
            # 尝试从文件名或标签文件获取真实标签（用于计算准确率）
            true_label = None
            # 尝试从标签文件读取
            label_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    true_label = int(f.read().strip())
            else:
                # 尝试从文件名提取（格式：xxx_数字.png）
                try:
                    parts = os.path.splitext(image_file)[0].split('_')
                    true_label = int(parts[-1])
                except:
                    pass
            
            if true_label is not None:
                total_predictions += 1
                if result['predicted_digit'] == true_label:
                    correct_predictions += 1
                    result['correct'] = True
                else:
                    result['correct'] = False
                    result['true_label'] = true_label
            else:
                result['correct'] = None
                result['true_label'] = None
            
        except Exception as e:
            print(f"处理 {image_file} 时出错: {str(e)}")
    
    # 打印统计信息
    print("\n预测统计:")
    print(f"总图片数: {len(results)}")
    if total_predictions > 0:
        accuracy = 100 * correct_predictions / total_predictions
        print(f"准确率: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
    
    # 打印每张图片的预测结果
    print("\n详细预测结果:")
    for result in results:
        status = ""
        if result['correct'] is True:
            status = "✓ 正确"
        elif result['correct'] is False:
            status = f"✗ 错误 (真实: {result['true_label']})"
        else:
            status = "? 未知"
        
        print(f"  {result['image_file']}: 预测={result['predicted_digit']}, "
              f"置信度={result['confidence']*100:.2f}%, {status}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='使用 U-Net 分类模型进行数字识别预测')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--image_path', type=str, default=None, help='单张图片路径')
    parser.add_argument('--image_dir', type=str, default=None, help='图片目录路径（批量预测）')
    parser.add_argument('--output_dir', type=str, default='predictions', help='输出目录')
    parser.add_argument('--image_size', type=int, default=256, help='图像尺寸')
    parser.add_argument('--num_classes', type=int, default=10, help='分类类别数（0-9）')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.image_path and not args.image_dir:
        print("错误: 必须提供 --image_path 或 --image_dir")
        return
    
    if args.image_path and args.image_dir:
        print("错误: 不能同时提供 --image_path 和 --image_dir")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    print(f"加载模型: {args.model_path}")
    model = load_model(args.model_path, device, args.num_classes)
    print("模型加载完成")
    
    # 执行预测
    if args.image_path:
        # 单张图片预测
        if not os.path.exists(args.image_path):
            print(f"错误: 图片文件不存在: {args.image_path}")
            return
        
        predict_single_image(
            model, args.image_path, device, 
            args.image_size, 
            save_output=True, output_dir=args.output_dir
        )
    else:
        # 批量预测
        if not os.path.exists(args.image_dir):
            print(f"错误: 图片目录不存在: {args.image_dir}")
            return
        
        predict_batch(
            model, args.image_dir, device,
            args.image_size, args.output_dir
        )
    
    print("\n预测完成！")


if __name__ == "__main__":
    main()
