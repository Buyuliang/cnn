# predict_rknn.py
"""
RKNN 模型推理 Demo（与 predict.py 使用方式一致）

使用转换好的 .rknn 模型对单张或批量图片进行数字识别（0-9）。
预处理、后处理与 predict.py 保持一致：灰度图、resize 256×256、归一化 [0,1]，
输出为类别与置信度。

依赖：pip install rknn-toolkit2
PC 上无 NPU 时使用 --target simulator 进行模拟推理。
"""
import argparse
import os
import numpy as np

try:
    from PIL import Image
except ImportError:
    raise ImportError("请安装 Pillow: pip install Pillow")

try:
    from rknn.api import RKNN
except ImportError:
    raise ImportError(
        "请安装 rknn-toolkit2：\n"
        "  https://github.com/airockchip/rknn-toolkit2\n"
        "  或 pip install rknn-toolkit2"
    )


# 与 model.py / predict.py 一致
DEFAULT_IMAGE_SIZE = 256
NUM_CLASSES = 10


def preprocess_image(image_path, image_size=DEFAULT_IMAGE_SIZE):
    """
    预处理输入图像，与 predict.py 一致。
    返回 (input_data, original_size, img_array_for_vis)。
    input_data: (1, 1, H, W) float32, 范围 [0, 1]，用于 RKNN 推理。
    """
    img = Image.open(image_path).convert("L")
    original_size = img.size
    img = img.resize((image_size, image_size))
    img_array = np.array(img, dtype=np.float32) / 255.0
    # NCHW, batch=1, channel=1
    input_data = np.expand_dims(np.expand_dims(img_array, 0), 0)
    input_data = np.ascontiguousarray(input_data)
    return input_data, original_size, img_array


def softmax(x):
    """沿最后一维做 softmax"""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_inference(rknn, input_data):
    """
    执行 RKNN 推理。
    input_data: (1, 1, H, W) float32。
    返回 (logits, probs, predicted_class, confidence)。
    """
    outputs = rknn.inference(inputs=[input_data])
    if not outputs:
        raise RuntimeError("RKNN inference 返回为空")
    logits = outputs[0]  # (1, 10)
    if logits.ndim == 2:
        logits = logits[0]
    probs = softmax(logits)
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])
    return logits, probs, predicted_class, confidence


def predict_single_image(rknn, image_path, image_size=DEFAULT_IMAGE_SIZE, verbose=True):
    """对单张图片进行 RKNN 预测，用法与 predict.py 的 predict_single_image 对齐。"""
    if verbose:
        print(f"处理图片: {image_path}")
    input_data, _, img_array = preprocess_image(image_path, image_size)
    _, probs, predicted_class, confidence = run_inference(rknn, input_data)
    if verbose:
        print(f"预测结果: 数字 {predicted_class}")
        print(f"置信度: {confidence:.4f} ({confidence*100:.2f}%)")
        print("所有类别概率:")
        for i in range(NUM_CLASSES):
            print(f"  数字 {i}: {probs[i]:.4f} ({probs[i]*100:.2f}%)")
    return {
        "predicted_digit": predicted_class,
        "confidence": confidence,
        "all_probabilities": probs,
        "image_path": image_path,
    }


def predict_batch(rknn, image_dir, image_size=DEFAULT_IMAGE_SIZE, verbose=True):
    """批量预测目录下图片，与 predict.py 的 predict_batch 用法一致。"""
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith(image_extensions)
    ]
    if not image_files:
        if verbose:
            print(f"在 {image_dir} 中未找到图像文件")
        return []
    if verbose:
        print(f"找到 {len(image_files)} 张图片")
    results = []
    correct = 0
    total_labeled = 0
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        try:
            result = predict_single_image(
                rknn, image_path, image_size, verbose=verbose
            )
            result["image_file"] = image_file
            results.append(result)
            # 可选：从同名校的 .txt 或文件名中的 _数字 解析真实标签
            true_label = None
            label_path = os.path.splitext(image_path)[0] + ".txt"
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    true_label = int(f.read().strip())
            else:
                try:
                    parts = os.path.splitext(image_file)[0].split("_")
                    true_label = int(parts[-1])
                except Exception:
                    pass
            if true_label is not None:
                total_labeled += 1
                result["true_label"] = true_label
                result["correct"] = result["predicted_digit"] == true_label
                if result["correct"]:
                    correct += 1
            else:
                result["true_label"] = None
                result["correct"] = None
        except Exception as e:
            if verbose:
                print(f"处理 {image_file} 时出错: {e}")
    if verbose and total_labeled > 0:
        print("\n预测统计:")
        print(f"总图片数: {len(results)}")
        print(f"准确率: {100*correct/total_labeled:.2f}% ({correct}/{total_labeled})")
        print("\n详细预测结果:")
        for r in results:
            status = (
                "✓ 正确"
                if r["correct"] is True
                else (
                    f"✗ 错误 (真实: {r['true_label']})"
                    if r["correct"] is False
                    else "? 未知"
                )
            )
            print(
                f"  {r['image_file']}: 预测={r['predicted_digit']}, "
                f"置信度={r['confidence']*100:.2f}%, {status}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(
        description="使用 RKNN 模型进行数字识别（与 predict.py 用法一致）"
    )
    parser.add_argument("--model_path", type=str, required=True, help=".rknn 模型路径")
    parser.add_argument("--image_path", type=str, default=None, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, default=None, help="图片目录（批量预测）")
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"输入尺寸（需与转 RKNN 时一致，默认 {DEFAULT_IMAGE_SIZE}）",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="simulator",
        choices=["simulator", "rk3568", "rk3566", "rk3588", "rk3562", "rv1106", "rv1103"],
        help="运行目标：PC 上无 NPU 用 simulator，板子上用对应型号（默认 simulator）",
    )
    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        print("错误: 必须提供 --image_path 或 --image_dir")
        return
    if args.image_path and args.image_dir:
        print("错误: 不能同时提供 --image_path 和 --image_dir")
        return
    if not os.path.isfile(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return

    print(f"加载 RKNN 模型: {args.model_path}")
    rknn = RKNN(verbose=False)
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        raise RuntimeError(f"load_rknn 失败: {ret}")
    print(f"初始化运行时 (target={args.target})...")
    ret = rknn.init_runtime(target=args.target)
    if ret != 0:
        raise RuntimeError(f"init_runtime 失败: {ret}")
    print("模型就绪，开始推理\n")

    try:
        if args.image_path:
            predict_single_image(
                rknn, args.image_path, args.image_size, verbose=True
            )
        else:
            if not os.path.isdir(args.image_dir):
                print(f"错误: 图片目录不存在: {args.image_dir}")
                return
            predict_batch(rknn, args.image_dir, args.image_size, verbose=True)
    finally:
        rknn.release()

    print("\n推理完成！")


if __name__ == "__main__":
    main()
