#!/usr/bin/env python3
"""
ONNX 转 RKNN 模型转换脚本（对应 model.py 中的 UNetClassifier）。

将 train.py 导出的 ONNX 模型（UNetClassifier，输入 1×1×256×256，input/output）
转换为 RKNN 格式，用于在 Rockchip NPU（RK3568、RK3588 等）上部署。

与 model.py 的对应关系：
  - 模型：UNetClassifier（in_channels=1, num_classes=10）
  - 输入名：input，形状 NCHW，默认 (1, 1, 256, 256)
  - 输出名：output，形状 (N, 10) 分类 logits

依赖：pip install rknn-toolkit2
或从 https://github.com/airockchip/rknn-toolkit2 安装对应平台 wheel。
"""

import argparse
import os
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from rknn.api import RKNN
except ImportError:
    raise ImportError(
        "rknn-toolkit2 not found. Install from:\n"
        "  https://github.com/airockchip/rknn-toolkit2\n"
        "  or: pip install rknn-toolkit2 (if available for your platform)"
    )


# Supported Rockchip NPU platforms
TARGET_PLATFORMS = ["rk3568", "rk3566", "rk3588", "rk3562", "rv1106", "rv1103"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to RKNN format for Rockchip NPU."
    )
    parser.add_argument(
        "onnx_path",
        type=str,
        help="Path to input ONNX model file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output RKNN model (default: same name as ONNX with .rknn).",
    )
    parser.add_argument(
        "-t",
        "--target",
        type=str,
        default="rk3588",
        choices=TARGET_PLATFORMS,
        help="Target NPU platform (default: rk3588).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization (faster inference, may need dataset).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset list file for quantization. "
             "Each line: path to image. Used only when --quantize.",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        default="1,1,256,256",
        help="Input shape N,C,H,W for building (default: 1,1,256,256).",
    )
    parser.add_argument(
        "--input-name",
        type=str,
        default="input",
        help="ONNX input name to bind shape (default: input).",
    )
    # Accuracy analysis（需在 build 之后、release 之前调用）
    parser.add_argument(
        "--accuracy-analysis",
        action="store_true",
        help="Run RKNN accuracy analysis after build (need calibration .npy inputs).",
    )
    parser.add_argument(
        "--accuracy-inputs",
        type=str,
        nargs="+",
        default=None,
        help="Calibration inputs for accuracy_analysis: image path(s) or .npy path(s) (e.g. img.png calib/1.npy).",
    )
    parser.add_argument(
        "--accuracy-output-dir",
        type=str,
        default="./snapshot",
        help="Output directory for accuracy analysis (default: ./snapshot).",
    )
    parser.add_argument(
        "--accuracy-device-id",
        type=str,
        default=None,
        help="Device ID for on-device accuracy analysis (e.g. 192.168.202.227:5555). "
             "If not set, analysis runs on simulator.",
    )
    return parser.parse_args()


def parse_input_size(size_str: str):
    try:
        parts = [int(x.strip()) for x in size_str.split(",")]
        if len(parts) != 4:
            raise ValueError("Need 4 values: N,C,H,W")
        return parts
    except Exception as e:
        raise ValueError(f"Invalid --input-size '{size_str}': {e}") from e


# 图片扩展名视为图片路径，否则视为 .npy
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def _is_image_path(path: str) -> bool:
    return path.lower().endswith(IMAGE_EXTENSIONS)


def _image_path_to_npy(image_path: str, input_size, save_path: str) -> str:
    """将一张图片按模型输入预处理并保存为 .npy，与 predict_rknn 预处理一致。"""
    if Image is None:
        raise ImportError("Accuracy 使用图片路径时需安装 Pillow: pip install Pillow")
    n, c, h, w = input_size
    img = Image.open(image_path).convert("L")
    img = img.resize((w, h))
    arr = np.array(img, dtype=np.float32) / 255.0
    data = np.expand_dims(np.expand_dims(arr, 0), 0)  # (1, 1, H, W)
    data = np.ascontiguousarray(data)
    np.save(save_path, data)
    return save_path


def _resolve_accuracy_inputs(paths, input_size, output_dir):
    """
    将 --accuracy-inputs 中的路径解析为 .npy 路径列表。
    图片路径会先预处理成 (1,1,H,W) float32 并保存到 output_dir 下的临时 .npy。
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_paths = []
    for i, p in enumerate(paths):
        p_abs = os.path.abspath(p)
        if not os.path.isfile(p_abs):
            raise FileNotFoundError(f"Accuracy analysis 输入文件不存在: {p_abs}")
        if p.lower().endswith(".npy"):
            npy_paths.append(p_abs)
        elif _is_image_path(p):
            save_path = os.path.join(output_dir, f"calib_from_image_{i}.npy")
            _image_path_to_npy(p_abs, input_size, save_path)
            npy_paths.append(os.path.abspath(save_path))
        else:
            raise ValueError(
                f"不支持的精度分析输入格式: {p}，请使用图片路径（.png/.jpg 等）或 .npy 路径。"
            )
    return npy_paths


def main():
    args = parse_args()

    onnx_path = os.path.abspath(args.onnx_path)
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    if args.output:
        rknn_path = os.path.abspath(args.output)
    else:
        base = os.path.splitext(onnx_path)[0]
        rknn_path = base + ".rknn"

    input_size = parse_input_size(args.input_size)
    print(f"Input shape: {input_size} (N,C,H,W)")

    # 量化时必须提供且存在 dataset 文件
    if args.quantize:
        if not args.dataset:
            raise FileNotFoundError(
                "启用 --quantize 时必须用 --dataset 指定校准数据集列表文件。\n"
                "示例：\n"
                "  find data/train -name '*.png' | head -200 > dataset.txt\n"
                "  python onnx2rknn.py ... --quantize --dataset dataset.txt -o ..."
            )
        dataset_path = os.path.abspath(args.dataset)
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"校准数据集文件不存在: {dataset_path}\n"
                "请先创建该文件，每行一个图片路径。示例：\n"
                "  find data/train -name '*.png' | head -200 > dataset.txt\n"
                "  find data/val -name '*.png' >> dataset.txt"
            )
        args.dataset = dataset_path

    rknn = RKNN(verbose=False)

    # Must call config() before load_onnx (required by rknn-toolkit2 2.x)
    print(f"Target platform: {args.target}")
    ret = rknn.config(target_platform=args.target)
    if ret != 0:
        raise RuntimeError(f"rknn.config failed: {ret}")

    # ONNX may have dynamic input (e.g. batch_size); RKNN requires fixed shape.
    # Pass inputs + input_size_list to bind the fixed shape to input name.
    print("Loading ONNX...")
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=[args.input_name],
        input_size_list=[input_size],
    )
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")

    print("Building RKNN model...")
    if args.quantize and args.dataset and os.path.isfile(args.dataset):
        ret = rknn.build(
            do_quantization=True,
            dataset=args.dataset,
            rknn_batch_size=input_size[0],
        )
    else:
        ret = rknn.build(
            do_quantization=bool(args.quantize),
            dataset=args.dataset if (args.quantize and args.dataset) else None,
            rknn_batch_size=input_size[0],
        )
    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")

    print(f"Exporting RKNN to {rknn_path}...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"export_rknn failed: {ret}")

    # Accuracy analysis（可选；--accuracy-inputs 支持图片路径或 .npy 路径，图片会按 --input-size 预处理）
    if args.accuracy_analysis:
        if not args.accuracy_inputs:
            print("W 未指定 --accuracy-inputs，跳过 Accuracy analysis。")
        else:
            inputs_abs = [os.path.abspath(p) for p in args.accuracy_inputs]
            missing = [p for p in inputs_abs if not os.path.isfile(p)]
            if missing:
                print("W 跳过 Accuracy analysis：以下输入文件不存在:")
                for p in missing:
                    print(f"    {p}")
                print("  请指定存在的图片路径或 .npy 路径，或去掉 --accuracy-analysis。")
            else:
                output_dir = os.path.abspath(args.accuracy_output_dir)
                try:
                    npy_paths = _resolve_accuracy_inputs(
                        inputs_abs, input_size, output_dir
                    )
                except (ImportError, ValueError, FileNotFoundError) as e:
                    print(f"W 跳过 Accuracy analysis: {e}")
                else:
                    print("--> Accuracy analysis")
                    if args.accuracy_device_id:
                        ret = rknn.accuracy_analysis(
                            inputs=npy_paths,
                            output_dir=output_dir,
                            target=args.target,
                            device_id=args.accuracy_device_id,
                        )
                    else:
                        ret = rknn.accuracy_analysis(
                            inputs=npy_paths,
                            output_dir=output_dir,
                        )
                    if ret != 0:
                        print("Accuracy analysis failed!")
                        rknn.release()
                        exit(ret)
                    print("Accuracy analysis done.")

    rknn.release()
    print("Done. RKNN model saved:", rknn_path)


if __name__ == "__main__":
    main()
