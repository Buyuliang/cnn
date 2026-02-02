  ────────────────────────────────────────

  依赖
  • 需安装 RKNN-Toolkit2（Rockchip 官方工具，在 PC 上做转换）
  • 从 airockchip/rknn-toolkit2 (​https://github.com/airockchip/rknn-toolkit2​) 下载对应你系统的
    wheel 安装，或按官方文档安装。


  ────────────────────────────────────────

  基本用法

  # 默认：ONNX → RKNN，目标 rk3588，FP 精度
  python onnx2rknn.py path/to/model.onnx
  # 指定输出路径
  python onnx2rknn.py model.onnx -o model.rknn
  # 指定板子（如 RK3568）
  python onnx2rknn.py model.onnx -t rk3568 -o model.rknn
  # 启用 INT8 量化（需提供校准数据集列表）
  python onnx2rknn.py model.onnx --quantize --dataset dataset.txt -o model.rknn


  ────────────────────────────────────────

  参数说明

  | 参数         | 说明                                                                       | 
  |--------------|----------------------------------------------------------------------------|
  | onnx_path    | 输入的 ONNX 模型路径                                                       | 
  | -o, --output | 输出 .rknn 路径（不写则与 ONNX 同目录、同名 .rknn）                        | 
  | -t, --target | 目标 NPU：rk3588 / rk3568 / rk3566 / rk3562 / rv1106 / rv1103，默认 rk3588 | 
  | --quantize   | 是否做 INT8 量化                                                           | 
  | --dataset    | 量化用数据集列表文件（每行一个图片路径），仅 --quantize 时有用             | 
  | --input-size | 输入形状 N,C,H,W，默认 1,1,256,256，与当前 ONNX 一致                       | 


  ────────────────────────────────────────

  量化用 dataset.txt 格式
  • 每行一个图片的绝对路径或相对路径。
  • 图片会作为校准数据用于 INT8 量化，建议用训练/验证集里的几十到几百张即可。

  示例（生成列表）：

  find /path/to/images -name "*.png" | head -200 > dataset.txt
