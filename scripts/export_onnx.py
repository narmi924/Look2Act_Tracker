"""
ONNX 模型导出脚本：将训练好的 PyTorch GazeNet 模型导出为 ONNX 格式。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python scripts/export_onnx.py
         conda run -n gaze-env python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output checkpoints/gaze_net.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 11,
    verify: bool = True,
) -> None:
    """将 PyTorch 模型导出为 ONNX 格式。
    
    参数:
        checkpoint_path: PyTorch checkpoint 路径
        output_path: ONNX 模型输出路径
        opset_version: ONNX opset 版本
        verify: 是否验证导出的模型
    """
    print(f"加载 PyTorch 模型: {checkpoint_path}")
    
    # 加载模型
    model = GazeNet()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Best val angle: {checkpoint.get('best_val_angle', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # 创建示例输入（batch=2，左右眼）
    dummy_input = torch.randn(2, 3, 128, 128)
    
    print(f"\n导出 ONNX 模型: {output_path}")
    print(f"  - Opset version: {opset_version}")
    print(f"  - Input shape: {dummy_input.shape}")
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )
    
    print(f"✓ ONNX 模型已导出")
    
    # 验证导出的模型
    if verify:
        print("\n验证 ONNX 模型...")
        import onnxruntime as ort
        
        # 加载 ONNX 模型
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        
        # 测试推理
        test_input = np.random.randn(2, 3, 128, 128).astype(np.float32)
        
        # PyTorch 推理
        with torch.no_grad():
            pytorch_output = model(torch.from_numpy(test_input)).numpy()
        
        # ONNX Runtime 推理
        onnx_output = session.run(
            ['output'],
            {'input': test_input}
        )[0]
        
        # 比较输出
        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()
        
        print(f"  - PyTorch output shape: {pytorch_output.shape}")
        print(f"  - ONNX output shape: {onnx_output.shape}")
        print(f"  - Max difference: {max_diff:.6e}")
        print(f"  - Mean difference: {mean_diff:.6e}")
        
        if max_diff < 1e-4:
            print("✓ 验证通过：ONNX 模型输出与 PyTorch 一致")
        else:
            print(f"⚠ 警告：输出差异较大 (max_diff={max_diff:.6e})")


def main():
    parser = argparse.ArgumentParser(description="导出 GazeNet 模型为 ONNX 格式")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="PyTorch checkpoint 路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/gaze_net.onnx",
        help="ONNX 模型输出路径",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="跳过验证步骤",
    )
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.checkpoint).exists():
        print(f"错误：checkpoint 文件不存在: {args.checkpoint}")
        sys.exit(1)
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 导出
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=str(output_path),
        opset_version=args.opset,
        verify=not args.no_verify,
    )
    
    print(f"\n完成！ONNX 模型已保存到: {output_path}")


if __name__ == "__main__":
    main()
