"""
ONNX 模型导出脚本：将训练好的 GazeNet/GazeNetV2 模型导出为 ONNX 格式。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python scripts/export_onnx.py
         conda run -n gaze-env python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet, GazeNetV2


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 11,
    verify: bool = True,
) -> None:
    """将 PyTorch 模型导出为 ONNX 格式。"""
    print(f"加载 PyTorch 模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 检测模型版本
    model_version = "v1"
    config = {}
    if isinstance(checkpoint, dict):
        model_version = checkpoint.get("model_version", "v1")
        config = checkpoint.get("config", {})
        print(f"  - 模型版本: {model_version}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Best val angle: {checkpoint.get('best_val_angle', 'N/A')}")

    # 构建模型
    model_cfg = config.get("model", {})
    channels = model_cfg.get("channels", [32, 64, 128, 256])

    if model_version == "v2":
        model = GazeNetV2(
            num_channels=channels,
            head_pose_dim=model_cfg.get("head_pose_dim", 3),
            fusion_dim=model_cfg.get("fusion_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        model = GazeNet(num_channels=channels)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"\n导出 ONNX 模型: {output_path}")
    print(f"  - Opset version: {opset_version}")

    if model_version == "v2":
        # V2：三个输入
        dummy_left = torch.randn(1, 3, 128, 128)
        dummy_right = torch.randn(1, 3, 128, 128)
        dummy_pose = torch.randn(1, 3)

        torch.onnx.export(
            model,
            (dummy_left, dummy_right, dummy_pose),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['left_eye', 'right_eye', 'head_pose'],
            output_names=['output'],
            dynamic_axes={
                'left_eye': {0: 'batch_size'},
                'right_eye': {0: 'batch_size'},
                'head_pose': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
        )
    else:
        # V1：单个输入
        dummy_input = torch.randn(2, 3, 128, 128)
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

    print("[Success] ONNX 模型已导出")

    # 验证导出的模型
    if verify:
        print("\n验证 ONNX 模型...")
        import onnxruntime as ort

        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])

        if model_version == "v2":
            test_left = np.random.randn(1, 3, 128, 128).astype(np.float32)
            test_right = np.random.randn(1, 3, 128, 128).astype(np.float32)
            test_pose = np.random.randn(1, 3).astype(np.float32)

            with torch.no_grad():
                pytorch_output = model(
                    torch.from_numpy(test_left),
                    torch.from_numpy(test_right),
                    torch.from_numpy(test_pose),
                ).numpy()

            onnx_output = session.run(
                ['output'],
                {
                    'left_eye': test_left,
                    'right_eye': test_right,
                    'head_pose': test_pose,
                }
            )[0]
        else:
            test_input = np.random.randn(2, 3, 128, 128).astype(np.float32)

            with torch.no_grad():
                pytorch_output = model(torch.from_numpy(test_input)).numpy()

            onnx_output = session.run(
                ['output'],
                {'input': test_input}
            )[0]

        max_diff = np.abs(pytorch_output - onnx_output).max()
        mean_diff = np.abs(pytorch_output - onnx_output).mean()

        print(f"  - PyTorch output shape: {pytorch_output.shape}")
        print(f"  - ONNX output shape: {onnx_output.shape}")
        print(f"  - Max difference: {max_diff:.6e}")
        print(f"  - Mean difference: {mean_diff:.6e}")

        if max_diff < 1e-4:
            print("[Pass] 验证通过：ONNX 模型输出与 PyTorch 一致")
        else:
            print(f"[Warning] 警告：输出差异较大 (max_diff={max_diff:.6e})")


def main():
    parser = argparse.ArgumentParser(description="导出 GazeNet 模型为 ONNX 格式")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="PyTorch checkpoint 路径",
    )
    parser.add_argument(
        "--output", type=str, default="checkpoints/gaze_net.onnx",
        help="ONNX 模型输出路径",
    )
    parser.add_argument(
        "--opset", type=int, default=11,
        help="ONNX opset 版本",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="跳过验证步骤",
    )
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"错误：checkpoint 文件不存在: {args.checkpoint}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=str(output_path),
        opset_version=args.opset,
        verify=not args.no_verify,
    )

    print(f"\n完成！ONNX 模型已保存到: {output_path}")


if __name__ == "__main__":
    main()
