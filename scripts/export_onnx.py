"""导出 PyTorch GazeNet 模型为 ONNX 格式。

用法:
    conda run -n gaze-env python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output checkpoints/gaze_net.onnx
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.gaze_net import GazeNet


def export_to_onnx(checkpoint_path: str, output_path: str, opset_version: int = 14):
    """导出模型为 ONNX 格式。
    
    参数:
        checkpoint_path: PyTorch 模型权重路径
        output_path: ONNX 模型输出路径
        opset_version: ONNX opset 版本
    """
    print("=" * 60)
    print("导出 GazeNet 为 ONNX 格式")
    print("=" * 60)
    
    # 加载模型
    print(f"\n1. 加载模型: {checkpoint_path}")
    model = GazeNet()
    
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   已加载权重 (epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            model.load_state_dict(checkpoint)
            print("   已加载模型权重")
    else:
        print(f"   模型文件不存在，使用随机初始化")
    
    model.eval()
    
    # 创建虚拟输入
    print("\n2. 创建虚拟输入...")
    dummy_input = torch.randn(1, 3, 128, 128)
    print(f"   输入形状: {dummy_input.shape}")
    
    # 测试前向传播
    print("\n3. 测试 PyTorch 模型...")
    with torch.no_grad():
        output_torch = model(dummy_input)
    print(f"   输出形状: {output_torch.shape}")
    print(f"   输出范数: {torch.norm(output_torch).item():.6f}")
    
    # 导出为 ONNX
    print(f"\n4. 导出为 ONNX: {output_path}")
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
            'output': {0: 'batch_size'}
        }
    )
    print("   ONNX 导出成功")
    
    # 验证 ONNX 模型
    print("\n5. 验证 ONNX 模型...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ONNX 模型验证通过")
    
    # 测试 ONNX Runtime 推理
    print("\n6. 测试 ONNX Runtime 推理...")
    import onnxruntime as ort
    
    session = ort.InferenceSession(
        output_path,
        providers=['CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    output_onnx = session.run(
        [output_name],
        {input_name: dummy_input.numpy()}
    )[0]
    
    print(f"   输出形状: {output_onnx.shape}")
    print(f"   输出范数: {torch.norm(torch.from_numpy(output_onnx)).item():.6f}")
    
    # 比较结果
    diff = torch.abs(output_torch - torch.from_numpy(output_onnx)).max().item()
    print(f"\n7. PyTorch vs ONNX 差异: {diff:.6f}")
    if diff < 1e-4:
        print("   结果匹配")
    else:
        print(f"   结果存在差异 (阈值: 1e-4)")
    
    # 文件信息
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n8. 文件信息:")
    print(f"   路径: {output_path}")
    print(f"   大小: {file_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("导出完成！")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="导出 GazeNet 为 ONNX 格式")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='PyTorch 模型权重路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='checkpoints/gaze_net.onnx',
        help='ONNX 模型输出路径'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset 版本'
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    export_to_onnx(args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
