"""
CPU 优化相关的 Property-Based Tests。

测试内容：
- Property 15: 模型格式转换输出一致性
- Property 16: 批处理与逐个推理一致性

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env pytest tests/test_optimization.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given, settings, strategies as st

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet


# ============================================================================
# Property 15: 模型格式转换输出一致性
# ============================================================================
# Feature: look2act-tracker, Property 15: 模型格式转换输出一致性
# **Validates: Requirements 10.2**

@given(
    batch_size=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, deadline=None)
def test_property_15_onnx_conversion_consistency(batch_size: int, seed: int):
    """Property 15: 模型格式转换输出一致性。
    
    对于任意输入张量，ONNX 格式的模型输出应与原始 PyTorch 模型输出近似一致（允许误差 1e-4）。
    
    验证需求: 10.2
    """
    # 设置随机种子以确保可重现性
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建模型（随机初始化）
    model = GazeNet()
    model.eval()
    
    # 生成随机输入
    input_tensor = torch.randn(batch_size, 3, 128, 128)
    
    # PyTorch 推理
    with torch.no_grad():
        pytorch_output = model(input_tensor).numpy()
    
    # 导出为 ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        onnx_path = tmp.name
    
    try:
        torch.onnx.export(
            model,
            input_tensor,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )
        
        # ONNX Runtime 推理
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        onnx_output = session.run(
            ['output'],
            {'input': input_tensor.numpy()}
        )[0]
        
        # 验证输出一致性
        max_diff = np.abs(pytorch_output - onnx_output).max()
        
        # 允许误差 1e-4
        assert max_diff < 1e-4, (
            f"ONNX 输出与 PyTorch 输出差异过大: max_diff={max_diff:.6e} "
            f"(batch_size={batch_size}, seed={seed})"
        )
        
        # 验证输出形状一致
        assert pytorch_output.shape == onnx_output.shape, (
            f"输出形状不一致: PyTorch={pytorch_output.shape}, ONNX={onnx_output.shape}"
        )
        
        # 验证输出仍为单位向量
        norms = np.linalg.norm(onnx_output, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6), (
            f"ONNX 输出不是单位向量: norms={norms}"
        )
        
    finally:
        # 清理临时文件
        Path(onnx_path).unlink(missing_ok=True)


# ============================================================================
# Property 16: 批处理与逐个推理一致性
# ============================================================================
# Feature: look2act-tracker, Property 16: 批处理与逐个推理一致性
# **Validates: Requirements 10.3**

@given(
    seed=st.integers(min_value=0, max_value=10000),
)
@settings(max_examples=100, deadline=None)
def test_property_16_batch_vs_individual_consistency(seed: int):
    """Property 16: 批处理与逐个推理一致性。
    
    对于任意左右眼图像对，将两张图像合并为 batch=2 进行推理的结果，
    应与分别对每张图像进行 batch=1 推理的结果一致（允许浮点误差 1e-6）。
    
    验证需求: 10.3
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建模型
    model = GazeNet()
    model.eval()
    
    # 生成左右眼图像（随机）
    left_eye = torch.randn(1, 3, 128, 128)
    right_eye = torch.randn(1, 3, 128, 128)
    
    # 方法 1: 批处理推理 (batch=2)
    batch = torch.cat([left_eye, right_eye], dim=0)  # (2, 3, 128, 128)
    with torch.no_grad():
        batch_output = model(batch).numpy()  # (2, 3)
    
    # 方法 2: 逐个推理 (batch=1 × 2)
    with torch.no_grad():
        left_output = model(left_eye).numpy()  # (1, 3)
        right_output = model(right_eye).numpy()  # (1, 3)
    individual_output = np.vstack([left_output, right_output])  # (2, 3)
    
    # 验证一致性
    max_diff = np.abs(batch_output - individual_output).max()
    
    assert max_diff < 1e-6, (
        f"批处理与逐个推理结果不一致: max_diff={max_diff:.6e} (seed={seed})"
    )
    
    # 验证形状一致
    assert batch_output.shape == individual_output.shape, (
        f"输出形状不一致: batch={batch_output.shape}, individual={individual_output.shape}"
    )
    
    # 验证每个输出都是单位向量
    batch_norms = np.linalg.norm(batch_output, axis=1)
    individual_norms = np.linalg.norm(individual_output, axis=1)
    
    assert np.allclose(batch_norms, 1.0, atol=1e-6), (
        f"批处理输出不是单位向量: norms={batch_norms}"
    )
    assert np.allclose(individual_norms, 1.0, atol=1e-6), (
        f"逐个推理输出不是单位向量: norms={individual_norms}"
    )


# ============================================================================
# 单元测试：IPEX 优化可用性检查
# ============================================================================

def test_ipex_availability():
    """单元测试：检查 IPEX 是否可用（仅在训练时使用）。
    
    注意：IPEX 仅在训练阶段使用，推理系统不依赖 IPEX 以确保跨平台兼容性。
    """
    try:
        import intel_extension_for_pytorch as ipex
        print(f"✓ IPEX 可用: {ipex.__version__}")
        
        # 测试 IPEX 优化
        model = GazeNet()
        model.eval()
        
        optimized_model = ipex.optimize(model)
        
        # 测试推理
        test_input = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            output = optimized_model(test_input)
        
        assert output.shape == (2, 3), f"IPEX 优化后输出形状错误: {output.shape}"
        
        # 验证输出仍为单位向量
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
            f"IPEX 优化后输出不是单位向量: norms={norms}"
        )
        
        print("✓ IPEX 优化测试通过")
        
    except (ImportError, AttributeError) as e:
        pytest.skip(f"IPEX 不可用，跳过测试: {e}")


# ============================================================================
# 单元测试：ONNX Runtime 可用性检查
# ============================================================================

def test_onnx_runtime_availability():
    """单元测试：检查 ONNX Runtime 是否可用。
    
    ONNX Runtime 是推理系统的默认推理引擎，确保跨平台兼容性。
    """
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime 可用: {ort.__version__}")
        print(f"  可用 providers: {ort.get_available_providers()}")
        
        # 验证 CPUExecutionProvider 可用
        assert 'CPUExecutionProvider' in ort.get_available_providers(), (
            "CPUExecutionProvider 不可用"
        )
        
    except ImportError as e:
        pytest.fail(f"ONNX Runtime 不可用: {e}")


# ============================================================================
# 单元测试：批处理推理性能对比
# ============================================================================

def test_batch_inference_performance():
    """单元测试：对比批处理与逐个推理的性能。
    
    批处理应该比逐个推理更快（或至少不慢）。
    """
    import time
    
    model = GazeNet()
    model.eval()
    
    # 生成测试数据
    left_eye = torch.randn(1, 3, 128, 128)
    right_eye = torch.randn(1, 3, 128, 128)
    
    # 预热
    with torch.no_grad():
        _ = model(torch.cat([left_eye, right_eye], dim=0))
    
    # 测试批处理推理
    num_iterations = 100
    
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            batch = torch.cat([left_eye, right_eye], dim=0)
            _ = model(batch)
    batch_time = time.perf_counter() - t0
    
    # 测试逐个推理
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(left_eye)
            _ = model(right_eye)
    individual_time = time.perf_counter() - t0
    
    print(f"\n性能对比 ({num_iterations} 次迭代):")
    print(f"  批处理 (batch=2): {batch_time:.4f}s ({batch_time/num_iterations*1000:.2f}ms/iter)")
    print(f"  逐个推理 (batch=1×2): {individual_time:.4f}s ({individual_time/num_iterations*1000:.2f}ms/iter)")
    print(f"  加速比: {individual_time/batch_time:.2f}x")
    
    # 批处理应该更快（或至少不慢太多）
    # 允许 10% 的性能波动
    assert batch_time <= individual_time * 1.1, (
        f"批处理推理比逐个推理慢: batch={batch_time:.4f}s, individual={individual_time:.4f}s"
    )
