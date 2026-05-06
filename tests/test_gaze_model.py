"""GazeNet 模型 Property-Based 测试。

Feature: look2act-tracker, Property 6: GazeModel 输出单位向量约束
"""

import torch
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.models.gaze_net import GazeNet, GazeNetPoG


# 生成随机 128x128 RGB 图像张量的策略
# 使用 float32，值域 [0, 1]，模拟归一化后的图像输入
def eye_image_tensors(batch_size: int = 1):
    """生成形状为 (B, 3, 128, 128) 的随机输入张量。"""
    return arrays(
        dtype=np.float32,
        shape=(batch_size, 3, 128, 128),
        elements=st.floats(min_value=0.0, max_value=1.0, width=32),
    )


class TestGazeNetUnitVector:
    """Property 6: GazeModel 输出单位向量约束

    对于任意 128×128×3 的输入张量，GazeModel 的输出必须是 3 维向量
    且满足 L2 范数为 1（即 gx² + gy² + gz² = 1，允许浮点误差 1e-6）。

    **Validates: Requirements 3.1, 3.4**
    """

    def setup_method(self):
        """每个测试方法前初始化模型（eval 模式）。"""
        self.model = GazeNet()
        self.model.eval()

    @given(data=eye_image_tensors(batch_size=1))
    @settings(max_examples=100)
    def test_single_input_unit_vector(self, data: np.ndarray):
        """单张图像输入，输出应为 3 维单位向量。

        **Validates: Requirements 3.1, 3.4**
        """
        x = torch.from_numpy(data)
        with torch.no_grad():
            y = self.model(x)

        # 输出形状必须是 (1, 3)
        assert y.shape == (1, 3), f"输出形状应为 (1, 3)，实际为 {y.shape}"

        # L2 范数应 ≈ 1
        norm = torch.norm(y, dim=1).item()
        assert abs(norm - 1.0) < 1e-6, f"L2 范数应为 1，实际为 {norm}"

    @given(data=eye_image_tensors(batch_size=4))
    @settings(max_examples=100)
    def test_batch_input_unit_vectors(self, data: np.ndarray):
        """批量输入，每个输出向量都应为单位向量。

        **Validates: Requirements 3.1, 3.4**
        """
        x = torch.from_numpy(data)
        with torch.no_grad():
            y = self.model(x)

        # 输出形状必须是 (4, 3)
        assert y.shape == (4, 3), f"输出形状应为 (4, 3)，实际为 {y.shape}"

        # 每个向量的 L2 范数应 ≈ 1
        norms = torch.norm(y, dim=1)
        for i, norm in enumerate(norms):
            assert abs(norm.item() - 1.0) < 1e-6, (
                f"第 {i} 个向量 L2 范数应为 1，实际为 {norm.item()}"
            )


def test_gazenet_pog_outputs_normalized_screen_points():
    model = GazeNetPoG()
    model.eval()
    left = torch.rand(2, 3, 128, 128)
    right = torch.rand(2, 3, 128, 128)
    pose = torch.zeros(2, 3)

    with torch.no_grad():
        out = model(left, right, pose)

    assert out.shape == (2, 2)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_gazenet_pog_linear_output_keeps_screen_point_shape_without_sigmoid():
    model = GazeNetPoG(num_channels=[8, 8, 8, 8], fusion_dim=16, dropout=0.0, output_activation="linear")
    model.eval()
    left = torch.rand(2, 3, 128, 128)
    right = torch.rand(2, 3, 128, 128)
    pose = torch.zeros(2, 3)

    with torch.no_grad():
        out = model(left, right, pose)

    assert out.shape == (2, 2)
    assert model.output_activation == "linear"
