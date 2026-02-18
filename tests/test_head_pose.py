"""HeadPoseEstimator Property-Based 测试。

Feature: look2act-tracker, Property 7: 旋转矩阵正交性不变量
"""
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.vision.head_pose import HeadPoseEstimator


# 基准关键点（正面人脸，640×480 图像）
_BASE_POINTS = {
    "nose_tip": (320.0, 240.0),
    "chin": (320.0, 400.0),
    "left_eye_outer": (250.0, 200.0),
    "right_eye_outer": (390.0, 200.0),
    "left_mouth": (280.0, 340.0),
    "right_mouth": (360.0, 340.0),
}


def perturbed_pnp_points():
    """生成带随机扰动的有效 PnP 关键点。

    在基准点基础上添加 [-40, 40] 像素的随机偏移，
    模拟不同头部姿态下的关键点位置变化。
    """
    return st.fixed_dictionaries({
        name: st.tuples(
            st.floats(min_value=base[0] - 40, max_value=base[0] + 40,
                      allow_nan=False, allow_infinity=False),
            st.floats(min_value=base[1] - 40, max_value=base[1] + 40,
                      allow_nan=False, allow_infinity=False),
        )
        for name, base in _BASE_POINTS.items()
    })


class TestRotationMatrixOrthogonality:
    """Property 7: 旋转矩阵正交性不变量

    对于任意有效的 6 个面部关键点输入，HeadPoseEstimator 输出的
    3×3 旋转矩阵 R 必须满足正交性：R^T × R ≈ I（允许元素误差 1e-4）。

    **Validates: Requirements 4.3**
    """

    def setup_method(self):
        self.estimator = HeadPoseEstimator(frame_size=(640, 480))

    @given(points=perturbed_pnp_points())
    @settings(max_examples=100)
    def test_rotation_matrix_orthogonality(self, points: dict):
        """旋转矩阵应满足 R^T × R ≈ I。

        **Validates: Requirements 4.3**
        """
        result = self.estimator.estimate(points)

        # 只验证 PnP 求解成功的情况
        assume(result.valid)

        R = result.rotation_matrix
        assert R is not None
        assert R.shape == (3, 3)

        # R^T × R 应近似单位矩阵
        RtR = R.T @ R
        identity = np.eye(3)
        assert np.allclose(RtR, identity, atol=1e-4), (
            f"R^T × R 不是单位矩阵:\n{RtR}"
        )

        # 行列式应为 +1（正交旋转矩阵）
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 1e-4, f"行列式应为 1，实际为 {det}"
