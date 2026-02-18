"""校准模块 Property-Based 测试。

Feature: look2act-tracker
- Property 11: 仿射校准拟合精度
- Property 12: 校准参数序列化 round-trip
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from src.calibration.calibrator import CalibrationModule


def affine_matrices():
    """生成随机 2×3 仿射矩阵。"""
    return arrays(
        dtype=np.float64,
        shape=(2, 3),
        elements=st.floats(min_value=-2.0, max_value=2.0,
                           allow_nan=False, allow_infinity=False),
    )


def raw_points(n: int = 9):
    """生成 N 个随机原始坐标点，确保不退化（点不全相同）。"""
    return st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1920.0,
                      allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1080.0,
                      allow_nan=False, allow_infinity=False),
        ),
        min_size=n, max_size=n,
    ).filter(lambda pts: _points_have_spread(pts))


def _points_have_spread(pts: list[tuple[float, float]]) -> bool:
    """检查点集是否有足够的分散度（非退化）。"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (max(xs) - min(xs)) > 10.0 and (max(ys) - min(ys)) > 10.0


class TestAffineCalibrationFit:
    """Property 11: 仿射校准拟合精度

    对于任意已知的 2×3 仿射矩阵 A 和由 A 生成的 N 个校准点对（N ≥ 9），
    CalibrationModule 拟合得到的仿射矩阵 A' 应与原始矩阵 A 近似相等
    （允许元素误差 1e-3）。

    **Validates: Requirements 6.3**
    """

    @given(A=affine_matrices(), pts=raw_points(12))
    @settings(max_examples=100)
    def test_affine_fit_recovers_matrix(
        self, A: np.ndarray, pts: list[tuple[float, float]]
    ):
        """从已知仿射矩阵生成的校准点应能还原该矩阵。

        **Validates: Requirements 6.3**
        """
        cm = CalibrationModule()

        for rx, ry in pts:
            # 用已知仿射矩阵计算目标点
            src = np.array([rx, ry, 1.0])
            target = A @ src
            cm.add_calibration_point((rx, ry), (float(target[0]), float(target[1])))

        residual = cm.calibrate()

        # 残差应接近 0（因为数据完全符合仿射变换）
        assert residual < 1e-3, f"残差应接近 0，实际为 {residual}"

        # 拟合矩阵应与原始矩阵近似
        assert cm.affine_matrix is not None
        assert np.allclose(cm.affine_matrix, A, atol=1e-3), (
            f"拟合矩阵与原始矩阵不一致:\n拟合: {cm.affine_matrix}\n原始: {A}"
        )


from src.calibration.serializer import save_calibration, load_calibration


class TestCalibrationSerializationRoundTrip:
    """Property 12: 校准参数序列化 round-trip

    对于任意有效的校准参数（仿射矩阵 + 校准点列表 + 残差），
    序列化保存到 JSON 文件后再反序列化加载，
    恢复的校准参数应与原始参数完全一致。

    **Validates: Requirements 6.4, 6.5**
    """

    @given(A=affine_matrices(), pts=raw_points(9))
    @settings(max_examples=100)
    def test_serialization_round_trip(
        self, A: np.ndarray, pts: list[tuple[float, float]]
    ):
        """序列化后反序列化应恢复原始校准参数。

        **Validates: Requirements 6.4, 6.5**
        """
        # 构建并校准
        cm_orig = CalibrationModule()
        for rx, ry in pts:
            src = np.array([rx, ry, 1.0])
            target = A @ src
            cm_orig.add_calibration_point((rx, ry), (float(target[0]), float(target[1])))
        cm_orig.calibrate()

        # 序列化
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            tmp_path = f.name

        save_calibration(cm_orig, tmp_path, screen_w=1536, screen_h=864)

        # 反序列化
        cm_loaded = CalibrationModule()
        load_calibration(cm_loaded, tmp_path)

        Path(tmp_path).unlink()

        # 验证仿射矩阵一致
        assert cm_orig.affine_matrix is not None
        assert cm_loaded.affine_matrix is not None
        assert np.allclose(cm_orig.affine_matrix, cm_loaded.affine_matrix, atol=1e-10), (
            f"仿射矩阵不一致:\n原始: {cm_orig.affine_matrix}\n恢复: {cm_loaded.affine_matrix}"
        )

        # 验证校准状态一致
        assert cm_loaded.is_calibrated == cm_orig.is_calibrated

        # 验证 apply 结果一致
        test_pt = (500.0, 300.0)
        orig_result = cm_orig.apply(test_pt)
        loaded_result = cm_loaded.apply(test_pt)
        assert abs(orig_result[0] - loaded_result[0]) < 1e-6
        assert abs(orig_result[1] - loaded_result[1]) < 1e-6
