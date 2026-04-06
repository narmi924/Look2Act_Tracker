"""视线校准模块。

支持两种校准映射函数：
1. 仿射变换 (affine): p_cal = A × [x, y, 1]^T，A ∈ R^{2×3}，6 参数
2. 二次多项式 (polynomial): p_cal = W × [x, y, xy, x², y², 1]^T，W ∈ R^{2×6}，12 参数

通过用户注视已知屏幕点，使用最小二乘法拟合映射参数，
将原始视线映射修正为校准后的屏幕坐标。
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np


class CalibrationMethod(Enum):
    """校准映射函数类型。"""
    AFFINE = "affine"
    POLYNOMIAL = "polynomial"


class CalibrationModule:
    """视线校准模块。支持 affine 和 polynomial 两种映射函数。

    参数:
        num_points: 校准点数量，默认 9
        max_residual_px: 残差阈值（像素），超过则提示重新校准
        method: 校准方法，"affine" 或 "polynomial"
    """

    def __init__(
        self,
        num_points: int = 9,
        max_residual_px: float = 300.0,
        method: Literal["affine", "polynomial"] = "affine",
    ):
        self.num_points = num_points
        self.max_residual_px = max_residual_px
        self.method = CalibrationMethod(method)

        # 校准数据
        self._raw_points: list[tuple[float, float]] = []
        self._target_points: list[tuple[float, float]] = []

        # 拟合结果矩阵：affine 为 2×3，polynomial 为 2×6
        self._transform_matrix: np.ndarray | None = None
        self._residual_mean: float = 0.0
        self._calibrated: bool = False

        # 归一化参数（用于 polynomial 数值稳定性）
        self._norm_mean: np.ndarray = np.zeros(2)
        self._norm_std: np.ndarray = np.ones(2)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def residual_mean(self) -> float:
        return self._residual_mean

    @property
    def affine_matrix(self) -> np.ndarray | None:
        """向后兼容：返回变换矩阵。"""
        return self._transform_matrix

    @property
    def transform_matrix(self) -> np.ndarray | None:
        return self._transform_matrix

    def _build_feature_row(self, x: float, y: float) -> list[float]:
        """根据校准方法构建特征行。

        affine:     [x, y, 1]
        polynomial: [x, y, xy, x², y², 1]
        """
        if self.method == CalibrationMethod.POLYNOMIAL:
            return [x, y, x * y, x * x, y * y, 1.0]
        else:
            return [x, y, 1.0]

    def add_calibration_point(
        self,
        raw_gaze: tuple[float, float],
        screen_target: tuple[float, float],
    ) -> None:
        """添加一个校准数据点。

        参数:
            raw_gaze: 原始视线映射坐标 (x, y)
            screen_target: 屏幕真实目标坐标 (x, y)
        """
        self._raw_points.append(raw_gaze)
        self._target_points.append(screen_target)

    def clear_points(self) -> None:
        """清除所有校准数据点。"""
        self._raw_points.clear()
        self._target_points.clear()
        self._transform_matrix = None
        self._calibrated = False
        self._residual_mean = 0.0
        self._norm_mean = np.zeros(2)
        self._norm_std = np.ones(2)

    def calibrate(self) -> float:
        """执行校准，拟合映射函数。

        affine:     min_A Σ ||A × [xi, yi, 1]^T - target_i||²
        polynomial: min_W Σ ||W × [xi, yi, xi*yi, xi², yi², 1]^T - target_i||²

        对 raw 数据进行归一化预处理，避免 polynomial 特征的数值不稳定。

        特殊情况：
        - 1-2 个点 + affine → 退化为纯平移修正（取平均偏移量）

        返回:
            平均残差（像素）。若超过 max_residual_px 则建议重新校准。

        异常:
            ValueError: 校准点不足
        """
        n = len(self._raw_points)
        if n < 1:
            raise ValueError("校准点不足：需要至少 1 个")

        # 特殊情况：1-2 个点只做平移修正
        if n < 3 and self.method == CalibrationMethod.AFFINE:
            raw_arr = np.array(self._raw_points, dtype=np.float64)
            tgt_arr = np.array(self._target_points, dtype=np.float64)
            offset = np.mean(tgt_arr - raw_arr, axis=0)  # (2,)
            # 构造平移仿射矩阵: [[1, 0, dx], [0, 1, dy]]
            self._transform_matrix = np.array([
                [1.0, 0.0, offset[0]],
                [0.0, 1.0, offset[1]],
            ], dtype=np.float64)
            # 归一化参数：无归一化
            self._norm_mean = np.zeros(2)
            self._norm_std = np.ones(2)
            predicted = raw_arr + offset
            residuals = np.sqrt(np.sum((predicted - tgt_arr) ** 2, axis=1))
            self._residual_mean = float(np.mean(residuals))
            self._calibrated = True
            return self._residual_mean

        min_points = 6 if self.method == CalibrationMethod.POLYNOMIAL else 3
        if n < min_points:
            raise ValueError(
                f"校准点不足：{self.method.value} 方法需要至少 {min_points} 个，当前 {n} 个"
            )

        # 归一化 raw 数据（避免 polynomial 特征数值爆炸）
        raw_arr = np.array(self._raw_points, dtype=np.float64)
        self._norm_mean = raw_arr.mean(axis=0)
        self._norm_std = raw_arr.std(axis=0)
        # 防止除零
        self._norm_std[self._norm_std < 1e-8] = 1.0
        raw_normed = (raw_arr - self._norm_mean) / self._norm_std

        # 构建特征矩阵（使用归一化后的坐标）
        src = np.array(
            [self._build_feature_row(p[0], p[1]) for p in raw_normed],
            dtype=np.float64,
        )

        dst = np.array(self._target_points, dtype=np.float64)  # (N, 2)

        # 最小二乘求解
        result, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        self._transform_matrix = result.T  # affine: (2,3), polynomial: (2,6)

        # 计算残差
        predicted = src @ self._transform_matrix.T
        residuals = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
        self._residual_mean = float(np.mean(residuals))
        self._calibrated = True

        return self._residual_mean

    def apply(self, raw_gaze: tuple[float, float]) -> tuple[float, float]:
        """应用校准变换，将原始视线映射为校准后的屏幕坐标。

        参数:
            raw_gaze: 原始视线坐标 (x, y)

        返回:
            校准后的屏幕坐标 (x, y)
        """
        if not self._calibrated or self._transform_matrix is None:
            return raw_gaze

        # 归一化（与 calibrate 时一致）
        normed_x = (raw_gaze[0] - self._norm_mean[0]) / self._norm_std[0]
        normed_y = (raw_gaze[1] - self._norm_mean[1]) / self._norm_std[1]

        feat = np.array(
            self._build_feature_row(normed_x, normed_y),
            dtype=np.float64,
        )
        result = self._transform_matrix @ feat
        return (float(result[0]), float(result[1]))

    def needs_recalibration(self) -> bool:
        """判断是否需要重新校准（残差超过阈值）。"""
        return self._residual_mean > self.max_residual_px

    def get_calibration_data(self) -> dict:
        """获取校准数据，用于序列化。"""
        return {
            "method": self.method.value,
            "transform_matrix": (
                self._transform_matrix.tolist()
                if self._transform_matrix is not None
                else None
            ),
            # 向后兼容
            "affine_matrix": (
                self._transform_matrix.tolist()
                if self._transform_matrix is not None
                else None
            ),
            "norm_mean": self._norm_mean.tolist(),
            "norm_std": self._norm_std.tolist(),
            "calibration_points": [
                {"raw": list(r), "target": list(t)}
                for r, t in zip(self._raw_points, self._target_points)
            ],
            "residual_mean_px": self._residual_mean,
        }

    def set_calibration_data(self, data: dict) -> None:
        """从反序列化数据恢复校准状态。"""
        # 恢复方法类型
        method_str = data.get("method", "affine")
        self.method = CalibrationMethod(method_str)

        # 恢复变换矩阵（兼容旧格式）
        matrix = data.get("transform_matrix") or data.get("affine_matrix")
        if matrix is not None:
            self._transform_matrix = np.array(matrix, dtype=np.float64)
            self._calibrated = True
        else:
            self._transform_matrix = None
            self._calibrated = False

        # 恢复归一化参数
        self._norm_mean = np.array(data.get("norm_mean", [0.0, 0.0]), dtype=np.float64)
        self._norm_std = np.array(data.get("norm_std", [1.0, 1.0]), dtype=np.float64)

        self._residual_mean = data.get("residual_mean_px", 0.0)

        self._raw_points = []
        self._target_points = []
        for pt in data.get("calibration_points", []):
            self._raw_points.append(tuple(pt["raw"]))
            self._target_points.append(tuple(pt["target"]))
