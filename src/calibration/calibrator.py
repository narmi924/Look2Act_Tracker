"""视线校准模块。

通过用户注视已知屏幕点，使用最小二乘法拟合仿射变换矩阵，
将原始视线映射修正为校准后的屏幕坐标。

仿射变换：[px_cal, py_cal]^T = A × [px_raw, py_raw, 1]^T
其中 A 是 2×3 仿射矩阵。
"""
from __future__ import annotations

import numpy as np


class CalibrationModule:
    """视线校准模块。使用最小二乘法拟合仿射变换。

    参数:
        num_points: 校准点数量，默认 9
        max_residual_px: 残差阈值（像素），超过则提示重新校准
    """

    def __init__(self, num_points: int = 9, max_residual_px: float = 50.0):
        self.num_points = num_points
        self.max_residual_px = max_residual_px

        # 校准数据
        self._raw_points: list[tuple[float, float]] = []
        self._target_points: list[tuple[float, float]] = []

        # 拟合结果：2×3 仿射矩阵
        self._affine_matrix: np.ndarray | None = None
        self._residual_mean: float = 0.0
        self._calibrated: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def residual_mean(self) -> float:
        return self._residual_mean

    @property
    def affine_matrix(self) -> np.ndarray | None:
        return self._affine_matrix

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
        self._affine_matrix = None
        self._calibrated = False
        self._residual_mean = 0.0

    def calibrate(self) -> float:
        """执行校准，拟合仿射变换矩阵。

        使用最小二乘法求解：
        min_A Σ ||A × [xi, yi, 1]^T - [xi_true, yi_true]^T||²

        返回:
            平均残差（像素）。若超过 max_residual_px 则建议重新校准。

        异常:
            ValueError: 校准点不足
        """
        n = len(self._raw_points)
        if n < 3:
            raise ValueError(f"校准点不足：需要至少 3 个，当前 {n} 个")

        # 构建矩阵 [x, y, 1]
        src = np.array(
            [[p[0], p[1], 1.0] for p in self._raw_points],
            dtype=np.float64,
        )  # (N, 3)

        dst = np.array(self._target_points, dtype=np.float64)  # (N, 2)

        # 最小二乘求解：dst = src @ A^T  =>  A^T = lstsq(src, dst)
        result, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
        self._affine_matrix = result.T  # (2, 3)

        # 计算残差
        predicted = src @ self._affine_matrix.T  # (N, 2)
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
        if not self._calibrated or self._affine_matrix is None:
            return raw_gaze

        pt = np.array([raw_gaze[0], raw_gaze[1], 1.0], dtype=np.float64)
        result = self._affine_matrix @ pt
        return (float(result[0]), float(result[1]))

    def needs_recalibration(self) -> bool:
        """判断是否需要重新校准（残差超过阈值）。"""
        return self._residual_mean > self.max_residual_px

    def get_calibration_data(self) -> dict:
        """获取校准数据，用于序列化。"""
        return {
            "affine_matrix": self._affine_matrix.tolist() if self._affine_matrix is not None else None,
            "calibration_points": [
                {"raw": list(r), "target": list(t)}
                for r, t in zip(self._raw_points, self._target_points)
            ],
            "residual_mean_px": self._residual_mean,
        }

    def set_calibration_data(self, data: dict) -> None:
        """从反序列化数据恢复校准状态。"""
        matrix = data.get("affine_matrix")
        if matrix is not None:
            self._affine_matrix = np.array(matrix, dtype=np.float64)
            self._calibrated = True
        else:
            self._affine_matrix = None
            self._calibrated = False

        self._residual_mean = data.get("residual_mean_px", 0.0)

        self._raw_points = []
        self._target_points = []
        for pt in data.get("calibration_points", []):
            self._raw_points.append(tuple(pt["raw"]))
            self._target_points.append(tuple(pt["target"]))
