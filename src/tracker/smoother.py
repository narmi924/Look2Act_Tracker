"""指数移动平均（EMA）时序平滑滤波器。

用于对连续帧的注视点进行平滑，减少抖动。
"""
from __future__ import annotations

from typing import Optional


class GazeSmoother:
    """指数移动平均（EMA）时序平滑滤波器。

    参数:
        alpha: 平滑系数 (0, 1]，越小越平滑。
               alpha=1 表示不平滑（直接使用新值）。
    """

    def __init__(self, alpha: float = 0.3):
        assert 0.0 < alpha <= 1.0, f"alpha 必须在 (0, 1] 范围内，当前为 {alpha}"
        self.alpha = alpha
        self._prev: Optional[tuple[float, float]] = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        """输入新的注视点，返回平滑后的注视点。

        EMA 公式：smoothed = alpha * new + (1 - alpha) * prev
        """
        if self._prev is None:
            self._prev = point
            return point

        sx = self.alpha * point[0] + (1.0 - self.alpha) * self._prev[0]
        sy = self.alpha * point[1] + (1.0 - self.alpha) * self._prev[1]
        self._prev = (sx, sy)
        return (sx, sy)

    def reset(self) -> None:
        """重置滤波器状态。"""
        self._prev = None
