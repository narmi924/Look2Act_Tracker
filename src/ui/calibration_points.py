"""Shared calibration target geometry."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CalibrationPoint:
    """校准点数据结构。"""
    x: float  # 屏幕像素坐标 X
    y: float  # 屏幕像素坐标 Y
    index: int  # 点序号


def generate_calibration_points_for_screen(
    screen_w: int,
    screen_h: int,
    *,
    num_points: int,
    margin_ratio: float = 0.1,
) -> list[CalibrationPoint]:
    """生成指定屏幕尺寸下的校准点。"""
    margin_x = screen_w * margin_ratio
    margin_y = screen_h * margin_ratio

    effective_w = screen_w - 2 * margin_x
    effective_h = screen_h - 2 * margin_y

    grid_size = 5 if num_points >= 25 else 3
    denom = max(grid_size - 1, 1)

    points: list[CalibrationPoint] = []
    index = 0
    for row in range(grid_size):
        for col in range(grid_size):
            x = margin_x + col * effective_w / denom
            y = margin_y + row * effective_h / denom
            points.append(CalibrationPoint(x=x, y=y, index=index))
            index += 1
    return points
