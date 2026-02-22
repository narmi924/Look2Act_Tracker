"""屏幕几何建模模块。

描述屏幕平面在摄像头坐标系中的位置，
提供射线-平面求交和 3D 交点到屏幕像素坐标的转换。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src.geometry.ray_plane import ray_plane_intersect


class ScreenGeometry:
    """屏幕几何模型。描述屏幕平面在摄像头坐标系中的位置。

    参数:
        screen_w_px: 屏幕宽度（像素）
        screen_h_px: 屏幕高度（像素）
        screen_w_mm: 屏幕物理宽度（毫米）
        screen_h_mm: 屏幕物理高度（毫米）
        camera_matrix: 3×3 相机内参矩阵
    """

    def __init__(
        self,
        screen_w_px: int,
        screen_h_px: int,
        screen_w_mm: float,
        screen_h_mm: float,
        camera_matrix: Optional[np.ndarray] = None,
    ):
        self.screen_w_px = screen_w_px
        self.screen_h_px = screen_h_px
        self.screen_w_mm = screen_w_mm
        self.screen_h_mm = screen_h_mm
        self.camera_matrix = camera_matrix

        # 屏幕平面参数（需调用 setup_plane 设置）
        self._plane_normal: Optional[np.ndarray] = None
        self._plane_origin: Optional[np.ndarray] = None
        self._screen_x_axis: Optional[np.ndarray] = None
        self._screen_y_axis: Optional[np.ndarray] = None

    def setup_plane(
        self,
        screen_origin_mm: np.ndarray,
        screen_normal: np.ndarray,
        screen_x_axis: Optional[np.ndarray] = None,
        screen_y_axis: Optional[np.ndarray] = None,
    ) -> None:
        """设置屏幕平面参数。

        参数:
            screen_origin_mm: 屏幕左上角在摄像头坐标系中的 3D 位置（mm）
            screen_normal: 屏幕平面法向量（指向摄像头方向）
            screen_x_axis: 屏幕 X 轴方向（向右），默认 [1,0,0]
            screen_y_axis: 屏幕 Y 轴方向（向下），默认 [0,1,0]
        """
        self._plane_origin = np.asarray(screen_origin_mm, dtype=np.float64).flatten()
        self._plane_normal = np.asarray(screen_normal, dtype=np.float64).flatten()

        # 归一化法向量
        n = np.linalg.norm(self._plane_normal)
        if n > 1e-12:
            self._plane_normal = self._plane_normal / n

        if screen_x_axis is not None:
            self._screen_x_axis = np.asarray(screen_x_axis, dtype=np.float64).flatten()
        else:
            self._screen_x_axis = np.array([1.0, 0.0, 0.0])

        if screen_y_axis is not None:
            self._screen_y_axis = np.asarray(screen_y_axis, dtype=np.float64).flatten()
        else:
            self._screen_y_axis = np.array([0.0, 1.0, 0.0])

    def ray_plane_intersect(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
    ) -> Optional[np.ndarray]:
        """射线-平面求交。返回交点的 3D 坐标，无交点返回 None。"""
        if self._plane_normal is None or self._plane_origin is None:
            return None

        return ray_plane_intersect(
            ray_origin, ray_direction,
            self._plane_origin, self._plane_normal,
        )

    def world_to_screen_px(
        self,
        point_3d: np.ndarray,
        clamp: bool = True,
    ) -> tuple[float, float]:
        """将 3D 交点转换为屏幕像素坐标 (px, py)。

        通过将交点投影到屏幕局部坐标系（X/Y 轴），
        再按物理尺寸与像素尺寸的比例转换。

        参数:
            point_3d: 3D 交点坐标（mm）
            clamp: 是否将结果 clamp 到屏幕范围内

        返回:
            (px, py) 屏幕像素坐标
        """
        if self._plane_origin is None:
            return 0.0, 0.0

        p = np.asarray(point_3d, dtype=np.float64).flatten()
        delta = p - self._plane_origin

        # 投影到屏幕局部坐标系
        local_x_mm = np.dot(delta, self._screen_x_axis)
        local_y_mm = np.dot(delta, self._screen_y_axis)

        # 物理坐标 → 像素坐标
        px = local_x_mm / self.screen_w_mm * self.screen_w_px
        py = local_y_mm / self.screen_h_mm * self.screen_h_px

        if clamp:
            px = float(np.clip(px, 0, self.screen_w_px - 1))
            py = float(np.clip(py, 0, self.screen_h_px - 1))

        return float(px), float(py)

    def get_gaze_point(
        self,
        ray_origin: np.ndarray,
        ray_direction: np.ndarray,
        clamp_to_screen: bool = True,
    ) -> Optional[tuple[float, float]]:
        """完整流程：射线求交 → 转换为屏幕像素坐标。

        参数:
            ray_origin: 射线起点
            ray_direction: 射线方向
            clamp_to_screen: 交点在屏幕外时是否 clamp 到边缘

        返回:
            (px, py) 屏幕像素坐标，无交点返回 None
        """
        intersection = self.ray_plane_intersect(ray_origin, ray_direction)
        if intersection is None:
            return None

        return self.world_to_screen_px(intersection, clamp=clamp_to_screen)
