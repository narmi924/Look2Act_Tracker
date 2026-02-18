"""坐标系转换模块。

将视线方向从眼球局部坐标系转换到摄像头坐标系。
"""
from __future__ import annotations

import numpy as np


def transform_gaze_to_camera(
    gaze_vector: np.ndarray,
    rotation_matrix: np.ndarray,
    translation_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """将视线方向从眼球局部坐标系转换到摄像头坐标系。

    使用旋转矩阵将视线方向旋转到摄像头坐标系，
    使用平移向量作为射线起点（眼球在摄像头坐标系中的位置）。

    参数:
        gaze_vector: 眼球局部坐标系中的视线方向向量 (3,)
        rotation_matrix: 3×3 旋转矩阵（从眼球坐标系到摄像头坐标系）
        translation_vec: 3×1 或 (3,) 平移向量（眼球在摄像头坐标系中的位置）

    返回:
        (ray_origin, ray_direction) 在摄像头坐标系中的射线
        ray_origin: (3,) 射线起点
        ray_direction: (3,) 射线方向（单位向量）
    """
    gaze = np.asarray(gaze_vector, dtype=np.float64).flatten()
    R = np.asarray(rotation_matrix, dtype=np.float64)
    t = np.asarray(translation_vec, dtype=np.float64).flatten()

    # 旋转视线方向到摄像头坐标系
    ray_direction = R @ gaze

    # 归一化为单位向量
    norm = np.linalg.norm(ray_direction)
    if norm > 1e-12:
        ray_direction = ray_direction / norm

    # 射线起点为平移向量（眼球位置）
    ray_origin = t.copy()

    return ray_origin, ray_direction
