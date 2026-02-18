"""射线-平面求交算法。

给定射线 P(t) = O + t*D 和平面 N·(P - P0) = 0：
  t = N·(P0 - O) / (N·D)
  若 |N·D| < epsilon，视线与屏幕平行，无交点
  若 t < 0，交点在射线反方向，无效
  交点 P = O + t*D
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def ray_plane_intersect(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    epsilon: float = 1e-8,
) -> Optional[np.ndarray]:
    """计算射线与平面的交点。

    参数:
        ray_origin: 射线起点 (3,)
        ray_direction: 射线方向 (3,)，不要求单位向量
        plane_point: 平面上一点 (3,)
        plane_normal: 平面法向量 (3,)，不要求单位向量
        epsilon: 平行判定阈值

    返回:
        交点 3D 坐标 (3,)，无交点时返回 None
    """
    O = np.asarray(ray_origin, dtype=np.float64).flatten()
    D = np.asarray(ray_direction, dtype=np.float64).flatten()
    P0 = np.asarray(plane_point, dtype=np.float64).flatten()
    N = np.asarray(plane_normal, dtype=np.float64).flatten()

    denom = np.dot(N, D)

    # 射线与平面平行
    if abs(denom) < epsilon:
        return None

    t = np.dot(N, P0 - O) / denom

    # 交点在射线反方向
    if t < 0:
        return None

    return O + t * D
