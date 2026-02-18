"""头部姿态估计模块（基于 PnP 算法）。

从 6 个面部关键点计算头部的旋转矩阵和欧拉角（yaw/pitch/roll）。
复用 Gaze_Dataset_Collector_Project 中的 PnP 逻辑。

输入：6 个 PnP 关键点像素坐标（nose_tip, chin, left_eye_outer,
     right_eye_outer, left_mouth, right_mouth）
输出：yaw/pitch/roll（度）、3×3 旋转矩阵、3×1 平移向量
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class HeadPoseResult:
    """头部姿态估计结果。"""
    valid: bool
    yaw: float       # 度
    pitch: float     # 度
    roll: float      # 度
    rotation_matrix: Optional[np.ndarray]  # 3×3 旋转矩阵
    translation_vec: Optional[np.ndarray]  # 3×1 平移向量


# PnP 所需的 6 个关键点名称
_REQUIRED_KEYS = (
    "nose_tip", "chin",
    "left_eye_outer", "right_eye_outer",
    "left_mouth", "right_mouth",
)

# 简化 3D 人脸模型点（单位：mm），鼻尖为原点
_MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # nose_tip
    (0.0, -330.0, -65.0),     # chin
    (-225.0, 170.0, -135.0),  # left_eye_outer
    (225.0, 170.0, -135.0),   # right_eye_outer
    (-150.0, -150.0, -125.0), # left_mouth
    (150.0, -150.0, -125.0),  # right_mouth
], dtype=np.float64)


def _invalid_result() -> HeadPoseResult:
    """返回无效的头姿结果。"""
    return HeadPoseResult(
        valid=False, yaw=0.0, pitch=0.0, roll=0.0,
        rotation_matrix=None, translation_vec=None,
    )


class HeadPoseEstimator:
    """头部姿态估计器。基于 solvePnP 算法。

    参数:
        frame_size: 图像尺寸 (width, height)，用于计算相机内参
    """

    def __init__(self, frame_size: tuple[int, int]):
        w, h = frame_size
        # 相机内参：焦距近似为图像宽度
        focal_length = float(w)
        center = (w / 2.0, h / 2.0)
        self.camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def estimate(self, pnp_points_2d: dict[str, tuple[float, float]]) -> HeadPoseResult:
        """从 6 个面部关键点估计头部姿态。

        参数:
            pnp_points_2d: 6 个关键点像素坐标字典

        返回:
            HeadPoseResult，包含 yaw/pitch/roll 和旋转矩阵
        """
        # 检查关键点完整性
        if any(k not in pnp_points_2d for k in _REQUIRED_KEYS):
            return _invalid_result()

        image_points = np.array(
            [pnp_points_2d[k] for k in _REQUIRED_KEYS],
            dtype=np.float64,
        )

        # PnP 求解
        ok, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS_3D,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return _invalid_result()

        # 旋转向量 → 旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # 旋转矩阵 → 欧拉角（度）
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        return HeadPoseResult(
            valid=True,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation_matrix=rotation_matrix,
            translation_vec=tvec,
        )
