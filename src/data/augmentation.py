"""
数据增强模块：对眼部图像和视线标签进行同步增强。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env

增强策略：
1. 随机水平翻转（同步翻转 gaze_x 取反）
2. 随机亮度抖动
3. 随机轻微旋转

关键约束：增强后视线标签必须保持一致性（单位向量、方向正确）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class AugmentedSample:
    """增强后的样本。"""
    eye_img: np.ndarray          # (128, 128, 3) BGR
    gaze_x: float
    gaze_y: float
    gaze_z: float


def horizontal_flip(
    eye_img: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    gaze_z: float,
) -> AugmentedSample:
    """水平翻转眼部图像，同步翻转视线标签。

    翻转规则：
    - 图像左右镜像
    - gaze_x 取反（水平方向反转）
    - gaze_y, gaze_z 不变

    Args:
        eye_img: (H, W, 3) BGR 眼部图像
        gaze_x, gaze_y, gaze_z: 3D 视线方向单位向量分量

    Returns:
        AugmentedSample，包含翻转后的图像和标签
    """
    flipped_img = cv2.flip(eye_img, 1)  # 1 = 水平翻转
    return AugmentedSample(
        eye_img=flipped_img,
        gaze_x=-gaze_x,
        gaze_y=gaze_y,
        gaze_z=gaze_z,
    )


def brightness_jitter(
    eye_img: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    gaze_z: float,
    delta_range: tuple[float, float] = (-30.0, 30.0),
    rng: Optional[np.random.RandomState] = None,
) -> AugmentedSample:
    """随机亮度抖动。视线标签不变。

    Args:
        eye_img: (H, W, 3) BGR 眼部图像
        gaze_x, gaze_y, gaze_z: 3D 视线方向单位向量分量
        delta_range: 亮度偏移范围（像素值）
        rng: 随机数生成器，None 时使用默认

    Returns:
        AugmentedSample
    """
    if rng is None:
        rng = np.random.RandomState()

    delta = rng.uniform(delta_range[0], delta_range[1])
    jittered = np.clip(eye_img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return AugmentedSample(
        eye_img=jittered,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        gaze_z=gaze_z,
    )


def slight_rotation(
    eye_img: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    gaze_z: float,
    max_angle: float = 5.0,
    rng: Optional[np.random.RandomState] = None,
) -> AugmentedSample:
    """随机轻微旋转图像。视线标签不变（角度很小，影响可忽略）。

    Args:
        eye_img: (H, W, 3) BGR 眼部图像
        gaze_x, gaze_y, gaze_z: 3D 视线方向单位向量分量
        max_angle: 最大旋转角度（度）
        rng: 随机数生成器

    Returns:
        AugmentedSample
    """
    if rng is None:
        rng = np.random.RandomState()

    angle = rng.uniform(-max_angle, max_angle)
    h, w = eye_img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        eye_img, M, (w, h),
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return AugmentedSample(
        eye_img=rotated,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        gaze_z=gaze_z,
    )


def augment_sample(
    eye_img: np.ndarray,
    gaze_x: float,
    gaze_y: float,
    gaze_z: float,
    flip_prob: float = 0.5,
    brightness_prob: float = 0.5,
    rotation_prob: float = 0.3,
    rng: Optional[np.random.RandomState] = None,
) -> AugmentedSample:
    """对单个样本应用随机数据增强组合。

    按概率依次应用：水平翻转 → 亮度抖动 → 轻微旋转。

    Args:
        eye_img: (H, W, 3) BGR 眼部图像
        gaze_x, gaze_y, gaze_z: 3D 视线方向单位向量分量
        flip_prob: 水平翻转概率
        brightness_prob: 亮度抖动概率
        rotation_prob: 轻微旋转概率
        rng: 随机数生成器

    Returns:
        AugmentedSample，增强后的样本
    """
    if rng is None:
        rng = np.random.RandomState()

    img = eye_img.copy()
    gx, gy, gz = gaze_x, gaze_y, gaze_z

    # 水平翻转
    if rng.random() < flip_prob:
        result = horizontal_flip(img, gx, gy, gz)
        img, gx, gy, gz = result.eye_img, result.gaze_x, result.gaze_y, result.gaze_z

    # 亮度抖动
    if rng.random() < brightness_prob:
        result = brightness_jitter(img, gx, gy, gz, rng=rng)
        img = result.eye_img

    # 轻微旋转
    if rng.random() < rotation_prob:
        result = slight_rotation(img, gx, gy, gz, rng=rng)
        img = result.eye_img

    return AugmentedSample(eye_img=img, gaze_x=gx, gaze_y=gy, gaze_z=gz)
