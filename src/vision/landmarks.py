"""Pure-numpy landmark helpers, importable without the MediaPipe native module."""
import numpy as np


def landmarks_array(lms, w: int, h: int) -> np.ndarray:
    """把 MediaPipe 归一化关键点转成像素坐标数组 (N, 3)；z 按 MediaPipe 约定乘以图像宽度。"""
    return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in lms], dtype=np.float32)
