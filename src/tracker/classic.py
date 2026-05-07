"""Classic 图像处理视线后端。

该链路使用 MediaPipe 提取眼部 ROI，再通过暗色瞳孔质心、相机归一化特征、
多项式校准和屏幕空间平滑完成注视点估计。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassicGazeFeature:
    """用于校准输入的相机归一化二维特征。"""

    x: float
    y: float
    confidence: float
    method: str = "classic_pupil"

    @property
    def point(self) -> tuple[float, float]:
        return (self.x, self.y)


def detect_pupil_centroid(eye_roi: np.ndarray) -> Optional[tuple[float, float]]:
    """返回眼部 ROI 内的暗色瞳孔质心坐标。

    先使用 Otsu 反向阈值和轮廓矩定位；当轮廓不可用时，使用暗像素加权质心作为
    备用估计。
    """
    if eye_roi is None or eye_roi.size == 0:
        return None

    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    k = max(3, int(min(eye_roi.shape[:2]) / 8) | 1)
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except cv2.error:
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    mk = max(3, int(min(eye_roi.shape[:2]) / 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= 10:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = float(moments["m10"] / moments["m00"])
                cy = float(moments["m01"] / moments["m00"])
                return (cx, cy)

    inv = 255.0 - gray.astype(np.float32)
    weight_sum = float(np.sum(inv)) + 1e-6
    yy, xx = np.indices(gray.shape)
    cx = float(np.sum(xx * inv) / weight_sum)
    cy = float(np.sum(yy * inv) / weight_sum)
    h, w = gray.shape
    if 0.0 <= cx < w and 0.0 <= cy < h:
        return (cx, cy)
    return None


def normalize_camera_point(
    point: tuple[float, float],
    camera_width: int,
    camera_height: int,
) -> tuple[float, float]:
    """按当前帧尺寸归一化相机坐标。"""
    width = max(int(camera_width), 1)
    height = max(int(camera_height), 1)
    return (
        float(np.clip(point[0] / float(width), 0.0, 1.0)),
        float(np.clip(point[1] / float(height), 0.0, 1.0)),
    )


def absolute_pupil_point(
    pupil: Optional[tuple[float, float]],
    roi_origin: Optional[tuple[int, int]],
) -> Optional[tuple[float, float]]:
    """将 ROI 内的瞳孔坐标转换为原始相机坐标。"""
    if pupil is None or roi_origin is None:
        return None
    return (float(roi_origin[0] + pupil[0]), float(roi_origin[1] + pupil[1]))


def fuse_eye_features(
    left_point: Optional[tuple[float, float]],
    right_point: Optional[tuple[float, float]],
    camera_width: int = 1,
    camera_height: int = 1,
    method: str = "classic_pupil",
) -> Optional[ClassicGazeFeature]:
    """融合可用的左右眼绝对坐标，并按相机尺寸归一化。"""
    points = [p for p in (left_point, right_point) if p is not None]
    if not points:
        return None

    arr = np.array(points, dtype=np.float64)
    mean = arr.mean(axis=0)
    norm_x, norm_y = normalize_camera_point(
        (float(mean[0]), float(mean[1])),
        camera_width,
        camera_height,
    )
    confidence = 0.65 if len(points) == 1 else 1.0
    return ClassicGazeFeature(norm_x, norm_y, confidence, method)


def normalize_crop_point(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """兼容旧实验数据的 ROI 归一化工具。"""
    if width <= 1 or height <= 1:
        return (0.5, 0.5)
    nx = float(np.clip(x / float(width - 1), 0.0, 1.0))
    ny = float(np.clip(y / float(height - 1), 0.0, 1.0))
    return (nx, ny)


def normalize_iris_offset(
    iris_center: Optional[tuple[float, float]],
    eye_center: Optional[tuple[float, float]],
    eye_width: Optional[float],
) -> Optional[tuple[float, float]]:
    """兼容旧实验数据的虹膜偏移归一化工具。"""
    if iris_center is None or eye_center is None or eye_width is None or eye_width <= 1e-6:
        return None
    dx = (iris_center[0] - eye_center[0]) / eye_width
    dy = (iris_center[1] - eye_center[1]) / eye_width
    return (float(np.clip(0.5 + dx, 0.0, 1.0)), float(np.clip(0.5 + dy, 0.0, 1.0)))


class ClassicKalmanSmoother:
    """屏幕空间常速度 Kalman 平滑器。"""

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._kf = cv2.KalmanFilter(4, 2)
        self._kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32
        )
        self._kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self._kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
        self._initialized = False

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        if not self._initialized:
            self._kf.statePost = np.array(
                [[measurement[0, 0]], [measurement[1, 0]], [0.0], [0.0]],
                dtype=np.float32,
            )
            self._kf.statePre = self._kf.statePost.copy()
            self._initialized = True
            return (float(measurement[0, 0]), float(measurement[1, 0]))
        self._kf.correct(measurement)
        prediction = self._kf.predict()
        return (float(prediction[0, 0]), float(prediction[1, 0]))

    def reset(self) -> None:
        self._initialized = False


class ClassicScreenSmoother:
    """Classic 后端的屏幕空间平滑器：Kalman 预测后叠加历史均值。"""

    def __init__(self, history_len: int = 60):
        self.kalman = ClassicKalmanSmoother()
        self.history: deque[tuple[float, float]] = deque(maxlen=history_len)
        self.current: Optional[tuple[float, float]] = None

    def reset(self) -> None:
        self.kalman.reset()
        self.history.clear()
        self.current = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        predicted = self.kalman.update(point)
        self.history.append(predicted)
        avg_x = sum(p[0] for p in self.history) / len(self.history)
        avg_y = sum(p[1] for p in self.history) / len(self.history)
        self.current = (float(avg_x), float(avg_y))
        return self.current
