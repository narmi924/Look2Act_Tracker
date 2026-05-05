"""Classic image-processing gaze backend.

This module provides the Eye_Touch-style runtime path used for smooth
coarse gaze interaction: extract a compact pupil/iris feature from eye
crops, map it with user calibration, and smooth the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class ClassicGazeFeature:
    """Normalized two-dimensional eye feature used as calibration input."""

    x: float
    y: float
    confidence: float
    method: str

    @property
    def point(self) -> tuple[float, float]:
        return (self.x, self.y)


def detect_pupil_centroid(eye_bgr: np.ndarray) -> Optional[tuple[float, float]]:
    """Detect the dark pupil centroid in a cropped eye image.

    Returns normalized coordinates in the crop, or ``None`` if the crop is
    empty. A weighted dark-pixel centroid is used as a robust fallback.
    """
    if eye_bgr is None or eye_bgr.size == 0:
        return None

    gray = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2GRAY)
    k = max(3, int(min(gray.shape[:2]) / 8) | 1)
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    try:
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    except cv2.error:
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    mk = max(3, int(min(gray.shape[:2]) / 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= 10:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                return normalize_crop_point(cx, cy, gray.shape[1], gray.shape[0])

    inv = 255.0 - gray.astype(np.float32)
    weight_sum = float(np.sum(inv))
    if weight_sum <= 1e-6:
        return None

    yy, xx = np.indices(gray.shape)
    cx = float(np.sum(xx * inv) / weight_sum)
    cy = float(np.sum(yy * inv) / weight_sum)
    return normalize_crop_point(cx, cy, gray.shape[1], gray.shape[0])


def normalize_crop_point(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Normalize a crop coordinate to [0, 1]."""
    if width <= 1 or height <= 1:
        return (0.5, 0.5)
    nx = float(np.clip(x / float(width - 1), 0.0, 1.0))
    ny = float(np.clip(y / float(height - 1), 0.0, 1.0))
    return (nx, ny)


def fuse_eye_features(
    left_norm: Optional[tuple[float, float]],
    right_norm: Optional[tuple[float, float]],
    method: str = "pupil_centroid",
) -> Optional[ClassicGazeFeature]:
    """Fuse normalized left/right crop features into one calibration point."""
    points = [p for p in (left_norm, right_norm) if p is not None]
    if not points:
        return None

    arr = np.array(points, dtype=np.float64)
    mean = arr.mean(axis=0)
    confidence = 0.65 if len(points) == 1 else 1.0
    return ClassicGazeFeature(
        x=float(mean[0]),
        y=float(mean[1]),
        confidence=confidence,
        method=method,
    )


def normalize_iris_offset(
    iris_center: Optional[tuple[float, float]],
    eye_center: Optional[tuple[float, float]],
    eye_width: Optional[float],
) -> Optional[tuple[float, float]]:
    """Normalize an iris center relative to eye center and eye width.

    The result is centered near 0.5/0.5 so it can be calibrated with the
    same 2D mapping as crop-based pupil features.
    """
    if iris_center is None or eye_center is None or eye_width is None or eye_width <= 1e-6:
        return None
    dx = (iris_center[0] - eye_center[0]) / eye_width
    dy = (iris_center[1] - eye_center[1]) / eye_width
    return (float(np.clip(0.5 + dx, 0.0, 1.0)), float(np.clip(0.5 + dy, 0.0, 1.0)))


class ClassicKalmanSmoother:
    """Small 2D constant-velocity Kalman smoother for gaze points."""

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
            self._initialized = True
        self._kf.correct(measurement)
        prediction = self._kf.predict()
        return (float(prediction[0, 0]), float(prediction[1, 0]))

    def reset(self) -> None:
        self._initialized = False
