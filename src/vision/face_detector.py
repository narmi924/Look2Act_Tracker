"""人脸检测与眼部区域裁剪模块。

基于 MediaPipe FaceMesh，封装人脸检测、关键点提取、眼部裁剪功能。
实时推理和训练预处理共用同一套关键点与眼部裁剪规则。

主要功能：
1. 检测人脸边界框和面部关键点
2. 裁剪左右眼区域图像，输出固定 128×128 尺寸
3. 多人脸时选择面积最大的人脸
4. 未检测到人脸时返回 confidence=0 的空结果
5. 提取 6 个 PnP 关键点用于头部姿态估计
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import cv2


@dataclass
class FaceDetectionResult:
    """人脸检测结果。"""
    detected: bool
    confidence: float
    face_bbox: Optional[tuple[int, int, int, int]]  # (x, y, w, h)
    landmarks_68: Optional[np.ndarray]  # (68, 2) 像素坐标（从 468 点中选取）
    left_eye_crop: Optional[np.ndarray]  # (128, 128, 3) BGR
    right_eye_crop: Optional[np.ndarray]  # (128, 128, 3) BGR
    pnp_points_2d: dict[str, tuple[float, float]]  # 6 个 PnP 关键点
    # 虹膜特征（用于校准）
    left_iris_center: Optional[tuple[float, float]] = None   # 左虹膜中心像素坐标
    right_iris_center: Optional[tuple[float, float]] = None  # 右虹膜中心像素坐标
    left_eye_center: Optional[tuple[float, float]] = None    # 左眼眶中心像素坐标
    right_eye_center: Optional[tuple[float, float]] = None   # 右眼眶中心像素坐标
    left_eye_width: Optional[float] = None
    right_eye_width: Optional[float] = None
    # Classic 后端需要未缩放的眼部 ROI 及其在原始画面中的左上角坐标。
    left_eye_roi: Optional[np.ndarray] = None
    right_eye_roi: Optional[np.ndarray] = None
    left_eye_origin: Optional[tuple[int, int]] = None
    right_eye_origin: Optional[tuple[int, int]] = None
    frame_size: Optional[tuple[int, int]] = None


# MediaPipe 468 点中对应传统 68 点的近似映射索引
# 选取关键的 68 个点覆盖脸部轮廓、眉毛、眼睛、鼻子、嘴巴
_MP_TO_68_INDICES = [
    # 脸部轮廓 (17 点)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    # 左眉 (5 点)
    70, 63, 105, 66, 107,
    # 右眉 (5 点)
    336, 296, 334, 293, 300,
    # 鼻梁 (4 点)
    168, 6, 197, 195,
    # 鼻底 (5 点)
    5, 4, 1, 275, 281,
    # 左眼 (6 点)
    33, 160, 158, 133, 153, 144,
    # 右眼 (6 点)
    362, 385, 387, 263, 373, 380,
    # 外嘴唇 (12 点)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    # 内嘴唇 (8 点)
    78, 82, 13, 312, 308, 317, 14, 87,
]

# PnP 头姿估计用的 6 个关键点索引
_PNP_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# 左右眼轮廓关键点索引（用于计算眼部裁剪区域）
_LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155]
_RIGHT_EYE_INDICES = [263, 362, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382]

_EYETOUCH_LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 155, 154, 153, 145, 144, 163, 7]
_EYETOUCH_RIGHT_EYE_INDICES = [362, 263, 387, 386, 385, 384, 398, 382, 381, 380, 373, 374, 390, 249]

# 虹膜关键点索引（refine_landmarks=True 时可用，共 10 个点）
# 左虹膜：468（中心），469-472（周围 4 点）
# 右虹膜：473（中心），474-477（周围 4 点）
_LEFT_IRIS_CENTER_IDX = 468
_RIGHT_IRIS_CENTER_IDX = 473

# 眼眶角点索引（用于计算眼眶中心，作为虹膜偏移的参考）
_LEFT_EYE_INNER_IDX = 133   # 左眼内眼角
_LEFT_EYE_OUTER_IDX = 33    # 左眼外眼角
_RIGHT_EYE_INNER_IDX = 362  # 右眼内眼角
_RIGHT_EYE_OUTER_IDX = 263  # 右眼外眼角


def _empty_result() -> FaceDetectionResult:
    """返回未检测到人脸的空结果。"""
    return FaceDetectionResult(
        detected=False,
        confidence=0.0,
        face_bbox=None,
        landmarks_68=None,
        left_eye_crop=None,
        right_eye_crop=None,
        pnp_points_2d={},
        left_iris_center=None,
        right_iris_center=None,
        left_eye_center=None,
        right_eye_center=None,
        left_eye_width=None,
        right_eye_width=None,
        left_eye_roi=None,
        right_eye_roi=None,
        left_eye_origin=None,
        right_eye_origin=None,
        frame_size=None,
    )


class FaceDetector:
    """人脸检测与眼部区域裁剪。基于 MediaPipe FaceMesh。

    参数:
        eye_crop_size: 眼部裁剪输出尺寸，默认 128
        max_num_faces: 最多检测人脸数，默认 3（多人脸时选最大）
        min_detection_confidence: 检测置信度阈值
        min_tracking_confidence: 跟踪置信度阈值
    """

    def __init__(
        self,
        eye_crop_size: int = 128,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        refine_landmarks: bool = False,
    ):
        self.eye_crop_size = eye_crop_size
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        """释放 MediaPipe 资源。"""
        self._mesh.close()

    def detect(self, frame_bgr: np.ndarray) -> FaceDetectionResult:
        """检测人脸，返回关键点、眼部裁剪图、人脸边界框。

        多人脸时选择面积最大的人脸。
        未检测到人脸时返回 confidence=0 的空结果。

        参数:
            frame_bgr: BGR 格式输入图像

        返回:
            FaceDetectionResult 检测结果
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return _empty_result()

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return _empty_result()

        # 多人脸时选择面积最大的
        best_lms = None
        best_area = -1.0

        for face_lms in result.multi_face_landmarks:
            lms = face_lms.landmark
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area > best_area:
                best_area = area
                best_lms = lms

        if best_lms is None:
            return _empty_result()

        # 计算人脸边界框
        xs = [lm.x * w for lm in best_lms]
        ys = [lm.y * h for lm in best_lms]
        x0 = int(np.clip(min(xs), 0, w - 1))
        y0 = int(np.clip(min(ys), 0, h - 1))
        x1 = int(np.clip(max(xs), 0, w - 1))
        y1 = int(np.clip(max(ys), 0, h - 1))
        face_bbox = (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

        # 置信度：使用人脸面积占比作为近似置信度
        confidence = min(1.0, best_area * 4.0)

        # 提取 68 个关键点
        landmarks_68 = self._extract_landmarks_68(best_lms, w, h)

        # 提取 PnP 关键点
        pnp_points = self._extract_pnp_points(best_lms, w, h)

        # 裁剪左右眼
        left_crop = self._crop_eye(frame_bgr, best_lms, _LEFT_EYE_INDICES, w, h)
        right_crop = self._crop_eye(frame_bgr, best_lms, _RIGHT_EYE_INDICES, w, h)
        left_roi, left_origin = self._extract_eye_roi(frame_bgr, best_lms, _EYETOUCH_LEFT_EYE_INDICES, w, h)
        right_roi, right_origin = self._extract_eye_roi(frame_bgr, best_lms, _EYETOUCH_RIGHT_EYE_INDICES, w, h)

        # 提取虹膜和眼眶中心（refine_landmarks=True 时有 478 个点）
        left_iris_center = None
        right_iris_center = None
        left_eye_center = None
        right_eye_center = None
        left_eye_width = None
        right_eye_width = None

        if len(best_lms) > _RIGHT_IRIS_CENTER_IDX:
            left_inner = np.array([
                best_lms[_LEFT_EYE_INNER_IDX].x * w,
                best_lms[_LEFT_EYE_INNER_IDX].y * h,
            ], dtype=np.float64)
            left_outer = np.array([
                best_lms[_LEFT_EYE_OUTER_IDX].x * w,
                best_lms[_LEFT_EYE_OUTER_IDX].y * h,
            ], dtype=np.float64)
            right_inner = np.array([
                best_lms[_RIGHT_EYE_INNER_IDX].x * w,
                best_lms[_RIGHT_EYE_INNER_IDX].y * h,
            ], dtype=np.float64)
            right_outer = np.array([
                best_lms[_RIGHT_EYE_OUTER_IDX].x * w,
                best_lms[_RIGHT_EYE_OUTER_IDX].y * h,
            ], dtype=np.float64)
            # 虹膜中心
            left_iris_center = (
                float(best_lms[_LEFT_IRIS_CENTER_IDX].x * w),
                float(best_lms[_LEFT_IRIS_CENTER_IDX].y * h),
            )
            right_iris_center = (
                float(best_lms[_RIGHT_IRIS_CENTER_IDX].x * w),
                float(best_lms[_RIGHT_IRIS_CENTER_IDX].y * h),
            )
            # 眼眶中心（内外眼角中点）
            left_mid = (left_inner + left_outer) / 2.0
            right_mid = (right_inner + right_outer) / 2.0
            left_eye_center = (
                float(left_mid[0]),
                float(left_mid[1]),
            )
            right_eye_center = (
                float(right_mid[0]),
                float(right_mid[1]),
            )
            left_eye_width = float(np.linalg.norm(left_inner - left_outer))
            right_eye_width = float(np.linalg.norm(right_inner - right_outer))

        return FaceDetectionResult(
            detected=True,
            confidence=confidence,
            face_bbox=face_bbox,
            landmarks_68=landmarks_68,
            left_eye_crop=left_crop,
            right_eye_crop=right_crop,
            pnp_points_2d=pnp_points,
            left_iris_center=left_iris_center,
            right_iris_center=right_iris_center,
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            left_eye_width=left_eye_width,
            right_eye_width=right_eye_width,
            left_eye_roi=left_roi,
            right_eye_roi=right_roi,
            left_eye_origin=left_origin,
            right_eye_origin=right_origin,
            frame_size=(w, h),
        )

    def _extract_eye_roi(
        self,
        frame_bgr: np.ndarray,
        lms,
        eye_indices: list[int],
        w: int,
        h: int,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[int, int]]]:
        """提取未缩放的眼部 ROI 及其原始画面坐标。"""
        pts = []
        for idx in eye_indices:
            if idx < len(lms):
                pts.append([int(lms[idx].x * w), int(lms[idx].y * h)])
        if not pts:
            return None, None

        pts_arr = np.array(pts, dtype=np.int32)
        x, y, box_w, box_h = cv2.boundingRect(pts_arr)
        x = max(0, x)
        y = max(0, y)
        box_w = max(0, min(box_w, w - x))
        box_h = max(0, min(box_h, h - y))
        if box_w <= 0 or box_h <= 0:
            return None, (x, y)

        roi = frame_bgr[y:y + box_h, x:x + box_w]
        if roi is None or roi.size == 0:
            return None, (x, y)
        return roi, (x, y)

    def _extract_landmarks_68(self, lms, w: int, h: int) -> np.ndarray:
        """从 MediaPipe 468 点中提取 68 个关键点的像素坐标。"""
        points = np.zeros((68, 2), dtype=np.float32)
        for i, idx in enumerate(_MP_TO_68_INDICES):
            if idx < len(lms):
                points[i, 0] = lms[idx].x * w
                points[i, 1] = lms[idx].y * h
        return points

    def _extract_pnp_points(self, lms, w: int, h: int) -> dict[str, tuple[float, float]]:
        """提取 6 个 PnP 关键点像素坐标。"""
        pnp_points = {}
        for name, idx in _PNP_INDICES.items():
            if idx < len(lms):
                px = float(np.clip(lms[idx].x * w, 0, w - 1))
                py = float(np.clip(lms[idx].y * h, 0, h - 1))
                pnp_points[name] = (px, py)
        return pnp_points

    def _crop_eye(
        self,
        frame_bgr: np.ndarray,
        lms,
        eye_indices: list[int],
        w: int,
        h: int,
    ) -> np.ndarray:
        """裁剪眼部区域并 resize 到固定尺寸。

        使用眼部关键点计算中心和范围，加 padding 后裁剪，
        最终 resize 到 eye_crop_size × eye_crop_size。

        返回:
            (eye_crop_size, eye_crop_size, 3) BGR 图像
        """
        # 收集眼部关键点像素坐标
        pts_x = []
        pts_y = []
        for idx in eye_indices:
            if idx < len(lms):
                pts_x.append(lms[idx].x * w)
                pts_y.append(lms[idx].y * h)

        if not pts_x:
            return np.zeros((self.eye_crop_size, self.eye_crop_size, 3), dtype=np.uint8)

        # 计算中心和范围
        cx = sum(pts_x) / len(pts_x)
        cy = sum(pts_y) / len(pts_y)
        eye_w = max(pts_x) - min(pts_x)
        eye_h = max(pts_y) - min(pts_y)

        # 与离线眼部裁剪规则保持一致：四周各留 50% 边距，总边长为眼框的 2 倍。
        side = max(eye_w, eye_h) * 2.0
        side = max(side, 40.0)  # 最小边长

        half = side / 2.0
        x0 = int(cx - half)
        y0 = int(cy - half)
        x1 = int(cx + half)
        y1 = int(cy + half)

        # 处理越界：使用 padding
        pad_left = max(0, -x0)
        pad_top = max(0, -y0)
        pad_right = max(0, x1 - w)
        pad_bottom = max(0, y1 - h)

        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)

        crop = frame_bgr[y0:y1, x0:x1]

        if crop.size == 0:
            return np.zeros((self.eye_crop_size, self.eye_crop_size, 3), dtype=np.uint8)

        # 添加 padding（黑色填充）
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

        # resize 到固定尺寸
        eye_img = cv2.resize(
            crop,
            (self.eye_crop_size, self.eye_crop_size),
            interpolation=cv2.INTER_AREA,
        )
        return eye_img
