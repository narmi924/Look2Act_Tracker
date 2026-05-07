"""
数据预处理模块：人脸检测、眼部区域裁剪、坐标归一化、3D 视线标签计算。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env

本模块保留训练和实时推理一致的 FaceMesh 关键点索引、眼部裁剪规则和几何标签算法。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from mediapipe.python.solutions import face_mesh as mp_face_mesh

logger = logging.getLogger(__name__)

# 眼部关键点索引（MediaPipe FaceMesh 468 点），需与实时检测模块保持一致。
LEFT_EYE_INDICES = [
    33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155
]
RIGHT_EYE_INDICES = [
    263, 362, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382
]

# PnP 头姿估计用的 6 个关键点索引
PNP_LANDMARK_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

# 默认眼部裁剪输出尺寸
DEFAULT_EYE_CROP_SIZE = 128


@dataclass
class EyeCropResult:
    """眼部裁剪结果。"""
    success: bool
    left_eye: Optional[np.ndarray] = None   # (128, 128, 3) BGR
    right_eye: Optional[np.ndarray] = None  # (128, 128, 3) BGR
    pnp_points_2d: Optional[dict[str, tuple[float, float]]] = None
    face_bbox: Optional[tuple[int, int, int, int]] = None


class EyeCropper:
    """基于 MediaPipe FaceMesh 的眼部区域裁剪器。

    用于离线数据预处理阶段，从采集的图像帧中裁剪眼部区域。
    使用 static_image_mode=True（逐帧独立检测，适合离线处理）。
    """

    def __init__(self, eye_crop_size: int = DEFAULT_EYE_CROP_SIZE):
        self.eye_crop_size = eye_crop_size
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def close(self):
        """释放 MediaPipe 资源。"""
        self._face_mesh.close()

    def crop_eyes(self, frame_bgr: np.ndarray) -> EyeCropResult:
        """从一帧图像中裁剪左右眼区域。

        Args:
            frame_bgr: BGR 格式图像，shape (H, W, 3)

        Returns:
            EyeCropResult，包含 128×128 的左右眼裁剪图
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return EyeCropResult(success=False)

        h, w = frame_bgr.shape[:2]

        # MediaPipe 需要 RGB 输入
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(frame_rgb)

        if not result.multi_face_landmarks:
            return EyeCropResult(success=False)

        landmarks = result.multi_face_landmarks[0].landmark

        # 提取 PnP 关键点
        pnp_points = {}
        for name, idx in PNP_LANDMARK_INDICES.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                pnp_points[name] = (
                    float(np.clip(lm.x * w, 0, w - 1)),
                    float(np.clip(lm.y * h, 0, h - 1)),
                )

        # 计算人脸边界框
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        face_bbox = (
            int(max(0, min(xs))),
            int(max(0, min(ys))),
            int(min(w, max(xs)) - max(0, min(xs))),
            int(min(h, max(ys)) - max(0, min(ys))),
        )

        # 裁剪左右眼
        left_eye = self._crop_single_eye(
            frame_bgr, landmarks, LEFT_EYE_INDICES, h, w
        )
        right_eye = self._crop_single_eye(
            frame_bgr, landmarks, RIGHT_EYE_INDICES, h, w
        )

        if left_eye is None or right_eye is None:
            return EyeCropResult(
                success=False,
                pnp_points_2d=pnp_points,
                face_bbox=face_bbox,
            )

        return EyeCropResult(
            success=True,
            left_eye=left_eye,
            right_eye=right_eye,
            pnp_points_2d=pnp_points,
            face_bbox=face_bbox,
        )

    def _crop_single_eye(
        self,
        frame_bgr: np.ndarray,
        landmarks,
        eye_indices: list[int],
        img_h: int,
        img_w: int,
    ) -> Optional[np.ndarray]:
        """裁剪单只眼睛区域并 resize 到固定尺寸。

        使用 50% padding 确保包含完整眼区（与 Collector 一致）。
        如果裁剪区域越界，使用黑色 padding 填充。

        Returns:
            (eye_crop_size, eye_crop_size, 3) 的 BGR 图像，或 None
        """
        # 提取眼部关键点像素坐标
        pts = []
        for idx in eye_indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append((lm.x * img_w, lm.y * img_h))

        if len(pts) < 5:
            return None

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        # 计算中心和基础宽度
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        base_w = max(xs) - min(xs)
        base_h = max(ys) - min(ys)

        if base_w < 1 or base_h < 1:
            return None

        # 带 50% padding 的正方形边长
        pad_ratio = 0.5
        side = max(base_w, base_h) * (1.0 + 2 * pad_ratio)
        side = max(side, 40.0)  # 最小 40px
        half = side / 2.0

        # 计算裁剪区域（允许越界，后续用 padding 处理）
        x1 = int(cx - half)
        y1 = int(cy - half)
        x2 = int(cx + half)
        y2 = int(cy + half)

        # 使用 copyMakeBorder 处理越界
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - img_h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - img_w)

        # 裁剪有效区域
        crop_y1 = max(0, y1)
        crop_y2 = min(img_h, y2)
        crop_x1 = max(0, x1)
        crop_x2 = min(img_w, x2)

        if crop_y2 <= crop_y1 or crop_x2 <= crop_x1:
            return None

        crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        # 添加黑色 padding（如果越界）
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = cv2.copyMakeBorder(
                crop, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

        # Resize 到固定尺寸
        eye_img = cv2.resize(
            crop,
            (self.eye_crop_size, self.eye_crop_size),
            interpolation=cv2.INTER_AREA,
        )

        return eye_img


# ============ 从 Collector 合成图中提取眼部区域 ============

def extract_eyes_from_mosaic(
    mosaic_bgr: np.ndarray,
    eye_crop_size: int = DEFAULT_EYE_CROP_SIZE,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """从 Collector 生成的隐私合成图中提取左右眼裁剪区域。

    Collector 的合成图布局（4 列）：
    - 第 1 列 [0:128, 0:128]：左眼裁剪
    - 第 2 列 [0:128, 128:256]：右眼裁剪
    - 第 3 列：ArUco marker
    - 第 4 列：骨架图

    合成图高度可能为 128（标准）或 240（旧版），宽度固定 640。
    眼部区域始终在前 128 行。

    Args:
        mosaic_bgr: Collector 生成的合成图，shape (H, 640, 3)，H 为 128 或 240
        eye_crop_size: 眼部裁剪尺寸，默认 128

    Returns:
        (left_eye, right_eye)，各为 (128, 128, 3) BGR 图像。
        如果合成图尺寸不符合预期，返回 (None, None)。
    """
    if mosaic_bgr is None or mosaic_bgr.size == 0:
        return None, None

    h, w = mosaic_bgr.shape[:2]

    # 验证合成图尺寸
    if w < 2 * eye_crop_size or h < eye_crop_size:
        logger.warning(
            f"合成图尺寸不符合预期: {h}x{w}，"
            f"需要至少 {eye_crop_size}x{2 * eye_crop_size}"
        )
        return None, None

    # 提取左右眼区域（始终在前 eye_crop_size 行）
    left_eye = mosaic_bgr[:eye_crop_size, :eye_crop_size].copy()
    right_eye = mosaic_bgr[:eye_crop_size, eye_crop_size:2 * eye_crop_size].copy()

    # 检查是否为全黑（表示 Collector 未检测到眼部）
    if np.mean(left_eye) < 1.0:
        left_eye = None
    if np.mean(right_eye) < 1.0:
        right_eye = None

    return left_eye, right_eye


# ============ 坐标归一化工具函数 ============

def normalize_screen_coords(
    target_x: float,
    target_y: float,
    screen_w: int,
    screen_h: int,
) -> tuple[float, float]:
    """将屏幕像素坐标归一化到 [0, 1] 范围。

    Args:
        target_x: 屏幕 X 坐标（像素）
        target_y: 屏幕 Y 坐标（像素）
        screen_w: 屏幕宽度（像素）
        screen_h: 屏幕高度（像素）

    Returns:
        (norm_x, norm_y) 归一化坐标
    """
    if screen_w <= 0 or screen_h <= 0:
        raise ValueError(f"屏幕尺寸无效: {screen_w}x{screen_h}")
    return target_x / screen_w, target_y / screen_h


def denormalize_screen_coords(
    norm_x: float,
    norm_y: float,
    screen_w: int,
    screen_h: int,
) -> tuple[float, float]:
    """将归一化坐标还原为屏幕像素坐标。

    Args:
        norm_x: 归一化 X 坐标 [0, 1]
        norm_y: 归一化 Y 坐标 [0, 1]
        screen_w: 屏幕宽度（像素）
        screen_h: 屏幕高度（像素）

    Returns:
        (target_x, target_y) 屏幕像素坐标
    """
    if screen_w <= 0 or screen_h <= 0:
        raise ValueError(f"屏幕尺寸无效: {screen_w}x{screen_h}")
    return norm_x * screen_w, norm_y * screen_h



# ============ 3D 视线标签计算 ============

@dataclass
class ScreenCameraGeometry:
    """屏幕-相机几何参数。

    描述相机相对于屏幕的位置关系，用于计算 3D 视线标签。
    坐标系定义（摄像头坐标系）：
    - 原点：相机光心
    - X 轴：向右
    - Y 轴：向下
    - Z 轴：向前（远离相机）

    默认假设：笔记本电脑，相机在屏幕正上方中央。
    """
    screen_w_mm: float = 344.0    # 屏幕物理宽度（mm），16 英寸笔记本约 344mm
    screen_h_mm: float = 215.0    # 屏幕物理高度（mm），16 英寸笔记本约 215mm
    cam_above_screen_mm: float = 5.0   # 相机在屏幕上边缘上方的距离（mm）
    screen_distance_mm: float = 500.0  # 屏幕平面到相机的默认距离（mm）


# 默认几何参数（16 英寸笔记本电脑）
DEFAULT_GEOMETRY = ScreenCameraGeometry()


def compute_gaze_vector(
    target_x: float,
    target_y: float,
    screen_w: int,
    screen_h: int,
    geometry: ScreenCameraGeometry = DEFAULT_GEOMETRY,
    distance_mm: Optional[float] = None,
) -> tuple[float, float, float]:
    """计算从相机（近似眼球位置）到屏幕目标点的 3D 视线方向向量。

    几何模型：
    1. 将屏幕像素坐标转换为屏幕物理坐标（mm）
    2. 将屏幕物理坐标转换为摄像头坐标系中的 3D 点
    3. 计算从相机原点到目标点的归一化方向向量

    坐标系约定（摄像头坐标系）：
    - 原点：相机光心
    - X 轴：向右
    - Y 轴：向下
    - Z 轴：向前（远离相机，朝向屏幕）

    屏幕在摄像头坐标系中的位置：
    - 屏幕中心在 Z 轴正方向 distance_mm 处
    - 屏幕上边缘在相机下方 cam_above_screen_mm 处（Y 轴正方向）
    - 屏幕左右对称于 X=0 平面

    Args:
        target_x: 屏幕目标 X 坐标（像素）
        target_y: 屏幕目标 Y 坐标（像素）
        screen_w: 屏幕宽度（像素）
        screen_h: 屏幕高度（像素）
        geometry: 屏幕-相机几何参数
        distance_mm: 用户到屏幕的距离（mm），None 时使用默认值

    Returns:
        (gaze_x, gaze_y, gaze_z) 归一化 3D 视线方向单位向量
    """
    if screen_w <= 0 or screen_h <= 0:
        raise ValueError(f"屏幕尺寸无效: {screen_w}x{screen_h}")

    dist = distance_mm if distance_mm is not None else geometry.screen_distance_mm

    # 步骤 1：屏幕像素 → 屏幕物理坐标（mm）
    # 屏幕左上角为 (0, 0)，右下角为 (screen_w, screen_h)
    screen_x_mm = (target_x / screen_w) * geometry.screen_w_mm
    screen_y_mm = (target_y / screen_h) * geometry.screen_h_mm

    # 步骤 2：屏幕物理坐标 → 摄像头坐标系 3D 点
    # 屏幕中心在摄像头坐标系中的位置：
    #   X = 0（屏幕水平居中）
    #   Y = cam_above_screen_mm + screen_h_mm / 2（屏幕中心在相机下方）
    #   Z = dist（屏幕在相机前方）
    #
    # 屏幕左上角在摄像头坐标系中：
    #   X = -screen_w_mm / 2
    #   Y = cam_above_screen_mm
    #   Z = dist
    target_3d_x = screen_x_mm - geometry.screen_w_mm / 2.0
    target_3d_y = geometry.cam_above_screen_mm + screen_y_mm
    target_3d_z = dist

    # 步骤 3：计算从相机原点 (0,0,0) 到目标点的方向向量并归一化
    vec = np.array([target_3d_x, target_3d_y, target_3d_z], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        # 目标点与相机重合（不应发生），返回正前方
        return 0.0, 0.0, 1.0

    unit_vec = vec / norm
    return float(unit_vec[0]), float(unit_vec[1]), float(unit_vec[2])


def estimate_distance_from_proxy(
    distance_proxy: float,
    tag_size_mm: float = 40.0,
    focal_length_px: float = 1920.0,
    scale: float = 1000.0,
) -> float:
    """从 Collector 的 distance_proxy 估算真实距离（mm）。

    distance_proxy = scale / marker_size_px
    marker_size_px = focal_length_px * tag_size_mm / real_distance_mm
    => real_distance_mm = focal_length_px * tag_size_mm * distance_proxy / scale

    Args:
        distance_proxy: Collector 记录的距离代理值
        tag_size_mm: ArUco marker 物理尺寸（mm）
        focal_length_px: 相机焦距（像素），近似为图像宽度
        scale: Collector 中使用的缩放因子

    Returns:
        估算的真实距离（mm）
    """
    if distance_proxy <= 0:
        return DEFAULT_GEOMETRY.screen_distance_mm

    # marker_size_px = scale / distance_proxy
    marker_size_px = scale / distance_proxy
    # real_distance = focal_length * tag_size_mm / marker_size_px
    real_distance_mm = focal_length_px * tag_size_mm / marker_size_px
    return float(real_distance_mm)


def compute_gaze_labels_for_row(
    target_x: float,
    target_y: float,
    screen_w: int,
    screen_h: int,
    distance_proxy: float = -1.0,
    tag_size_mm: float = 40.0,
    focal_length_px: float = 1920.0,
    geometry: ScreenCameraGeometry = DEFAULT_GEOMETRY,
) -> dict:
    """为 labels.csv 的一行计算完整的视线标签。

    Args:
        target_x: 屏幕目标 X 坐标（像素）
        target_y: 屏幕目标 Y 坐标（像素）
        screen_w: 屏幕宽度（像素）
        screen_h: 屏幕高度（像素）
        distance_proxy: Collector 的距离代理值，-1 表示不可用
        tag_size_mm: ArUco marker 物理尺寸（mm）
        focal_length_px: 相机焦距（像素）
        geometry: 屏幕-相机几何参数

    Returns:
        包含 gaze_x, gaze_y, gaze_z, norm_target_x, norm_target_y 的字典
    """
    # 估算距离
    distance_mm = None
    if distance_proxy > 0:
        distance_mm = estimate_distance_from_proxy(
            distance_proxy, tag_size_mm, focal_length_px
        )

    # 计算 3D 视线向量
    gx, gy, gz = compute_gaze_vector(
        target_x, target_y, screen_w, screen_h,
        geometry=geometry, distance_mm=distance_mm,
    )

    # 计算归一化屏幕坐标
    norm_x, norm_y = normalize_screen_coords(target_x, target_y, screen_w, screen_h)

    return {
        "gaze_x": gx,
        "gaze_y": gy,
        "gaze_z": gz,
        "norm_target_x": norm_x,
        "norm_target_y": norm_y,
    }
