"""
PyTorch Dataset 类：加载预处理后的标准化数据用于训练/验证/测试。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env

数据格式：
- dataset_processed/{split}/images/ 下的眼部图像（128×128 BGR）
- dataset_processed/{split}/labels.csv 包含标签信息

也支持直接从 dataset_raw/ 加载（在线预处理模式），
此时从合成图中提取眼部区域并计算视线标签。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.augmentation import augment_sample
from data.preprocessing import (
    extract_eyes_from_mosaic,
    compute_gaze_labels_for_row,
    normalize_screen_coords,
    ScreenCameraGeometry,
    DEFAULT_GEOMETRY,
)

logger = logging.getLogger(__name__)


class GazeDataset(Dataset):
    """视线估计 PyTorch Dataset。

    支持两种数据源：
    1. 预处理模式：从 dataset_processed/ 加载已处理的图像和标签
    2. 在线模式：从 dataset_raw/ 的合成图中实时提取眼部区域

    输出格式：
    - eye_img: (3, 128, 128) float32 张量，像素值归一化到 [0, 1]
    - gaze: (3,) float32 张量，3D 视线方向单位向量
    - meta: dict，包含 norm_target_x, norm_target_y, session_id 等
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_root: Optional[Path] = None,
        augment: bool = False,
        augment_rng_seed: Optional[int] = None,
        geometry: ScreenCameraGeometry = DEFAULT_GEOMETRY,
    ):
        """初始化 Dataset。

        Args:
            labels_df: 标签 DataFrame，必须包含以下列：
                在线模式：img_path, target_x, target_y, screen_w, screen_h,
                         head_yaw, head_pitch, head_roll, session_id, user_id,
                         distance_proxy
                预处理模式：eye_img_path, gaze_x, gaze_y, gaze_z,
                           norm_target_x, norm_target_y, session_id, user_id
            image_root: 图像根目录（在线模式下为 session 目录的父目录）
            augment: 是否启用数据增强
            augment_rng_seed: 增强随机种子（None 则每次不同）
            geometry: 屏幕-相机几何参数（在线模式用）
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.image_root = image_root
        self.augment = augment
        self.geometry = geometry

        # 判断数据模式
        self._online_mode = "gaze_x" not in labels_df.columns

        if augment_rng_seed is not None:
            self._rng = np.random.RandomState(augment_rng_seed)
        else:
            self._rng = np.random.RandomState()

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.labels_df.iloc[idx]

        if self._online_mode:
            return self._load_online(row)
        else:
            return self._load_processed(row)

    def _load_online(self, row: pd.Series) -> dict:
        """从 dataset_raw 的合成图中在线加载和预处理。"""
        # 加载合成图
        img_path = row["img_path"]
        if self.image_root is not None:
            full_path = self.image_root / img_path
        else:
            full_path = Path(img_path)

        mosaic = cv2.imread(str(full_path))
        if mosaic is None:
            logger.warning(f"无法读取图像: {full_path}")
            return self._empty_sample()

        # 从合成图提取左眼（训练时使用左眼）
        left_eye, right_eye = extract_eyes_from_mosaic(mosaic)
        if left_eye is None:
            return self._empty_sample()

        # 计算视线标签
        labels = compute_gaze_labels_for_row(
            float(row["target_x"]),
            float(row["target_y"]),
            int(row["screen_w"]),
            int(row["screen_h"]),
            distance_proxy=float(row.get("distance_proxy", -1)),
            geometry=self.geometry,
        )

        gx, gy, gz = labels["gaze_x"], labels["gaze_y"], labels["gaze_z"]

        # 数据增强
        if self.augment:
            aug = augment_sample(left_eye, gx, gy, gz, rng=self._rng)
            left_eye = aug.eye_img
            gx, gy, gz = aug.gaze_x, aug.gaze_y, aug.gaze_z

        # 转换为张量
        eye_tensor = self._img_to_tensor(left_eye)
        gaze_tensor = torch.tensor([gx, gy, gz], dtype=torch.float32)

        return {
            "eye_img": eye_tensor,
            "gaze": gaze_tensor,
            "meta": {
                "norm_target_x": labels["norm_target_x"],
                "norm_target_y": labels["norm_target_y"],
                "session_id": str(row.get("session_id", "")),
                "user_id": str(row.get("user_id", "")),
            },
        }

    def _load_processed(self, row: pd.Series) -> dict:
        """从预处理后的数据加载。"""
        # 加载眼部图像
        img_path = row["eye_img_path"]
        if self.image_root is not None:
            full_path = self.image_root / img_path
        else:
            full_path = Path(img_path)

        eye_img = cv2.imread(str(full_path))
        if eye_img is None:
            logger.warning(f"无法读取图像: {full_path}")
            return self._empty_sample()

        gx = float(row["gaze_x"])
        gy = float(row["gaze_y"])
        gz = float(row["gaze_z"])

        # 数据增强
        if self.augment:
            aug = augment_sample(eye_img, gx, gy, gz, rng=self._rng)
            eye_img = aug.eye_img
            gx, gy, gz = aug.gaze_x, aug.gaze_y, aug.gaze_z

        eye_tensor = self._img_to_tensor(eye_img)
        gaze_tensor = torch.tensor([gx, gy, gz], dtype=torch.float32)

        return {
            "eye_img": eye_tensor,
            "gaze": gaze_tensor,
            "meta": {
                "norm_target_x": float(row.get("norm_target_x", 0)),
                "norm_target_y": float(row.get("norm_target_y", 0)),
                "session_id": str(row.get("session_id", "")),
                "user_id": str(row.get("user_id", "")),
            },
        }

    @staticmethod
    def _img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 图像 → (3, H, W) float32 张量，归一化到 [0, 1]。"""
        # BGR → RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # (H, W, 3) → (3, H, W)，归一化
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return tensor

    def _empty_sample(self) -> dict:
        """返回空样本（图像加载失败时的 fallback）。"""
        return {
            "eye_img": torch.zeros(3, 128, 128, dtype=torch.float32),
            "gaze": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
            "meta": {
                "norm_target_x": 0.0,
                "norm_target_y": 0.0,
                "session_id": "",
                "user_id": "",
            },
        }
