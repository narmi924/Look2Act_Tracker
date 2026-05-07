"""
PyTorch Dataset 类：加载预处理后的标准化数据用于训练/验证/测试。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env

数据格式：
- dataset_processed/{split}/images/ 下的眼部图像（128×128 BGR）
- dataset_processed/{split}/labels.csv 包含标签信息

支持两种模型版本：
- V1 (GazeNet)：单眼输入，输出 eye_img + gaze
- V2 (GazeNetV2)：双眼 + head pose 输入，输出 left_eye + right_eye + head_pose + gaze
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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

    输出格式（V2 模式，model_version="v2"）：
    - left_eye: (3, 128, 128) float32 张量
    - right_eye: (3, 128, 128) float32 张量
    - head_pose: (3,) float32 张量 (yaw, pitch, roll)
    - gaze: (3,) float32 张量，3D 视线方向单位向量
    - pog: (2,) float32 张量，归一化屏幕坐标 [x, y]
    - meta: dict

    输出格式（V1 模式，model_version="v1"）：
    - eye_img: (3, 128, 128) float32 张量
    - gaze: (3,) float32 张量
    - meta: dict
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        image_root: Optional[Path] = None,
        augment: bool = False,
        augment_rng_seed: Optional[int] = None,
        geometry: ScreenCameraGeometry = DEFAULT_GEOMETRY,
        model_version: str = "v2",
        target_mode: str = "gaze3d",
        head_pose_mode: str = "stored",
    ):
        """初始化 Dataset。

        Args:
            labels_df: 标签 DataFrame
            image_root: 图像根目录
            augment: 是否启用数据增强
            augment_rng_seed: 增强随机种子
            geometry: 屏幕-相机几何参数（在线模式用）
            model_version: "v1" 或 "v2"，决定输出格式
            target_mode: "gaze3d" 或 "pog2d"，决定训练脚本使用的目标
            head_pose_mode: "stored" 使用标签姿态，"zero" 用零向量做消融/稳健 baseline
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.image_root = image_root
        self.augment = augment
        self.geometry = geometry
        self.model_version = model_version
        self.target_mode = target_mode
        self.head_pose_mode = head_pose_mode if head_pose_mode in {"stored", "zero"} else "stored"

        # 判断数据模式
        self._online_mode = "gaze_x" not in labels_df.columns
        # 判断是否有右眼数据
        self._has_right_eye = "right_eye_img_path" in labels_df.columns

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
        img_path = row["img_path"]
        if self.image_root is not None:
            full_path = self.image_root / img_path
        else:
            full_path = Path(img_path)

        mosaic = cv2.imread(str(full_path))
        if mosaic is None:
            logger.warning(f"无法读取图像: {full_path}")
            return self._empty_sample()

        left_eye, right_eye = extract_eyes_from_mosaic(mosaic)
        if left_eye is None:
            return self._empty_sample()

        labels = compute_gaze_labels_for_row(
            float(row["target_x"]),
            float(row["target_y"]),
            int(row["screen_w"]),
            int(row["screen_h"]),
            distance_proxy=float(row.get("distance_proxy", -1)),
            geometry=self.geometry,
        )

        gx, gy, gz = labels["gaze_x"], labels["gaze_y"], labels["gaze_z"]

        if self.augment:
            aug = augment_sample(left_eye, gx, gy, gz, rng=self._rng)
            left_eye = aug.eye_img
            gx, gy, gz = aug.gaze_x, aug.gaze_y, aug.gaze_z

        gaze_tensor = torch.tensor([gx, gy, gz], dtype=torch.float32)
        pog_tensor = torch.tensor(
            [labels["norm_target_x"], labels["norm_target_y"]],
            dtype=torch.float32,
        )
        meta = {
            "norm_target_x": labels["norm_target_x"],
            "norm_target_y": labels["norm_target_y"],
            "session_id": str(row.get("session_id", "")),
            "user_id": str(row.get("user_id", "")),
        }

        if self.model_version in {"v2", "pog_v1"}:
            if right_eye is None:
                right_eye = cv2.flip(left_eye, 1)
            head_pose = self._head_pose_tensor(row)
            return {
                "left_eye": self._img_to_tensor(left_eye),
                "right_eye": self._img_to_tensor(right_eye),
                "head_pose": head_pose,
                "gaze": gaze_tensor,
                "pog": pog_tensor,
                "meta": meta,
            }
        else:
            return {
                "eye_img": self._img_to_tensor(left_eye),
                "gaze": gaze_tensor,
                "pog": pog_tensor,
                "meta": meta,
            }

    def _load_processed(self, row: pd.Series) -> dict:
        """从预处理后的数据加载。"""
        # 加载左眼图像
        img_path = row["eye_img_path"]
        if self.image_root is not None:
            full_path = self.image_root / img_path
        else:
            full_path = Path(img_path)

        left_eye = cv2.imread(str(full_path))
        if left_eye is None:
            logger.warning(f"无法读取图像: {full_path}")
            return self._empty_sample()

        gx = float(row["gaze_x"])
        gy = float(row["gaze_y"])
        gz = float(row["gaze_z"])

        if self.augment:
            aug = augment_sample(left_eye, gx, gy, gz, rng=self._rng)
            left_eye = aug.eye_img
            gx, gy, gz = aug.gaze_x, aug.gaze_y, aug.gaze_z

        gaze_tensor = torch.tensor([gx, gy, gz], dtype=torch.float32)
        pog_tensor = torch.tensor(
            [
                float(row.get("norm_target_x", 0)),
                float(row.get("norm_target_y", 0)),
            ],
            dtype=torch.float32,
        )
        meta = {
            "norm_target_x": float(row.get("norm_target_x", 0)),
            "norm_target_y": float(row.get("norm_target_y", 0)),
            "session_id": str(row.get("session_id", "")),
            "user_id": str(row.get("user_id", "")),
        }

        if self.model_version in {"v2", "pog_v1"}:
            # 加载右眼图像
            right_eye = None
            if self._has_right_eye:
                right_path = row.get("right_eye_img_path", "")
                if right_path and self.image_root is not None:
                    right_full = self.image_root / right_path
                    right_eye = cv2.imread(str(right_full))

            # 右眼不可用时用左眼水平翻转近似
            if right_eye is None:
                right_eye = cv2.flip(left_eye, 1)

            # 右眼也做相同增强（亮度/旋转，但不翻转）
            if self.augment:
                aug_r = augment_sample(right_eye, gx, gy, gz, rng=self._rng)
                right_eye = aug_r.eye_img

            head_pose = self._head_pose_tensor(row)

            return {
                "left_eye": self._img_to_tensor(left_eye),
                "right_eye": self._img_to_tensor(right_eye),
                "head_pose": head_pose,
                "gaze": gaze_tensor,
                "pog": pog_tensor,
                "meta": meta,
            }
        else:
            return {
                "eye_img": self._img_to_tensor(left_eye),
                "gaze": gaze_tensor,
                "pog": pog_tensor,
                "meta": meta,
            }

    @staticmethod
    def _img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 图像 → (3, H, W) float32 张量，归一化到 [0, 1]。"""
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        return tensor

    def _head_pose_tensor(self, row: pd.Series) -> torch.Tensor:
        if self.head_pose_mode == "zero":
            return torch.zeros(3, dtype=torch.float32)
        return torch.tensor([
            float(row.get("head_yaw", 0)),
            float(row.get("head_pitch", 0)),
            float(row.get("head_roll", 0)),
        ], dtype=torch.float32)

    def _empty_sample(self) -> dict:
        """返回空样本，用于图像加载失败时的保护处理。"""
        if self.model_version in {"v2", "pog_v1"}:
            return {
                "left_eye": torch.zeros(3, 128, 128, dtype=torch.float32),
                "right_eye": torch.zeros(3, 128, 128, dtype=torch.float32),
                "head_pose": torch.zeros(3, dtype=torch.float32),
                "gaze": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
                "pog": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "meta": {
                    "norm_target_x": 0.0,
                    "norm_target_y": 0.0,
                    "session_id": "",
                    "user_id": "",
                },
            }
        else:
            return {
                "eye_img": torch.zeros(3, 128, 128, dtype=torch.float32),
                "gaze": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32),
                "pog": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "meta": {
                    "norm_target_x": 0.0,
                    "norm_target_y": 0.0,
                    "session_id": "",
                    "user_id": "",
                },
            }
