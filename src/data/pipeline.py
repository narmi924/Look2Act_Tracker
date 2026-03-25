"""
数据处理管道：将 dataset_raw/ 中的原始 Session 数据转换为标准化训练数据。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# labels.csv 必需的 24 个字段
REQUIRED_COLUMNS = [
    "user_id", "session_id", "frame_idx", "timestamp_ms",
    "img_path", "target_id", "target_x", "target_y",
    "target_elapsed_ms", "valid", "face_score", "lm_score",
    "head_yaw", "head_pitch", "head_roll",
    "distance_proxy", "illumination_mean",
    "device_id", "camera_name", "glass_id",
    "screen_w", "screen_h", "frame_w", "frame_h",
]

# 标准化输出 CSV 的字段（V2 新增 right_eye_img_path）
PROCESSED_COLUMNS = [
    "eye_img_path", "right_eye_img_path",
    "gaze_x", "gaze_y", "gaze_z",
    "head_yaw", "head_pitch", "head_roll",
    "norm_target_x", "norm_target_y",
    "session_id", "user_id",
]


@dataclass
class SessionMeta:
    """Session 元数据，来自 meta.json。"""
    user_id: str
    session_id: str
    glass_id: str
    device_id: str
    camera_name: str
    screen_w: int
    screen_h: int
    frame_w: int
    frame_h: int
    eye_crop_size: int
    distance_init_proxy: float
    grid_rows: int
    grid_cols: int
    raw: dict = field(default_factory=dict)  # 原始 JSON 数据


@dataclass
class SessionData:
    """一个完整的 Session 数据。"""
    meta: SessionMeta
    labels: pd.DataFrame  # 原始 labels.csv 数据
    session_dir: Path      # Session 目录路径
    valid_count: int = 0   # valid=1 的样本数
    skipped_count: int = 0 # valid=0 被跳过的样本数
    skip_reasons: list = field(default_factory=list)  # 跳过原因列表


@dataclass
class ProcessedSample:
    """预处理后的单个样本。"""
    eye_img_path: str      # 裁剪后眼部图像路径
    gaze_x: float          # 3D 视线方向向量 X 分量
    gaze_y: float          # 3D 视线方向向量 Y 分量
    gaze_z: float          # 3D 视线方向向量 Z 分量
    head_yaw: float        # 头部偏航角（度）
    head_pitch: float      # 头部俯仰角（度）
    head_roll: float       # 头部翻滚角（度）
    norm_target_x: float   # 归一化目标 X（0~1）
    norm_target_y: float   # 归一化目标 Y（0~1）
    session_id: str
    user_id: str


class DataPipeline:
    """数据处理管道，将原始 Session 数据转换为标准化训练数据。"""

    def __init__(self, dataset_raw_dir: str = "dataset_raw"):
        self.dataset_raw_dir = Path(dataset_raw_dir)

    def load_session(self, session_dir: str | Path) -> SessionData:
        """加载单个 Session 目录，解析 labels.csv 和 meta.json。

        Args:
            session_dir: Session 目录路径（如 dataset_raw/4/）

        Returns:
            SessionData 对象

        Raises:
            FileNotFoundError: labels.csv 或 meta.json 不存在
            ValueError: 数据完整性验证失败
        """
        session_dir = Path(session_dir)

        # 检查必需文件
        labels_path = session_dir / "labels.csv"
        meta_path = session_dir / "meta.json"

        if not labels_path.exists():
            raise FileNotFoundError(f"labels.csv 不存在: {labels_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json 不存在: {meta_path}")

        # 解析 meta.json
        meta = self._parse_meta(meta_path)

        # 解析 labels.csv
        labels = self._parse_labels(labels_path)

        # 验证数据完整性
        self._validate_labels(labels, session_dir)

        # 过滤 valid=0 的样本
        valid_mask = labels["valid"] == 1
        skipped = labels[~valid_mask]
        skip_reasons = []

        for idx, row in skipped.iterrows():
            reason = f"frame_idx={row['frame_idx']}, valid=0"
            if row.get("face_score", 1.0) < 0.5:
                reason += ", face_score 过低"
            if row.get("lm_score", 1.0) < 0.5:
                reason += ", lm_score 过低"
            skip_reasons.append(reason)

        skipped_count = len(skipped)
        if skipped_count > 0:
            logger.info(
                f"Session {meta.session_id}: 跳过 {skipped_count} 个无效样本"
            )
            for reason in skip_reasons[:5]:  # 最多打印 5 条
                logger.debug(f"  跳过原因: {reason}")

        return SessionData(
            meta=meta,
            labels=labels,
            session_dir=session_dir,
            valid_count=int(valid_mask.sum()),
            skipped_count=skipped_count,
            skip_reasons=skip_reasons,
        )

    def load_all_sessions(self) -> list[SessionData]:
        """加载 dataset_raw/ 下所有 Session 目录。"""
        sessions = []
        if not self.dataset_raw_dir.exists():
            raise FileNotFoundError(
                f"数据集目录不存在: {self.dataset_raw_dir}"
            )

        for session_dir in sorted(self.dataset_raw_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            # 跳过没有 labels.csv 的目录
            if not (session_dir / "labels.csv").exists():
                logger.warning(f"跳过目录（无 labels.csv）: {session_dir}")
                continue
            try:
                session = self.load_session(session_dir)
                sessions.append(session)
                logger.info(
                    f"已加载 Session: {session.meta.session_id} "
                    f"(有效={session.valid_count}, 跳过={session.skipped_count})"
                )
            except Exception as e:
                logger.error(f"加载 Session 失败: {session_dir}, 错误: {e}")

        logger.info(f"共加载 {len(sessions)} 个 Session")
        return sessions

    def split_dataset(
        self,
        sessions: list[SessionData],
        ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> tuple[list[str], list[str], list[str]]:
        """按 Session 级别划分 train/val/test，返回 session_id 列表。

        确保同一 Session 的所有样本完全属于同一个集合。

        Args:
            sessions: SessionData 列表
            ratios: (train, val, test) 比例，总和应为 1.0
            seed: 随机种子

        Returns:
            (train_ids, val_ids, test_ids) 三个 session_id 列表
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, f"划分比例之和必须为 1.0，当前为 {sum(ratios)}"

        rng = np.random.RandomState(seed)
        session_ids = [s.meta.session_id for s in sessions]
        rng.shuffle(session_ids)

        n = len(session_ids)
        n_train = max(1, int(n * ratios[0]))
        n_val = max(1, int(n * ratios[1]))
        # test 取剩余
        train_ids = session_ids[:n_train]
        val_ids = session_ids[n_train:n_train + n_val]
        test_ids = session_ids[n_train + n_val:]

        # 如果 test 为空（Session 数太少），从 train 中分一个
        if not test_ids and len(train_ids) > 2:
            test_ids = [train_ids.pop()]

        logger.info(
            f"数据集划分: train={len(train_ids)}, "
            f"val={len(val_ids)}, test={len(test_ids)}"
        )
        return train_ids, val_ids, test_ids

    def get_valid_labels(self, session: SessionData) -> pd.DataFrame:
        """获取 Session 中 valid=1 的标签数据。"""
        return session.labels[session.labels["valid"] == 1].copy()

    # ==================== 内部方法 ====================

    def _parse_meta(self, meta_path: Path) -> SessionMeta:
        """解析 meta.json 文件。"""
        with open(meta_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        grid = raw.get("grid", {})
        return SessionMeta(
            user_id=str(raw["user_id"]),
            session_id=raw["session_id"],
            glass_id=raw.get("glass_id", "unknown"),
            device_id=raw.get("device_id", "unknown"),
            camera_name=raw.get("camera_name", "unknown"),
            screen_w=int(raw["screen_w"]),
            screen_h=int(raw["screen_h"]),
            frame_w=int(raw["frame_w"]),
            frame_h=int(raw["frame_h"]),
            eye_crop_size=int(raw.get("eye_crop_size", 128)),
            distance_init_proxy=float(raw.get("distance_init_proxy", 0.0)),
            grid_rows=int(grid.get("rows", 5)),
            grid_cols=int(grid.get("cols", 5)),
            raw=raw,
        )

    def _parse_labels(self, labels_path: Path) -> pd.DataFrame:
        """解析 labels.csv 文件，验证 24 个必需字段。"""
        df = pd.read_csv(labels_path)

        # 检查必需字段
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"labels.csv 缺少必需字段: {missing}\n"
                f"文件: {labels_path}\n"
                f"现有字段: {list(df.columns)}"
            )

        return df

    def _validate_labels(
        self, labels: pd.DataFrame, session_dir: Path
    ) -> None:
        """验证 labels 数据完整性。"""
        # 检查数据类型
        numeric_cols = [
            "target_x", "target_y", "head_yaw", "head_pitch", "head_roll",
            "distance_proxy", "illumination_mean", "face_score", "lm_score",
        ]
        for col in numeric_cols:
            if col in labels.columns and not pd.api.types.is_numeric_dtype(labels[col]):
                raise ValueError(
                    f"字段 {col} 应为数值类型，实际为 {labels[col].dtype}\n"
                    f"Session: {session_dir}"
                )

        # 检查 valid 字段值域
        valid_values = labels["valid"].unique()
        invalid = set(valid_values) - {0, 1}
        if invalid:
            raise ValueError(
                f"valid 字段包含非法值: {invalid}，仅允许 0 和 1\n"
                f"Session: {session_dir}"
            )

        # 检查有效样本的图像路径（valid=0 的行可能没有图像，允许为空）
        valid_rows = labels[labels["valid"] == 1]
        if valid_rows["img_path"].isna().any():
            na_count = valid_rows["img_path"].isna().sum()
            raise ValueError(
                f"valid=1 的样本中 img_path 存在 {na_count} 个空值\n"
                f"Session: {session_dir}"
            )

        # 检查屏幕尺寸一致性
        screen_ws = labels["screen_w"].unique()
        screen_hs = labels["screen_h"].unique()
        if len(screen_ws) > 1 or len(screen_hs) > 1:
            logger.warning(
                f"Session {session_dir} 中屏幕尺寸不一致: "
                f"screen_w={screen_ws}, screen_h={screen_hs}"
            )
