"""
数据预处理入口脚本：加载所有 Session → 预处理 → 划分 → 导出。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python scripts/preprocess.py
         conda run -n gaze-env python scripts/preprocess.py --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.pipeline import DataPipeline, SessionData, PROCESSED_COLUMNS
from data.preprocessing import (
    extract_eyes_from_mosaic,
    compute_gaze_labels_for_row,
    normalize_screen_coords,
    ScreenCameraGeometry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preprocess_session(
    session: SessionData,
    output_dir: Path,
    geometry: ScreenCameraGeometry,
) -> list[dict]:
    """预处理单个 Session：提取眼部图像 + 计算视线标签。

    Args:
        session: 已加载的 SessionData
        output_dir: 输出目录（如 dataset_processed/train/）
        geometry: 屏幕-相机几何参数

    Returns:
        处理后的标签行列表
    """
    pipeline = DataPipeline()
    valid_labels = pipeline.get_valid_labels(session)
    meta = session.meta
    tag_size_mm = meta.raw.get("tag_size_mm", 40.0)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = []
    skipped = 0

    for idx, (_, row) in enumerate(valid_labels.iterrows()):
        # 加载合成图
        img_path = session.session_dir / row["img_path"]
        if not img_path.exists():
            skipped += 1
            continue

        mosaic = cv2.imread(str(img_path))
        if mosaic is None:
            skipped += 1
            continue

        # 提取左右眼
        left_eye, right_eye = extract_eyes_from_mosaic(mosaic)
        if left_eye is None:
            skipped += 1
            continue

        # 保存左眼图像
        left_filename = f"{meta.session_id}_{int(row['frame_idx']):06d}_L.jpg"
        cv2.imwrite(str(images_dir / left_filename), left_eye)

        # 保存右眼图像（V2 双眼输入需要）
        right_filename = f"{meta.session_id}_{int(row['frame_idx']):06d}_R.jpg"
        if right_eye is not None:
            cv2.imwrite(str(images_dir / right_filename), right_eye)
        else:
            # 右眼不可用时复制左眼（水平翻转作为近似）
            cv2.imwrite(str(images_dir / right_filename), cv2.flip(left_eye, 1))

        # 计算视线标签
        labels = compute_gaze_labels_for_row(
            float(row["target_x"]),
            float(row["target_y"]),
            meta.screen_w,
            meta.screen_h,
            distance_proxy=float(row.get("distance_proxy", -1)),
            tag_size_mm=tag_size_mm,
            focal_length_px=float(meta.frame_w),
            geometry=geometry,
        )

        processed_rows.append({
            "eye_img_path": f"images/{left_filename}",
            "right_eye_img_path": f"images/{right_filename}",
            "gaze_x": labels["gaze_x"],
            "gaze_y": labels["gaze_y"],
            "gaze_z": labels["gaze_z"],
            "head_yaw": float(row["head_yaw"]),
            "head_pitch": float(row["head_pitch"]),
            "head_roll": float(row["head_roll"]),
            "norm_target_x": labels["norm_target_x"],
            "norm_target_y": labels["norm_target_y"],
            "session_id": meta.session_id,
            "user_id": meta.user_id,
        })

    if skipped > 0:
        logger.warning(
            f"Session {meta.session_id}: 跳过 {skipped} 个无法处理的样本"
        )

    return processed_rows


def main():
    parser = argparse.ArgumentParser(description="Look2Act 数据预处理")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="清除已有的 dataset_processed 目录后重新处理",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    data_cfg = config.get("data", {})

    raw_dir = data_cfg.get("dataset_raw_dir", "dataset_raw")
    output_dir = Path(data_cfg.get("dataset_processed_dir", "dataset_processed"))
    train_ratio = data_cfg.get("train_split", 0.7)
    val_ratio = data_cfg.get("val_split", 0.15)
    test_ratio = data_cfg.get("test_split", 0.15)

    geometry = ScreenCameraGeometry()

    # 清除旧数据
    if args.clean and output_dir.exists():
        logger.info(f"清除旧数据: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载所有 Session
    logger.info(f"从 {raw_dir} 加载数据...")
    pipeline = DataPipeline(raw_dir)
    sessions = pipeline.load_all_sessions()

    if not sessions:
        logger.error("未找到任何有效 Session")
        sys.exit(1)

    total_valid = sum(s.valid_count for s in sessions)
    logger.info(f"共 {len(sessions)} 个 Session，{total_valid} 个有效样本")

    # 划分数据集
    train_ids, val_ids, test_ids = pipeline.split_dataset(
        sessions, ratios=(train_ratio, val_ratio, test_ratio)
    )

    # 按划分分组
    split_map = {}
    for sid in train_ids:
        split_map[sid] = "train"
    for sid in val_ids:
        split_map[sid] = "val"
    for sid in test_ids:
        split_map[sid] = "test"

    # 预处理每个 Session
    split_rows = {"train": [], "val": [], "test": []}

    for session in sessions:
        sid = session.meta.session_id
        split = split_map.get(sid, "train")
        split_dir = output_dir / split

        logger.info(
            f"处理 Session {sid} → {split} "
            f"(有效={session.valid_count})"
        )

        rows = preprocess_session(session, split_dir, geometry)
        split_rows[split].extend(rows)

    # 导出标签 CSV
    for split_name, rows in split_rows.items():
        if not rows:
            continue
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(rows, columns=PROCESSED_COLUMNS)
        csv_path = split_dir / "labels.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"{split_name}: {len(rows)} 样本 → {csv_path}")

    # 保存划分信息
    split_info = {
        "train_sessions": train_ids,
        "val_sessions": val_ids,
        "test_sessions": test_ids,
        "total_samples": {
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
    }
    with open(output_dir / "split_info.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)

    logger.info("预处理完成")
    for split_name, rows in split_rows.items():
        logger.info(f"  {split_name}: {len(rows)} 样本")


if __name__ == "__main__":
    main()
