"""验证 GazeDataset 在真实数据上的加载。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
from data.pipeline import DataPipeline
from data.dataset import GazeDataset

# 加载一个 session
pipeline = DataPipeline("dataset_raw")
session = pipeline.load_session("dataset_raw/13")
valid_labels = pipeline.get_valid_labels(session)
print(f"Session 13: {len(valid_labels)} valid rows")

# 创建在线模式 Dataset
ds = GazeDataset(
    labels_df=valid_labels,
    image_root=session.session_dir,
    augment=False,
)
print(f"Dataset length: {len(ds)}")

# 加载第一个样本
sample = ds[0]
print(f"eye_img shape: {sample['eye_img'].shape}, dtype: {sample['eye_img'].dtype}")
print(f"eye_img range: [{sample['eye_img'].min():.3f}, {sample['eye_img'].max():.3f}]")
print(f"gaze: {sample['gaze']}")
print(f"gaze norm: {sample['gaze'].norm():.6f}")
print(f"meta: {sample['meta']}")

# 测试数据增强模式
ds_aug = GazeDataset(
    labels_df=valid_labels,
    image_root=session.session_dir,
    augment=True,
    augment_rng_seed=42,
)
sample_aug = ds_aug[0]
print(f"\nAugmented gaze: {sample_aug['gaze']}")
print(f"Augmented gaze norm: {sample_aug['gaze'].norm():.6f}")

# 测试 split_dataset
sessions = pipeline.load_all_sessions()
train_ids, val_ids, test_ids = pipeline.split_dataset(sessions)
print(f"\nSplit: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
print(f"Train IDs: {train_ids[:3]}...")

print("\nOK")
