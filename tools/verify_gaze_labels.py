"""验证 3D 视线标签计算在真实数据上的效果。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import numpy as np
import pandas as pd
from data.preprocessing import compute_gaze_labels_for_row, estimate_distance_from_proxy

# 加载真实数据
df = pd.read_csv("dataset_raw/13/labels.csv")
meta = json.load(open("dataset_raw/13/meta.json"))
tag_size_mm = meta.get("tag_size_mm", 40)

valid_rows = df[df["valid"] == 1]
print(f"Session 13: {len(valid_rows)} valid rows")
print(f"Screen: {meta['screen_w']}x{meta['screen_h']}")

# 测试前 5 行
for i, (_, row) in enumerate(valid_rows.head(5).iterrows()):
    labels = compute_gaze_labels_for_row(
        row["target_x"], row["target_y"],
        meta["screen_w"], meta["screen_h"],
        row["distance_proxy"], tag_size_mm,
    )
    v = np.array([labels["gaze_x"], labels["gaze_y"], labels["gaze_z"]])
    norm = np.linalg.norm(v)
    dist = estimate_distance_from_proxy(row["distance_proxy"], tag_size_mm)
    print(
        f"  [{i}] target=({row['target_x']:.0f}, {row['target_y']:.0f}) "
        f"gaze=({labels['gaze_x']:.4f}, {labels['gaze_y']:.4f}, {labels['gaze_z']:.4f}) "
        f"norm={norm:.6f} dist={dist:.0f}mm"
    )

# 验证所有 valid 行的视线向量都是单位向量
norms = []
for _, row in valid_rows.iterrows():
    labels = compute_gaze_labels_for_row(
        row["target_x"], row["target_y"],
        meta["screen_w"], meta["screen_h"],
        row["distance_proxy"], tag_size_mm,
    )
    v = np.array([labels["gaze_x"], labels["gaze_y"], labels["gaze_z"]])
    norms.append(np.linalg.norm(v))

norms = np.array(norms)
print(f"\nAll {len(norms)} vectors: norm min={norms.min():.8f} max={norms.max():.8f}")
assert np.allclose(norms, 1.0, atol=1e-6), "Not all unit vectors!"
print("OK: all gaze vectors are unit vectors")
