"""Evaluate whether a direct PoG model preserves screen topology.

The regular PoG evaluator reports pixel error. This script answers a separate
question that matters for calibration: do raw model outputs keep the 5x5 screen
grid ordering, or have they collapsed into a small unordered blob?

Usage:
    conda run --no-capture-output -n gaze-env python scripts/evaluate_pog_topology.py \
        --checkpoint checkpoints/deep_pog_topology/best_model.pth \
        --output evaluation_results/deep_pog_topology/topology.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.dataset import GazeDataset
from scripts.evaluate_pog import evaluate_pog, load_config, load_model


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return 0.0
    value = a.corr(b)
    return 0.0 if pd.isna(value) else float(value)


def _extent(values: pd.Series) -> float:
    return float(values.max() - values.min()) if not values.empty else 0.0


def _monotonic_fraction_by_group(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    pred_col: str,
) -> float:
    ok = 0
    total = 0
    for _, group in df.sort_values([group_col, order_col]).groupby(group_col):
        vals = group[pred_col].to_numpy(dtype=np.float64)
        if vals.shape[0] < 2:
            continue
        diffs = np.diff(vals)
        ok += int(np.sum(diffs > 0))
        total += int(diffs.shape[0])
    return float(ok / total) if total else 0.0


def analyze_prediction_topology(
    pred_points: np.ndarray,
    true_points: np.ndarray,
    target_decimals: int = 4,
) -> dict[str, object]:
    """Analyze topology after averaging predictions per target point."""
    if pred_points.size == 0 or true_points.size == 0:
        return {"available": False, "points": 0}

    df = pd.DataFrame(
        {
            "pred_x": pred_points[:, 0],
            "pred_y": pred_points[:, 1],
            "target_x": true_points[:, 0],
            "target_y": true_points[:, 1],
        }
    )
    df["target_x_round"] = df["target_x"].round(target_decimals)
    df["target_y_round"] = df["target_y"].round(target_decimals)
    grouped = (
        df.groupby(["target_x_round", "target_y_round"], as_index=False)
        .agg(
            pred_x=("pred_x", "mean"),
            pred_y=("pred_y", "mean"),
            pred_x_std=("pred_x", "std"),
            pred_y_std=("pred_y", "std"),
            target_x=("target_x", "mean"),
            target_y=("target_y", "mean"),
            count=("pred_x", "size"),
        )
        .fillna(0.0)
    )

    pred_w = _extent(grouped["pred_x"])
    pred_h = _extent(grouped["pred_y"])
    target_w = _extent(grouped["target_x"])
    target_h = _extent(grouped["target_y"])
    pred_area = max(pred_w, 1e-9) * max(pred_h, 1e-9)
    target_area = max(target_w, 1e-9) * max(target_h, 1e-9)

    return {
        "available": True,
        "points": int(len(grouped)),
        "prediction_extent": {"width": pred_w, "height": pred_h, "area": pred_area},
        "target_extent": {"width": target_w, "height": target_h, "area": target_area},
        "pred_to_target_area_ratio": float(pred_area / target_area),
        "mean_per_target_std": {
            "x": float(grouped["pred_x_std"].mean()),
            "y": float(grouped["pred_y_std"].mean()),
        },
        "corr": {
            "pred_x_vs_target_x": _safe_corr(grouped["pred_x"], grouped["target_x"]),
            "pred_y_vs_target_y": _safe_corr(grouped["pred_y"], grouped["target_y"]),
            "pred_x_vs_target_y": _safe_corr(grouped["pred_x"], grouped["target_y"]),
            "pred_y_vs_target_x": _safe_corr(grouped["pred_y"], grouped["target_x"]),
        },
        "monotonic": {
            "rows_pred_x_increases_with_target_x": _monotonic_fraction_by_group(
                grouped, "target_y_round", "target_x_round", "pred_x"
            ),
            "cols_pred_y_increases_with_target_y": _monotonic_fraction_by_group(
                grouped, "target_x_round", "target_y_round", "pred_y"
            ),
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    if not summary.get("available"):
        print("[pog-topology] no prediction points")
        return
    corr = summary["corr"]
    mono = summary["monotonic"]
    extent = summary["prediction_extent"]
    print(
        "[pog-topology] "
        f"points={summary['points']} "
        f"pred_extent={extent['width']:.4f}x{extent['height']:.4f} "
        f"area_ratio={summary['pred_to_target_area_ratio']:.4f}"
    )
    print(
        "[pog-topology] "
        f"corr_x={corr['pred_x_vs_target_x']:.3f} "
        f"corr_y={corr['pred_y_vs_target_y']:.3f} "
        f"mono_x={mono['rows_pred_x_increases_with_target_x']:.3f} "
        f"mono_y={mono['cols_pred_y_increases_with_target_y']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate direct PoG topology")
    parser.add_argument("--config", default="configs/train_pog_topology_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/deep_pog_topology/best_model.pth")
    parser.add_argument("--output", default="evaluation_results/deep_pog_topology/topology.json")
    parser.add_argument("--screen-w", type=int, default=1920)
    parser.add_argument("--screen-h", type=int, default=1080)
    parser.add_argument("--processed-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[pog-topology] checkpoint 不存在: {ckpt_path}")
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    data_cfg = ckpt_config.get("data", config.get("data", {}))
    test_dir = Path(args.processed_dir or data_cfg.get("dataset_processed_dir", "dataset_processed")) / "test"
    labels_path = test_dir / "labels.csv"
    if not labels_path.exists():
        print(f"[pog-topology] 测试数据不存在: {labels_path}")
        return 2

    df = pd.read_csv(labels_path)
    dataset = GazeDataset(
        df,
        image_root=test_dir,
        model_version="v2",
        target_mode="pog2d",
        head_pose_mode=data_cfg.get("head_pose_mode", "stored"),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model = load_model(str(ckpt_path), config)
    results = evaluate_pog(model, loader, screen_w=args.screen_w, screen_h=args.screen_h)
    topology = analyze_prediction_topology(results["pred_points"], results["true_points"])
    metrics = dict(results["metrics"])
    summary = {"metrics": metrics, "topology": topology}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(topology)
    print(f"[pog-topology] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
