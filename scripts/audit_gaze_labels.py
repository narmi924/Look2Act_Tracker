"""Audit processed gaze labels and coordinate-space assumptions.

This is a finite, non-training research diagnostic. It reads labels.csv files
from dataset_processed and reports gaze vector, target, and head-pose statistics
that help decide whether the deep model should be treated as camera-space or
head-local.

Usage:
    conda run --no-capture-output -n gaze-env python scripts/audit_gaze_labels.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUIRED_COLUMNS = {
    "gaze_x",
    "gaze_y",
    "gaze_z",
    "norm_target_x",
    "norm_target_y",
}

POSE_COLUMNS = ["head_yaw", "head_pitch", "head_roll"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Look2Act processed labels.")
    parser.add_argument("--processed-dir", default="dataset_processed")
    parser.add_argument("--output", default="evaluation_results/label_audit/summary.json")
    parser.add_argument("--csv", default="evaluation_results/label_audit/split_summary.csv")
    return parser.parse_args()


def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Build Rz(roll) @ Ry(yaw) @ Rx(pitch), matching the ablation script."""
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def describe_series(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p50": float(values.quantile(0.50)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def load_split_labels(processed_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        labels_path = split_dir / "labels.csv"
        if labels_path.exists():
            splits[split_dir.name] = pd.read_csv(labels_path)
    return splits


def _safe_target_duplicate_ratio(df: pd.DataFrame) -> float:
    if not {"norm_target_x", "norm_target_y"}.issubset(df.columns) or df.empty:
        return 0.0
    rounded = df[["norm_target_x", "norm_target_y"]].round(4)
    unique_count = rounded.drop_duplicates().shape[0]
    return float(1.0 - unique_count / max(len(df), 1))


def _unit_vectors(df: pd.DataFrame) -> np.ndarray:
    gaze = df[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(gaze, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return gaze / norms


def compute_head_local_stats(df: pd.DataFrame) -> dict[str, object]:
    """Estimate what labels look like if current gaze is camera-space.

    If current labels are camera-space vectors, head-local labels for training
    would be roughly R^-1 @ gaze_camera. Large instability here indicates that
    the stored Euler angles may not be safe to use directly.
    """
    if not set(POSE_COLUMNS).issubset(df.columns) or df.empty:
        return {"available": False}

    gaze = _unit_vectors(df)
    head_local = []
    for vec, (_, row) in zip(gaze, df.iterrows()):
        r = euler_to_rotation_matrix(
            float(row.get("head_yaw", 0.0)),
            float(row.get("head_pitch", 0.0)),
            float(row.get("head_roll", 0.0)),
        )
        converted = r.T @ vec
        norm = np.linalg.norm(converted)
        head_local.append(converted / norm if norm > 1e-12 else converted)
    arr = np.vstack(head_local)
    return {
        "available": True,
        "x": describe_series(pd.Series(arr[:, 0])),
        "y": describe_series(pd.Series(arr[:, 1])),
        "z": describe_series(pd.Series(arr[:, 2])),
        "negative_z_ratio": float(np.mean(arr[:, 2] < 0.0)),
    }


def audit_split(name: str, df: pd.DataFrame) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    summary: dict[str, object] = {
        "split": name,
        "rows": int(len(df)),
        "missing_required_columns": missing,
    }
    if missing:
        return summary

    gaze = df[["gaze_x", "gaze_y", "gaze_z"]].to_numpy(dtype=np.float64)
    norms = np.linalg.norm(gaze, axis=1)
    summary.update(
        {
            "gaze_norm": describe_series(pd.Series(norms)),
            "gaze_x": describe_series(df["gaze_x"]),
            "gaze_y": describe_series(df["gaze_y"]),
            "gaze_z": describe_series(df["gaze_z"]),
            "norm_target_x": describe_series(df["norm_target_x"]),
            "norm_target_y": describe_series(df["norm_target_y"]),
            "target_duplicate_ratio": _safe_target_duplicate_ratio(df),
            "out_of_bounds_target_ratio": float(
                np.mean(
                    (df["norm_target_x"] < 0.0)
                    | (df["norm_target_x"] > 1.0)
                    | (df["norm_target_y"] < 0.0)
                    | (df["norm_target_y"] > 1.0)
                )
            ),
        }
    )

    for col in POSE_COLUMNS:
        if col in df.columns:
            summary[col] = describe_series(df[col])
    if set(POSE_COLUMNS).issubset(df.columns):
        summary["pose_abs_gt_90_ratio"] = {
            col: float(np.mean(np.abs(pd.to_numeric(df[col], errors="coerce")) > 90.0))
            for col in POSE_COLUMNS
        }
    summary["head_local_if_camera_space"] = compute_head_local_stats(df)
    return summary


def flatten_for_csv(summary: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "split": summary.get("split"),
        "rows": summary.get("rows"),
        "missing_required_columns": ",".join(summary.get("missing_required_columns", [])),
        "target_duplicate_ratio": summary.get("target_duplicate_ratio"),
        "out_of_bounds_target_ratio": summary.get("out_of_bounds_target_ratio"),
    }
    for key in ("gaze_norm", "gaze_x", "gaze_y", "gaze_z", "norm_target_x", "norm_target_y"):
        stats = summary.get(key, {})
        if isinstance(stats, dict):
            row[f"{key}_mean"] = stats.get("mean")
            row[f"{key}_std"] = stats.get("std")
            row[f"{key}_min"] = stats.get("min")
            row[f"{key}_max"] = stats.get("max")
    for key in POSE_COLUMNS:
        stats = summary.get(key, {})
        if isinstance(stats, dict):
            row[f"{key}_mean"] = stats.get("mean")
            row[f"{key}_min"] = stats.get("min")
            row[f"{key}_max"] = stats.get("max")
    pose_ratios = summary.get("pose_abs_gt_90_ratio", {})
    if isinstance(pose_ratios, dict):
        for key, value in pose_ratios.items():
            row[f"{key}_abs_gt_90_ratio"] = value
    head_local = summary.get("head_local_if_camera_space", {})
    if isinstance(head_local, dict):
        row["head_local_negative_z_ratio"] = head_local.get("negative_z_ratio")
    return row


def print_findings(summaries: Iterable[dict[str, object]]) -> None:
    for summary in summaries:
        split = summary.get("split")
        rows = summary.get("rows")
        print(f"[audit] split={split} rows={rows}")
        missing = summary.get("missing_required_columns", [])
        if missing:
            print(f"[audit]   missing columns: {missing}")
            continue
        norm = summary["gaze_norm"]
        print(
            "[audit]   gaze_norm "
            f"mean={norm['mean']:.6f} min={norm['min']:.6f} max={norm['max']:.6f}"
        )
        print(
            "[audit]   target duplicate ratio="
            f"{summary['target_duplicate_ratio']:.3f} "
            f"out_of_bounds={summary['out_of_bounds_target_ratio']:.3f}"
        )
        pose_ratios = summary.get("pose_abs_gt_90_ratio")
        if pose_ratios:
            print(f"[audit]   abs(pose)>90 ratios: {pose_ratios}")
        head_local = summary.get("head_local_if_camera_space", {})
        if head_local.get("available"):
            print(
                "[audit]   R^-1 camera-label z<0 ratio="
                f"{head_local.get('negative_z_ratio'):.3f}"
            )


def main() -> int:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        print(f"[audit] processed dataset not found: {processed_dir}")
        return 2

    split_frames = load_split_labels(processed_dir)
    if not split_frames:
        print(f"[audit] no labels.csv files found under: {processed_dir}")
        return 2

    summaries = [audit_split(name, df) for name, df in split_frames.items()]
    print_findings(summaries)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"splits": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([flatten_for_csv(s) for s in summaries]).to_csv(csv_path, index=False)

    print(f"[audit] wrote {output_path}")
    print(f"[audit] wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
