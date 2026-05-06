"""Audit Look2Act data/runtime contracts without training.

This diagnostic answers a narrow question: before changing model capacity, do
the dataset, runtime eye crop, target grid, screen geometry, and saved
calibration points preserve the same contract?

Usage:
    conda run --no-capture-output -n gaze-env python scripts/analyze_domain_contract.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_calibration_topology import analyze_calibration


def describe(values: Iterable[float]) -> dict[str, float]:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def load_csvs(pattern: str) -> pd.DataFrame | None:
    frames = []
    for path in sorted(Path().glob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["_source"] = str(path)
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def audit_raw_dataset(raw_dir: Path) -> dict[str, object]:
    if not raw_dir.exists():
        return {"available": False, "path": str(raw_dir)}
    frames = []
    for labels_path in sorted(raw_dir.glob("*/labels.csv")):
        try:
            df = pd.read_csv(labels_path)
        except Exception:
            continue
        df["_session_dir"] = labels_path.parent.name
        frames.append(df)
    if not frames:
        return {"available": False, "path": str(raw_dir)}
    raw = pd.concat(frames, ignore_index=True)
    out: dict[str, object] = {
        "available": True,
        "path": str(raw_dir),
        "rows": int(len(raw)),
        "sessions": int(raw["session_id"].nunique()) if "session_id" in raw else 0,
        "users": int(raw["user_id"].nunique()) if "user_id" in raw else 0,
    }
    for col in ["screen_w", "screen_h", "frame_w", "frame_h", "target_x", "target_y", "distance_proxy"]:
        if col in raw:
            out[col] = describe(raw[col])
    if {"target_x", "target_y", "screen_w", "screen_h"}.issubset(raw.columns):
        nx = raw["target_x"] / raw["screen_w"]
        ny = raw["target_y"] / raw["screen_h"]
        out["norm_target_x"] = describe(nx)
        out["norm_target_y"] = describe(ny)
        out["target_count_rounded"] = int(pd.DataFrame({"x": nx.round(4), "y": ny.round(4)}).drop_duplicates().shape[0])
    for col in ["head_yaw", "head_pitch", "head_roll"]:
        if col in raw:
            out[col] = describe(raw[col])
    if {"head_yaw", "head_pitch", "head_roll"}.issubset(raw.columns):
        out["pose_abs_gt_90_ratio"] = {
            col: float(np.mean(np.abs(pd.to_numeric(raw[col], errors="coerce")) > 90.0))
            for col in ["head_yaw", "head_pitch", "head_roll"]
        }
    return out


def audit_processed_dataset(processed_dir: Path) -> dict[str, object]:
    if not processed_dir.exists():
        return {"available": False, "path": str(processed_dir)}
    out: dict[str, object] = {"available": True, "path": str(processed_dir), "splits": {}}
    for split_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        labels_path = split_dir / "labels.csv"
        if not labels_path.exists():
            continue
        df = pd.read_csv(labels_path)
        split: dict[str, object] = {"rows": int(len(df))}
        if {"norm_target_x", "norm_target_y"}.issubset(df.columns):
            targets = df[["norm_target_x", "norm_target_y"]].round(4)
            counts = targets.value_counts().reset_index(name="count")
            split["target_count"] = int(len(counts))
            split["frames_per_target"] = describe(counts["count"])
            split["norm_target_x"] = describe(df["norm_target_x"])
            split["norm_target_y"] = describe(df["norm_target_y"])
        if {"eye_img_path", "right_eye_img_path"}.issubset(df.columns):
            split["has_right_eye"] = True
        for col in ["head_yaw", "head_pitch", "head_roll"]:
            if col in df:
                split[col] = describe(df[col])
        out["splits"][split_dir.name] = split
    return out


def audit_crop_contract() -> dict[str, object]:
    face_detector_path = ROOT / "src" / "vision" / "face_detector.py"
    preprocessing_path = ROOT / "src" / "data" / "preprocessing.py"
    face_src = face_detector_path.read_text(encoding="utf-8")
    prep_src = preprocessing_path.read_text(encoding="utf-8")
    runtime_side = re.findall(r"side\s*=\s*max\(eye_w,\s*eye_h\)\s*\*\s*([0-9.]+)", face_src)
    offline_pad = re.findall(r"pad_ratio\s*=\s*([0-9.]+)", prep_src)
    offline_side_factor = None
    if offline_pad:
        offline_side_factor = 1.0 + 2.0 * float(offline_pad[0])
    runtime_side_factor = float(runtime_side[0]) if runtime_side else None
    return {
        "runtime_side_factor": runtime_side_factor,
        "offline_side_factor": offline_side_factor,
        "matches": bool(runtime_side_factor is not None and offline_side_factor is not None and abs(runtime_side_factor - offline_side_factor) < 1e-6),
        "runtime_file": str(face_detector_path.relative_to(ROOT)),
        "offline_file": str(preprocessing_path.relative_to(ROOT)),
    }


def audit_calibration(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"available": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = analyze_calibration(data)
    summary["path"] = str(path)
    return summary


def build_summary(args: argparse.Namespace) -> dict[str, object]:
    return {
        "raw_dataset": audit_raw_dataset(Path(args.raw_dir)),
        "processed_dataset": audit_processed_dataset(Path(args.processed_dir)),
        "crop_contract": audit_crop_contract(),
        "calibration": audit_calibration(Path(args.calibration)),
    }


def print_summary(summary: dict[str, object]) -> None:
    raw = summary["raw_dataset"]
    proc = summary["processed_dataset"]
    crop = summary["crop_contract"]
    calib = summary["calibration"]
    print(f"[domain] raw available={raw.get('available')} rows={raw.get('rows')} sessions={raw.get('sessions')}")
    if raw.get("pose_abs_gt_90_ratio"):
        print(f"[domain] raw pose_abs_gt_90={raw['pose_abs_gt_90_ratio']}")
    print(f"[domain] processed available={proc.get('available')} splits={list(proc.get('splits', {}).keys())}")
    print(
        "[domain] crop contract "
        f"runtime_factor={crop.get('runtime_side_factor')} offline_factor={crop.get('offline_side_factor')} "
        f"matches={crop.get('matches')}"
    )
    if calib.get("available"):
        corr = calib.get("corr", {})
        mono = calib.get("monotonic", {})
        print(
            "[domain] calibration "
            f"residual={calib.get('residual_mean_px'):.2f}px "
            f"area_ratio={calib.get('raw_to_target_area_ratio'):.4f} "
            f"corr_x={corr.get('raw_x_vs_target_x'):.3f} corr_y={corr.get('raw_y_vs_target_y'):.3f} "
            f"mono_x={mono.get('rows_raw_x_increases_with_target_x'):.3f} "
            f"mono_y={mono.get('cols_raw_y_increases_with_target_y'):.3f}"
        )
    else:
        print(f"[domain] calibration unavailable path={calib.get('path')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Look2Act domain contracts.")
    parser.add_argument("--raw-dir", default="dataset_raw")
    parser.add_argument("--processed-dir", default="dataset_processed")
    parser.add_argument("--calibration", default="calibration_deep_pog.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary = build_summary(args)
    print_summary(summary)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
