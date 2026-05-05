"""Summarize tracker diagnostic CSV files.

Usage:
    conda run --no-capture-output -n gaze-env python scripts/analyze_tracker_diagnostics.py diagnostics_classic.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Look2Act tracker diagnostic CSV files.")
    parser.add_argument("csv_files", nargs="+")
    parser.add_argument("--output", default="evaluation_results/tracker_diagnostics_summary.json")
    return parser.parse_args()


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def describe(values: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    return {
        "count": int(values.shape[0]),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "p50": float(values.quantile(0.5)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def bool_ratio(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or df.empty:
        return 0.0
    normalized = df[column].astype(str).str.lower().isin({"true", "1", "yes"})
    return float(normalized.mean())


def point_jitter(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    x = numeric_series(df, x_col)
    y = numeric_series(df, y_col)
    common = pd.concat([x, y], axis=1).dropna()
    if common.shape[0] < 2:
        return {"count": int(common.shape[0])}
    pts = common.to_numpy(dtype=np.float64)
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return describe(pd.Series(deltas))


def point_extent(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, float | int]:
    x = numeric_series(df, x_col)
    y = numeric_series(df, y_col)
    common = pd.concat([x, y], axis=1).dropna()
    if common.empty:
        return {"count": 0}
    pts = common.to_numpy(dtype=np.float64)
    return {
        "count": int(common.shape[0]),
        "x_min": float(np.min(pts[:, 0])),
        "x_max": float(np.max(pts[:, 0])),
        "y_min": float(np.min(pts[:, 1])),
        "y_max": float(np.max(pts[:, 1])),
        "width": float(np.max(pts[:, 0]) - np.min(pts[:, 0])),
        "height": float(np.max(pts[:, 1]) - np.min(pts[:, 1])),
    }


def analyze_file(csv_path: Path) -> dict[str, object]:
    df = pd.read_csv(csv_path)
    summary: dict[str, object] = {
        "file": str(csv_path),
        "rows": int(len(df)),
        "backend_counts": df["backend"].value_counts(dropna=False).to_dict() if "backend" in df else {},
        "valid_ratio": bool_ratio(df, "valid"),
        "face_detected_ratio": bool_ratio(df, "face_detected"),
        "fps": describe(numeric_series(df, "fps")),
        "raw_extent": point_extent(df, "raw_x", "raw_y"),
        "calibrated_extent": point_extent(df, "calibrated_x", "calibrated_y"),
        "raw_step_px": point_jitter(df, "raw_x", "raw_y"),
        "calibrated_step_px": point_jitter(df, "calibrated_x", "calibrated_y"),
    }
    if "error" in df.columns:
        errors = df["error"].fillna("").astype(str)
        non_empty = errors[errors != ""]
        summary["error_counts"] = non_empty.value_counts().head(20).to_dict()
        summary["error_ratio"] = float((errors != "").mean()) if len(errors) else 0.0
    return summary


def print_summary(summary: dict[str, object]) -> None:
    print(f"[diag-summary] {summary['file']} rows={summary['rows']}")
    print(
        "[diag-summary]   valid={:.3f} face={:.3f} error={:.3f}".format(
            summary.get("valid_ratio", 0.0),
            summary.get("face_detected_ratio", 0.0),
            summary.get("error_ratio", 0.0),
        )
    )
    fps = summary.get("fps", {})
    if isinstance(fps, dict) and fps.get("count", 0):
        print(f"[diag-summary]   fps mean={fps['mean']:.1f} p50={fps['p50']:.1f} p95={fps['p95']:.1f}")
    raw_extent = summary.get("raw_extent", {})
    if isinstance(raw_extent, dict) and raw_extent.get("count", 0):
        print(
            "[diag-summary]   raw extent "
            f"w={raw_extent['width']:.3f} h={raw_extent['height']:.3f}"
        )
    raw_step = summary.get("raw_step_px", {})
    if isinstance(raw_step, dict) and raw_step.get("count", 0):
        print(f"[diag-summary]   raw step mean={raw_step['mean']:.3f} p95={raw_step['p95']:.3f}")
    errors = summary.get("error_counts", {})
    if errors:
        print(f"[diag-summary]   top errors={errors}")


def main() -> int:
    args = parse_args()
    summaries = []
    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"[diag-summary] missing CSV: {csv_path}")
            return 2
        summary = analyze_file(csv_path)
        print_summary(summary)
        summaries.append(summary)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"files": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diag-summary] wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
