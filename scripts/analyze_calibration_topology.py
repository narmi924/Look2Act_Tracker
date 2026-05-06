"""Analyze whether calibration raw points preserve screen topology.

This is useful for deep_pog debugging: a low residual is only possible when
raw points keep a roughly monotonic relation with target screen positions.

Usage:
    conda run --no-capture-output -n gaze-env python scripts/analyze_calibration_topology.py calibration_deep_pog.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return 0.0
    value = a.corr(b)
    return 0.0 if pd.isna(value) else float(value)


def _extent(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.max() - values.min())


def _monotonic_fraction_by_group(
    df: pd.DataFrame,
    group_col: str,
    order_col: str,
    raw_col: str,
) -> float:
    """Return fraction of adjacent pairs whose raw order matches target order."""
    ok = 0
    total = 0
    for _, group in df.sort_values([group_col, order_col]).groupby(group_col):
        vals = group[raw_col].to_numpy(dtype=np.float64)
        if vals.shape[0] < 2:
            continue
        diffs = np.diff(vals)
        ok += int(np.sum(diffs > 0))
        total += int(diffs.shape[0])
    return float(ok / total) if total else 0.0


def calibration_points_to_frame(data: dict) -> pd.DataFrame:
    rows = []
    for point in data.get("calibration_points", []):
        raw = point.get("raw", [None, None])
        target = point.get("target", [None, None])
        rows.append(
            {
                "raw_x": raw[0],
                "raw_y": raw[1],
                "target_x": target[0],
                "target_y": target[1],
            }
        )
    return pd.DataFrame(rows).dropna()


def analyze_calibration(data: dict) -> dict[str, object]:
    df = calibration_points_to_frame(data)
    if df.empty:
        return {"available": False, "points": 0}

    raw_w = _extent(df["raw_x"])
    raw_h = _extent(df["raw_y"])
    target_w = _extent(df["target_x"])
    target_h = _extent(df["target_y"])
    raw_area = max(raw_w, 1e-9) * max(raw_h, 1e-9)
    target_area = max(target_w, 1e-9) * max(target_h, 1e-9)

    return {
        "available": True,
        "points": int(len(df)),
        "residual_mean_px": float(data.get("residual_mean_px", 0.0)),
        "method": data.get("method", "unknown"),
        "raw_extent": {"width": raw_w, "height": raw_h, "area": raw_area},
        "target_extent": {"width": target_w, "height": target_h, "area": target_area},
        "raw_to_target_area_ratio": float(raw_area / target_area),
        "corr": {
            "raw_x_vs_target_x": _safe_corr(df["raw_x"], df["target_x"]),
            "raw_y_vs_target_y": _safe_corr(df["raw_y"], df["target_y"]),
            "raw_x_vs_target_y": _safe_corr(df["raw_x"], df["target_y"]),
            "raw_y_vs_target_x": _safe_corr(df["raw_y"], df["target_x"]),
        },
        "monotonic": {
            "rows_raw_x_increases_with_target_x": _monotonic_fraction_by_group(
                df, "target_y", "target_x", "raw_x"
            ),
            "cols_raw_y_increases_with_target_y": _monotonic_fraction_by_group(
                df, "target_x", "target_y", "raw_y"
            ),
        },
    }


def print_summary(summary: dict[str, object]) -> None:
    if not summary.get("available"):
        print("[calib-topology] no calibration points found")
        return
    raw = summary["raw_extent"]
    target = summary["target_extent"]
    corr = summary["corr"]
    mono = summary["monotonic"]
    print(
        "[calib-topology] "
        f"points={summary['points']} method={summary['method']} "
        f"residual={summary['residual_mean_px']:.2f}px"
    )
    print(
        "[calib-topology] "
        f"raw_extent={raw['width']:.1f}x{raw['height']:.1f} "
        f"target_extent={target['width']:.1f}x{target['height']:.1f} "
        f"area_ratio={summary['raw_to_target_area_ratio']:.4f}"
    )
    print(
        "[calib-topology] "
        f"corr raw_x~target_x={corr['raw_x_vs_target_x']:.3f} "
        f"raw_y~target_y={corr['raw_y_vs_target_y']:.3f} "
        f"cross raw_x~target_y={corr['raw_x_vs_target_y']:.3f} "
        f"raw_y~target_x={corr['raw_y_vs_target_x']:.3f}"
    )
    print(
        "[calib-topology] "
        f"row_monotonic_x={mono['rows_raw_x_increases_with_target_x']:.3f} "
        f"col_monotonic_y={mono['cols_raw_y_increases_with_target_y']:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze calibration raw/target topology.")
    parser.add_argument("calibration_json", nargs="?", default="calibration_deep_pog.json")
    parser.add_argument("--json", dest="json_out", default=None)
    args = parser.parse_args()

    path = Path(args.calibration_json)
    if not path.exists():
        print(f"[calib-topology] file not found: {path}")
        return 2

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = analyze_calibration(data)
    print_summary(summary)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
