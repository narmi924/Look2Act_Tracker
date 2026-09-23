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


def bool_counts(df: pd.DataFrame, column: str) -> dict[str, object]:
    if column not in df.columns:
        return dict(true=0, denominator=0, unknown_rows=len(df), ratio=None)
    normalized = df[column].astype('string').str.strip().str.lower()
    positive = normalized.isin({'true', '1', '1.0', 'yes'})
    known = normalized.isin({'true', '1', '1.0', 'yes', 'false', '0', '0.0', 'no'})
    denominator = int(known.sum())
    numerator = int(positive.sum())
    return dict(true=numerator, denominator=denominator, unknown_rows=len(df) - denominator,
                ratio=numerator / denominator if denominator else None)


def bool_ratio(df: pd.DataFrame, column: str) -> float | None:
    return bool_counts(df, column)['ratio']


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
    raw_units = df['raw_units'].dropna().unique().tolist() if 'raw_units' in df else []
    raw_units = raw_units[0] if len(raw_units) == 1 else 'unknown'
    valid_column = 'source_valid' if 'source_valid' in df else 'valid'
    valid_counts = bool_counts(df, valid_column)
    face_counts = bool_counts(df, 'face_detected')
    fps = describe(numeric_series(df, 'fps'))
    fps['denominator'] = len(df)
    fps['unavailable_count'] = len(df) - fps['count']
    if fps['count'] == 0:
        fps['status'] = 'unavailable'
    summary: dict[str, object] = {
        "file": str(csv_path),
        "rows": int(len(df)),
        "backend_counts": df["backend"].value_counts(dropna=False).to_dict() if "backend" in df else {},
        "valid_ratio": valid_counts['ratio'],
        "valid_counts": valid_counts,
        "source_valid_ratio": bool_ratio(df, "source_valid") if 'source_valid' in df else None,
        "face_detected_ratio": face_counts['ratio'],
        "face_detected_counts": face_counts,
        "fps": fps,
        "raw_extent": point_extent(df, "raw_x", "raw_y"),
        "calibrated_extent": point_extent(df, "calibrated_x", "calibrated_y"),
        "raw_units": raw_units,
        "gate_state_counts": df['gate_state'].value_counts().to_dict() if 'gate_state' in df else {'unknown': len(df)},
        "raw_step": dict(continuous_steps(df, 'raw_x', 'raw_y'), units=raw_units),
        "calibrated_step": dict(continuous_steps(df, 'calibrated_x', 'calibrated_y'), units='screen_px'),
    }
    if "error" in df.columns:
        errors = df["error"].fillna("").astype(str)
        non_empty = errors[errors != ""]
        summary["error_counts"] = non_empty.value_counts().head(20).to_dict()
        summary["error_ratio"] = float((errors != "").mean()) if len(errors) else 0.0
        summary["error_count"] = len(non_empty)
        summary["error_denominator"] = len(errors)
    else:
        summary["error_ratio"] = None
    return summary


def continuous_steps(df, x_col, y_col):
    """Per-frame displacement, NOT static fixation jitter. Never bridge rejections."""
    required = {'gate_state', 'session', 'sequence', 'continuity', 'reset', x_col, y_col}
    if not required.issubset(df.columns):
        return {'count': 0, 'status': 'unknown_missing_gate_or_identity'}
    steps, previous, seen = [], None, set()
    for _, row in df.iterrows():
        if row['gate_state'] == 'duplicate':
            continue
        if row['gate_state'] != 'new':
            previous = None
            continue
        point = np.array([row[x_col], row[y_col]], dtype=float)
        identity = (row['session'], row['sequence'])
        key = (row['session'], row['continuity'])
        rejected = any(pd.notna(row.get(k)) and str(row.get(k)) not in ('', 'None')
                       for k in ('screen_rejection', 'dispatch_rejection'))
        reset = str(row['reset']).lower() in ('true', '1')
        if not np.all(np.isfinite(point)) or rejected or identity in seen:
            previous = None
            continue
        seen.add(identity)
        if not reset and previous is not None and previous[0] == key:
            steps.append(float(np.linalg.norm(point - previous[1])))
        previous = (key, point)
    return describe(pd.Series(steps, dtype=float))


def print_summary(summary: dict[str, object]) -> None:
    print(f"[diag-summary] {summary['file']} rows={summary['rows']}")
    def ratio_text(counts):
        ratio = counts['ratio']
        value = 'unknown' if ratio is None else f'{ratio:.3f}'
        return f"{value} ({counts['true']}/{counts['denominator']}; unknown={counts['unknown_rows']})"
    error_ratio = summary.get('error_ratio')
    error_text = 'unknown' if error_ratio is None else (
        f"{error_ratio:.3f} ({summary['error_count']}/{summary['error_denominator']})")
    print(
        f"[diag-summary]   valid={ratio_text(summary['valid_counts'])} "
        f"face={ratio_text(summary['face_detected_counts'])} error={error_text}"
    )
    fps = summary.get("fps", {})
    if isinstance(fps, dict) and fps.get("count", 0):
        print(f"[diag-summary]   fps mean={fps['mean']:.1f} p50={fps['p50']:.1f} p95={fps['p95']:.1f} "
              f"(n={fps['count']}/{fps['denominator']})")
    else:
        print('[diag-summary]   fps=unknown')
    raw_extent = summary.get("raw_extent", {})
    if isinstance(raw_extent, dict) and raw_extent.get("count", 0):
        print(
            "[diag-summary]   raw extent "
            f"w={raw_extent['width']:.3f} h={raw_extent['height']:.3f}"
        )
    raw_step = summary.get("raw_step", {})
    if isinstance(raw_step, dict) and raw_step.get("count", 0):
        print(f"[diag-summary]   raw step [{raw_step['units']}] mean={raw_step['mean']:.3f} p95={raw_step['p95']:.3f}")
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
