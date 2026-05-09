"""Analyze saved Look2Act calibration JSON files.

This script summarizes runtime calibration files and recomputes per-point fit
residuals from saved transform matrices. Missing files are reported explicitly.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "calibration_final"


def calibration_candidates() -> list[tuple[str, Path, bool]]:
    appdata = os.environ.get("APPDATA")
    candidates: list[tuple[str, Path, bool]] = []
    if appdata:
        base = Path(appdata) / "Look2Act" / "calibration"
        candidates.extend(
            [
                ("runtime_classic", base / "calibration_classic.json", True),
                ("runtime_deep", base / "calibration_deep.json", True),
            ]
        )
    candidates.extend(
        [
            ("bin_classic", PROJECT_ROOT / "bin" / "calibration_classic.json", False),
            ("legacy_root", PROJECT_ROOT / "calibration.json", False),
        ]
    )
    return candidates


def feature_row(method: str, x: float, y: float) -> list[float]:
    if method == "polynomial":
        return [x, y, x * y, x * x, y * y, 1.0]
    return [x, y, 1.0]


def infer_screen_from_targets(points: list[dict[str, Any]]) -> tuple[str, str]:
    targets = [pt.get("target") for pt in points if pt.get("target")]
    if not targets:
        return "", ""
    xs = sorted({round(float(t[0]), 3) for t in targets})
    ys = sorted({round(float(t[1]), 3) for t in targets})
    if len(xs) >= 2 and len(ys) >= 2:
        # Calibration points are generated with a 10 percent margin.
        inferred_w = round((xs[-1] - xs[0]) / 0.8)
        inferred_h = round((ys[-1] - ys[0]) / 0.8)
        return f"{inferred_w}x{inferred_h}", "inferred_from_target_points"
    return "", ""


def analyze_file(label: str, path: Path, is_runtime_default: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    base_row: dict[str, Any] = {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "is_runtime_default": is_runtime_default,
        "method": "",
        "point_count": 0,
        "stored_mean_residual_px": "",
        "computed_mean_residual_px": "",
        "computed_median_residual_px": "",
        "computed_max_residual_px": "",
        "screen_resolution": "",
        "screen_resolution_source": "",
        "timestamp": "",
    }
    residual_rows: list[dict[str, Any]] = []
    if not path.exists():
        return base_row, residual_rows

    data = json.loads(path.read_text(encoding="utf-8"))
    method = str(data.get("method", "affine"))
    points = data.get("calibration_points", []) or []
    matrix = data.get("transform_matrix") or data.get("affine_matrix")
    norm_mean = data.get("norm_mean", [0.0, 0.0])
    norm_std = data.get("norm_std", [1.0, 1.0])

    base_row.update(
        {
            "method": method,
            "point_count": len(points),
            "stored_mean_residual_px": data.get("residual_mean_px", ""),
            "timestamp": data.get("timestamp", ""),
        }
    )

    screen_w = data.get("screen_w", "")
    screen_h = data.get("screen_h", "")
    if screen_w and screen_h and int(screen_w) > 0 and int(screen_h) > 0:
        base_row["screen_resolution"] = f"{screen_w}x{screen_h}"
        base_row["screen_resolution_source"] = "json"
    else:
        inferred, source = infer_screen_from_targets(points)
        base_row["screen_resolution"] = inferred
        base_row["screen_resolution_source"] = source

    residuals: list[float] = []
    if matrix is not None:
        mat = np.asarray(matrix, dtype=np.float64)
        mean = np.asarray(norm_mean, dtype=np.float64)
        std = np.asarray(norm_std, dtype=np.float64)
        std[std < 1e-12] = 1.0
        for index, pt in enumerate(points):
            raw = pt.get("raw")
            target = pt.get("target")
            if raw is None or target is None:
                continue
            x, y = float(raw[0]), float(raw[1])
            tx, ty = float(target[0]), float(target[1])
            xn = (x - mean[0]) / std[0]
            yn = (y - mean[1]) / std[1]
            feat = np.asarray(feature_row(method, xn, yn), dtype=np.float64)
            pred = mat @ feat
            residual = math.hypot(float(pred[0]) - tx, float(pred[1]) - ty)
            residuals.append(residual)
            residual_rows.append(
                {
                    "label": label,
                    "point_index": index,
                    "raw_x": x,
                    "raw_y": y,
                    "target_x": tx,
                    "target_y": ty,
                    "pred_x": float(pred[0]),
                    "pred_y": float(pred[1]),
                    "residual_px": residual,
                }
            )

    if residuals:
        base_row["computed_mean_residual_px"] = float(np.mean(residuals))
        base_row["computed_median_residual_px"] = float(np.median(residuals))
        base_row["computed_max_residual_px"] = float(np.max(residuals))

    return base_row, residual_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_residuals(rows: list[dict[str, Any]], label: str, title: str, filename: str) -> None:
    selected_rows = [r for r in rows if r["label"] == label]
    if not selected_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(r["point_index"]) + 1 for r in selected_rows]
    y = [float(r["residual_px"]) for r in selected_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(x, y, color="#4C78A8")
    ax.axhline(np.mean(y), color="#E45756", linestyle="--", linewidth=1.5, label=f"Mean {np.mean(y):.2f}px")
    ax.set_xlabel("Calibration point index")
    ax.set_ylabel("Fit residual (px)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for bar, value in zip(bars, y):
        if value == max(y):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=180)
    plt.close(fig)


def write_manual_templates() -> None:
    template = OUTPUT_DIR / "deep_25pt_manual_test_template.csv"
    template.write_text(
        "mode,screen_resolution,camera_resolution,calibration_points,method,valid_points,"
        "mean_residual_px,max_residual_px,notes\n"
        "Deep,,,,25,polynomial,,,,\n",
        encoding="utf-8",
    )
    readme = OUTPUT_DIR / "README_25pt_calibration_test.md"
    readme.write_text(
        "# 25-point Deep Calibration Manual Test\n\n"
        "1. Activate the project environment: `conda activate gaze-env`.\n"
        "2. Start Deep mode: `python main.py --config configs/deep.yaml`.\n"
        "3. Open camera preview and confirm the requested camera resolution, face detection, and eye crops.\n"
        "4. Run the 25-point calibration and save the result.\n"
        "5. Confirm the file exists at `%APPDATA%/Look2Act/calibration/calibration_deep.json`.\n"
        "6. Run `python scripts/analyze_calibration_results.py` and copy the Deep row into the paper table.\n\n"
        "Do not report the Deep values as measured until `calibration_deep.json` exists.\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for label, path, is_runtime_default in calibration_candidates():
        summary, residuals = analyze_file(label, path, is_runtime_default)
        summary_rows.append(summary)
        residual_rows.extend(residuals)

    write_csv(OUTPUT_DIR / "calibration_summary.csv", summary_rows)
    write_csv(OUTPUT_DIR / "classic_25pt_residuals.csv", [r for r in residual_rows if r["label"] == "runtime_classic"])
    write_csv(OUTPUT_DIR / "deep_25pt_residuals.csv", [r for r in residual_rows if r["label"] == "runtime_deep"])
    (OUTPUT_DIR / "calibration_summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_residuals(
        residual_rows,
        "runtime_classic",
        "Classic 25-point Polynomial Calibration Fit Residuals",
        "classic_25pt_residuals.png",
    )
    plot_residuals(
        residual_rows,
        "runtime_deep",
        "Deep 25-point Polynomial Calibration Fit Residuals",
        "deep_25pt_residuals.png",
    )
    write_manual_templates()
    print(f"Wrote calibration analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
