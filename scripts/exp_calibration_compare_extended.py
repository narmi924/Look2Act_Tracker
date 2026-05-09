"""Extended calibration strategy comparison for the thesis.

This script extends the existing calibration hold-out experiment to include
13-point and 25-point strategies, then generates a thesis-ready comparison
figure that separates hold-out errors from runtime fit residuals.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.exp_calibration_compare import (  # noqa: E402
    _precompute_predictions,
    simulate_calibration_cached,
)


DEFAULT_CONFIGS: tuple[tuple[int, str], ...] = (
    (0, "none"),
    (1, "affine"),
    (3, "affine"),
    (5, "affine"),
    (9, "affine"),
    (9, "polynomial"),
    (13, "affine"),
    (13, "polynomial"),
    (25, "affine"),
    (25, "polynomial"),
)


def run_extended_from_arrays(
    all_raw_px: np.ndarray,
    all_true_px: np.ndarray,
    n_repeats: int,
) -> list[dict[str, Any]]:
    n_samples = len(all_raw_px)
    all_indices = list(range(n_samples))
    rng = np.random.RandomState(42)
    results: list[dict[str, Any]] = []

    for num_points, method in DEFAULT_CONFIGS:
        repeat_rows = []
        for repeat in range(n_repeats):
            if num_points == 0:
                calib_idx: list[int] = []
                test_idx = all_indices
            else:
                shuffled = rng.permutation(n_samples)
                calib_idx = shuffled[:num_points].tolist()
                test_idx = shuffled[num_points:].tolist()
            row = simulate_calibration_cached(
                all_raw_px,
                all_true_px,
                calib_idx,
                test_idx,
                method=method if method != "none" else "affine",
            )
            if row is not None:
                row["repeat"] = repeat
                repeat_rows.append(row)
        if repeat_rows:
            results.append(
                {
                    "method": method,
                    "num_calib_points": num_points,
                    "holdout_error_px_mean": float(np.mean([r["holdout_error_px"] for r in repeat_rows])),
                    "holdout_error_px_std": float(np.std([r["holdout_error_px"] for r in repeat_rows])),
                    "holdout_median_px_mean": float(np.mean([r["holdout_median_px"] for r in repeat_rows])),
                    "holdout_p90_px_mean": float(np.mean([r["holdout_p90_px"] for r in repeat_rows])),
                    "fit_error_px_mean": float(np.mean([r["fit_error_px"] for r in repeat_rows])),
                    "n_repeats": len(repeat_rows),
                }
            )
    return results


def run_extended_experiment(
    df: pd.DataFrame,
    model: Any,
    n_repeats: int,
    model_version: str,
    screen_w: int = 1536,
    screen_h: int = 864,
) -> list[dict[str, Any]]:
    all_raw_px, all_true_px = _precompute_predictions(df, model, screen_w, screen_h, model_version)
    return run_extended_from_arrays(all_raw_px, all_true_px, n_repeats)


def load_error_analysis_arrays(
    path: Path,
    split: str = "test",
    screen_w: int = 1536,
    screen_h: int = 864,
) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == split].copy()
    required = {"pred_nx", "pred_ny", "norm_target_x", "norm_target_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    raw = np.column_stack([df["pred_nx"].to_numpy(dtype=float) * screen_w, df["pred_ny"].to_numpy(dtype=float) * screen_h])
    true = np.column_stack(
        [df["norm_target_x"].to_numpy(dtype=float) * screen_w, df["norm_target_y"].to_numpy(dtype=float) * screen_h]
    )
    return raw, true


def load_model(config_path: Path, checkpoint_path: Path) -> tuple[Any, str]:
    import torch
    import yaml

    from models.gaze_net import GazeNet, GazeNetV2

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_version = "v1"
    ckpt_config = {}
    if isinstance(ckpt, dict):
        model_version = ckpt.get("model_version", "v1")
        ckpt_config = ckpt.get("config", {})
    model_cfg = ckpt_config.get("model", config.get("model", {}))
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    if model_version == "v2":
        model = GazeNetV2(
            num_channels=channels,
            head_pose_dim=model_cfg.get("head_pose_dim", 3),
            fusion_dim=model_cfg.get("fusion_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        model = GazeNet(num_channels=channels)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, model_version


def runtime_classic_residual() -> dict[str, Any] | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    summary_path = PROJECT_ROOT / "evaluation_results" / "calibration_final" / "calibration_summary.json"
    if not summary_path.exists():
        return None
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("label") == "runtime_classic" and row.get("exists"):
            return row
    return None


def write_results(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


def plot_strategy_matrix(output_dir: Path, results: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    selected = [
        r
        for r in results
        if (r["num_calib_points"], r["method"]) in {(0, "none"), (9, "affine"), (9, "polynomial"), (25, "affine"), (25, "polynomial")}
    ]
    labels = [
        "无校准" if r["num_calib_points"] == 0 else f"{r['num_calib_points']}点 {r['method']}"
        for r in selected
    ]
    y = [r["holdout_error_px_mean"] for r in selected]
    yerr = [r["holdout_error_px_std"] for r in selected]
    colors = ["#7A869A", "#4C78A8", "#F58518", "#72B7B2", "#54A24B"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]})

    ax = axes[0]
    bars = ax.bar(labels, y, yerr=yerr, color=colors[: len(labels)], capsize=4)
    ax.set_title("A. 离线 hold-out 校准策略对比")
    ax.set_ylabel("Hold-out 像素误差 (px)")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, y):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:.1f}", ha="center", fontsize=8)

    ax2 = axes[1]
    classic = runtime_classic_residual()
    if classic is not None and classic.get("computed_mean_residual_px") != "":
        mean = float(classic["computed_mean_residual_px"])
        max_res = float(classic["computed_max_residual_px"])
        bars2 = ax2.bar(["Classic\n25点 polynomial", "Deep\n25点 polynomial"], [mean, 0.0], color=["#4C78A8", "#CCCCCC"])
        ax2.errorbar([0], [mean], yerr=[[0], [max_res - mean]], fmt="none", ecolor="#E45756", capsize=6, linewidth=1.8)
        ax2.text(bars2[0].get_x() + bars2[0].get_width() / 2, mean + 6, f"均值 {mean:.2f}\n最大 {max_res:.2f}", ha="center", fontsize=8)
        ax2.text(1, max(mean * 0.45, 20), "待实测", ha="center", va="center", fontsize=10, color="#555555")
        ax2.set_ylim(0, max(max_res * 1.25, max(y) * 0.45))
    else:
        ax2.bar(["Classic\n25点 polynomial", "Deep\n25点 polynomial"], [0.0, 0.0], color="#CCCCCC")
        ax2.text(0.5, 0.5, "实时校准文件缺失", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_title("B. 最终实时 25 点校准拟合残差")
    ax2.set_ylabel("拟合残差 (px)")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("不同校准策略与最终 25 点校准结果对比", fontsize=14)
    fig.text(
        0.5,
        0.015,
        "A 区为 hold-out 像素误差；B 区为校准拟合残差。两类指标不可直接等同。",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_dir / "calibration_strategy_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended calibration comparison")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="evaluation_results/calibration_compare_extended")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--source", default="evaluation_results/error_analysis/per_sample_errors.csv")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    source = PROJECT_ROOT / args.source
    labels = PROJECT_ROOT / "dataset_processed" / "test" / "labels.csv"
    if source.exists():
        all_raw_px, all_true_px = load_error_analysis_arrays(source, split=args.split)
        results = run_extended_from_arrays(all_raw_px, all_true_px, args.repeats)
    else:
        model, model_version = load_model(PROJECT_ROOT / args.config, PROJECT_ROOT / args.checkpoint)
        if not labels.exists():
            raise FileNotFoundError(labels)
        df = pd.read_csv(labels)
        results = run_extended_experiment(df, model, args.repeats, model_version)
    output_dir = PROJECT_ROOT / args.output
    write_results(output_dir, results)
    plot_strategy_matrix(output_dir, results)
    print(f"Wrote extended calibration comparison to {output_dir}")


if __name__ == "__main__":
    main()
