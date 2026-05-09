"""Generate publication-style candidate figures for Look2Act thesis results.

The script writes review-only figures to evaluation_results/publication_figures.
It does not overwrite docs/final-project-paper/引用图片.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from calibration.calibrator import CalibrationModule  # noqa: E402

OUT = PROJECT_ROOT / "evaluation_results" / "publication_figures"

PALETTE = {
    "neutral_dark": "#3F3F46",
    "neutral_mid": "#8A8F98",
    "baseline": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "affine": "#0F4D92",
    "poly": "#E4CCD8",
    "poly_dark": "#B64342",
    "final": "#42949E",
    "classic": "#0F4D92",
    "deep": "#42949E",
    "gain": "#2E9E44",
    "warn": "#B64342",
    "ink": "#242933",
    "grid": "#E8EBF2",
    "paper_bg": "#FFFFFF",
    "soft_gray": "#A9AFBB",
    "muted_blue": "#2F6690",
    "muted_teal": "#3D9A95",
    "muted_rose": "#C66B6B",
    "muted_gold": "#C69C4A",
}


def setup_matplotlib():
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial",
                "Helvetica",
                "DejaVu Sans",
                "sans-serif",
            ],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "legend.frameon": False,
            "axes.unicode_minus": False,
        }
    )
    return plt


def save_all(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight")


def clean_non_png_outputs() -> None:
    if not OUT.exists():
        return
    for path in OUT.iterdir():
        if path.is_file() and path.suffix.lower() != ".png":
            path.unlink()


def load_per_sample_dataframe() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "evaluation_results" / "error_analysis" / "per_sample_errors.csv")


def load_error_analysis_arrays(path: Path, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "split" in df.columns:
        df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError(f"No rows found in {path} for split={split}")
    screen_w, screen_h = 1536.0, 864.0
    raw = df[["pred_nx", "pred_ny"]].to_numpy(dtype=np.float64) * np.array([screen_w, screen_h])
    target = df[["norm_target_x", "norm_target_y"]].to_numpy(dtype=np.float64) * np.array([screen_w, screen_h])
    return raw, target


def evaluate_calibration_repeats(
    raw_px: np.ndarray,
    target_px: np.ndarray,
    configs: list[tuple[int, str, str]],
    repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows: list[dict[str, Any]] = []
    n = len(raw_px)
    for num_points, method, label in configs:
        for repeat in range(repeats):
            if num_points == 0:
                test_idx = np.arange(n)
                pred = raw_px
                fit_mean = 0.0
            else:
                shuffled = rng.permutation(n)
                calib_idx = shuffled[:num_points]
                test_idx = shuffled[num_points:]
                calibrator = CalibrationModule(num_points=num_points, method=method)
                for raw_point, target_point in zip(raw_px[calib_idx], target_px[calib_idx]):
                    calibrator.add_calibration_point(tuple(raw_point), tuple(target_point))
                calibrator.calibrate()
                pred = np.asarray([calibrator.apply(tuple(p)) for p in raw_px[test_idx]], dtype=np.float64)
                fit_pred = np.asarray([calibrator.apply(tuple(p)) for p in raw_px[calib_idx]], dtype=np.float64)
                fit_err = np.linalg.norm(fit_pred - target_px[calib_idx], axis=1)
                fit_mean = float(np.mean(fit_err))
            err = np.linalg.norm(pred - target_px[test_idx], axis=1)
            rows.append(
                {
                    "repeat": repeat,
                    "num_points": num_points,
                    "method": method,
                    "label": label,
                    "holdout_mean_px": float(np.mean(err)),
                    "holdout_median_px": float(np.median(err)),
                    "holdout_p90_px": float(np.percentile(err, 90)),
                    "fit_mean_px": fit_mean,
                }
            )
    return pd.DataFrame(rows)


def summarize_repeats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (num_points, method, label), group in df.groupby(["num_points", "method", "label"], sort=False):
        rows.append(
            {
                "num_points": num_points,
                "method": method,
                "label": label,
                "holdout_mean_px": float(np.mean(group["holdout_mean_px"])),
                "holdout_std_px": float(np.std(group["holdout_mean_px"], ddof=0)),
                "holdout_median_px": float(np.mean(group["holdout_median_px"])),
                "holdout_p90_px": float(np.mean(group["holdout_p90_px"])),
                "fit_mean_px": float(np.mean(group["fit_mean_px"])),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def load_runtime_summary() -> dict[str, dict[str, Any]]:
    rows = json.loads((PROJECT_ROOT / "evaluation_results" / "calibration_final" / "calibration_summary.json").read_text(encoding="utf-8"))
    return {row["label"]: row for row in rows if row.get("exists")}


def fig6_6_publication() -> None:
    plt = setup_matplotlib()
    raw, target = load_error_analysis_arrays(PROJECT_ROOT / "evaluation_results" / "error_analysis" / "per_sample_errors.csv")
    configs = [
        (0, "none", "无校准"),
        (1, "affine", "1点 affine"),
        (3, "affine", "3点 affine"),
        (5, "affine", "5点 affine"),
        (9, "affine", "9点 affine"),
        (9, "polynomial", "9点 polynomial"),
        (13, "affine", "13点 affine"),
        (13, "polynomial", "13点 polynomial"),
        (25, "affine", "25点 affine"),
        (25, "polynomial", "25点 polynomial"),
    ]
    repeat_df = evaluate_calibration_repeats(raw, target, configs)
    summary = summarize_repeats(repeat_df)
    selected_labels = ["无校准", "9点 affine", "9点 polynomial", "25点 affine", "25点 polynomial"]
    plot_summary = summary[summary["label"].isin(selected_labels)].copy()
    repeat_df.to_csv(OUT / "fig6_6_calibration_repeats.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "fig6_6_calibration_summary.csv", index=False, encoding="utf-8-sig")

    order = selected_labels
    label_to_y = {label: i for i, label in enumerate(order)}
    colors = {
        "无校准": PALETTE["neutral_mid"],
        "9点 affine": PALETTE["baseline"],
        "9点 polynomial": PALETTE["poly_dark"],
        "25点 affine": PALETTE["affine"],
        "25点 polynomial": PALETTE["final"],
    }

    fig = plt.figure(figsize=(7.3, 5.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.05, 0.95], height_ratios=[1.0, 0.9], wspace=0.45, hspace=0.55)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[:, 2])

    for _, row in plot_summary.iterrows():
        y = label_to_y[row["label"]]
        ax_a.errorbar(
            row["holdout_mean_px"],
            y,
            xerr=row["holdout_std_px"],
            fmt="o",
            markersize=5.5,
            color=colors[row["label"]],
            ecolor=colors[row["label"]],
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
        ax_a.text(row["holdout_mean_px"] + 10, y, f"{row['holdout_mean_px']:.1f}", va="center", fontsize=7)
    ax_a.set_yticks(range(len(order)))
    ax_a.set_yticklabels(order)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("hold-out 平均像素误差 (px)")
    ax_a.set_title("a  校准策略均值±标准差", loc="left", fontweight="bold")
    ax_a.axvline(314.54, color="#B9BDC7", lw=0.8, ls="--", zorder=0)
    ax_a.grid(axis="x", color="#E6E8EF", lw=0.55)
    ax_a.set_xlim(150, 520)
    ax_a.annotate(
        "本轮最低",
        xy=(184.57, label_to_y["25点 affine"]),
        xytext=(250, label_to_y["25点 affine"] - 0.45),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=PALETTE["gain"]),
        color=PALETTE["gain"],
        fontsize=7,
    )

    subset = repeat_df[repeat_df["label"].isin(["9点 affine", "9点 polynomial", "25点 affine", "25点 polynomial"])].copy()
    box_data = [subset[subset["label"] == label]["holdout_mean_px"].to_numpy() for label in order[1:]]
    positions = np.arange(len(box_data))
    bp = ax_b.boxplot(
        box_data,
        positions=positions,
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#272727", lw=1.1),
        boxprops=dict(linewidth=0.8),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, label in zip(bp["boxes"], order[1:]):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.62)
        patch.set_edgecolor("#272727")
    for i, label in enumerate(order[1:]):
        vals = subset[subset["label"] == label]["holdout_mean_px"].to_numpy()
        jitter = np.linspace(-0.11, 0.11, len(vals))
        ax_b.scatter(np.full(len(vals), i) + jitter, vals, s=8, color=colors[label], alpha=0.75, linewidth=0, zorder=3)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels(order[1:], rotation=35, ha="right")
    ax_b.set_ylabel("20次重复的 hold-out 误差 (px)")
    ax_b.set_title("b  重复抽样稳定性", loc="left", fontweight="bold")
    ax_b.grid(axis="y", color="#E6E8EF", lw=0.55)
    ax_b.set_ylim(120, 780)
    ax_b.text(2.5, 705, "25点策略分布更集中", ha="center", fontsize=7, color=PALETTE["neutral_dark"])

    runtime = load_runtime_summary()
    modes = [("runtime_classic", "Classic"), ("runtime_deep", "Deep")]
    x = np.arange(len(modes))
    mean_vals = [float(runtime[key]["computed_mean_residual_px"]) for key, _ in modes]
    med_vals = [float(runtime[key]["computed_median_residual_px"]) for key, _ in modes]
    max_vals = [float(runtime[key]["computed_max_residual_px"]) for key, _ in modes]
    for i, ((_, label), mean, med, max_v) in enumerate(zip(modes, mean_vals, med_vals, max_vals)):
        color = PALETTE["classic"] if label == "Classic" else PALETTE["deep"]
        ax_c.vlines(i, med, max_v, color=color, lw=2.2, alpha=0.7)
        ax_c.scatter(i, mean, s=52, color=color, edgecolor="white", linewidth=0.8, zorder=4, label="mean" if i == 0 else None)
        ax_c.scatter(i, med, s=26, color="white", edgecolor=color, linewidth=1.0, zorder=4, label="median" if i == 0 else None)
        ax_c.scatter(i, max_v, marker="_", s=160, color=color, linewidth=2.0, zorder=4)
        ax_c.text(i + 0.08, mean, f"均值 {mean:.1f}", va="center", fontsize=7)
        ax_c.text(i + 0.08, max_v, f"最大 {max_v:.1f}", va="center", fontsize=7)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([label for _, label in modes])
    ax_c.set_ylabel("校准拟合残差 (px)")
    ax_c.set_title("c  最终实时校准文件", loc="left", fontweight="bold")
    ax_c.grid(axis="y", color="#E6E8EF", lw=0.55)
    ax_c.set_ylim(0, max(max_vals) * 1.18)
    ax_c.text(0.5, -0.24, "c 为拟合残差，不能等同于 hold-out 误差", transform=ax_c.transAxes, ha="center", fontsize=6.8)

    fig.suptitle("不同校准策略与最终25点校准结果", y=0.99, fontsize=10.5, fontweight="bold")
    save_all(fig, "fig6_6_publication_calibration")
    plt.close(fig)


def fig6_6_simple_publication() -> None:
    """A simpler candidate for direct thesis replacement of Figure 6-6."""
    plt = setup_matplotlib()
    summary = pd.read_csv(PROJECT_ROOT / "evaluation_results" / "calibration_compare_extended" / "results.csv")
    runtime = load_runtime_summary()

    selected = [
        ("none", 0, "无校准", PALETTE["soft_gray"]),
        ("affine", 9, "9点 affine", PALETTE["muted_blue"]),
        ("polynomial", 9, "9点 polynomial", PALETTE["muted_rose"]),
        ("affine", 25, "25点 affine", PALETTE["muted_teal"]),
        ("polynomial", 25, "25点 polynomial", PALETTE["muted_gold"]),
    ]
    rows = []
    for method, points, label, color in selected:
        row = summary[(summary["method"] == method) & (summary["num_calib_points"] == points)].iloc[0]
        rows.append({**row.to_dict(), "label": label, "color": color})
    plot_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.35), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.32})
    fig.patch.set_facecolor(PALETTE["paper_bg"])

    ax = axes[0]
    x = np.arange(len(plot_df))
    bars = ax.bar(
        x,
        plot_df["holdout_error_px_mean"],
        yerr=plot_df["holdout_error_px_std"],
        capsize=3.5,
        color=plot_df["color"],
        edgecolor="#FFFFFF",
        linewidth=0.8,
    )
    ax.set_title("A. 离线校准对比：hold-out 像素误差", loc="left", fontweight="bold")
    ax.set_ylabel("平均误差 (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 720)
    for bar, value in zip(bars, plot_df["holdout_error_px_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 18, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    affine25 = plot_df[plot_df["label"] == "25点 affine"].iloc[0]
    ax.annotate(
        "本轮最低",
        xy=(3, affine25["holdout_error_px_mean"]),
        xytext=(3.55, 105),
        arrowprops=dict(arrowstyle="-|>", lw=0.85, color=PALETTE["muted_teal"]),
        color=PALETTE["muted_teal"],
        fontsize=7,
        ha="left",
    )
    ax.text(
        4.05,
        255,
        "25点 polynomial\n优于9点策略",
        color=PALETTE["muted_gold"],
        fontsize=7,
        ha="center",
    )

    ax2 = axes[1]
    runtime_rows = [
        ("Classic", runtime["runtime_classic"], PALETTE["muted_blue"]),
        ("Deep", runtime["runtime_deep"], PALETTE["muted_teal"]),
    ]
    labels = [r[0] for r in runtime_rows]
    means = [float(r[1]["computed_mean_residual_px"]) for r in runtime_rows]
    medians = [float(r[1]["computed_median_residual_px"]) for r in runtime_rows]
    maxes = [float(r[1]["computed_max_residual_px"]) for r in runtime_rows]
    colors = [r[2] for r in runtime_rows]
    x2 = np.arange(len(labels))
    bars2 = ax2.bar(x2, means, color=colors, edgecolor="#FFFFFF", linewidth=0.8, width=0.58)
    for i, (mean, median, max_v, color) in enumerate(zip(means, medians, maxes, colors)):
        ax2.plot([i, i], [median, max_v], color=color, lw=1.7, alpha=0.75)
        ax2.scatter(i, max_v, marker="_", s=130, color=color, lw=1.8, zorder=4)
        ax2.scatter(i, median, s=28, facecolor="white", edgecolor=color, lw=1.0, zorder=4)
        ax2.text(i, mean + 8, f"均值 {mean:.1f}", ha="center", fontsize=7)
        ax2.text(i + 0.12, max_v + 2, f"最大 {max_v:.1f}", ha="left", va="center", fontsize=6.8, color=PALETTE["ink"])
    ax2.set_title("B. 最终实时校准：拟合残差", loc="left", fontweight="bold")
    ax2.set_ylabel("残差 (px)")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 300)
    ax2.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax2.set_axisbelow(True)
    ax2.text(0.5, -0.28, "B区为校准拟合残差，不能等同于A区hold-out误差", transform=ax2.transAxes, ha="center", fontsize=6.8, color="#565B66")

    fig.suptitle("不同校准策略与最终25点校准结果对比", y=1.03, fontsize=10.5, fontweight="bold")
    save_all(fig, "fig6_6_publication_simple")
    plt.close(fig)


def loo_dataframe() -> pd.DataFrame:
    data = json.loads((PROJECT_ROOT / "evaluation_results" / "leave_one_out_resume" / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for item in data["per_user"]:
        metrics = item["test_metrics"]
        rows.append(
            {
                "user": int(item["left_out_user"]),
                "test_samples": int(item["test_samples"]),
                "mean_angle_error": float(metrics["mean_angle_error"]),
                "std_angle_error": float(metrics["std_angle_error"]),
                "mean_pixel_error": float(metrics["mean_pixel_error"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fig6_4_6_5_loo_source_data.csv", index=False, encoding="utf-8-sig")
    return df


def fig6_4_6_5_publication() -> None:
    plt = setup_matplotlib()
    df = loo_dataframe()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 4.6), gridspec_kw={"wspace": 0.36})

    angle_df = df.sort_values("mean_angle_error", ascending=True).reset_index(drop=True)
    y = np.arange(len(angle_df))
    colors = np.where(angle_df["mean_angle_error"] <= 3.0, PALETTE["baseline"], PALETTE["warn"])
    axes[0].hlines(y, 0, angle_df["mean_angle_error"], color="#D8DAE2", lw=0.8)
    axes[0].scatter(angle_df["mean_angle_error"], y, s=22, color=colors, edgecolor="white", linewidth=0.4, zorder=3)
    axes[0].axvline(3.0, color=PALETTE["gain"], lw=1.0, ls="--")
    axes[0].axvline(angle_df["mean_angle_error"].mean(), color=PALETTE["neutral_dark"], lw=0.9)
    axes[0].set_yticks(y[::3])
    axes[0].set_yticklabels([str(v) for v in angle_df["user"].iloc[::3]])
    axes[0].set_xlabel("平均角度误差 (°)")
    axes[0].set_ylabel("留一用户 ID（按误差排序）")
    axes[0].set_title("a  LOO 角度误差", loc="left", fontweight="bold")
    axes[0].grid(axis="x", color="#E6E8EF", lw=0.55)
    axes[0].text(3.05, 2.0, "3°阈值", color=PALETTE["gain"], fontsize=7, rotation=90, va="bottom")
    axes[0].text(angle_df["mean_angle_error"].mean() + 0.05, 27, f"均值 {angle_df['mean_angle_error'].mean():.2f}°", fontsize=7)

    pixel_df = df.sort_values("mean_pixel_error", ascending=True).reset_index(drop=True)
    y = np.arange(len(pixel_df))
    axes[1].hlines(y, 250, pixel_df["mean_pixel_error"], color="#D8DAE2", lw=0.8)
    axes[1].scatter(pixel_df["mean_pixel_error"], y, s=22, color=PALETTE["affine"], edgecolor="white", linewidth=0.4, zorder=3)
    axes[1].axvline(pixel_df["mean_pixel_error"].mean(), color=PALETTE["neutral_dark"], lw=0.9)
    axes[1].set_yticks(y[::3])
    axes[1].set_yticklabels([str(v) for v in pixel_df["user"].iloc[::3]])
    axes[1].set_xlabel("平均屏幕像素误差 (px)")
    axes[1].set_title("b  LOO 屏幕像素误差", loc="left", fontweight="bold")
    axes[1].grid(axis="x", color="#E6E8EF", lw=0.55)
    axes[1].text(pixel_df["mean_pixel_error"].mean() + 1.5, 27, f"均值 {pixel_df['mean_pixel_error'].mean():.1f}px", fontsize=7)
    axes[1].set_xlim(250, max(pixel_df["mean_pixel_error"]) + 25)

    fig.suptitle("31名用户留一交叉验证结果", y=0.99, fontsize=10.5, fontweight="bold")
    save_all(fig, "fig6_4_6_5_publication_loo_combined")
    plt.close(fig)

    # Separate candidates for one-to-one replacement of existing Figure 6-4 and 6-5.
    for metric, stem, xlabel, title, threshold in [
        ("mean_angle_error", "fig6_4_publication_loo_angle", "平均角度误差 (°)", "31名用户 LOO 角度误差分布", 3.0),
        ("mean_pixel_error", "fig6_5_publication_loo_pixel", "平均屏幕像素误差 (px)", "31名用户 LOO 屏幕像素误差分布", None),
    ]:
        data = df.sort_values(metric, ascending=True).reset_index(drop=True)
        fig2, ax = plt.subplots(figsize=(6.5, 3.4))
        y = np.arange(len(data))
        point_colors = np.where(data[metric] <= threshold, PALETTE["baseline"], PALETTE["warn"]) if threshold else PALETTE["affine"]
        xmin = 0 if metric == "mean_angle_error" else 250
        ax.hlines(y, xmin, data[metric], color="#D8DAE2", lw=0.8)
        ax.scatter(data[metric], y, s=24, color=point_colors, edgecolor="white", linewidth=0.4, zorder=3)
        ax.axvline(data[metric].mean(), color=PALETTE["neutral_dark"], lw=0.9)
        if threshold:
            ax.axvline(threshold, color=PALETTE["gain"], lw=1.0, ls="--")
        ax.set_yticks(y[::3])
        ax.set_yticklabels([str(v) for v in data["user"].iloc[::3]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("留一用户 ID（按误差排序）")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", color="#E6E8EF", lw=0.55)
        save_all(fig2, stem)
        plt.close(fig2)


def head_pose_summary() -> pd.DataFrame:
    data = json.loads((PROJECT_ROOT / "evaluation_results" / "head_pose_ablation" / "summary.json").read_text(encoding="utf-8"))
    order = [
        ("no_pose", "无旋转"),
        ("no_pitch", "消融 pitch"),
        ("no_yaw", "消融 yaw"),
        ("full_pose", "完整 pose"),
    ]
    rows = []
    for key, label in order:
        item = data["conditions"][key]
        rows.append(
            {
                "condition": key,
                "label": label,
                "mean_pixel_error": float(item["mean_pixel_error"]),
                "median_pixel_error": float(item["median_pixel_error"]),
                "std_pixel_error": float(item["std_pixel_error"]),
                "num_samples": int(item["num_samples"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "fig6_7_head_pose_summary.csv", index=False, encoding="utf-8-sig")
    return df


def fig6_7_publication() -> None:
    plt = setup_matplotlib()
    df = head_pose_summary()
    contrib = json.loads((PROJECT_ROOT / "evaluation_results" / "head_pose_ablation" / "summary.json").read_text(encoding="utf-8"))["head_pose_contribution"]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = np.arange(len(df))
    colors = [PALETTE["warn"], PALETTE["poly_dark"], PALETTE["baseline_soft"], PALETTE["affine"]]
    ax.errorbar(
        df["mean_pixel_error"],
        y,
        xerr=df["std_pixel_error"],
        fmt="none",
        ecolor="#C6CAD3",
        elinewidth=1.8,
        capsize=3,
        zorder=1,
    )
    ax.scatter(df["mean_pixel_error"], y, s=70, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.invert_yaxis()
    ax.set_xlabel("平均屏幕像素误差 (px)")
    ax.set_title("头姿几何链路消融结果", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E6E8EF", lw=0.55)
    for yi, row in zip(y, df.itertuples()):
        ax.text(row.mean_pixel_error + 8, yi, f"{row.mean_pixel_error:.1f}", va="center", fontsize=7)

    no_pose = float(contrib["no_pose_px"])
    full_pose = float(contrib["full_pose_px"])
    reduction = float(contrib["pixel_error_reduction"])
    pct = float(contrib["pixel_error_reduction_pct"])
    ax.annotate(
        f"降低 {reduction:.1f}px\n({pct:.1f}%)",
        xy=(full_pose, 3),
        xytext=(430, 2.45),
        arrowprops=dict(arrowstyle="-|>", lw=1.0, color=PALETTE["gain"]),
        color=PALETTE["gain"],
        ha="center",
        fontsize=8,
    )
    ax.plot([full_pose, no_pose], [3, 0], color=PALETTE["gain"], lw=1.2, alpha=0.8, zorder=2)
    ax.set_xlim(250, 720)
    ax.text(0.99, 0.04, "误差条为样本标准差，n=1390", transform=ax.transAxes, ha="right", fontsize=6.8, color=PALETTE["neutral_dark"])
    save_all(fig, "fig6_7_publication_head_pose_ablation")
    plt.close(fig)


def fig6_7_simple_publication() -> None:
    """A cleaner bar-based candidate for direct thesis replacement of Figure 6-7."""
    plt = setup_matplotlib()
    df = head_pose_summary()
    contrib = json.loads((PROJECT_ROOT / "evaluation_results" / "head_pose_ablation" / "summary.json").read_text(encoding="utf-8"))["head_pose_contribution"]

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    order = ["no_pose", "no_pitch", "no_yaw", "full_pose"]
    labels = ["无旋转", "消融 pitch", "消融 yaw", "完整 pose"]
    df = df.set_index("condition").loc[order].reset_index()
    x = np.arange(len(df))
    colors = [PALETTE["muted_rose"], "#D6A1A1", PALETTE["baseline_soft"], PALETTE["muted_blue"]]
    bars = ax.bar(
        x,
        df["mean_pixel_error"],
        yerr=df["std_pixel_error"],
        capsize=3.5,
        color=colors,
        edgecolor="#FFFFFF",
        linewidth=0.8,
        width=0.62,
    )
    ax.set_title("头姿几何链路消融结果", loc="left", fontweight="bold")
    ax.set_ylabel("平均屏幕像素误差 (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 760)
    for bar, value in zip(bars, df["mean_pixel_error"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 18, f"{value:.1f}", ha="center", fontsize=7)

    full_idx = labels.index("完整 pose")
    nopose_idx = labels.index("无旋转")
    ax.annotate(
        f"完整pose相对无旋转降低 {contrib['pixel_error_reduction']:.1f}px ({contrib['pixel_error_reduction_pct']:.1f}%)",
        xy=(full_idx, contrib["full_pose_px"]),
        xytext=(1.65, 650),
        arrowprops=dict(arrowstyle="-|>", lw=0.95, color=PALETTE["gain"]),
        color=PALETTE["gain"],
        fontsize=7.5,
        ha="center",
    )
    ax.plot([nopose_idx, full_idx], [contrib["no_pose_px"], contrib["full_pose_px"]], color=PALETTE["gain"], lw=1.0, alpha=0.9)
    ax.text(0.99, 0.03, "误差条为样本标准差，n=1390", transform=ax.transAxes, ha="right", fontsize=6.8, color="#565B66")
    save_all(fig, "fig6_7_publication_simple")
    plt.close(fig)


def fig6_1_dataset_split_publication() -> None:
    plt = setup_matplotlib()
    df = load_per_sample_dataframe()
    order = ["train", "val", "test"]
    labels = ["训练集", "验证集", "固定测试集"]
    counts = df.groupby("split").size().reindex(order).astype(int)
    colors = [PALETTE["muted_blue"], PALETTE["muted_teal"], PALETTE["muted_gold"]]

    fig, ax = plt.subplots(figsize=(5.9, 3.2))
    x = np.arange(len(order))
    bars = ax.bar(x, counts.to_numpy(), color=colors, width=0.58, edgecolor="#FFFFFF", linewidth=0.8)
    ax.set_title("数据集划分样本数量分布", loc="left", fontweight="bold")
    ax.set_ylabel("样本数量")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(0, int(counts.max() * 1.22))
    total = int(counts.sum())
    for bar, value in zip(bars, counts):
        pct = value / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, value + counts.max() * 0.035, f"{value}\n{pct:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.text(0.99, 0.94, f"总样本数 n={total}", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#565B66")
    save_all(fig, "fig6_1_publication_dataset_split")
    plt.close(fig)


def fig6_2_error_distribution_publication() -> None:
    plt = setup_matplotlib()
    df = load_per_sample_dataframe()
    test = df[df["split"] == "test"].copy()
    values = test["angle_error"].to_numpy(dtype=np.float64)
    mean = float(np.mean(values))
    median = float(np.median(values))
    p90 = float(np.percentile(values, 90))

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bins = np.linspace(0, max(12.0, values.max()), 28)
    ax.hist(values, bins=bins, color=PALETTE["muted_blue"], alpha=0.82, edgecolor="white", linewidth=0.35)
    ax.axvline(mean, color=PALETTE["muted_rose"], lw=1.15, label=f"均值 {mean:.2f}°")
    ax.axvline(median, color=PALETTE["muted_teal"], lw=1.15, ls="--", label=f"中位数 {median:.2f}°")
    ax.axvline(p90, color=PALETTE["muted_gold"], lw=1.15, ls=":", label=f"P90 {p90:.2f}°")
    ax.set_title("固定测试集角度误差分布", loc="left", fontweight="bold")
    ax.set_xlabel("角度误差 (°)")
    ax.set_ylabel("样本数")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    ax.text(0.02, 0.94, f"n={len(values)}", transform=ax.transAxes, ha="left", va="top", fontsize=7, color="#565B66")
    save_all(fig, "fig6_2_publication_angle_error_distribution")
    plt.close(fig)


def fig6_3_target_prediction_publication() -> None:
    plt = setup_matplotlib()
    from torch.utils.data import DataLoader
    from data.dataset import GazeDataset
    from evaluate_pog import evaluate_pog, load_config, load_model

    config = load_config(str(PROJECT_ROOT / "configs" / "train_pog_config.yaml"))
    data_cfg = config.get("data", {})
    test_dir = PROJECT_ROOT / data_cfg.get("dataset_processed_dir", "dataset_processed") / "test"
    labels_path = test_dir / "labels.csv"
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "deep_pog_zero" / "best_model.pth"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing processed test labels: {labels_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Deep PoG checkpoint: {checkpoint_path}")

    df = pd.read_csv(labels_path)
    dataset = GazeDataset(
        df,
        image_root=test_dir,
        model_version="v2",
        target_mode="pog2d",
        head_pose_mode=data_cfg.get("head_pose_mode", "zero"),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model = load_model(str(checkpoint_path), config)
    results = evaluate_pog(model, loader, screen_w=1920, screen_h=1080)

    screen_w, screen_h = 1920.0, 1080.0
    target = np.asarray(results["true_points"], dtype=np.float64) * np.array([screen_w, screen_h])
    pred = np.asarray(results["pred_points"], dtype=np.float64) * np.array([screen_w, screen_h])
    target_unique = pd.DataFrame(target, columns=["x", "y"]).round(3).drop_duplicates().to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 3.75))
    ax.scatter(pred[:, 0], pred[:, 1], s=7, color=PALETTE["muted_blue"], alpha=0.26, linewidth=0, label="预测点")
    ax.scatter(target_unique[:, 0], target_unique[:, 1], s=34, facecolor=PALETTE["muted_gold"], edgecolor="white", linewidth=0.55, label="目标点")
    ax.set_title("固定测试集目标点与预测点分布", loc="left", fontweight="bold")
    ax.set_xlabel("屏幕 x 坐标 (px)")
    ax.set_ylabel("屏幕 y 坐标 (px)")
    ax.set_xlim(0, screen_w)
    ax.set_ylim(screen_h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color=PALETTE["grid"], lw=0.55)
    ax.legend(loc="lower right")
    ax.text(0.01, 0.04, f"预测样本 n={len(pred)}，目标点 {len(target_unique)} 个", transform=ax.transAxes, ha="left", fontsize=7, color="#565B66")
    save_all(fig, "fig6_3_publication_target_prediction")
    plt.close(fig)


def fig5_6_smoothing_publication() -> None:
    plt = setup_matplotlib()
    data = json.loads((PROJECT_ROOT / "evaluation_results" / "smoothing_compare" / "results.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(data)
    selected_names = ["No smoothing", "EMA α=0.2", "EMA α=0.3", "1Euro mc=0.5 β=0.005", "1Euro mc=1.0 β=0.007"]
    plot_df = df[df["name"].isin(selected_names)].copy()
    plot_df["label"] = plot_df["name"].replace(
        {
            "No smoothing": "无平滑",
            "EMA α=0.2": "EMA 0.2",
            "EMA α=0.3": "EMA 0.3",
            "1Euro mc=0.5 β=0.005": "1Euro 0.5",
            "1Euro mc=1.0 β=0.007": "1Euro 1.0",
        }
    )
    plot_df = plot_df.set_index("name").loc[selected_names].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35), gridspec_kw={"wspace": 0.34})
    colors = [PALETTE["soft_gray"], PALETTE["muted_blue"], PALETTE["muted_teal"], PALETTE["muted_gold"], PALETTE["muted_rose"]]
    x = np.arange(len(plot_df))

    ax = axes[0]
    bars = ax.bar(x, plot_df["fixation_jitter"], color=colors, width=0.58, edgecolor="#FFFFFF", linewidth=0.8)
    ax.set_title("A. 静止注视抖动", loc="left", fontweight="bold")
    ax.set_ylabel("抖动均值 (px)")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["label"], rotation=25, ha="right")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, plot_df["fixation_jitter"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.55, f"{value:.1f}", ha="center", fontsize=6.8)

    ax2 = axes[1]
    ax2.scatter(plot_df["latency_proxy"], plot_df["rmse"], s=64, color=colors, edgecolor="white", linewidth=0.7)
    for row in plot_df.itertuples():
        ax2.text(row.latency_proxy + 2.0, row.rmse, row.label, va="center", fontsize=6.8)
    ax2.set_title("B. 延迟与跟踪误差折中", loc="left", fontweight="bold")
    ax2.set_xlabel("延迟代理指标")
    ax2.set_ylabel("RMSE (px)")
    ax2.grid(color=PALETTE["grid"], lw=0.7)
    ax2.set_axisbelow(True)
    ax2.set_xlim(20, max(plot_df["latency_proxy"]) + 28)
    ax2.set_ylim(15, max(plot_df["rmse"]) + 12)

    fig.suptitle("平滑策略对实时注视轨迹的影响", y=1.03, fontsize=10.5, fontweight="bold")
    save_all(fig, "fig5_6_publication_smoothing_summary")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clean_non_png_outputs()
    fig5_6_smoothing_publication()
    fig6_1_dataset_split_publication()
    fig6_2_error_distribution_publication()
    fig6_3_target_prediction_publication()
    fig6_6_publication()
    fig6_6_simple_publication()
    fig6_4_6_5_publication()
    fig6_7_publication()
    fig6_7_simple_publication()
    manifest = {
        "fig6_6": "fig6_6_publication_calibration",
        "fig6_6_simple": "fig6_6_publication_simple",
        "fig6_4_6_5_combined": "fig6_4_6_5_publication_loo_combined",
        "fig6_4": "fig6_4_publication_loo_angle",
        "fig6_5": "fig6_5_publication_loo_pixel",
        "fig6_7": "fig6_7_publication_head_pose_ablation",
        "fig6_7_simple": "fig6_7_publication_simple",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    clean_non_png_outputs()
    print(f"Wrote publication-style figures to {OUT}")


if __name__ == "__main__":
    main()
