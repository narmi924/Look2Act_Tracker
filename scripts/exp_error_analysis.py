"""
条件分桶误差分析脚本。

实验目标（对应 Paper1 MPIIGaze、Paper2 Full-Face、Paper5 RT-GENE、Paper7 OpenGaze 的建议）：
1. 误差 vs 头部姿态（yaw/pitch 分桶）
2. 误差 vs session（跨 session 稳定性）
3. 屏幕空间误差热图（中心 vs 边角）
4. 误差 vs 用户（跨用户泛化）

工作目录：Look2Act_Tracker_Project/
运行方式：conda run -n gaze-env python scripts/exp_error_analysis.py

输出：evaluation_results/error_analysis/
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet
from data.dataset import GazeDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_per_sample_errors(
    model: GazeNet,
    df: pd.DataFrame,
    image_root: Path,
    screen_w: int = 1536,
    screen_h: int = 864,
) -> pd.DataFrame:
    """计算每个样本的角度误差和像素误差，附带元数据。

    注意：head_yaw/pitch/roll 直接从 DataFrame 读取，
    因为 GazeDataset 的 meta 不包含这些字段。
    """
    ds = GazeDataset(df, image_root=image_root, augment=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    records = []
    sample_idx = 0
    with torch.no_grad():
        for batch in loader:
            preds = model(batch["eye_img"])
            gaze_targets = batch["gaze"]
            meta = batch["meta"]
            batch_size = preds.shape[0]

            for i in range(batch_size):
                pred_vec = preds[i].numpy()
                true_vec = gaze_targets[i].numpy()

                # 角度误差
                cos_sim = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_sim))

                # 像素误差
                true_nx = float(meta["norm_target_x"][i])
                true_ny = float(meta["norm_target_y"][i])
                if abs(pred_vec[2]) > 1e-6:
                    pred_nx = np.clip(0.5 + pred_vec[0] / pred_vec[2] * 0.5, 0, 1)
                    pred_ny = np.clip(0.5 + pred_vec[1] / pred_vec[2] * 0.5, 0, 1)
                else:
                    pred_nx, pred_ny = 0.5, 0.5

                px_err = np.sqrt(
                    ((pred_nx - true_nx) * screen_w) ** 2
                    + ((pred_ny - true_ny) * screen_h) ** 2
                )

                # 从原始 DataFrame 读取 head pose（Dataset meta 不含这些字段）
                row = df.iloc[sample_idx]
                records.append({
                    "angle_error": angle_deg,
                    "pixel_error": px_err,
                    "head_yaw": float(row.get("head_yaw", 0)),
                    "head_pitch": float(row.get("head_pitch", 0)),
                    "head_roll": float(row.get("head_roll", 0)),
                    "norm_target_x": true_nx,
                    "norm_target_y": true_ny,
                    "pred_nx": pred_nx,
                    "pred_ny": pred_ny,
                    "session_id": str(row.get("session_id", "")),
                    "user_id": str(row.get("user_id", "")),
                })
                sample_idx += 1

    return pd.DataFrame(records)


def generate_error_analysis_plots(result_df: pd.DataFrame, output_dir: Path) -> None:
    """生成条件分桶误差分析的全部图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # ========== 图 1: 误差 vs Head Yaw 分桶 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # yaw 分桶
    yaw_vals = result_df["head_yaw"].values
    yaw_bins = np.linspace(yaw_vals.min(), yaw_vals.max(), 8)
    result_df["yaw_bin"] = pd.cut(result_df["head_yaw"], bins=yaw_bins)
    yaw_grouped = result_df.groupby("yaw_bin", observed=True)["angle_error"]
    yaw_means = yaw_grouped.mean()
    yaw_stds = yaw_grouped.std()
    yaw_counts = yaw_grouped.count()

    ax = axes[0]
    x_labels = [f"{iv.left:.0f}~{iv.right:.0f}" for iv in yaw_means.index]
    x_pos = range(len(x_labels))
    ax.bar(x_pos, yaw_means.values, yerr=yaw_stds.values, capsize=3, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Head Yaw (degrees)")
    ax.set_ylabel("Mean Angle Error (degrees)")
    ax.set_title("Error vs Head Yaw")
    # 在柱上标注样本数
    for j, (cnt, mean_v) in enumerate(zip(yaw_counts.values, yaw_means.values)):
        ax.text(j, mean_v + 0.2, f"n={cnt}", ha="center", fontsize=7)

    # pitch 分桶
    pitch_vals = result_df["head_pitch"].values
    pitch_bins = np.linspace(pitch_vals.min(), pitch_vals.max(), 8)
    result_df["pitch_bin"] = pd.cut(result_df["head_pitch"], bins=pitch_bins)
    pitch_grouped = result_df.groupby("pitch_bin", observed=True)["angle_error"]
    pitch_means = pitch_grouped.mean()
    pitch_stds = pitch_grouped.std()
    pitch_counts = pitch_grouped.count()

    ax = axes[1]
    x_labels = [f"{iv.left:.0f}~{iv.right:.0f}" for iv in pitch_means.index]
    x_pos = range(len(x_labels))
    ax.bar(x_pos, pitch_means.values, yerr=pitch_stds.values, capsize=3, alpha=0.7, color="orange")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Head Pitch (degrees)")
    ax.set_ylabel("Mean Angle Error (degrees)")
    ax.set_title("Error vs Head Pitch")
    for j, (cnt, mean_v) in enumerate(zip(pitch_counts.values, pitch_means.values)):
        ax.text(j, mean_v + 0.2, f"n={cnt}", ha="center", fontsize=7)

    fig.suptitle("Gaze Error vs Head Pose", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_head_pose.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  已保存: error_vs_head_pose.png")

    # ========== 图 2: 跨 Session 误差对比 ==========
    fig, ax = plt.subplots(figsize=(10, 5))
    session_grouped = result_df.groupby("session_id")["angle_error"]
    session_means = session_grouped.mean().sort_values()
    session_stds = session_grouped.std()
    session_counts = session_grouped.count()

    x_pos = range(len(session_means))
    short_labels = []
    for sid in session_means.index:
        # 提取日期和设备关键信息
        parts = sid.split("_")
        date_part = parts[0][1:9] if len(parts) > 0 else sid[:8]
        dev_part = parts[2][:8] if len(parts) > 2 else ""
        glass_part = parts[-1] if len(parts) > 3 else ""
        short_labels.append(f"{date_part}\n{dev_part}\n{glass_part}")

    bars = ax.bar(x_pos, session_means.values,
                  yerr=session_stds[session_means.index].values,
                  capsize=3, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_labels, fontsize=7, ha="center")
    ax.set_xlabel("Session")
    ax.set_ylabel("Mean Angle Error (degrees)")
    ax.set_title("Per-Session Gaze Error (All Splits)")

    # 标注样本数
    for j, sid in enumerate(session_means.index):
        cnt = session_counts[sid]
        ax.text(j, session_means[sid] + 0.15, f"n={cnt}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_session.png", dpi=150)
    plt.close(fig)
    logger.info("  已保存: error_vs_session.png")

    # ========== 图 3: 跨用户误差对比 ==========
    fig, ax = plt.subplots(figsize=(8, 5))
    user_grouped = result_df.groupby("user_id")["angle_error"]
    user_means = user_grouped.mean().sort_values()
    user_stds = user_grouped.std()
    user_counts = user_grouped.count()

    x_pos = range(len(user_means))
    bars = ax.bar(x_pos, user_means.values,
                  yerr=user_stds[user_means.index].values,
                  capsize=4, alpha=0.7, color="green")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"User {uid}" for uid in user_means.index], fontsize=10)
    ax.set_xlabel("User ID")
    ax.set_ylabel("Mean Angle Error (degrees)")
    ax.set_title("Per-User Gaze Error")

    for j, uid in enumerate(user_means.index):
        cnt = user_counts[uid]
        ax.text(j, user_means[uid] + 0.15, f"n={cnt}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "error_vs_user.png", dpi=150)
    plt.close(fig)
    logger.info("  已保存: error_vs_user.png")

    # ========== 图 4: 屏幕空间误差热图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：目标点分布 + 误差颜色
    ax = axes[0]
    sc = ax.scatter(
        result_df["norm_target_x"], result_df["norm_target_y"],
        c=result_df["angle_error"], cmap="YlOrRd",
        s=15, alpha=0.6, edgecolors="none",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y 轴反转（屏幕坐标）
    ax.set_xlabel("Normalized Screen X")
    ax.set_ylabel("Normalized Screen Y")
    ax.set_title("Screen-Space Error Distribution")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Angle Error (deg)")

    # 右图：网格化平均误差热图
    ax = axes[1]
    grid_size = 5
    x_edges = np.linspace(0, 1, grid_size + 1)
    y_edges = np.linspace(0, 1, grid_size + 1)
    error_grid = np.full((grid_size, grid_size), np.nan)
    count_grid = np.zeros((grid_size, grid_size), dtype=int)

    for _, row in result_df.iterrows():
        xi = min(int(row["norm_target_x"] * grid_size), grid_size - 1)
        yi = min(int(row["norm_target_y"] * grid_size), grid_size - 1)
        if np.isnan(error_grid[yi, xi]):
            error_grid[yi, xi] = 0.0
        error_grid[yi, xi] += row["angle_error"]
        count_grid[yi, xi] += 1

    # 计算平均
    mask = count_grid > 0
    error_grid[mask] /= count_grid[mask]

    im = ax.imshow(error_grid, cmap="YlOrRd", origin="upper",
                   extent=[0, 1, 1, 0], aspect="equal")
    # 在每个格子里标注数值
    for yi in range(grid_size):
        for xi in range(grid_size):
            if count_grid[yi, xi] > 0:
                val = error_grid[yi, xi]
                ax.text(
                    (xi + 0.5) / grid_size, (yi + 0.5) / grid_size,
                    f"{val:.1f}\nn={count_grid[yi, xi]}",
                    ha="center", va="center", fontsize=8,
                    color="white" if val > np.nanmean(error_grid) else "black",
                )
    ax.set_xlabel("Normalized Screen X")
    ax.set_ylabel("Normalized Screen Y")
    ax.set_title("Grid-Averaged Error Heatmap")
    fig.colorbar(im, ax=ax, label="Mean Angle Error (deg)")

    fig.suptitle("Screen-Space Error Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "screen_error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  已保存: screen_error_heatmap.png")

    # ========== 图 5: 预测 vs 真实 散点图（偏差方向可视化）==========
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        result_df["norm_target_x"], result_df["norm_target_y"],
        c="blue", s=10, alpha=0.4, label="Target", zorder=2,
    )
    ax.scatter(
        result_df["pred_nx"], result_df["pred_ny"],
        c="red", s=10, alpha=0.4, label="Prediction", zorder=2,
    )
    # 画连线表示偏差方向
    for _, row in result_df.sample(min(100, len(result_df)), random_state=42).iterrows():
        ax.plot(
            [row["norm_target_x"], row["pred_nx"]],
            [row["norm_target_y"], row["pred_ny"]],
            color="gray", alpha=0.3, linewidth=0.5, zorder=1,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("Normalized Screen X")
    ax.set_ylabel("Normalized Screen Y")
    ax.set_title("Target vs Prediction (with offset lines)")
    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "target_vs_prediction.png", dpi=150)
    plt.close(fig)
    logger.info("  已保存: target_vs_prediction.png")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="条件分桶误差分析")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="evaluation_results/error_analysis")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"],
                        help="分析哪个数据划分")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载模型
    model_cfg = config.get("model", {})
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    model = GazeNet(num_channels=channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"已加载模型: {args.checkpoint}")

    # 加载数据
    processed_dir = Path("dataset_processed")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_dfs = []
    for split in splits:
        labels_path = processed_dir / split / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            df["split"] = split
            all_dfs.append((split, df, processed_dir / split))
            logger.info(f"  {split}: {len(df)} 样本")

    # 逐 split 计算误差
    all_results = []
    for split, df, img_root in all_dfs:
        logger.info(f"计算 {split} 集误差...")
        result_df = compute_per_sample_errors(model, df, img_root)
        result_df["split"] = split
        all_results.append(result_df)

    combined_df = pd.concat(all_results, ignore_index=True)
    logger.info(f"总计 {len(combined_df)} 样本")

    # 保存逐样本结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_dir / "per_sample_errors.csv", index=False)
    logger.info(f"逐样本结果已保存: {output_dir / 'per_sample_errors.csv'}")

    # 保存汇总统计
    summary = {
        "total_samples": len(combined_df),
        "overall": {
            "mean_angle_error": float(combined_df["angle_error"].mean()),
            "median_angle_error": float(combined_df["angle_error"].median()),
            "std_angle_error": float(combined_df["angle_error"].std()),
            "mean_pixel_error": float(combined_df["pixel_error"].mean()),
        },
        "per_split": {},
        "per_user": {},
        "per_session": {},
    }
    for split in combined_df["split"].unique():
        sub = combined_df[combined_df["split"] == split]
        summary["per_split"][split] = {
            "n": len(sub),
            "mean_angle_error": float(sub["angle_error"].mean()),
            "median_angle_error": float(sub["angle_error"].median()),
        }
    for uid in combined_df["user_id"].unique():
        sub = combined_df[combined_df["user_id"] == uid]
        summary["per_user"][str(uid)] = {
            "n": len(sub),
            "mean_angle_error": float(sub["angle_error"].mean()),
        }
    for sid in combined_df["session_id"].unique():
        sub = combined_df[combined_df["session_id"] == sid]
        summary["per_session"][sid] = {
            "n": len(sub),
            "mean_angle_error": float(sub["angle_error"].mean()),
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"汇总统计已保存: {output_dir / 'summary.json'}")

    # 生成图表
    generate_error_analysis_plots(combined_df, output_dir)
    logger.info("误差分析完成。")


if __name__ == "__main__":
    main()
