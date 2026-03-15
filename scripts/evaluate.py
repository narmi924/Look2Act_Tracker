"""
评估入口脚本：在测试集上评估 GazeNet 模型并生成可视化图表。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python scripts/evaluate.py
         conda run -n gaze-env python scripts/evaluate.py --checkpoint checkpoints/best_model.pth

输出指标：平均角度误差、中位角度误差、屏幕像素误差均值
输出图表：误差分布直方图、注视点热力图、各 Session 误差对比柱状图
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet
from models.losses import angular_loss
from data.dataset import GazeDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, config: dict) -> GazeNet:
    """从 checkpoint 加载模型。"""
    model_cfg = config.get("model", {})
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    model = GazeNet(num_channels=channels)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    logger.info(f"已加载模型: {checkpoint_path}")
    return model


@torch.no_grad()
def evaluate_on_test(
    model: GazeNet,
    test_loader: DataLoader,
    screen_w: int = 1536,
    screen_h: int = 864,
) -> dict:
    """在测试集上计算评估指标。

    返回:
        包含指标和逐样本结果的字典
    """
    all_angle_errors = []  # 角度误差（度）
    all_pixel_errors = []  # 屏幕像素误差
    all_session_ids = []
    all_pred_nx = []  # 预测归一化 x（近似）
    all_pred_ny = []  # 预测归一化 y（近似）
    all_true_nx = []
    all_true_ny = []

    for batch in test_loader:
        eye_imgs = batch["eye_img"]
        gaze_targets = batch["gaze"]
        meta = batch["meta"]

        preds = model(eye_imgs)

        # 逐样本计算角度误差
        for i in range(preds.shape[0]):
            pred_vec = preds[i].numpy()
            true_vec = gaze_targets[i].numpy()

            # 角度误差（度）
            cos_sim = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_sim))
            all_angle_errors.append(angle_deg)

            # 屏幕像素误差（基于归一化目标坐标的近似计算）
            true_nx = float(meta["norm_target_x"][i])
            true_ny = float(meta["norm_target_y"][i])

            # 从视线向量近似推算归一化屏幕坐标
            # 简化：使用 gaze_x/gaze_z 和 gaze_y/gaze_z 的比值
            if abs(pred_vec[2]) > 1e-6:
                pred_nx = 0.5 + pred_vec[0] / pred_vec[2] * 0.5
                pred_ny = 0.5 + pred_vec[1] / pred_vec[2] * 0.5
            else:
                pred_nx, pred_ny = 0.5, 0.5

            pred_nx = np.clip(pred_nx, 0, 1)
            pred_ny = np.clip(pred_ny, 0, 1)

            px_err = np.sqrt(
                ((pred_nx - true_nx) * screen_w) ** 2
                + ((pred_ny - true_ny) * screen_h) ** 2
            )
            all_pixel_errors.append(px_err)

            all_session_ids.append(str(meta["session_id"][i]))
            all_pred_nx.append(pred_nx)
            all_pred_ny.append(pred_ny)
            all_true_nx.append(true_nx)
            all_true_ny.append(true_ny)

    angle_errors = np.array(all_angle_errors)
    pixel_errors = np.array(all_pixel_errors)

    metrics = {
        "mean_angle_error": float(np.mean(angle_errors)),
        "median_angle_error": float(np.median(angle_errors)),
        "mean_pixel_error": float(np.mean(pixel_errors)),
        "std_angle_error": float(np.std(angle_errors)),
        "num_samples": len(angle_errors),
    }

    results = {
        "metrics": metrics,
        "angle_errors": angle_errors,
        "pixel_errors": pixel_errors,
        "session_ids": all_session_ids,
        "pred_nx": np.array(all_pred_nx),
        "pred_ny": np.array(all_pred_ny),
        "true_nx": np.array(all_true_nx),
        "true_ny": np.array(all_true_ny),
    }
    return results


def generate_plots(results: dict, output_dir: Path) -> None:
    """生成可视化图表。

    1. 误差分布直方图
    2. 注视点热力图
    3. 各 Session 误差对比柱状图
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    angle_errors = results["angle_errors"]
    session_ids = results["session_ids"]
    true_nx = results["true_nx"]
    true_ny = results["true_ny"]

    # 1. 误差分布直方图
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(angle_errors, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Angle Error (degrees)")
    ax.set_ylabel("Count")
    ax.set_title("Gaze Angle Error Distribution")
    mean_err = results["metrics"]["mean_angle_error"]
    median_err = results["metrics"]["median_angle_error"]
    ax.axvline(mean_err, color="red", linestyle="--", label=f"Mean: {mean_err:.2f}°")
    ax.axvline(median_err, color="orange", linestyle="--", label=f"Median: {median_err:.2f}°")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "error_histogram.png", dpi=150)
    plt.close(fig)

    # 2. 注视点热力图
    fig, ax = plt.subplots(figsize=(8, 5))
    hb = ax.hexbin(true_nx, true_ny, gridsize=20, cmap="YlOrRd", mincnt=1)
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title("Gaze Target Heatmap")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    fig.colorbar(hb, ax=ax, label="Sample Count")
    fig.tight_layout()
    fig.savefig(output_dir / "gaze_heatmap.png", dpi=150)
    plt.close(fig)

    # 3. 各 Session 误差对比柱状图
    df = pd.DataFrame({"session_id": session_ids, "angle_error": angle_errors})
    session_means = df.groupby("session_id")["angle_error"].mean().sort_values()

    if len(session_means) > 0:
        fig, ax = plt.subplots(figsize=(max(8, len(session_means) * 0.8), 5))
        bars = ax.bar(range(len(session_means)), session_means.values, alpha=0.7)
        ax.set_xticks(range(len(session_means)))
        labels = [sid[:12] + "..." if len(sid) > 12 else sid for sid in session_means.index]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Session ID")
        ax.set_ylabel("Mean Angle Error (degrees)")
        ax.set_title("Per-Session Mean Angle Error")
        fig.tight_layout()
        fig.savefig(output_dir / "session_errors.png", dpi=150)
        plt.close(fig)

    logger.info(f"图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Look2Act GazeNet 评估")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results",
        help="评估结果输出目录",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    data_cfg = config.get("data", {})

    # 加载模型
    model = load_model(args.checkpoint, config)

    # 加载测试数据
    processed_dir = Path(data_cfg.get("dataset_processed_dir", "dataset_processed"))
    test_dir = processed_dir / "test"
    labels_path = test_dir / "labels.csv"

    if not labels_path.exists():
        logger.error(f"测试数据不存在: {labels_path}")
        sys.exit(1)

    df = pd.read_csv(labels_path)
    logger.info(f"测试集: {len(df)} 样本")

    test_ds = GazeDataset(df, image_root=test_dir, augment=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # 评估
    results = evaluate_on_test(model, test_loader)
    metrics = results["metrics"]

    logger.info("=== 评估结果 ===")
    logger.info(f"  平均角度误差: {metrics['mean_angle_error']:.2f}°")
    logger.info(f"  中位角度误差: {metrics['median_angle_error']:.2f}°")
    logger.info(f"  屏幕像素误差均值: {metrics['mean_pixel_error']:.1f} px")
    logger.info(f"  角度误差标准差: {metrics['std_angle_error']:.2f}°")
    logger.info(f"  样本数: {metrics['num_samples']}")

    # 保存指标 JSON
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 生成可视化图表
    generate_plots(results, output_dir)

    logger.info(f"评估完成，结果保存到: {output_dir}")


if __name__ == "__main__":
    main()
