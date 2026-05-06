"""Evaluate a deep PoG checkpoint on normalized screen-point targets.

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：
    conda run --no-capture-output -n gaze-env python scripts/evaluate_pog.py --checkpoint checkpoints/deep_pog/best_model.pth
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.dataset import GazeDataset
from models.gaze_net import GazeNetPoG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: str, fallback_config: dict) -> GazeNetPoG:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = ckpt_config.get("model", fallback_config.get("model", {}))
    model = GazeNetPoG(
        num_channels=model_cfg.get("channels", [32, 64, 128, 256]),
        head_pose_dim=model_cfg.get("head_pose_dim", 3),
        fusion_dim=model_cfg.get("fusion_dim", 128),
        dropout=model_cfg.get("dropout", 0.3),
        output_activation=model_cfg.get("output_activation", "sigmoid"),
    )
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def evaluate_pog(
    model: GazeNetPoG,
    loader: DataLoader,
    screen_w: int,
    screen_h: int,
) -> dict[str, object]:
    pred_points = []
    true_points = []
    pixel_errors = []
    norm_errors = []
    session_ids = []

    for batch in loader:
        preds = model(batch["left_eye"], batch["right_eye"], batch["head_pose"])
        targets = batch["pog"]
        meta = batch["meta"]
        diff = preds - targets
        px = diff[:, 0] * screen_w
        py = diff[:, 1] * screen_h
        batch_pixel_errors = torch.sqrt(px * px + py * py).cpu().numpy()
        batch_norm_errors = torch.sqrt(torch.sum(diff * diff, dim=1)).cpu().numpy()

        pred_points.append(preds.cpu().numpy())
        true_points.append(targets.cpu().numpy())
        pixel_errors.extend(batch_pixel_errors.tolist())
        norm_errors.extend(batch_norm_errors.tolist())
        session_ids.extend([str(x) for x in meta["session_id"]])

    pred_arr = np.vstack(pred_points) if pred_points else np.empty((0, 2))
    true_arr = np.vstack(true_points) if true_points else np.empty((0, 2))
    pixel_arr = np.asarray(pixel_errors, dtype=np.float64)
    norm_arr = np.asarray(norm_errors, dtype=np.float64)

    metrics = {
        "num_samples": int(pixel_arr.shape[0]),
        "mean_pixel_error": float(np.mean(pixel_arr)) if pixel_arr.size else 0.0,
        "median_pixel_error": float(np.median(pixel_arr)) if pixel_arr.size else 0.0,
        "p95_pixel_error": float(np.quantile(pixel_arr, 0.95)) if pixel_arr.size else 0.0,
        "mean_norm_error": float(np.mean(norm_arr)) if norm_arr.size else 0.0,
        "median_norm_error": float(np.median(norm_arr)) if norm_arr.size else 0.0,
    }
    return {
        "metrics": metrics,
        "pred_points": pred_arr,
        "true_points": true_arr,
        "pixel_errors": pixel_arr,
        "session_ids": session_ids,
    }


def generate_plots(results: dict[str, object], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    errors = results["pixel_errors"]
    true_points = results["true_points"]
    pred_points = results["pred_points"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=50, edgecolor="black", alpha=0.75)
    ax.set_xlabel("Pixel Error")
    ax.set_ylabel("Count")
    ax.set_title("Deep PoG Pixel Error Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "pog_error_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    if len(true_points):
        ax.scatter(true_points[:, 0], true_points[:, 1], s=10, alpha=0.45, label="target")
        ax.scatter(pred_points[:, 0], pred_points[:, 1], s=10, alpha=0.45, label="prediction")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title("Deep PoG Target vs Prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pog_target_vs_prediction.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Look2Act deep PoG model")
    parser.add_argument("--config", default="configs/train_pog_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/deep_pog/best_model.pth")
    parser.add_argument("--output", default="evaluation_results/deep_pog")
    parser.add_argument("--screen-w", type=int, default=1920)
    parser.add_argument("--screen-h", type=int, default=1080)
    parser.add_argument("--processed-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config.get("data", {})
    test_dir = Path(args.processed_dir or data_cfg.get("dataset_processed_dir", "dataset_processed")) / "test"
    labels_path = test_dir / "labels.csv"
    if not labels_path.exists():
        logger.error("测试数据不存在: %s", labels_path)
        return 2
    if not Path(args.checkpoint).exists():
        logger.error("checkpoint 不存在: %s", args.checkpoint)
        return 2

    df = pd.read_csv(labels_path)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    data_cfg = ckpt_config.get("data", data_cfg)
    dataset = GazeDataset(
        df,
        image_root=test_dir,
        model_version="v2",
        target_mode="pog2d",
        head_pose_mode=data_cfg.get("head_pose_mode", "stored"),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    model = load_model(args.checkpoint, config)
    results = evaluate_pog(model, loader, screen_w=args.screen_w, screen_h=args.screen_h)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = results["metrics"]
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    generate_plots(results, output_dir)

    logger.info("=== Deep PoG 评估结果 ===")
    logger.info("样本数: %s", metrics["num_samples"])
    logger.info("平均像素误差: %.1f px", metrics["mean_pixel_error"])
    logger.info("中位像素误差: %.1f px", metrics["median_pixel_error"])
    logger.info("P95 像素误差: %.1f px", metrics["p95_pixel_error"])
    logger.info("结果目录: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
