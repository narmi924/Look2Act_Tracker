"""
跨用户 Leave-One-User-Out 交叉验证实验。

对每个用户：留出该用户全部数据做测试，其余用户数据做训练。
训练完成后评估留出用户的角度误差和像素误差。

工作目录：Look2Act_Tracker_Project/
运行方式：conda run -n gaze-env python scripts/exp_leave_one_out.py
         conda run -n gaze-env python scripts/exp_leave_one_out.py --epochs 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet
from models.losses import angular_loss
from data.dataset import GazeDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_all_data(processed_dir: Path) -> pd.DataFrame:
    """加载所有 split 的数据，合并为一个 DataFrame。"""
    all_dfs = []
    for split in ["train", "val", "test"]:
        labels_path = processed_dir / split / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            df["split"] = split
            df["image_root"] = str(processed_dir / split)
            all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError(f"未找到数据: {processed_dir}")
    return pd.concat(all_dfs, ignore_index=True)


def train_model(
    train_df: pd.DataFrame,
    config: dict,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
) -> GazeNet:
    """训练一个 GazeNet 模型。

    参数:
        train_df: 训练数据 DataFrame（需包含 image_root 列）
        config: 模型配置
        device: 训练设备
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率

    返回:
        训练好的模型（最佳验证 epoch 的权重）
    """
    channels = config.get("model", {}).get("channels", [32, 64, 128, 256])
    model = GazeNet(num_channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 按 image_root 分组创建 dataset（因为不同 split 的图像在不同目录）
    datasets = []
    for root, group_df in train_df.groupby("image_root"):
        ds = GazeDataset(group_df, image_root=Path(root), augment=True)
        datasets.append(ds)

    combined_ds = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(combined_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            eye_imgs = batch["eye_img"].to(device)
            gaze_targets = batch["gaze"].to(device)
            optimizer.zero_grad()
            preds = model(eye_imgs)
            loss = angular_loss(preds, gaze_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = " *"  # 标记最佳 epoch
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0 or (epoch + 1) == epochs:
            logger.info(
                f"    Epoch {epoch+1:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  best={best_loss:.4f}  "
                f"lr={current_lr:.6f}{marker}"
            )

    # 恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_model(
    model: GazeNet,
    test_df: pd.DataFrame,
    device: torch.device,
    screen_w: int = 1536,
    screen_h: int = 864,
) -> dict:
    """评估模型在测试集上的表现。"""
    datasets = []
    for root, group_df in test_df.groupby("image_root"):
        ds = GazeDataset(group_df, image_root=Path(root), augment=False)
        datasets.append(ds)

    combined_ds = torch.utils.data.ConcatDataset(datasets)
    loader = DataLoader(combined_ds, batch_size=64, shuffle=False, num_workers=0)

    angle_errors = []
    pixel_errors = []

    for batch in loader:
        eye_imgs = batch["eye_img"].to(device)
        gaze_targets = batch["gaze"]
        preds = model(eye_imgs).cpu().numpy()
        targets = gaze_targets.numpy()

        for i in range(preds.shape[0]):
            pred_vec = preds[i]
            true_vec = targets[i]

            # 角度误差
            cos_sim = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_sim)))
            angle_errors.append(angle_deg)

            # 像素误差
            true_nx = float(batch["meta"]["norm_target_x"][i])
            true_ny = float(batch["meta"]["norm_target_y"][i])
            if abs(pred_vec[2]) > 1e-6:
                pred_nx = np.clip(0.5 + pred_vec[0] / pred_vec[2] * 0.5, 0, 1)
                pred_ny = np.clip(0.5 + pred_vec[1] / pred_vec[2] * 0.5, 0, 1)
            else:
                pred_nx, pred_ny = 0.5, 0.5
            px_err = float(np.sqrt(
                ((pred_nx - true_nx) * screen_w) ** 2
                + ((pred_ny - true_ny) * screen_h) ** 2
            ))
            pixel_errors.append(px_err)

    return {
        "mean_angle_error": float(np.mean(angle_errors)),
        "median_angle_error": float(np.median(angle_errors)),
        "std_angle_error": float(np.std(angle_errors)),
        "mean_pixel_error": float(np.mean(pixel_errors)),
        "num_samples": len(angle_errors),
    }


def generate_loo_plots(results: list[dict], output_dir: Path) -> None:
    """生成 leave-one-out 可视化图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    user_ids = [r["left_out_user"] for r in results]
    angle_errors = [r["test_metrics"]["mean_angle_error"] for r in results]
    pixel_errors = [r["test_metrics"]["mean_pixel_error"] for r in results]
    train_sizes = [r["train_samples"] for r in results]
    test_sizes = [r["test_samples"] for r in results]

    # --- 图 1：各用户角度误差柱状图 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(user_ids))
    bars = ax.bar(x, angle_errors, alpha=0.8, color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"User {u}" for u in user_ids], rotation=45, ha="right")
    ax.set_ylabel("Mean Angle Error (degrees)")
    ax.set_title("Leave-One-User-Out: Per-User Angle Error")

    # 标注数值和样本数
    for bar, err, n in zip(bars, angle_errors, test_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{err:.1f}°\n(n={n})", ha="center", va="bottom", fontsize=7)

    # 画均值线
    mean_err = np.mean(angle_errors)
    ax.axhline(mean_err, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_err:.2f}°")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loo_angle_error_per_user.png", dpi=150)
    plt.close(fig)

    # --- 图 2：各用户像素误差柱状图 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, pixel_errors, alpha=0.8, color="#FF5722")
    ax.set_xticks(x)
    ax.set_xticklabels([f"User {u}" for u in user_ids], rotation=45, ha="right")
    ax.set_ylabel("Mean Pixel Error (px)")
    ax.set_title("Leave-One-User-Out: Per-User Pixel Error")

    for bar, err, n in zip(bars, pixel_errors, test_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{err:.0f}px\n(n={n})", ha="center", va="bottom", fontsize=7)

    mean_px = np.mean(pixel_errors)
    ax.axhline(mean_px, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_px:.0f}px")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loo_pixel_error_per_user.png", dpi=150)
    plt.close(fig)

    # --- 图 3：训练集大小 vs 测试误差散点图 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(train_sizes, angle_errors, s=80, c="#4CAF50", alpha=0.8)
    for i, uid in enumerate(user_ids):
        ax.annotate(f"User {uid}", (train_sizes[i], angle_errors[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Test Angle Error (degrees)")
    ax.set_title("Training Size vs Test Error")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "loo_train_size_vs_error.png", dpi=150)
    plt.close(fig)

    logger.info(f"图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="跨用户 Leave-One-Out 交叉验证")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--output", default="evaluation_results/leave_one_out")
    parser.add_argument("--epochs", type=int, default=50, help="每折训练轮数")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="xpu", choices=["cpu", "xpu"],
                        help="训练设备（默认 xpu，使用 Intel Arc GPU 加速）")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设备
    device = torch.device(args.device)
    if args.device == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            if not torch.xpu.is_available():
                logger.warning("XPU 不可用，回退到 CPU")
                device = torch.device("cpu")
        except ImportError:
            logger.warning("IPEX 未安装，回退到 CPU")
            device = torch.device("cpu")
    logger.info(f"训练设备: {device}")

    # 加载全部数据
    processed_dir = Path("dataset_processed")
    all_df = load_all_data(processed_dir)
    user_ids = sorted(all_df["user_id"].unique())
    logger.info(f"总计 {len(all_df)} 样本, {len(user_ids)} 个用户: {user_ids}")

    # Leave-One-Out 循环
    results = []
    for left_out_user in user_ids:
        logger.info(f"\n{'='*50}")
        logger.info(f"留出用户: {left_out_user}")

        test_df = all_df[all_df["user_id"] == left_out_user].copy()
        train_df = all_df[all_df["user_id"] != left_out_user].copy()

        logger.info(f"  训练: {len(train_df)} 样本 ({len(train_df['user_id'].unique())} 用户)")
        logger.info(f"  测试: {len(test_df)} 样本")

        # 训练
        t0 = time.time()
        model = train_model(
            train_df, config, device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        train_time = time.time() - t0
        logger.info(f"  训练耗时: {train_time:.1f}s")

        # 评估
        metrics = evaluate_model(model, test_df, device)
        logger.info(f"  角度误差: {metrics['mean_angle_error']:.2f}° (±{metrics['std_angle_error']:.2f}°)")
        logger.info(f"  像素误差: {metrics['mean_pixel_error']:.1f} px")

        results.append({
            "left_out_user": int(left_out_user),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "train_time_s": round(train_time, 1),
            "test_metrics": metrics,
        })

        # 释放模型内存
        del model
        if device.type == "xpu":
            torch.xpu.empty_cache()

    # 汇总
    all_angles = [r["test_metrics"]["mean_angle_error"] for r in results]
    all_pixels = [r["test_metrics"]["mean_pixel_error"] for r in results]

    summary = {
        "overall": {
            "mean_angle_error": float(np.mean(all_angles)),
            "std_angle_error": float(np.std(all_angles)),
            "mean_pixel_error": float(np.mean(all_pixels)),
            "std_pixel_error": float(np.std(all_pixels)),
            "num_users": len(user_ids),
            "total_samples": len(all_df),
        },
        "per_user": results,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
        },
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'='*50}")
    logger.info(f"=== Leave-One-Out 汇总 ===")
    logger.info(f"  跨用户平均角度误差: {np.mean(all_angles):.2f}° (±{np.std(all_angles):.2f}°)")
    logger.info(f"  跨用户平均像素误差: {np.mean(all_pixels):.1f} px (±{np.std(all_pixels):.1f})")
    logger.info(f"  最佳用户: User {results[np.argmin(all_angles)]['left_out_user']} ({min(all_angles):.2f}°)")
    logger.info(f"  最差用户: User {results[np.argmax(all_angles)]['left_out_user']} ({max(all_angles):.2f}°)")

    # 生成图表
    generate_loo_plots(results, output_dir)
    logger.info("Leave-One-Out 实验完成。")


if __name__ == "__main__":
    main()
