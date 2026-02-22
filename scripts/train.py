"""
训练入口脚本：从标准化数据集训练 GazeNet 模型。

工作目录：Look2Act_Tracker_Project/
conda 环境：gaze-env
运行方式：conda run -n gaze-env python scripts/train.py
         conda run -n gaze-env python scripts/train.py --config configs/train_config.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def build_model(config: dict) -> GazeNet:
    """根据配置构建 GazeNet 模型。"""
    model_cfg = config.get("model", {})
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    model = GazeNet(num_channels=channels)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {total_params:,}")
    return model


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """根据配置构建优化器。"""
    train_cfg = config.get("training", {})
    lr = train_cfg.get("learning_rate", 0.001)
    wd = train_cfg.get("weight_decay", 0.0001)
    opt_name = train_cfg.get("optimizer", "Adam")

    if opt_name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {opt_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """根据配置构建学习率调度器。"""
    train_cfg = config.get("training", {})
    sched_name = train_cfg.get("scheduler", "CosineAnnealingLR")
    epochs = train_cfg.get("epochs", 100)

    if sched_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif sched_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif sched_name == "none":
        return None
    else:
        raise ValueError(f"不支持的调度器: {sched_name}")


def load_split_data(
    processed_dir: Path, split: str, augment: bool = False
) -> GazeDataset | None:
    """加载指定划分的数据集。"""
    split_dir = processed_dir / split
    labels_path = split_dir / "labels.csv"

    if not labels_path.exists():
        logger.warning(f"未找到 {split} 数据: {labels_path}")
        return None

    df = pd.read_csv(labels_path)
    logger.info(f"{split} 数据: {len(df)} 样本")
    return GazeDataset(df, image_root=split_dir, augment=augment)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        eye_imgs = batch["eye_img"].to(device)
        gaze_targets = batch["gaze"].to(device)

        optimizer.zero_grad()
        preds = model(eye_imgs)
        loss = angular_loss(preds, gaze_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """在验证集上评估，返回 (平均损失, 平均角度误差/度)。"""
    model.eval()
    total_loss = 0.0
    total_angle_deg = 0.0
    num_batches = 0

    for batch in loader:
        eye_imgs = batch["eye_img"].to(device)
        gaze_targets = batch["gaze"].to(device)

        preds = model(eye_imgs)
        loss = angular_loss(preds, gaze_targets)

        # 角度误差转换为度
        angle_rad = loss.item()
        angle_deg = np.degrees(angle_rad)

        total_loss += loss.item()
        total_angle_deg += angle_deg
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_angle = total_angle_deg / max(num_batches, 1)
    return avg_loss, avg_angle


def main():
    parser = argparse.ArgumentParser(description="Look2Act GazeNet 训练")
    parser.add_argument(
        "--config", type=str, default="configs/train_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="从指定 checkpoint 恢复训练",
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    train_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    ckpt_cfg = config.get("checkpoint", {})
    log_cfg = config.get("logging", {})

    # 设备配置
    device_name = train_cfg.get("device", "cpu")
    use_ipex = train_cfg.get("use_ipex", False)
    
    # 初始化设备
    if device_name == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device = torch.device("xpu")
                logger.info(f"使用设备: XPU (Intel Arc GPU)")
                logger.info(f"IPEX 版本: {ipex.__version__}")
            else:
                logger.warning("XPU 不可用，回退到 CPU")
                device = torch.device("cpu")
                use_ipex = False
        except ImportError:
            logger.warning("IPEX 未安装，回退到 CPU")
            device = torch.device("cpu")
            use_ipex = False
    else:
        device = torch.device("cpu")
        logger.info(f"使用设备: CPU")

    # 构建模型
    model = build_model(config)
    model.to(device)

    # 构建优化器和调度器
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # IPEX 优化（训练加速）
    if use_ipex and device.type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(model, optimizer=optimizer)
            logger.info("✓ IPEX 训练优化已启用")
        except Exception as e:
            logger.warning(f"IPEX 优化失败，继续使用标准训练: {e}")

    # 恢复训练
    start_epoch = 0
    best_val_angle = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_angle = ckpt.get("best_val_angle", float("inf"))
        logger.info(f"从 epoch {start_epoch} 恢复训练，最佳验证角度误差: {best_val_angle:.2f}°")

    # 加载数据
    processed_dir = Path(data_cfg.get("dataset_processed_dir", "dataset_processed"))
    augment = data_cfg.get("augmentation", True)
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)

    train_ds = load_split_data(processed_dir, "train", augment=augment)
    val_ds = load_split_data(processed_dir, "val", augment=False)

    if train_ds is None:
        logger.error("训练数据不存在，请先运行 scripts/preprocess.py")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=False,
        )

    # 准备输出目录
    ckpt_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(log_cfg.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_interval = log_cfg.get("log_interval", 1)
    epochs = train_cfg.get("epochs", 100)

    # 训练日志 CSV
    log_csv_path = log_dir / "training_log.csv"
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "val_angle_error"])

    logger.info(f"开始训练: {epochs} epochs, batch_size={batch_size}")

    # 训练循环
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # 验证
        val_loss = 0.0
        val_angle = 0.0
        if val_loader is not None:
            val_loss, val_angle = validate(model, val_loader, device)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0

        # 记录日志
        if (epoch + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_angle={val_angle:.2f}° | "
                f"lr={lr:.6f} | "
                f"time={elapsed:.1f}s"
            )

        # 写入 CSV
        csv_writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{val_angle:.4f}",
        ])
        csv_file.flush()

        # 保存最优模型
        if val_angle < best_val_angle:
            best_val_angle = val_angle
            best_path = ckpt_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_angle": best_val_angle,
                "config": config,
            }, best_path)
            logger.info(f"保存最优模型: {best_path} (val_angle={best_val_angle:.2f}°)")

    csv_file.close()

    # 保存最终模型
    final_path = ckpt_dir / "final_model.pth"
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_angle": best_val_angle,
        "config": config,
    }, final_path)

    logger.info(f"训练完成。最佳验证角度误差: {best_val_angle:.2f}°")
    logger.info(f"训练日志: {log_csv_path}")
    logger.info(f"最优模型: {ckpt_dir / 'best_model.pth'}")
    logger.info(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
