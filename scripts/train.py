"""
训练入口脚本：从标准化数据集训练 GazeNet / GazeNetV2 模型。

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
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models.gaze_net import GazeNet, GazeNetPoG, GazeNetV2
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


def build_model(config: dict) -> nn.Module:
    """根据配置构建模型（V1 或 V2）。"""
    model_cfg = config.get("model", {})
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    version = model_cfg.get("version", "v1")

    if version == "pog_v1":
        model = GazeNetPoG(
            num_channels=channels,
            head_pose_dim=model_cfg.get("head_pose_dim", 3),
            fusion_dim=model_cfg.get("fusion_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
            output_activation=model_cfg.get("output_activation", "sigmoid"),
        )
        logger.info(
            "模型: GazeNetPoG (双眼 + head pose -> 2D PoG, output=%s)",
            model_cfg.get("output_activation", "sigmoid"),
        )
    elif version == "v2":
        model = GazeNetV2(
            num_channels=channels,
            head_pose_dim=model_cfg.get("head_pose_dim", 3),
            fusion_dim=model_cfg.get("fusion_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )
        logger.info("模型: GazeNetV2 (双眼 + head pose 融合)")
    else:
        model = GazeNet(num_channels=channels)
        logger.info("模型: GazeNet V1 (单眼)")

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
    processed_dir: Path, split: str, augment: bool = False,
    model_version: str = "v2",
    target_mode: str = "gaze3d",
    head_pose_mode: str = "stored",
) -> GazeDataset | None:
    """加载指定划分的数据集。"""
    split_dir = processed_dir / split
    labels_path = split_dir / "labels.csv"

    if not labels_path.exists():
        logger.warning(f"未找到 {split} 数据: {labels_path}")
        return None

    df = pd.read_csv(labels_path)
    logger.info(f"{split} 数据: {len(df)} 样本")
    return GazeDataset(
        df, image_root=split_dir, augment=augment,
        model_version="v2" if model_version == "pog_v1" else model_version,
        target_mode=target_mode,
        head_pose_mode=head_pose_mode,
    )


def target_mode_for_model(model_version: str) -> str:
    """Return the dataset target mode for a model version."""
    return "pog2d" if model_version == "pog_v1" else "gaze3d"


def target_key_for_mode(target_mode: str) -> str:
    return "pog" if target_mode == "pog2d" else "gaze"


def compute_training_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    target_mode: str,
    loss_cfg: dict | None = None,
) -> torch.Tensor:
    if target_mode == "pog2d":
        base = nn.functional.smooth_l1_loss(preds, targets)
        loss_cfg = loss_cfg or {}
        topology_weight = float(loss_cfg.get("topology_weight", 0.0))
        if topology_weight <= 0:
            return base
        return base + topology_weight * pog_topology_loss(
            preds,
            targets,
            epsilon=float(loss_cfg.get("topology_epsilon", 0.03)),
            margin=float(loss_cfg.get("topology_margin", 0.01)),
        )
    return angular_loss(preds, targets)


def pog_topology_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 0.03,
    margin: float = 0.01,
) -> torch.Tensor:
    """Pairwise ordering loss for screen-point targets.

    The direct PoG model must preserve the screen topology: if target A is
    right/below target B, the predicted x/y should keep that ordering. This
    term does not replace pixel regression; it only discourages collapsed raw
    predictions that a calibration polynomial cannot repair.
    """
    if preds.shape[0] < 2:
        return preds.new_tensor(0.0)

    losses = []
    for dim in (0, 1):
        target_diff = targets[:, dim].unsqueeze(1) - targets[:, dim].unsqueeze(0)
        pred_diff = preds[:, dim].unsqueeze(1) - preds[:, dim].unsqueeze(0)
        mask = torch.abs(target_diff) > epsilon
        if torch.any(mask):
            signed_order = torch.sign(target_diff[mask])
            losses.append(torch.relu(margin - signed_order * pred_diff[mask]).mean())

    if not losses:
        return preds.new_tensor(0.0)
    return torch.stack(losses).mean()


def build_balanced_target_sampler(dataset: GazeDataset, decimals: int = 4) -> WeightedRandomSampler | None:
    """Build inverse-frequency sampler over normalized target grid cells."""
    df = getattr(dataset, "labels_df", None)
    if df is None or not {"norm_target_x", "norm_target_y"}.issubset(df.columns):
        return None
    targets = df[["norm_target_x", "norm_target_y"]].round(decimals)
    counts = targets.value_counts()
    weights = []
    for _, row in targets.iterrows():
        weights.append(1.0 / float(counts.loc[(row["norm_target_x"], row["norm_target_y"])]))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_version: str = "v2",
    target_mode: str = "gaze3d",
    loss_cfg: dict | None = None,
) -> float:
    """训练一个 epoch，返回平均损失。"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        targets = batch[target_key_for_mode(target_mode)].to(device)

        optimizer.zero_grad()

        if model_version in {"v2", "pog_v1"}:
            left_eye = batch["left_eye"].to(device)
            right_eye = batch["right_eye"].to(device)
            head_pose = batch["head_pose"].to(device)
            preds = model(left_eye, right_eye, head_pose)
        else:
            eye_imgs = batch["eye_img"].to(device)
            preds = model(eye_imgs)

        loss = compute_training_loss(preds, targets, target_mode, loss_cfg)
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
    model_version: str = "v2",
    target_mode: str = "gaze3d",
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> tuple[float, float]:
    """在验证集上评估，返回 (平均损失, 主要指标)。

    3D gaze 模式的主要指标是角度误差（度），PoG 模式是像素误差。
    """
    model.eval()
    total_loss = 0.0
    total_metric = 0.0
    num_batches = 0

    for batch in loader:
        targets = batch[target_key_for_mode(target_mode)].to(device)

        if model_version in {"v2", "pog_v1"}:
            left_eye = batch["left_eye"].to(device)
            right_eye = batch["right_eye"].to(device)
            head_pose = batch["head_pose"].to(device)
            preds = model(left_eye, right_eye, head_pose)
        else:
            eye_imgs = batch["eye_img"].to(device)
            preds = model(eye_imgs)

        loss = compute_training_loss(preds, targets, target_mode)
        if target_mode == "pog2d":
            dx = (preds[:, 0] - targets[:, 0]) * screen_w
            dy = (preds[:, 1] - targets[:, 1]) * screen_h
            metric = torch.sqrt(dx * dx + dy * dy).mean().item()
        else:
            metric = np.degrees(loss.item())

        total_loss += loss.item()
        total_metric += metric
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_metric = total_metric / max(num_batches, 1)
    return avg_loss, avg_metric


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
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    ckpt_cfg = config.get("checkpoint", {})
    log_cfg = config.get("logging", {})

    model_version = model_cfg.get("version", "v1")
    target_mode = data_cfg.get("target_mode", target_mode_for_model(model_version))
    head_pose_mode = data_cfg.get("head_pose_mode", "stored")
    metric_name = "val_pixel_error_px" if target_mode == "pog2d" else "val_angle_error"

    # 设备配置
    device_name = train_cfg.get("device", "cpu")
    use_ipex = train_cfg.get("use_ipex", False)

    # 初始化设备
    if device_name == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                device = torch.device("xpu")
                logger.info("使用设备: XPU (Intel Arc GPU)")
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
        logger.info("使用设备: CPU")

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
    best_val_metric = float("inf")
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_metric = ckpt.get("best_val_metric", ckpt.get("best_val_angle", float("inf")))
        logger.info(
            f"从 epoch {start_epoch} 恢复训练，"
            f"最佳验证指标: {best_val_metric:.2f}"
        )

    # 加载数据
    processed_dir = Path(data_cfg.get("dataset_processed_dir", "dataset_processed"))
    augment = data_cfg.get("augmentation", True)
    batch_size = train_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)

    train_ds = load_split_data(
        processed_dir, "train", augment=augment, model_version=model_version,
        target_mode=target_mode, head_pose_mode=head_pose_mode,
    )
    val_ds = load_split_data(
        processed_dir, "val", augment=False, model_version=model_version,
        target_mode=target_mode, head_pose_mode=head_pose_mode,
    )

    if train_ds is None:
        logger.error("训练数据不存在，请先运行 scripts/preprocess.py")
        sys.exit(1)

    sampler = None
    if target_mode == "pog2d" and data_cfg.get("balanced_targets", False):
        sampler = build_balanced_target_sampler(
            train_ds,
            decimals=int(data_cfg.get("target_balance_decimals", 4)),
        )
        if sampler is not None:
            logger.info("启用 PoG target 平衡采样")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler,
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
    csv_writer.writerow(["epoch", "train_loss", "val_loss", metric_name])

    logger.info(f"开始训练: {epochs} epochs, batch_size={batch_size}")

    # 训练循环
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, model_version, target_mode, loss_cfg,
        )

        # 验证
        val_loss = 0.0
        val_metric = 0.0
        if val_loader is not None:
            val_loss, val_metric = validate(
                model, val_loader, device, model_version, target_mode,
            )

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
                f"{metric_name}={val_metric:.2f} | "
                f"lr={lr:.6f} | "
                f"time={elapsed:.1f}s"
            )

        # 写入 CSV
        csv_writer.writerow([
            epoch + 1,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}",
            f"{val_metric:.4f}",
        ])
        csv_file.flush()

        # 保存最优模型
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_path = ckpt_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_angle": best_val_metric if target_mode == "gaze3d" else None,
                "best_val_metric": best_val_metric,
                "target_mode": target_mode,
                "metric_name": metric_name,
                "config": config,
                "model_version": model_version,
            }, best_path)
            logger.info(
                f"保存最优模型: {best_path} "
                f"({metric_name}={best_val_metric:.2f})"
            )

    csv_file.close()

    # 保存最终模型
    final_path = ckpt_dir / "final_model.pth"
    torch.save({
        "epoch": epochs - 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_angle": best_val_metric if target_mode == "gaze3d" else None,
        "best_val_metric": best_val_metric,
        "target_mode": target_mode,
        "metric_name": metric_name,
        "config": config,
        "model_version": model_version,
    }, final_path)

    logger.info(f"训练完成。最佳验证指标 {metric_name}: {best_val_metric:.2f}")
    logger.info(f"训练日志: {log_csv_path}")
    logger.info(f"最优模型: {ckpt_dir / 'best_model.pth'}")
    logger.info(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
