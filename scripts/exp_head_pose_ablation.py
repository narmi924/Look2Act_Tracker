"""
头部姿态消融实验：量化 PnP 几何补偿对注视估计精度的贡献。

实验条件：
  A. full_pose    — 完整 head pose 旋转（R @ gaze_vector）
  B. no_pose      — 无旋转（I @ gaze_vector，跳过 PnP 几何补偿）
  C. no_yaw       — 消融 yaw（yaw=0，保留 pitch/roll）
  D. no_pitch     — 消融 pitch（pitch=0，保留 yaw/roll）

流程：
  1. GazeNet 预测 gaze vector
  2. 从 labels.csv 的 head_yaw/pitch/roll 重建旋转矩阵
  3. transform_gaze_to_camera → ray-plane intersect → 屏幕坐标
  4. 与 ground truth 屏幕坐标对比，计算像素误差和角度误差

工作目录：Look2Act_Tracker_Project/
运行方式：conda run -n gaze-env python scripts/exp_head_pose_ablation.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 同时加入项目根目录和 src 目录，兼容两种 import 风格
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_project_root / "src"))

from models.gaze_net import GazeNet, GazeNetV2
from data.dataset import GazeDataset
from src.geometry.coordinate import transform_gaze_to_camera
from src.geometry.screen_geometry import ScreenGeometry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 几何工具函数
# ---------------------------------------------------------------------------

def euler_to_rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """从欧拉角（度）构建 3×3 旋转矩阵。

    旋转顺序：Rz(roll) @ Ry(yaw) @ Rx(pitch)，与 cv2.RQDecomp3x3 输出一致。
    """
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)],
    ])
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    Rz = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


def build_screen_geometry(
    screen_w_px: int = 1536,
    screen_h_px: int = 864,
    screen_w_mm: float = 344.0,
    screen_h_mm: float = 215.0,
    screen_distance_mm: float = 500.0,
    cam_above_screen_mm: float = 5.0,
) -> ScreenGeometry:
    """构建屏幕几何模型。

    假设相机在屏幕正上方中央，屏幕平面垂直于 Z 轴。
    """
    sg = ScreenGeometry(
        screen_w_px=screen_w_px,
        screen_h_px=screen_h_px,
        screen_w_mm=screen_w_mm,
        screen_h_mm=screen_h_mm,
    )
    # 屏幕左上角在相机坐标系中的位置
    # 相机在屏幕上边缘上方 cam_above_screen_mm 处
    # X: 屏幕左边缘 = -screen_w_mm/2
    # Y: 屏幕上边缘 = cam_above_screen_mm（相机下方）
    # Z: 屏幕距离
    screen_origin = np.array([
        -screen_w_mm / 2.0,
        cam_above_screen_mm,
        screen_distance_mm,
    ])
    sg.setup_plane(
        screen_origin_mm=screen_origin,
        screen_normal=np.array([0.0, 0.0, -1.0]),  # 法向量指向相机
        screen_x_axis=np.array([1.0, 0.0, 0.0]),
        screen_y_axis=np.array([0.0, 1.0, 0.0]),
    )
    return sg


# ---------------------------------------------------------------------------
# 消融条件定义
# ---------------------------------------------------------------------------

ABLATION_CONDITIONS = {
    "full_pose": "完整 head pose 旋转",
    "no_pose": "无旋转（单位矩阵）",
    "no_yaw": "消融 yaw（yaw=0）",
    "no_pitch": "消融 pitch（pitch=0）",
}


def get_rotation_for_condition(
    condition: str,
    yaw: float,
    pitch: float,
    roll: float,
) -> np.ndarray:
    """根据消融条件返回旋转矩阵。"""
    if condition == "full_pose":
        return euler_to_rotation_matrix(yaw, pitch, roll)
    elif condition == "no_pose":
        return np.eye(3)
    elif condition == "no_yaw":
        return euler_to_rotation_matrix(0.0, pitch, roll)
    elif condition == "no_pitch":
        return euler_to_rotation_matrix(yaw, 0.0, roll)
    else:
        raise ValueError(f"未知条件: {condition}")


# ---------------------------------------------------------------------------
# 核心评估函数
# ---------------------------------------------------------------------------

def evaluate_ablation(
    model,
    df: pd.DataFrame,
    image_root: Path,
    screen_geom: ScreenGeometry,
    screen_w_px: int = 1536,
    screen_h_px: int = 864,
    screen_distance_mm: float = 500.0,
    model_version: str = "v1",
) -> dict[str, pd.DataFrame]:
    """对所有消融条件进行评估。

    返回:
        {condition_name: DataFrame}，每个 DataFrame 包含逐样本误差
    """
    ds = GazeDataset(df, image_root=image_root, augment=False, model_version=model_version)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    # 先收集所有模型预测
    all_preds = []
    all_targets = []
    all_meta = []
    with torch.no_grad():
        for batch in loader:
            if model_version == "v2":
                preds = model(batch["left_eye"], batch["right_eye"], batch["head_pose"])
            else:
                preds = model(batch["eye_img"])
            all_preds.append(preds.numpy())
            all_targets.append(batch["gaze"].numpy())
            # meta 是 dict of lists
            batch_size = preds.shape[0]
            for i in range(batch_size):
                all_meta.append({
                    "norm_target_x": float(batch["meta"]["norm_target_x"][i]),
                    "norm_target_y": float(batch["meta"]["norm_target_y"][i]),
                    "session_id": str(batch["meta"]["session_id"][i]),
                    "user_id": str(batch["meta"]["user_id"][i]),
                })

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 对每个条件计算误差
    results = {}
    for condition in ABLATION_CONDITIONS:
        records = []
        for i in range(len(all_preds)):
            pred_vec = all_preds[i]
            true_vec = all_targets[i]
            row = df.iloc[i]
            meta = all_meta[i]

            # 角度误差（与条件无关，纯模型输出）
            cos_sim = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_sim)))

            # 获取 head pose
            yaw = float(row.get("head_yaw", 0))
            pitch = float(row.get("head_pitch", 0))
            roll = float(row.get("head_roll", 0))

            # 构建旋转矩阵
            R = get_rotation_for_condition(condition, yaw, pitch, roll)

            # 模拟平移向量（眼球位置近似为相机前方 screen_distance_mm * 0.8）
            t = np.array([0.0, 0.0, screen_distance_mm * 0.8])

            # 几何 pipeline：旋转 → 射线求交 → 屏幕坐标
            ray_origin, ray_direction = transform_gaze_to_camera(pred_vec, R, t)
            gaze_point = screen_geom.get_gaze_point(
                ray_origin=ray_origin,
                ray_direction=ray_direction,
                clamp_to_screen=True,
            )

            if gaze_point is not None:
                pred_px, pred_py = gaze_point
            else:
                pred_px, pred_py = screen_w_px / 2.0, screen_h_px / 2.0

            # ground truth 屏幕坐标
            true_px = meta["norm_target_x"] * screen_w_px
            true_py = meta["norm_target_y"] * screen_h_px

            # 像素误差
            pixel_err = float(np.sqrt((pred_px - true_px) ** 2 + (pred_py - true_py) ** 2))

            records.append({
                "angle_error": angle_deg,
                "pixel_error": pixel_err,
                "pred_px": pred_px,
                "pred_py": pred_py,
                "true_px": true_px,
                "true_py": true_py,
                "head_yaw": yaw,
                "head_pitch": pitch,
                "session_id": meta["session_id"],
                "user_id": meta["user_id"],
            })

        results[condition] = pd.DataFrame(records)
        mean_px = results[condition]["pixel_error"].mean()
        mean_ang = results[condition]["angle_error"].mean()
        logger.info(f"  {condition:12s}: 像素误差 {mean_px:.1f} px, 角度误差 {mean_ang:.2f}°")

    return results


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def generate_ablation_plots(
    results: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """生成消融实验可视化图表。

    1. 各条件像素误差对比柱状图
    2. 各条件按 head yaw 分桶的误差曲线
    3. 各条件按 head pitch 分桶的误差曲线
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = list(results.keys())
    labels_cn = [ABLATION_CONDITIONS[c] for c in conditions]

    # --- 图 1：各条件像素误差对比柱状图 ---
    means = [results[c]["pixel_error"].mean() for c in conditions]
    stds = [results[c]["pixel_error"].std() for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                  color=["#2196F3", "#FF5722", "#FFC107", "#4CAF50"])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9)
    ax.set_ylabel("Mean Pixel Error (px)")
    ax.set_title("Head Pose Ablation: Pixel Error Comparison")

    # 在柱子上标注数值
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{m:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "ablation_pixel_error_comparison.png", dpi=150)
    plt.close(fig)

    # --- 图 2：按 head yaw 分桶的误差曲线 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # 根据实际数据范围自动确定分桶
    all_yaw = pd.concat([results[c]["head_yaw"] for c in conditions])
    yaw_lo, yaw_hi = all_yaw.min() - 1, all_yaw.max() + 1
    yaw_bins = np.linspace(yaw_lo, yaw_hi, min(13, max(5, int((yaw_hi - yaw_lo) / 3))))
    colors = ["#2196F3", "#FF5722", "#FFC107", "#4CAF50"]
    has_legend = False

    for idx, cond in enumerate(conditions):
        df_c = results[cond]
        bin_means = []
        bin_centers = []
        for j in range(len(yaw_bins) - 1):
            lo, hi = yaw_bins[j], yaw_bins[j + 1]
            mask = (df_c["head_yaw"] >= lo) & (df_c["head_yaw"] < hi)
            if mask.sum() >= 3:  # 至少 3 个样本才有统计意义
                bin_means.append(df_c.loc[mask, "pixel_error"].mean())
                bin_centers.append((lo + hi) / 2)
        if bin_centers:
            ax.plot(bin_centers, bin_means, "o-", label=cond, color=colors[idx], markersize=4)
            has_legend = True

    ax.set_xlabel("Head Yaw (degrees)")
    ax.set_ylabel("Mean Pixel Error (px)")
    ax.set_title("Pixel Error vs Head Yaw (by ablation condition)")
    if has_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_error_vs_yaw.png", dpi=150)
    plt.close(fig)

    # --- 图 3：按 head pitch 分桶的误差曲线 ---
    fig, ax = plt.subplots(figsize=(10, 6))
    all_pitch = pd.concat([results[c]["head_pitch"] for c in conditions])
    pitch_lo, pitch_hi = all_pitch.min() - 1, all_pitch.max() + 1
    pitch_bins = np.linspace(pitch_lo, pitch_hi, min(13, max(5, int((pitch_hi - pitch_lo) / 10))))
    has_legend = False

    for idx, cond in enumerate(conditions):
        df_c = results[cond]
        bin_means = []
        bin_centers = []
        for j in range(len(pitch_bins) - 1):
            lo, hi = pitch_bins[j], pitch_bins[j + 1]
            mask = (df_c["head_pitch"] >= lo) & (df_c["head_pitch"] < hi)
            if mask.sum() >= 3:
                bin_means.append(df_c.loc[mask, "pixel_error"].mean())
                bin_centers.append((lo + hi) / 2)
        if bin_centers:
            ax.plot(bin_centers, bin_means, "o-", label=cond, color=colors[idx], markersize=4)
            has_legend = True

    ax.set_xlabel("Head Pitch (degrees)")
    ax.set_ylabel("Mean Pixel Error (px)")
    ax.set_title("Pixel Error vs Head Pitch (by ablation condition)")
    if has_legend:
        ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_error_vs_pitch.png", dpi=150)
    plt.close(fig)

    logger.info(f"图表已保存到: {output_dir}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="头部姿态消融实验")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="evaluation_results/head_pose_ablation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--screen-w-px", type=int, default=1536)
    parser.add_argument("--screen-h-px", type=int, default=864)
    parser.add_argument("--screen-w-mm", type=float, default=344.0)
    parser.add_argument("--screen-h-mm", type=float, default=215.0)
    parser.add_argument("--screen-distance-mm", type=float, default=500.0)
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载模型（自动检测 V1/V2）
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
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
    logger.info(f"已加载模型: {args.checkpoint} (版本: {model_version})")

    # 构建屏幕几何
    screen_geom = build_screen_geometry(
        screen_w_px=args.screen_w_px,
        screen_h_px=args.screen_h_px,
        screen_w_mm=args.screen_w_mm,
        screen_h_mm=args.screen_h_mm,
        screen_distance_mm=args.screen_distance_mm,
    )

    # 加载数据
    processed_dir = Path("dataset_processed")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_dfs = []
    for split in splits:
        labels_path = processed_dir / split / "labels.csv"
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            all_dfs.append((split, df, processed_dir / split))
            logger.info(f"  {split}: {len(df)} 样本")

    if not all_dfs:
        logger.error("未找到数据")
        sys.exit(1)

    # 合并数据
    combined_df = pd.concat([df for _, df, _ in all_dfs], ignore_index=True)
    # 使用第一个 split 的 image_root（预处理模式下图像路径是相对路径）
    # 对于多 split，需要修正路径
    if len(all_dfs) == 1:
        image_root = all_dfs[0][2]
    else:
        # 多 split 时，为每个样本添加 split 前缀
        offset = 0
        for split, df, img_root in all_dfs:
            for i in range(len(df)):
                idx = offset + i
                orig_path = combined_df.at[idx, "eye_img_path"]
                combined_df.at[idx, "eye_img_path"] = f"../{split}/{orig_path}"
            offset += len(df)
        image_root = all_dfs[0][2]

    logger.info(f"总计 {len(combined_df)} 样本，开始消融实验...")

    # 运行消融评估
    results = evaluate_ablation(
        model=model,
        df=combined_df,
        image_root=image_root,
        screen_geom=screen_geom,
        screen_w_px=args.screen_w_px,
        screen_h_px=args.screen_h_px,
        screen_distance_mm=args.screen_distance_mm,
        model_version=model_version,
    )

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 汇总 JSON
    summary = {"conditions": {}}
    for cond, df_c in results.items():
        summary["conditions"][cond] = {
            "description": ABLATION_CONDITIONS[cond],
            "mean_pixel_error": float(df_c["pixel_error"].mean()),
            "median_pixel_error": float(df_c["pixel_error"].median()),
            "std_pixel_error": float(df_c["pixel_error"].std()),
            "mean_angle_error": float(df_c["angle_error"].mean()),
            "num_samples": len(df_c),
        }
        # 保存逐样本结果
        df_c.to_csv(output_dir / f"per_sample_{cond}.csv", index=False)

    # 计算 head pose 贡献（full_pose vs no_pose 的改善）
    full_px = summary["conditions"]["full_pose"]["mean_pixel_error"]
    no_px = summary["conditions"]["no_pose"]["mean_pixel_error"]
    improvement = no_px - full_px
    improvement_pct = improvement / no_px * 100 if no_px > 0 else 0
    summary["head_pose_contribution"] = {
        "pixel_error_reduction": float(improvement),
        "pixel_error_reduction_pct": float(improvement_pct),
        "full_pose_px": float(full_px),
        "no_pose_px": float(no_px),
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"\n=== 消融实验结果 ===")
    for cond, stats in summary["conditions"].items():
        logger.info(f"  {cond:12s}: {stats['mean_pixel_error']:.1f} px (±{stats['std_pixel_error']:.1f})")
    logger.info(f"\nHead pose 贡献: 像素误差减少 {improvement:.1f} px ({improvement_pct:.1f}%)")

    # 生成图表
    generate_ablation_plots(results, output_dir)
    logger.info("消融实验完成。")


if __name__ == "__main__":
    main()
