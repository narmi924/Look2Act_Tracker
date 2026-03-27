"""
校准方法对比实验脚本。

实验目标（对应 Paper6 CalibMe、Paper7 OpenGaze、Paper8 Few-Shot 的建议）：
1. 比较不同校准点数（0/1/3/5/9）对精度的影响
2. 比较 affine vs polynomial 映射函数
3. 使用 hold-out 评估（不只报 fit residual）
4. 生成校准点数 vs 误差曲线

工作目录：Look2Act_Tracker_Project/
运行方式：conda run -n gaze-env python scripts/exp_calibration_compare.py

输出：evaluation_results/calibration_compare/
"""
from __future__ import annotations

import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from calibration.calibrator import CalibrationModule


def generate_grid_points(
    num_points: int,
    screen_w: int = 1536,
    screen_h: int = 864,
) -> list[tuple[float, float]]:
    """生成屏幕上均匀分布的校准目标点。

    参数:
        num_points: 校准点数量（1, 3, 5, 9, 25）
        screen_w: 屏幕宽度像素
        screen_h: 屏幕高度像素

    返回:
        校准点坐标列表
    """
    margin_x = screen_w * 0.1
    margin_y = screen_h * 0.1
    usable_w = screen_w - 2 * margin_x
    usable_h = screen_h - 2 * margin_y

    if num_points == 1:
        # 屏幕中心
        return [(screen_w / 2, screen_h / 2)]
    elif num_points == 3:
        # 上中、左下、右下
        return [
            (screen_w / 2, margin_y),
            (margin_x, screen_h - margin_y),
            (screen_w - margin_x, screen_h - margin_y),
        ]
    elif num_points == 5:
        # 四角 + 中心
        return [
            (margin_x, margin_y),
            (screen_w - margin_x, margin_y),
            (screen_w / 2, screen_h / 2),
            (margin_x, screen_h - margin_y),
            (screen_w - margin_x, screen_h - margin_y),
        ]
    elif num_points == 9:
        # 3×3 网格
        points = []
        for row in range(3):
            for col in range(3):
                x = margin_x + col * usable_w / 2
                y = margin_y + row * usable_h / 2
                points.append((x, y))
        return points
    elif num_points == 25:
        # 5×5 网格
        points = []
        for row in range(5):
            for col in range(5):
                x = margin_x + col * usable_w / 4
                y = margin_y + row * usable_h / 4
                points.append((x, y))
        return points
    else:
        raise ValueError(f"不支持的校准点数: {num_points}")


def simulate_calibration_from_data(
    df: pd.DataFrame,
    model,
    calib_indices: list[int],
    test_indices: list[int],
    method: str = "affine",
    screen_w: int = 1536,
    screen_h: int = 864,
) -> dict:
    """用数据集中的样本模拟校准过程。

    从数据集中选取 calib_indices 作为校准点，test_indices 作为测试点。
    用模型预测 raw gaze，再用校准映射修正，计算 hold-out 误差。

    参数:
        df: 数据集 DataFrame（含 gaze_x/y/z, norm_target_x/y）
        model: GazeNet 模型
        calib_indices: 用于校准的样本索引
        test_indices: 用于测试的样本索引
        method: "affine" 或 "polynomial"
        screen_w, screen_h: 屏幕分辨率

    返回:
        包含 fit_error, holdout_error, raw_error 等指标的字典
    """
    import torch
    from data.dataset import GazeDataset
    from torch.utils.data import DataLoader, Subset

    # 构建数据集
    processed_dir = Path("dataset_processed/test")
    full_ds = GazeDataset(df, image_root=processed_dir, augment=False)

    # 获取所有样本的 raw prediction
    all_raw_px = []  # 未校准的屏幕像素坐标
    all_true_px = []  # 真实屏幕像素坐标

    loader = DataLoader(full_ds, batch_size=64, shuffle=False, num_workers=0)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            preds = model(batch["eye_img"])
            meta = batch["meta"]
            for i in range(preds.shape[0]):
                pred_vec = preds[i].numpy()
                true_nx = float(meta["norm_target_x"][i])
                true_ny = float(meta["norm_target_y"][i])

                # 从 3D gaze 向量近似推算归一化屏幕坐标
                if abs(pred_vec[2]) > 1e-6:
                    pred_nx = 0.5 + pred_vec[0] / pred_vec[2] * 0.5
                    pred_ny = 0.5 + pred_vec[1] / pred_vec[2] * 0.5
                else:
                    pred_nx, pred_ny = 0.5, 0.5
                pred_nx = np.clip(pred_nx, 0, 1)
                pred_ny = np.clip(pred_ny, 0, 1)

                raw_px = (pred_nx * screen_w, pred_ny * screen_h)
                true_px = (true_nx * screen_w, true_ny * screen_h)
                all_raw_px.append(raw_px)
                all_true_px.append(true_px)
                idx += 1

    all_raw_px = np.array(all_raw_px)
    all_true_px = np.array(all_true_px)

    # 无校准时的误差
    if len(calib_indices) == 0:
        # 不做校准，直接计算 raw 误差
        test_raw = all_raw_px[test_indices]
        test_true = all_true_px[test_indices]
        errors = np.sqrt(np.sum((test_raw - test_true) ** 2, axis=1))
        return {
            "method": "none",
            "num_calib_points": 0,
            "fit_error_px": 0.0,
            "holdout_error_px": float(np.mean(errors)),
            "holdout_median_px": float(np.median(errors)),
            "holdout_p90_px": float(np.percentile(errors, 90)),
            "holdout_std_px": float(np.std(errors)),
            "num_test_points": len(test_indices),
        }

    # 用校准点拟合
    calibrator = CalibrationModule(
        num_points=len(calib_indices),
        method=method,
    )
    for ci in calib_indices:
        calibrator.add_calibration_point(
            raw_gaze=tuple(all_raw_px[ci]),
            screen_target=tuple(all_true_px[ci]),
        )

    try:
        fit_residual = calibrator.calibrate()
    except ValueError as e:
        logger.warning(f"校准失败: {e}")
        return None

    # 在测试点上计算 hold-out 误差
    holdout_errors = []
    for ti in test_indices:
        calibrated = calibrator.apply(tuple(all_raw_px[ti]))
        true = all_true_px[ti]
        err = np.sqrt((calibrated[0] - true[0]) ** 2 + (calibrated[1] - true[1]) ** 2)
        holdout_errors.append(err)

    holdout_errors = np.array(holdout_errors)

    return {
        "method": method,
        "num_calib_points": len(calib_indices),
        "fit_error_px": fit_residual,
        "holdout_error_px": float(np.mean(holdout_errors)),
        "holdout_median_px": float(np.median(holdout_errors)),
        "holdout_p90_px": float(np.percentile(holdout_errors, 90)),
        "holdout_std_px": float(np.std(holdout_errors)),
        "num_test_points": len(test_indices),
    }


def run_calibration_experiment(
    df: pd.DataFrame,
    model,
    screen_w: int = 1536,
    screen_h: int = 864,
    n_repeats: int = 20,
) -> list[dict]:
    """运行完整的校准对比实验。

    对每种校准配置（点数 × 方法），随机采样校准点和测试点，
    重复 n_repeats 次取平均。

    参数:
        df: 测试集 DataFrame
        model: GazeNet 模型
        screen_w, screen_h: 屏幕分辨率
        n_repeats: 每种配置的重复次数

    返回:
        所有实验结果列表
    """
    n_samples = len(df)
    all_indices = list(range(n_samples))
    rng = np.random.RandomState(42)

    # 实验配置：(校准点数, 方法)
    configs = [
        (0, "none"),       # 无校准 baseline
        (1, "affine"),     # 1 点平移修正（affine 退化为平移）
        (3, "affine"),     # 3 点仿射
        (5, "affine"),     # 5 点仿射
        (9, "affine"),     # 9 点仿射（当前默认）
        (9, "polynomial"), # 9 点多项式
    ]

    all_results = []

    for num_points, method in configs:
        logger.info(f"--- 配置: {num_points} 点 {method} ---")
        repeat_results = []

        for rep in range(n_repeats):
            if num_points == 0:
                # 无校准：所有样本都是测试点
                calib_idx = []
                test_idx = all_indices
            else:
                # 随机选择校准点，剩余作为测试点
                shuffled = rng.permutation(n_samples)
                calib_idx = shuffled[:num_points].tolist()
                test_idx = shuffled[num_points:].tolist()

            result = simulate_calibration_from_data(
                df, model, calib_idx, test_idx,
                method=method if method != "none" else "affine",
                screen_w=screen_w, screen_h=screen_h,
            )
            if result is not None:
                result["repeat"] = rep
                repeat_results.append(result)

        if repeat_results:
            # 汇总统计
            avg_result = {
                "method": method,
                "num_calib_points": num_points,
                "holdout_error_px_mean": float(np.mean([r["holdout_error_px"] for r in repeat_results])),
                "holdout_error_px_std": float(np.std([r["holdout_error_px"] for r in repeat_results])),
                "holdout_median_px_mean": float(np.mean([r["holdout_median_px"] for r in repeat_results])),
                "holdout_p90_px_mean": float(np.mean([r["holdout_p90_px"] for r in repeat_results])),
                "fit_error_px_mean": float(np.mean([r["fit_error_px"] for r in repeat_results])),
                "n_repeats": len(repeat_results),
            }
            all_results.append(avg_result)
            logger.info(
                f"  hold-out 误差: {avg_result['holdout_error_px_mean']:.1f} ± "
                f"{avg_result['holdout_error_px_std']:.1f} px "
                f"(fit: {avg_result['fit_error_px_mean']:.1f} px)"
            )

    return all_results


def generate_calibration_plots(results: list[dict], output_dir: Path) -> None:
    """生成校准对比实验的可视化图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置中文字体（尝试多种方案）
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # 图 1：校准点数 vs hold-out 误差
    fig, ax = plt.subplots(figsize=(9, 6))

    # 按方法分组
    affine_results = [r for r in results if r["method"] in ("none", "affine")]
    poly_results = [r for r in results if r["method"] == "polynomial"]

    # affine 线
    if affine_results:
        x_aff = [r["num_calib_points"] for r in affine_results]
        y_aff = [r["holdout_error_px_mean"] for r in affine_results]
        yerr_aff = [r["holdout_error_px_std"] for r in affine_results]
        ax.errorbar(x_aff, y_aff, yerr=yerr_aff, marker="o", capsize=4,
                     label="Affine", linewidth=2, markersize=8)

    # polynomial 线
    if poly_results:
        x_poly = [r["num_calib_points"] for r in poly_results]
        y_poly = [r["holdout_error_px_mean"] for r in poly_results]
        yerr_poly = [r["holdout_error_px_std"] for r in poly_results]
        ax.errorbar(x_poly, y_poly, yerr=yerr_poly, marker="s", capsize=4,
                     label="Polynomial", linewidth=2, markersize=8)

    ax.set_xlabel("Number of Calibration Points", fontsize=12)
    ax.set_ylabel("Hold-out Pixel Error (px)", fontsize=12)
    ax.set_title("Calibration Points vs Hold-out Error", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([0, 1, 3, 5, 9])
    fig.tight_layout()
    fig.savefig(output_dir / "calib_points_vs_error.png", dpi=150)
    plt.close(fig)
    logger.info(f"  已保存: calib_points_vs_error.png")

    # 图 2：fit error vs hold-out error 对比（揭示过拟合）
    fig, ax = plt.subplots(figsize=(9, 6))
    methods_with_calib = [r for r in results if r["num_calib_points"] > 0]
    if methods_with_calib:
        labels = [f"{r['num_calib_points']}pt {r['method']}" for r in methods_with_calib]
        fit_errs = [r["fit_error_px_mean"] for r in methods_with_calib]
        holdout_errs = [r["holdout_error_px_mean"] for r in methods_with_calib]

        x_pos = np.arange(len(labels))
        width = 0.35
        ax.bar(x_pos - width / 2, fit_errs, width, label="Fit Error", alpha=0.8)
        ax.bar(x_pos + width / 2, holdout_errs, width, label="Hold-out Error", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel("Pixel Error (px)", fontsize=12)
        ax.set_title("Fit Error vs Hold-out Error (Overfitting Check)", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "fit_vs_holdout_error.png", dpi=150)
    plt.close(fig)
    logger.info(f"  已保存: fit_vs_holdout_error.png")

    # 图 3：汇总表格图
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table_data = []
    headers = ["Config", "Hold-out Mean(px)", "Hold-out Median(px)", "P90(px)", "Fit(px)"]
    for r in results:
        label = f"{r['num_calib_points']}pt {r['method']}"
        table_data.append([
            label,
            f"{r['holdout_error_px_mean']:.1f} ± {r['holdout_error_px_std']:.1f}",
            f"{r['holdout_median_px_mean']:.1f}",
            f"{r['holdout_p90_px_mean']:.1f}",
            f"{r['fit_error_px_mean']:.1f}",
        ])
    table = ax.table(
        cellText=table_data, colLabels=headers,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("Calibration Comparison Summary", fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(output_dir / "calib_summary_table.png", dpi=150)
    plt.close(fig)
    logger.info(f"  已保存: calib_summary_table.png")


def main():
    import argparse
    import torch
    import yaml

    parser = argparse.ArgumentParser(description="校准方法对比实验")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output", default="evaluation_results/calibration_compare")
    parser.add_argument("--repeats", type=int, default=20, help="每种配置的重复次数")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 加载模型
    from models.gaze_net import GazeNet
    model_cfg = config.get("model", {})
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    model = GazeNet(num_channels=channels)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"已加载模型: {args.checkpoint}")

    # 加载测试数据
    test_labels = Path("dataset_processed/test/labels.csv")
    if not test_labels.exists():
        logger.error(f"测试数据不存在: {test_labels}")
        sys.exit(1)
    df = pd.read_csv(test_labels)
    logger.info(f"测试集: {len(df)} 样本")

    # 运行实验
    results = run_calibration_experiment(
        df, model, n_repeats=args.repeats,
    )

    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_dir / 'results.json'}")

    # 生成图表
    generate_calibration_plots(results, output_dir)

    logger.info("校准对比实验完成。")


if __name__ == "__main__":
    main()
