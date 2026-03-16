"""
平滑算法对比实验脚本。

实验目标（对应 Paper3 Gaze360 的建议）：
1. EMA α 参数扫描（0.1, 0.2, 0.3, 0.5, 0.7, 1.0）
2. 引入 One Euro Filter 作为对比
3. 评估平滑对抖动和延迟的影响

工作目录：Look2Act_Tracker_Project/
运行方式：conda run -n gaze-env python scripts/exp_smoothing_compare.py

输出：evaluation_results/smoothing_compare/
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# 直接导入 smoother 模块，避免 tracker/__init__.py 的循环导入
import importlib.util
_smoother_path = Path(__file__).resolve().parent.parent / "src" / "tracker" / "smoother.py"
_spec = importlib.util.spec_from_file_location("smoother", _smoother_path)
_smoother_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoother_mod)
GazeSmoother = _smoother_mod.GazeSmoother


class OneEuroFilter:
    """One Euro Filter 实现。

    自适应低通滤波器：静止时强平滑（低截止频率），
    快速移动时弱平滑（高截止频率），减少延迟。

    参考: Casiez et al., "1€ Filter", CHI 2012
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        fps: float = 30.0,
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.fps = fps
        self._prev_x: float | None = None
        self._prev_y: float | None = None
        self._prev_dx: float = 0.0
        self._prev_dy: float = 0.0

    def _alpha(self, cutoff: float) -> float:
        te = 1.0 / self.fps
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        if self._prev_x is None:
            self._prev_x = x
            self._prev_y = y
            return point

        # 计算速度（导数）
        dx = (x - self._prev_x) * self.fps
        dy = (y - self._prev_y) * self.fps

        # 平滑速度
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self._prev_dx
        dy_hat = a_d * dy + (1 - a_d) * self._prev_dy

        # 自适应截止频率
        speed = np.sqrt(dx_hat ** 2 + dy_hat ** 2)
        cutoff = self.min_cutoff + self.beta * speed

        # 平滑位置
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self._prev_x
        y_hat = a * y + (1 - a) * self._prev_y

        self._prev_x = x_hat
        self._prev_y = y_hat
        self._prev_dx = dx_hat
        self._prev_dy = dy_hat

        return (x_hat, y_hat)

    def reset(self) -> None:
        self._prev_x = None
        self._prev_y = None
        self._prev_dx = 0.0
        self._prev_dy = 0.0


def generate_synthetic_gaze_sequence(
    n_frames: int = 300,
    fps: float = 30.0,
    noise_std: float = 30.0,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成模拟注视点序列：真实轨迹 + 高斯噪声。

    模拟用户在屏幕上的注视行为：
    - 前 100 帧：注视固定点（fixation）
    - 100-120 帧：快速跳转（saccade）
    - 120-220 帧：注视另一个固定点
    - 220-240 帧：缓慢移动（smooth pursuit）
    - 240-300 帧：注视第三个点

    返回:
        (noisy_points, true_points)，形状均为 (n_frames, 2)
    """
    if rng is None:
        rng = np.random.RandomState(42)

    true_points = np.zeros((n_frames, 2))
    # fixation 1
    true_points[:100] = [400, 300]
    # saccade
    for i in range(100, 120):
        t = (i - 100) / 20.0
        true_points[i] = [400 + t * 600, 300 + t * 200]
    # fixation 2
    true_points[120:220] = [1000, 500]
    # smooth pursuit
    for i in range(220, 240):
        t = (i - 220) / 20.0
        true_points[i] = [1000 - t * 300, 500 - t * 100]
    # fixation 3
    true_points[240:] = [700, 400]

    noise = rng.normal(0, noise_std, size=(n_frames, 2))
    noisy_points = true_points + noise

    return noisy_points, true_points


def evaluate_smoother(
    noisy: np.ndarray,
    true: np.ndarray,
    smoother_fn,
) -> dict:
    """评估平滑器的性能。

    指标:
    - jitter: 连续帧间距离的标准差（越小越平滑）
    - rmse: 平滑后与真实轨迹的均方根误差（越小越准）
    - latency_proxy: 在 saccade 段的平均延迟（平滑值落后真实值的距离）
    """
    n = len(noisy)
    smoothed = np.zeros_like(noisy)

    for i in range(n):
        smoothed[i] = smoother_fn((noisy[i, 0], noisy[i, 1]))

    # jitter: 连续帧间距离的标准差
    diffs = np.sqrt(np.sum(np.diff(smoothed, axis=0) ** 2, axis=1))
    jitter = float(np.std(diffs))

    # RMSE
    errors = np.sqrt(np.sum((smoothed - true) ** 2, axis=1))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # latency proxy: saccade 段 (100-120) 的平均误差
    saccade_errors = errors[100:125]
    latency_proxy = float(np.mean(saccade_errors))

    # fixation jitter: fixation 段的抖动
    fix1_jitter = float(np.std(np.sqrt(
        np.sum(np.diff(smoothed[10:90], axis=0) ** 2, axis=1)
    )))

    return {
        "jitter": jitter,
        "fixation_jitter": fix1_jitter,
        "rmse": rmse,
        "latency_proxy": latency_proxy,
    }


def run_smoothing_experiment() -> list[dict]:
    """运行平滑对比实验。"""
    noisy, true = generate_synthetic_gaze_sequence(noise_std=30.0)

    configs = []

    # EMA α 扫描
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        smoother = GazeSmoother(alpha=alpha)
        metrics = evaluate_smoother(noisy, true, smoother.update)
        configs.append({
            "name": f"EMA α={alpha}",
            "type": "ema",
            "alpha": alpha,
            **metrics,
        })
        logger.info(
            f"  EMA α={alpha}: jitter={metrics['jitter']:.1f}, "
            f"rmse={metrics['rmse']:.1f}, latency={metrics['latency_proxy']:.1f}"
        )

    # One Euro Filter 参数扫描
    for min_cutoff, beta in [(0.5, 0.005), (1.0, 0.007), (2.0, 0.01), (5.0, 0.02)]:
        filt = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        metrics = evaluate_smoother(noisy, true, filt.update)
        configs.append({
            "name": f"1Euro mc={min_cutoff} β={beta}",
            "type": "one_euro",
            "min_cutoff": min_cutoff,
            "beta": beta,
            **metrics,
        })
        logger.info(
            f"  1Euro mc={min_cutoff} β={beta}: jitter={metrics['jitter']:.1f}, "
            f"rmse={metrics['rmse']:.1f}, latency={metrics['latency_proxy']:.1f}"
        )

    # 无平滑 baseline
    metrics_raw = evaluate_smoother(noisy, true, lambda p: p)
    configs.append({
        "name": "No smoothing",
        "type": "none",
        **metrics_raw,
    })

    return configs, noisy, true


def generate_smoothing_plots(
    results: list[dict],
    noisy: np.ndarray,
    true: np.ndarray,
    output_dir: Path,
) -> None:
    """生成平滑对比实验的可视化图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # 图 1: jitter vs latency trade-off 散点图
    fig, ax = plt.subplots(figsize=(9, 6))
    ema_results = [r for r in results if r["type"] == "ema"]
    euro_results = [r for r in results if r["type"] == "one_euro"]
    none_results = [r for r in results if r["type"] == "none"]

    if ema_results:
        ax.scatter(
            [r["fixation_jitter"] for r in ema_results],
            [r["latency_proxy"] for r in ema_results],
            s=100, marker="o", label="EMA", zorder=3,
        )
        for r in ema_results:
            ax.annotate(r["name"], (r["fixation_jitter"], r["latency_proxy"]),
                       fontsize=7, textcoords="offset points", xytext=(5, 5))

    if euro_results:
        ax.scatter(
            [r["fixation_jitter"] for r in euro_results],
            [r["latency_proxy"] for r in euro_results],
            s=100, marker="s", label="One Euro", zorder=3,
        )
        for r in euro_results:
            ax.annotate(r["name"], (r["fixation_jitter"], r["latency_proxy"]),
                       fontsize=7, textcoords="offset points", xytext=(5, 5))

    if none_results:
        ax.scatter(
            [r["fixation_jitter"] for r in none_results],
            [r["latency_proxy"] for r in none_results],
            s=100, marker="x", color="red", label="No smoothing", zorder=3,
        )

    ax.set_xlabel("Fixation Jitter (px, lower=smoother)", fontsize=11)
    ax.set_ylabel("Saccade Latency Proxy (px, lower=faster)", fontsize=11)
    ax.set_title("Smoothing: Jitter vs Latency Trade-off", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "jitter_vs_latency.png", dpi=150)
    plt.close(fig)
    logger.info("  已保存: jitter_vs_latency.png")

    # 图 2: 轨迹对比（选几个代表性配置）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    demo_configs = [
        ("EMA α=0.3 (current)", GazeSmoother(alpha=0.3).update),
        ("EMA α=0.1 (smooth)", GazeSmoother(alpha=0.1).update),
        ("EMA α=1.0 (no smooth)", GazeSmoother(alpha=1.0).update),
        ("One Euro (mc=1.0)", OneEuroFilter(min_cutoff=1.0, beta=0.007).update),
    ]

    for ax, (name, fn) in zip(axes.flat, demo_configs):
        smoothed = np.zeros_like(noisy)
        for i in range(len(noisy)):
            smoothed[i] = fn((noisy[i, 0], noisy[i, 1]))

        ax.plot(true[:, 0], label="True", linewidth=2, alpha=0.8)
        ax.plot(noisy[:, 0], label="Noisy", alpha=0.3, linewidth=0.5)
        ax.plot(smoothed[:, 0], label="Smoothed", linewidth=1.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Frame")
        ax.set_ylabel("X position (px)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Smoothing Trajectory Comparison (X-axis)", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "trajectory_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("  已保存: trajectory_comparison.png")

    # 图 3: 汇总柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = [r["name"] for r in results]
    x_pos = range(len(names))

    ax = axes[0]
    ax.barh(x_pos, [r["fixation_jitter"] for r in results], alpha=0.7)
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Fixation Jitter (px)")
    ax.set_title("Jitter (lower=better)")

    ax = axes[1]
    ax.barh(x_pos, [r["rmse"] for r in results], alpha=0.7, color="orange")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("RMSE (px)")
    ax.set_title("Accuracy (lower=better)")

    ax = axes[2]
    ax.barh(x_pos, [r["latency_proxy"] for r in results], alpha=0.7, color="green")
    ax.set_yticks(x_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Saccade Latency (px)")
    ax.set_title("Responsiveness (lower=better)")

    fig.suptitle("Smoothing Method Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "smoothing_summary.png", dpi=150)
    plt.close(fig)
    logger.info("  已保存: smoothing_summary.png")


def main():
    output_dir = Path("evaluation_results/smoothing_compare")
    logger.info("开始平滑对比实验...")

    results, noisy, true = run_smoothing_experiment()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"结果已保存: {output_dir / 'results.json'}")

    generate_smoothing_plots(results, noisy, true, output_dir)
    logger.info("平滑对比实验完成。")


if __name__ == "__main__":
    main()
