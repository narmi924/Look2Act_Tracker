# Experiment Log

## Exp-2026-03-16-CalibrationCompare

### 目标
比较不同校准点数（0/1/3/5/9）和映射函数（affine/polynomial）对屏幕像素误差的影响。
重点：使用 hold-out 评估而非 fit residual，揭示过拟合风险。

### 配置
- dataset: dataset_processed/test (447 samples)
- model: checkpoints/best_model.pth (GazeNet, channels=[32,64,128,256])
- screen: 1536×864
- 校准配置:
  - 0pt (no calibration) — baseline
  - 1pt affine (退化为平移修正)
  - 3pt affine
  - 5pt affine
  - 9pt affine (当前默认)
  - 9pt polynomial (二次多项式)
- 每种配置随机采样 20 次取平均
- 评估方式: hold-out (校准点不参与测试)

### 指标
- hold-out mean pixel error (px)
- hold-out median pixel error (px)
- hold-out P90 pixel error (px)
- fit error (px) — 用于对比过拟合程度

### 结果

| 配置 | Hold-out Mean (px) | Std | Fit Error (px) | 备注 |
|------|-------------------|-----|----------------|------|
| 0pt 无校准 | 369.4 ± 0.0 | — | — | baseline |
| 1pt 平移 | 472.5 ± 78.2 | 高 | ~0 | 比无校准更差 |
| 3pt affine | 980.8 ± 865.1 | 极高 | ~0 | 严重过拟合 |
| 5pt affine | 542.7 ± 286.9 | 高 | 182.2 | 不稳定 |
| 9pt affine | 387.0 ± 28.8 | 低 | 281.1 | 最稳定 |
| 9pt polynomial | 589.0 ± 190.7 | 高 | 116.0 | 过拟合 |

### 结论
1. 3pt affine 严重过拟合（6 参数 / 3 点 = 完美拟合无泛化）
2. 9pt affine 是唯一稳定配置，但 hold-out 仅略优于无校准
3. polynomial 过拟合：fit 低但 hold-out 高
4. 当前实验在测试集随机采样校准点，非真实校准场景
5. 后续需设计固定网格校准 + 随机测试点的更真实实验

### 文件位置
- 脚本: scripts/exp_calibration_compare.py
- 结果: evaluation_results/calibration_compare/

---

## Exp-2026-03-16-ErrorAnalysis

### 目标
对模型预测误差进行条件分桶分析，识别系统性弱点。

### 配置
- dataset: dataset_processed/all (train+val+test, 2193 samples)
- model: checkpoints/best_model.pth
- 分析维度:
  - 误差 vs head yaw (7 bins)
  - 误差 vs head pitch (7 bins)
  - 误差 vs session (10 sessions)
  - 误差 vs user (3 users)
  - 屏幕空间误差热图 (5×5 grid)
  - 预测 vs 真实散点图 (偏差方向)

### 指标
- per-bin mean angle error (degrees)
- per-session mean angle error
- per-user mean angle error
- screen-space grid-averaged error

### 结果

**Overall (2193 samples)**:
- Mean Angle Error: 4.10° (train 3.02°, val 5.27°, test 7.28°)
- Median Angle Error: 3.10°
- Mean Pixel Error: 342.6 px

**Per-User 误差差异显著**:
- User 13: 1.89° (最好，273 samples)
- User 14: 2.43°
- User 15: 2.30°
- User 16: 9.01° (最差，268 samples，测试集)
- User 25: 5.27°
- User 26: 4.99°
- User 27: 4.68°

**关键发现**:
1. 训练集 vs 测试集差距大（3.02° vs 7.28°），存在明显泛化 gap
2. User 16 误差 9.01° 远高于其他用户，可能是跨设备/跨用户泛化问题
3. 同一设备（DEVZjesur）的 3 个 session 误差都很低（1.89°~2.43°），说明设备内一致性好
4. 跨设备（DESKTOP-69KFPLC）误差最高（9.01°），是泛化瓶颈

### 文件位置
- 脚本: scripts/exp_error_analysis.py
- 结果: evaluation_results/error_analysis/


---

## Exp-2026-03-16-SmoothingCompare

### 目标
比较 EMA 不同 α 参数和 One Euro Filter 在 jitter/accuracy/latency 三个维度的表现。

### 配置
- 合成轨迹: 300 帧, 30 FPS, 噪声 σ=30px
- 轨迹包含: fixation → saccade → fixation → smooth pursuit → fixation
- EMA α: 0.1, 0.2, 0.3, 0.5, 0.7, 1.0
- One Euro: (mc=0.5,β=0.005), (mc=1.0,β=0.007), (mc=2.0,β=0.01), (mc=5.0,β=0.02)

### 结果

| 配置 | Jitter | RMSE (px) | Saccade Latency |
|------|--------|-----------|-----------------|
| No smoothing | 26.0 | 41.2 | 40.1 |
| EMA α=0.1 | 5.6 | 62.4 | 166.4 |
| EMA α=0.2 | 6.9 | 33.6 | 91.2 |
| EMA α=0.3 (当前) | 8.2 | 24.7 | 56.9 |
| EMA α=0.5 | 11.8 | 24.3 | 32.7 |
| EMA α=0.7 | 16.6 | 30.0 | 31.5 |
| 1Euro mc=0.5 β=0.005 | 12.9 | 20.5 | 31.4 |
| 1Euro mc=1.0 β=0.007 | 14.0 | 22.6 | 31.0 |

### 结论
1. One Euro Filter (mc=0.5) 在 RMSE 上最优（20.5px），同时延迟低（31.4）
2. 当前 EMA α=0.3 的 jitter 最低之一（8.2），但延迟较高（56.9）
3. EMA α=0.5 是 EMA 系列中 accuracy-latency 最佳平衡点
4. One Euro Filter 的自适应特性使其在 saccade 时响应快、fixation 时平滑
5. 建议：考虑将默认平滑器从 EMA α=0.3 切换为 One Euro Filter

### 文件位置
- 脚本: scripts/exp_smoothing_compare.py
- 结果: evaluation_results/smoothing_compare/
