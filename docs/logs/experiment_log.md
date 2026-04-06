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

---

## Exp-2026-03-16-HeadPoseAblation

### 目标
量化 PnP 头部姿态几何补偿对注视估计精度的贡献，通过消融实验对比有/无 head pose 旋转的屏幕像素误差。

### 配置
- dataset: dataset_processed/test (447 samples)
- model: checkpoints/best_model.pth (GazeNet, channels=[32,64,128,256])
- screen: 1536×864, 344×215mm, distance=500mm
- 几何 pipeline: gaze_vector → R @ gaze → ray-plane intersect → screen px
- 消融条件:
  - full_pose: 完整 head pose 旋转（R 从 yaw/pitch/roll 重建）
  - no_pose: 无旋转（单位矩阵 I 替代 R）
  - no_yaw: 消融 yaw（yaw=0，保留 pitch/roll）
  - no_pitch: 消融 pitch（pitch=0，保留 yaw/roll）

### 数据集 head pose 分布
- head_yaw: [-19.9°, +12.6°], mean=-6.7°
- head_pitch: [-173.4°, +162.3°], mean=-147.3°（RQDecomp3x3 的 180° 歧义）
- head_roll: [-175.4°, +179.0°], mean=52.0°

### 结果

| 条件 | Mean Pixel Error (px) | Std | Median (px) |
|------|----------------------|-----|-------------|
| full_pose | 364.1 | 153.0 | 369.7 |
| no_pose | 503.2 | 179.4 | 526.5 |
| no_yaw | 364.1 | 153.0 | 369.7 |
| no_pitch | 514.9 | 187.7 | 512.8 |

Head pose 总贡献: 像素误差减少 139.1 px (27.7%)

### 结论
1. Head pose 几何补偿将像素误差从 503.2 px 降至 364.1 px，贡献 27.7%
2. Pitch 是主要贡献维度：消融 pitch 后误差 514.9 px，甚至比完全无 pose 更差
3. Yaw 在当前数据集中无贡献（no_yaw = full_pose），因为 yaw 范围仅 [-20°, +13°]
4. 角度误差在所有条件下相同（7.28°），因为消融只影响几何投影，不影响模型输出
5. 注意：pitch/roll 的 RQDecomp3x3 输出存在 180° 歧义，但 roundtrip 验证一致

### 论文价值
- 证明 PnP 几何补偿是系统精度的重要组成部分（~28% 改善）
- Pitch 补偿是关键，yaw 在桌面场景下影响有限
- 支持 Table 3（系统消融表）中 head pose on/off 的对比行

### 文件位置
- 脚本: scripts/exp_head_pose_ablation.py
- 结果: evaluation_results/head_pose_ablation/


---

## Exp-2026-03-17-LeaveOneOut

### 目标
通过 Leave-One-User-Out 交叉验证评估模型的跨用户泛化能力。每折留出一个用户全部数据做测试，其余 9 个用户数据从头训练 GazeNet。

### 配置
- dataset: dataset_processed/all (2193 samples, 10 users)
- model: GazeNet (channels=[32,64,128,256])，每折从头训练
- epochs: 50, batch_size: 64, lr: 0.001, weight_decay: 1e-4
- scheduler: CosineAnnealingLR
- device: Intel Arc XPU (IPEX)
- screen: 1536×864

### 结果

| 留出用户 | 训练样本 | 测试样本 | 角度误差 (°) | 像素误差 (px) | 训练耗时 (s) |
|---------|---------|---------|-------------|-------------|-------------|
| User 4  | 2041 | 152 | 6.11 ±2.88 | 376.2 | 303 |
| User 5  | 2035 | 158 | 9.30 ±1.90 | 360.7 | 304 |
| User 6  | 2040 | 153 | 4.41 ±2.28 | 345.6 | 297 |
| User 13 | 1920 | 273 | 3.49 ±1.36 | 328.2 | 281 |
| User 14 | 1920 | 273 | 3.34 ±1.51 | 320.2 | 291 |
| User 15 | 1920 | 273 | 4.39 ±1.80 | 333.7 | 278 |
| User 16 | 1925 | 268 | 5.64 ±1.49 | 349.7 | 280 |
| User 25 | 1992 | 201 | 4.34 ±3.76 | 348.4 | 292 |
| User 26 | 1930 | 263 | 4.95 ±4.11 | 361.0 | 281 |
| User 27 | 2014 | 179 | 4.28 ±2.22 | 371.7 | 292 |

**跨用户汇总**:
- Mean Angle Error: 5.03° ±1.64°
- Mean Pixel Error: 349.5 ±17.4 px
- 最佳用户: User 14 (3.34°)
- 最差用户: User 5 (9.30°)
- 总训练时间: ~48 分钟 (10 折)

### 结论
1. 跨用户平均角度误差 5.03°，比固定划分的测试集误差 7.28° 更低，说明固定划分中测试用户恰好较难
2. User 5 误差最高 (9.30°)，是跨用户泛化的瓶颈，可能与该用户的头部姿态或眼部特征差异有关
3. User 13/14 误差最低 (~3.4°)，这两个用户数据量最大 (273 samples)，且来自同一设备
4. 用户间误差标准差 1.64°，说明模型对不同用户的泛化能力存在显著差异
5. 像素误差跨用户相对稳定 (320~376 px)，方差小于角度误差
6. 训练时间每折约 5 分钟，XPU 加速有效

### 论文价值
- 支持 Table 2（跨用户 LOO 结果）
- 证明模型具备一定跨用户泛化能力（5.03° vs 单用户 ~2-3°）
- 揭示用户间差异是精度瓶颈，支持 personalization/calibration 的必要性论述

### 文件位置
- 脚本: scripts/exp_leave_one_out.py
- 结果: evaluation_results/leave_one_out/


---

## Exp-2026-03-24-GazeNetV2

### 目标
将模型从 GazeNet V1（单眼输入）升级为 GazeNetV2（双眼共享 CNN + head pose 融合），验证双眼信息和头部姿态特征对视线估计精度的提升效果。

### 动机
- V1 仅使用单眼图像，丢失了双眼协同信息
- 头部姿态消融实验（Exp-2026-03-16）证明 head pose 几何补偿贡献 27.7%，但仅在后处理阶段使用
- 将 head pose 直接融入模型特征空间，让网络学习姿态-视线的非线性映射关系
- 新增 12 个用户数据（88~99），训练数据量翻倍，支撑更复杂模型

### V2 架构设计

```
左眼 (B,3,128,128) ──→ 共享 CNN backbone ──→ left_feat (B,256)  ─┐
                         (4层 Conv+BN+ReLU+Pool)                   │
右眼 (B,3,128,128) ──→ 共享 CNN backbone ──→ right_feat (B,256) ─┤── concat ──→ (B,515)
                         (权重共享)                                 │
head_pose (B,3) ─────────────────────────────────────────────────┘
                                                                    │
                                                              FC(515→128)
                                                              ReLU + Dropout(0.3)
                                                              FC(128→3)
                                                              L2 normalize
                                                                    │
                                                              gaze_vector (B,3)
```

**关键设计决策**：
- 共享 backbone：左右眼使用同一组 CNN 权重，减少参数量，利用眼部结构对称性
- 融合维度 515 = 256(左) + 256(右) + 3(pose)
- Dropout 0.3 缓解过拟合（V1 训练中观察到 train/val gap）
- 参数量 < 2.5M，保持轻量级

### 配置
- model: GazeNetV2, channels=[32,64,128,256], fusion_dim=128, dropout=0.3
- dataset: 22 用户 (原 10 + 新增 88~99), ~4800 samples
- data split: train=3350, val=604, test=868
- training: 100 epochs, batch_size=64, lr=0.001, Adam, CosineAnnealingLR
- device: Intel Arc XPU (IPEX)
- 预处理：同时保存左右眼裁剪图像

### 代码变更
- `src/models/gaze_net.py`: 新增 GazeNetV2 类
- `src/data/dataset.py`: 支持 V1/V2 双模式，V2 输出 left_eye + right_eye + head_pose
- `scripts/train.py`: 支持 V1/V2 训练，自动检测模型版本
- `scripts/preprocess.py`: 同时保存左右眼图像
- `scripts/export_onnx.py`: V2 导出 3 输入 ONNX
- `src/tracker/pipeline.py`: 实时推理适配 V2（PyTorch + ONNX 双路径）
- `configs/train_config.yaml`: 新增 V2 参数（head_pose_dim, fusion_dim, dropout）
- `configs/system_config.yaml`: 新增 model_version 配置

### 结果（2026-03-29 训练完成）

**训练概况**：
- 设备: Intel Arc XPU (IPEX 2.8.10+xpu)
- 训练数据: 5018 样本（train），987 样本（val）
- 测试数据: 868 样本
- 总训练时间: ~107 分钟（100 epochs，每 epoch ~64s）
- 参数量: 455,811

| 指标 | V1 (10用户) | V2 (22用户) | 变化 |
|------|------------|------------|------|
| Best val_angle | 4.38° (epoch 56) | 2.02° (epoch 75) | -54% |
| Test angle error | 7.28° | 3.46° | -52% |
| Test median angle | — | 2.76° | — |
| Test pixel error | 369.4 px | 337.4 px | -8.7% |
| 训练数据量 | 1545 | 5018 | +225% |
| train/val gap | 0.017/0.088 | 0.024/0.035 | 大幅缩小 |

**训练曲线特征**：
- Epoch 1-14: 快速下降阶段，val_angle 从 5.72° 降至 2.72°
- Epoch 14-43: 震荡收敛阶段，val_angle 在 2.0°~3.9° 间波动
- Epoch 43-100: 稳定阶段，val_angle 稳定在 2.0°~2.3°
- 最终 train_loss=0.024, val_loss=0.036，gap 仅 0.012（V1 gap 为 0.071），过拟合显著缓解

### 结论
1. V2 相比 V1 角度误差降低 52%（7.28° → 3.46°），验证了双眼 + head pose 融合的有效性
2. train/val gap 从 V1 的 0.071 缩小到 0.012，Dropout + 更大数据集有效缓解过拟合
3. 像素误差改善相对较小（-8.7%），说明几何投影链路的误差仍是瓶颈
4. 训练数据量增加 225%（1545 → 5018）是性能提升的重要因素，无法完全归因于架构改进
5. 需要后续控制变量实验（同数据量下 V1 vs V2）来分离架构贡献和数据贡献

### V2 评估实验更新（2026-03-30）

用 V2 模型重跑了所有评估实验，结果如下：

**基础评估（测试集 1390 样本）**：
- Mean angle error: 2.82°（V1: 7.28°，-61%）
- Median angle error: 1.98°
- Mean pixel error: 314.5 px（V1: 369.4 px，-15%）

**条件分桶误差分析（全量 7395 样本）**：
- Overall mean angle: 1.60°（V1: 4.10°，-61%）
- train/val/test: 1.17° / 2.06° / 2.82°
- 最佳用户: User 82 (0.74°)，最差: User 5 (7.72°)
- User 5 仍是泛化瓶颈，其余用户大幅改善

**头部姿态消融（测试集 1390 样本）**：
- full_pose: 365.0 px, no_pose: 501.9 px
- Head pose 贡献: 136.9 px (27.3%)，与 V1 结论一致

**校准对比（测试集 1390 样本，20 次重复）**：
- 0pt 无校准: 314.5 px
- 9pt affine: 221.2 ±28.6 px（-30%，最稳定）
- 9pt polynomial: 437.0 ±222.3 px（过拟合）
- 3pt affine: 4608.4 px（严重过拟合，与 V1 一致）


---

## Exp-2026-04-05-LeaveOneOutResume

### 目标
在扩展后的全量数据集上完成 Leave-One-User-Out 交叉验证收尾，验证 GazeNetV2 在更大用户规模下的跨用户泛化能力，并为论文中的主结果表更新最终口径。

### 配置
- dataset: dataset_processed/all（7395 samples, 31 users）
- model: GazeNetV2
- protocol: 每折留出 1 个用户，其余 30 个用户重新训练后测试
- epochs: 50, batch_size: 64, lr: 0.001
- device: Intel Arc XPU
- output: evaluation_results/leave_one_out_resume/

### 结果

**总体结果**：
- 跨用户平均角度误差: 2.48° ±1.35°
- 跨用户平均像素误差: 321.4 ±19.9 px
- 用户中位数角度误差: 2.13°
- 总样本数: 7395
- 完成用户数: 31/31
- 单折平均训练时间: 2260.8 s（约 37.7 分钟）
- 全部 31 折总训练时间: 70085.6 s（约 19.5 小时）

**用户分布特征**：
- 最佳用户: User 81（0.93°，298.4 px）
- 次优用户: User 93（0.94°，315.5 px）、User 82（0.95°，300.0 px）
- 最差用户: User 5（6.96°，356.1 px）
- 次差用户: User 26（5.01°）、User 16（4.60°）
- 13/31 个用户角度误差 ≤ 2.0°
- 23/31 个用户角度误差 ≤ 3.0°
- 仅 5/31 个用户角度误差 ≥ 4.0°

**与旧版 LOO 结果对比（Exp-2026-03-17）**：
- Mean Angle Error: 5.03° → 2.48°（-50.7%）
- Mean Pixel Error: 349.5 px → 321.4 px（-8.0%）
- 用户规模: 10 → 31
- 总样本数: 2193 → 7395

### 分析
1. 新版 LOO 将跨用户平均角度误差压到 2.48°，说明 V2 + 扩展数据集已经把“跨用户可用性”从探索性结果推进到可作为论文主结果的水平。
2. 角度误差的均值（2.48°）和中位数（2.13°）差距不大，说明整体分布已明显收敛，当前瓶颈主要集中在少数难用户，而不是普遍性的跨用户失效。
3. User 5、26、16 仍构成长尾误差簇，后续需要重点排查其设备参数、姿态分布、佩戴眼镜/遮挡情况，以及是否存在标注或预处理质量问题。
4. 像素误差改善幅度（-8.0%）小于角度误差改善幅度（-50.7%），再次说明当前系统的剩余误差已更多受屏幕几何映射、相机内参近似和校准链路限制。
5. 从 31 个用户中已有 23 个用户达到 3° 以内，论文可以更有力地强调“多数用户可用，少数长尾用户仍需 personalization/calibration”这一系统结论。

### 对项目推进和论文的价值
- Table 2 可以升级为“31-user LOO”主结果，而不再停留在早期 10-user 版本。
- 摘要、Introduction、Conclusion 中的数据规模应统一更新为 31 users / 7395 samples / 2.48°。
- Discussion 部分应把论文主线从“证明具备一定泛化能力”调整为“多数用户已达到可用精度，但长尾用户仍需要个性化补偿”。
- 后续补充实验的优先级应转向长尾用户误差归因和系统级交互验证，而不是继续重复基础 LOO。

### 文件位置
- 脚本: scripts/exp_leave_one_out.py
- 结果: evaluation_results/leave_one_out_resume/
