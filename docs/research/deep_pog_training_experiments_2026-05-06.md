# Deep PoG 屏幕点训练实验记录（2026-05-06）

本记录对应分支 `research/deep-pog-screen-point`，目标是让深度模型直接输出屏幕归一化注视点 `[norm_x, norm_y]`，用于后续校准、验证和交互。Classic tracker 只作为体验保底，本实验不做无语义 `deep_feature` backend。

## 实验设置

共同设置：

- 模型：`GazeNetPoG`
- 输出：屏幕归一化点 `[x, y]`
- 输入：左右眼 crop + zero head pose
- 数据：`dataset_processed`
- 训练设备：Intel XPU + IPEX
- batch size：64
- optimizer：Adam
- scheduler：CosineAnnealingLR
- sampling：target-balanced sampling
- 评估：`val_pixel_error_px`、test pixel metrics、test topology metrics

候选配置：

| 实验 | 输出激活 | topology weight | 轮数 | 目的 |
| --- | --- | ---: | ---: | --- |
| `deep_pog_linear_topology_e10` | linear | 0.35 | 10 | 强拓扑约束，检查排序是否改善 |
| `deep_pog_linear_base_e10` | linear | 0.00 | 10 | 无拓扑约束，检查输出范围和像素误差 |
| `deep_pog_tanh_topology_e10` | tanh01 | 0.35 | 10 | 受限输出对照 |
| `deep_pog_linear_topology010_e10` | linear | 0.10 | 10 | 轻量拓扑约束折中 |
| `deep_pog_linear_base_e60` | linear | 0.00 | 60 | 根据短跑结果选出的主实验 |

## 10 epoch 短跑结果

| 实验 | best val px | test mean px | test median px | test P95 px | pred area ratio | corr x | corr y | mono x | mono y |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_topology_e10` | 389.12 | 341.02 | 322.02 | 677.87 | 0.231 | 0.570 | 0.549 | 0.807 | 0.741 |
| `linear_base_e10` | 377.98 | 334.64 | 297.37 | 688.17 | 0.429 | 0.491 | 0.447 | 0.719 | 0.776 |
| `tanh_topology_e10` | 452.53 | 445.31 | 448.08 | 700.86 | 0.002 | 0.215 | 0.293 | 0.649 | 0.621 |
| `linear_topology010_e10` | 396.10 | 355.90 | 336.95 | 670.35 | 0.262 | 0.526 | 0.529 | 0.719 | 0.724 |

短跑结论：

- `tanh01 + topology` 明显失败，预测范围几乎压成一点，淘汰。
- `linear + topology_weight=0.35` 的 x/y 相关性更均衡，但输出面积被压缩。
- `linear + topology_weight=0.10` 未达到预期折中，像素误差和面积都不如 base。
- `linear + base + balanced` 的 10 epoch 像素误差最低、预测面积最大，更适合作为 60 epoch 主实验。

## 60 epoch 主实验

配置：

```text
configs/experiments/train_pog_linear_base_e60.yaml
```

训练趋势：

| epoch | train loss | val loss | val px |
| ---: | ---: | ---: | ---: |
| 1 | 0.034010 | 0.028789 | 481.60 |
| 10 | 0.012900 | 0.017500 | 364.96 |
| 20 | 0.007900 | 0.013000 | 288.46 |
| 33 | 0.005400 | 0.008700 | 228.29 |
| 49 | 0.003700 | 0.008500 | 226.02 |
| 58 | 0.003449 | 0.008622 | 221.76 |
| 60 | 0.003392 | 0.008512 | 222.36 |

最佳 checkpoint：

```text
checkpoints/experiments/deep_pog_linear_base_e60/best_model.pth
```

ONNX：

```text
checkpoints/experiments/gaze_pog_linear_base_e60.onnx
```

ONNX 一致性：

```text
max diff = 2.384186e-07
mean diff = 1.192093e-07
```

test pixel metrics：

```text
num_samples = 1390
mean_pixel_error = 246.12 px
median_pixel_error = 170.24 px
p95_pixel_error = 720.68 px
mean_norm_error = 0.1679
median_norm_error = 0.1095
```

test topology metrics：

```text
points = 71
prediction_extent = 0.5546 x 0.8135
pred_to_target_area_ratio = 1.0979
corr pred_x ~ target_x = 0.7410
corr pred_y ~ target_y = 0.3424
row monotonic x = 0.7719
column monotonic y = 0.7414
```

## 阶段判断

这次 60 epoch 实验说明 direct PoG 并非完全不可救：

- validation pixel error 从 10 epoch 约 378px 降到最佳约 222px。
- test median pixel error 达到约 170px。
- prediction extent 不再被压缩，面积比约 1.10，说明模型输出不会像失败配置那样坍缩成小团。
- x 方向相关性较好，`pred_x ~ target_x ≈ 0.741`。

但它还没有达到可直接信任的实时交互模型：

- y 方向相关性仍弱，`pred_y ~ target_y ≈ 0.342`。
- P95 pixel error 仍高，约 721px。
- row/column monotonic 约 0.74-0.77，还没有达到实时 25 点校准所需的强单调。

因此下一步不应该只继续盲目增加 epoch。更合理的方向是：

1. 用该 ONNX 进入实时 25 点 deep_pog 校准，观察 raw topology 是否比旧模型明显改善。
2. 如果实时 raw area 和 x/y correlation 明显提升，再考虑继续扩展到 100-120 epoch 或增加数据。
3. 如果实时仍 y 轴崩坏，应优先研究 y 轴标签/屏幕高度/采集姿态/眼部 crop 域差异，而不是先改模型容量。
4. topology loss 需要重新设计为不压缩输出面积的形式；当前简单 pairwise loss 会改善部分排序，但有范围压缩副作用。

## 复现实验命令

```powershell
conda activate gaze-env

python scripts/train.py --config configs/experiments/train_pog_linear_base_e60.yaml
python scripts/export_onnx.py --checkpoint checkpoints/experiments/deep_pog_linear_base_e60/best_model.pth --output checkpoints/experiments/gaze_pog_linear_base_e60.onnx
python scripts/evaluate_pog.py --checkpoint checkpoints/experiments/deep_pog_linear_base_e60/best_model.pth --output evaluation_results/experiments/deep_pog_linear_base_e60
python scripts/evaluate_pog_topology.py --checkpoint checkpoints/experiments/deep_pog_linear_base_e60/best_model.pth --output evaluation_results/experiments/deep_pog_linear_base_e60/topology.json
```
