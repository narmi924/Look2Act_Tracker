# Look2Act 3D Gaze-to-Screen 阶段总结

日期：2026-05-06

## 结论

`Look2Act 3D Gaze-to-Screen 契约修复计划` 的第一阶段已经完成：我们找到了能让现有 3D 模型实时输出可校准屏幕注视点的 runtime 契约。

当前最佳配置不是旧的 `head-space + PnP rotation + face_translation + fixed 500mm`，而是：

```yaml
tracker.backend: deep
model.deep_gaze_space: camera
model.deep_ray_origin: zero_origin
model.deep_pose_input: zero
model.deep_eye_input_mode: swap
geometry.screen_distance_mm: 720
calibration.num_points: 25
smoother.type: ema
```

对应推荐命令：

```bash
python main.py --config configs/experiments/system_deep_camera_zero_720_pose_zero_swap_ema.yaml
```

## 已修复的问题

### 1. 3D 投影契约

离线 label-space 诊断证明：现有 3D gaze label 可以通过合适 screen distance 投影回屏幕点。旧 runtime 最大问题是把契约写成了 head-space/PnP rotation，并使用不合适的 ray origin / screen distance。

修复后：

- 使用 camera-space gaze。
- 不再默认乘 PnP rotation。
- 使用 zero origin。
- screen distance 以 720mm 作为当前实时候选。

### 2. 深度校准点数

实验 YAML 写了 25 点，但校准页原来对 `deep` backend 硬编码为 9 点 affine。已修复为按 `SystemConfig.calibration` 创建校准模块。

现在 deep 实验可正常使用 25 点 polynomial 校准，并保留每点 55 帧、丢弃前 10 帧的采样策略。

### 3. 左右眼输入契约

实时消融结果证明：最大突破来自 `deep_eye_input_mode: swap`。这说明训练集/实时推理之间存在左右眼输入顺序契约错位。

对比：

- `swap`：可用，残差最低到 152.33 px，验证阶段能随眼球方向正确变化。
- `flip`：上下强，但横向反向。
- `swap_flip`：基本折叠。
- `normal`：横向弱或折叠。

### 4. 校准阶段平滑污染

EMA 配置一开始会在校准阶段污染采样。已修复为：

- 校准模式下 pipeline 跳过 smoother。
- 校准页优先采 `raw_point`。
- EMA/Kalman 只影响验证和交互，不影响校准拟合。

## 关键实验记录

### 旧链路和中间方案

旧 deep 链路表现为 9 点或 25 点校准残差 300px 左右，raw topology 折叠，验证效果很差。

`camera + zero_origin + 720mm + pose_zero` 能恢复部分上下拓扑，但横向仍不稳定。

`swap_live` 在轻微转头时会压缩上下方向，说明现有实时 head pose 输入暂不适合作为默认。

### eye input 消融

| 配置 | residual | corr_x | corr_y | mono_x | mono_y | 结论 |
|---|---:|---:|---:|---:|---:|---|
| swap | 160.67 | 0.773 | 0.880 | 0.85 | 0.95 | 首个可用 3D runtime 契约 |
| flip | 167.06 | -0.528 | 0.968 | 0.15 | 1.00 | 横向反向 |
| swap_flip | 377.31 | -0.192 | 0.176 | 0.45 | 0.60 | 折叠 |
| swap + 1920x1080 | 未记录 residual | 0.855 | 0.909 | 0.85 | 0.95 | 拓扑最强 |
| swap + EMA 修复后 | 152.33 | 0.536 | 0.947 | 0.80 | 1.00 | 当前最低残差，验证可用 |

最新 `swap + EMA` 记录：

- residual：152.33 px
- corr_x：0.536
- corr_y：0.947
- mono_x：0.80
- mono_y：1.00
- area_ratio：0.415

这组横向 Pearson correlation 不如之前 1920x1080 那组高，但 raw 覆盖面积最大、残差最低，验证体验已经进入可演示状态。

## 是否完成

已完成：

- 找到让现有 3D 模型实时可工作的 runtime 契约。
- 证明 deep 模型不是完全失败，主要失败原因是 runtime 契约错位。
- 明确 head-local/PnP rotation 暂不成立。
- 明确 live head pose 暂不适合作为默认演示输入。
- 得到可演示配置：`system_deep_camera_zero_720_pose_zero_swap_ema.yaml`。

未完成：

- 真正鲁棒的头动补偿。
- 重新训练一个从训练阶段就采用正确左右眼契约、正确 camera-space contract 的模型。
- 把 `swap` 从实验开关变成正式默认 deep runtime 契约。
- 论文叙事需要从“head-local gaze”调整为“camera-space gaze-to-screen contract + runtime mismatch analysis”。

## 下一步建议

1. 将 `deep_eye_input_mode: swap` 固化为 deep backend 默认，或至少作为 3D 实验默认配置。
2. 用当前最佳配置做一次正式录屏和校准日志备份。
3. 重新导出/训练前先修正数据管线命名，确保训练和 runtime 的 left/right eye 语义一致。
4. 如果继续做 3D ML 主线，优先训练 camera-space gaze，暂不做 head-local relabel。
5. 头动补偿单独作为研究问题处理，不再阻塞“模型可演示”目标。
