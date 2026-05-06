# Look2Act 深度视线定位恢复实验纪要（2026-05-05 至 2026-05-06）

本文记录 Look2Act 在体验线恢复、direct PoG baseline、领域契约诊断、3D geometry mismatch 诊断中的实验事实和阶段结论。目标不是替代 classic tracker，而是恢复“机器学习模型输出屏幕内注视位置，并可用于交互”的研究主线。

## 一、问题背景

Look2Act 当前有两条路线：

- 产品体验线：使用 Eye_Touch 风格 classic tracker，保证校准、验证、交互窗口可演示。
- 机器学习研究线：使用深度模型预测用户看屏幕哪里，最终输出必须是屏幕内注视位置，而不是无语义 feature。

Classic tracker 已经恢复到可用体验，但 deep / deep_pog 在实时校准后仍出现 raw 点拓扑折叠、残差高、验证页跟随差的问题。因此下一阶段不能盲目加数据或调 UI，而要先证明数据、crop、屏幕坐标、head pose、ray origin 和 screen plane 的契约是否一致。

## 二、体验线结论

已经完成的体验线改造：

- 引入 Eye_Touch 风格 classic backend，作为默认体验路径。
- 校准点改为 5x5；每点采样 55 帧，并丢弃前 10 帧，降低视线移动过渡帧影响。
- 校准完成后停在结果页，不再自动开启 overlay 追踪。
- 验证和交互改为独立全屏窗口，不默认启动系统级透明 overlay。
- 交互页九宫格和 5x5 井字棋已经重做；井字棋结束后显示结果并倒计时返回九宫格。
- 主界面、校准、追踪、交互、设置做了中英双语和语言选择入口。

体验线结论：

- Classic 路线可以作为毕业设计/演示保底路径。
- Classic 不削弱研究创新性，因为它不是论文模型主线，而是工程体验兜底。
- 后续研究线可以专注 deep 模型，不需要再用 UI 问题掩盖模型链路问题。

## 三、Direct PoG baseline 结论

Direct PoG 的目标是：

```text
eye crops + optional head pose -> normalized screen point [norm_x, norm_y] -> calibration -> smoothing -> fullscreen validation/interaction
```

这一路线不做 `deep_feature`，因为无语义 feature 不能直接回答“用户看屏幕哪里”，也不能直接支撑人机交互。

已实现内容：

- 新增 `deep_pog` backend，模型输出屏幕归一化点 `[x, y]`。
- 校准文件隔离为 `calibration_deep_pog.json`。
- ONNX runtime 继续 CPU 推理；训练阶段使用 Intel XPU + IPEX。
- 增加 `deep_pog_zero` 对照，避免异常 head pose 输入污染 direct PoG baseline。
- 新增校准拓扑诊断脚本 `scripts/analyze_calibration_topology.py`。

用户实时测试事实：

```text
25 点校准残差多次约 200-335 px
raw 点会动，但验证页效果很差
raw 点相对 target 点发生拓扑压缩/折叠
```

一次保存的 `calibration_deep_pog.json` 诊断结果：

```text
points = 25
residual ≈ 296.60 px
raw_extent ≈ 274.9 x 266.7
target_extent ≈ 1152 x 720
raw_to_target_area_ratio ≈ 0.0884
raw_x ~ target_x corr ≈ 0.260
raw_y ~ target_y corr ≈ 0.769
row_monotonic_x ≈ 0.500
col_monotonic_y ≈ 0.700
```

该结果说明：高残差不是简单 polynomial calibration 参数问题，而是 raw model outputs 没有保持屏幕 5x5 网格拓扑。校准只能修连续、单调、可逆的映射；如果 raw 点本身折叠，校准无法救回来。

## 四、领域契约诊断结论

新增分支：

```text
research/domain-contract-diagnostics
PR #16: 研究：增加领域契约诊断工具
```

新增工具：

```text
scripts/analyze_domain_contract.py
```

它检查：

- `dataset_raw` 行数、session/user 数量、屏幕/摄像头分辨率分布。
- `dataset_processed` train/val/test target 分布和每点样本数。
- 离线 preprocessing crop 与实时 runtime crop 是否一致。
- 当前 calibration raw 点拓扑是否保持屏幕网格关系。

真实数据审计结果：

```text
raw rows = 7463
sessions = 31
users = 31
processed splits = train / val / test
runtime crop factor = 2.0
offline crop factor = 2.0
crop contract matches = True
head_pitch abs > 90 ratio ≈ 0.9999
```

结论：

- 当前实时 deep_pog collapse 不能优先归因于 eye crop padding 不一致，因为离线与实时 crop factor 已一致。
- head pose 数据约定异常非常明显，不能直接用于 head-local relabel 或作为强输入。
- direct PoG 的主要问题更可能来自模型输出拓扑、训练采样/损失、实时域输入分布和显示/校准契约的组合。

## 五、Direct PoG 拓扑训练实验设计

新增分支：

```text
research/deep-pog-screen-point
PR #17: 研究：增加屏幕点拓扑训练实验
```

新增或扩展内容：

- `GazeNetPoG` 支持 `output_activation`：
  - `sigmoid`
  - `linear`
  - `tanh01`
- `scripts/train.py` 支持 PoG topology ordering loss。
- `scripts/train.py` 支持按 target point inverse-frequency balanced sampling。
- 新增配置：

```text
configs/train_pog_topology_config.yaml
```

- 新增评估：

```text
scripts/evaluate_pog_topology.py
```

该评估不只看 pixel error，还看：

- prediction extent
- pred/target area ratio
- `pred_x ~ target_x`
- `pred_y ~ target_y`
- row/column monotonic

已有 `deep_pog_zero` checkpoint 的离线 test 拓扑结果：

```text
points = 71
pred_extent ≈ 0.5891 x 0.6384
pred_to_target_area_ratio ≈ 0.9152
corr_x ≈ 0.703
corr_y ≈ 0.378
mono_x ≈ 0.772
mono_y ≈ 0.741
```

结论：

- 离线 direct PoG 不是完全不可学，预测范围没有完全压成一点。
- 但 y 轴相关性弱，行列拓扑不够稳定。
- 实时 25 点校准比离线 test 更差，说明需要同时比较离线拓扑和实时 calibration topology，不能只报告 mean pixel error。

下一步实验优先级：

1. `linear + topology loss + balanced sampling`
2. `linear + base loss + balanced sampling`
3. `tanh01 + topology loss + balanced sampling`
4. 必要时再做 `sigmoid + topology loss` 对照

短跑先用 10 epoch 看趋势；再选择最值得的配置跑 60 epoch。

## 六、3D gaze-to-screen 几何契约结论

新增分支：

```text
research/deep-3d-gaze-to-screen
PR #18: 研究：增加3D几何契约诊断
```

新增工具：

```text
scripts/analyze_3d_geometry_contract.py
```

它验证：

```text
eye crops -> 3D gaze/ray -> screen intersection -> screen point
```

并明确区分：

- camera-space gaze
- head-local gaze
- PnP rotation
- ray origin
- fixed screen plane
- per-sample implied screen distance

合成闭环结论：

```text
compute_gaze_vector 生成的 camera-space gaze
使用同一 screen geometry 反投影
可精确回到原始屏幕点
```

真实 processed test 诊断结果：

```text
rows = 1390
head_pitch abs > 90 ratio = 1.0
R^-1 @ gaze 后 local z negative ratio = 1.0
camera_space_zero_origin + fixed 500mm plane:
  mean_pixel_error ≈ 269.5 px
  p95_pixel_error ≈ 580.0 px
camera_space + implied per-sample screen distance:
  mean_pixel_error ≈ 0 px
implied_screen_distance_mm:
  min ≈ 500.0
  p50 ≈ 718.5
  mean ≈ 923.1
  max ≈ 1835.5
face_translation_z500 origin:
  in_screen_ratio = 0.0
head rotation variants:
  no valid forward screen intersection
```

结论：

- 3D label 本身是 camera-space，可以通过正确距离反投影到屏幕点。
- 当前 runtime 固定 500mm screen plane 与训练标签中的 per-sample distance 不一致，这是 3D 路线的主要系统误差来源之一。
- 当前 PnP Euler/head pose 约定不能直接用于 head-local relabel；直接做 `R^-1 @ gaze` 会得到 local z 全负。
- 因此 3D 路线下一步不应立即重训 head-local 模型，而应先修 screen distance / screen plane / ray origin 契约。

## 七、当前推荐路线

短期目标：让机器学习路径先达到“能像 classic 一样完整跑通和可演示”。

推荐顺序：

1. 继续 direct PoG 主线，训练 `linear + topology loss + balanced sampling`。
2. 用 `evaluate_pog.py` 和 `evaluate_pog_topology.py` 同时看 pixel error 与拓扑指标。
3. 如果 10 epoch 趋势好，再跑 60 epoch，并导出 ONNX 进入实时校准验证。
4. 3D 线暂不重训；先设计 runtime/label distance 契约修复实验。
5. 若 direct PoG 仍失败，再考虑把 distance/head geometry 作为显式输入或辅助任务，而不是直接回到旧的 head-pose rotation 方案。

## 八、成功标准

Direct PoG 或 3D gaze-to-screen 模型必须满足：

- 输出是屏幕内注视位置，能用于交互。
- 实时 25 点校准 raw 拓扑不折叠。
- `raw_x ~ target_x`、`raw_y ~ target_y` 明显正相关。
- row/column monotonic 接近 1。
- 验证页中光标能稳定跟随用户看向上下左右和大致区域。

如果离线 pixel error 看起来不错，但实时 raw topology 仍然折叠，则不能认为模型链路恢复成功。

## 九、常用诊断命令

短命令可由 Codex 运行：

```powershell
conda run --no-capture-output -n gaze-env python scripts/analyze_domain_contract.py --raw-dir dataset_raw --processed-dir dataset_processed --calibration calibration_deep_pog.json --output tmp/domain_contract_summary.json
conda run --no-capture-output -n gaze-env python scripts/evaluate_pog_topology.py --checkpoint checkpoints/deep_pog_zero/best_model.pth --output evaluation_results/deep_pog_zero/topology.json
conda run --no-capture-output -n gaze-env python scripts/analyze_3d_geometry_contract.py --processed-dir dataset_processed --split test --output tmp/geometry_contract.json
```

训练命令优先在对应研究分支运行：

```powershell
conda activate gaze-env
python scripts/train.py --config configs/train_pog_topology_config.yaml
python scripts/export_onnx.py --checkpoint checkpoints/deep_pog_topology/best_model.pth --output checkpoints/gaze_pog_topology.onnx
python scripts/evaluate_pog.py --checkpoint checkpoints/deep_pog_topology/best_model.pth --output evaluation_results/deep_pog_topology
python scripts/evaluate_pog_topology.py --checkpoint checkpoints/deep_pog_topology/best_model.pth --output evaluation_results/deep_pog_topology/topology.json
```

## 十、2026-05-06 追加训练实验结果

在 `research/deep-pog-screen-point` 分支中，已按“先短跑、再选择最值得配置跑 60 epoch”的策略完成一轮 direct PoG 实验。

10 epoch 候选结果：

| 实验 | best val px | test mean px | pred area ratio | corr x | corr y | mono x | mono y | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `linear_topology_e10` | 389.12 | 341.02 | 0.231 | 0.570 | 0.549 | 0.807 | 0.741 | 排序较稳，但输出范围被压缩 |
| `linear_base_e10` | 377.98 | 334.64 | 0.429 | 0.491 | 0.447 | 0.719 | 0.776 | 像素误差最低，输出范围最大 |
| `tanh_topology_e10` | 452.53 | 445.31 | 0.002 | 0.215 | 0.293 | 0.649 | 0.621 | 基本坍缩，淘汰 |
| `linear_topology010_e10` | 396.10 | 355.90 | 0.262 | 0.526 | 0.529 | 0.719 | 0.724 | 轻量拓扑未达到折中效果 |

根据短跑结果，选择 `linear_base_e60` 继续跑 60 epoch。训练结果：

```text
config = configs/experiments/train_pog_linear_base_e60.yaml
best epoch = 58
best val_pixel_error_px = 221.76
epoch 60 val_pixel_error_px = 222.36
checkpoint = checkpoints/experiments/deep_pog_linear_base_e60/best_model.pth
onnx = checkpoints/experiments/gaze_pog_linear_base_e60.onnx
ONNX max diff = 2.384186e-07
```

60 epoch test pixel metrics：

```text
num_samples = 1390
mean_pixel_error = 246.12 px
median_pixel_error = 170.24 px
p95_pixel_error = 720.68 px
mean_norm_error = 0.1679
median_norm_error = 0.1095
```

60 epoch test topology metrics：

```text
points = 71
prediction_extent = 0.5546 x 0.8135
pred_to_target_area_ratio = 1.0979
corr pred_x ~ target_x = 0.7410
corr pred_y ~ target_y = 0.3424
row monotonic x = 0.7719
column monotonic y = 0.7414
```

阶段判断：

- Direct PoG 不是完全不可救：60 epoch 后 validation pixel error 从短跑约 378px 降到约 222px，test median pixel error 约 170px。
- 输出范围不再坍缩，`pred_to_target_area_ratio ≈ 1.10`，这比 topology-loss 短跑更适合实时 25 点校准。
- 但模型仍未真正达到可交互标准：y 方向相关性只有约 0.342，P95 仍高达约 721px。
- 下一步最重要的是用 `gaze_pog_linear_base_e60.onnx` 做实时 25 点 deep_pog 校准，检查 raw topology 是否比旧模型明显改善；如果实时仍 y 轴崩坏，应优先研究 y 轴标签、屏幕高度、采集姿态和实时域差异。
