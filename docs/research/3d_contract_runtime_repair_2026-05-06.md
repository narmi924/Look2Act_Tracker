# Look2Act 3D Gaze-to-Screen 契约修复记录

日期：2026-05-06

## 目标

本轮只处理 3D 机器学习研究线，不影响 classic 体验保底线。目标是先判断现有 3D label、训练输出、实时投影之间的数学契约是否闭合，再决定是否重训。

当前结论按实验事实修正为：现有 3D 路线更适合解释为 camera-space gaze + screen projection，而不是 head-local gaze + PnP rotation。head-local relabel 暂不允许进入重训 gate。

## 已实现

- 扩展 `scripts/analyze_3d_geometry_contract.py`：
  - 比较 fixed 500/720/900 mm、dataset/session/user implied median distance、oracle implied distance。
  - 输出按 split 的投影误差、valid ratio、in-screen ratio、P50/P95、topology 指标。
  - 保留 head rotation / inverse local 诊断，但明确标记为不能作为重训依据。
- 新增 `scripts/evaluate_3d_projection_variants.py`：
  - 加载现有 3D checkpoint 或 ONNX。
  - 比较 camera-space no-rotation、head-space PnP rotation、zero origin、face translation、不同 screen distance。
  - 输出 `evaluation_results/3d_contract_runtime/variant_summary.json` 与 CSV。
- Runtime 新增 3 个实验配置：
  - `configs/experiments/system_deep_camera_zero_500.yaml`
  - `configs/experiments/system_deep_camera_zero_720.yaml`
  - `configs/experiments/system_deep_camera_zero_900.yaml`
- `TrackerPipeline` 增加 `_compute_deep_ray()`，用于单测固定验证：
  - `deep_gaze_space=camera` 时不乘 head pose rotation。
  - `deep_gaze_space=head` 时才乘 PnP rotation。

## 关键诊断结果

### Label-space 几何闭合

在 `dataset_processed` 的标签空间里，oracle implied distance 可以把 3D gaze 几乎精确反投影回屏幕点；session/user median distance 的误差也很小。这说明现有 3D label 本身不是完全无效，主要问题在 runtime 固定距离、ray origin 和 head rotation 契约。

代表性结果：

- test split:
  - fixed 500 mm：mean 269.5 px，P95 580.0 px
  - fixed 720 mm：mean 171.3 px，P95 465.8 px
  - fixed 900 mm：mean 241.4 px，P95 564.7 px
  - dataset median：mean 171.0 px，P95 467.1 px
  - session/user median：mean 6.7 px，P95 19.4 px
  - oracle implied distance：0 px

### Head-local gate

现有 Euler/PnP 坐标暂不能用于 head-local relabel：

- `abs(head_pitch) > 90` 的比例接近 1。
- `R^-1 @ gaze` 后 local z 为负的比例接近 1。
- 现有 checkpoint 使用 `head-space + PnP rotation` 的投影评估 valid ratio 为 0。

因此短期不做 head-local 训练，不把论文叙事强行写成 head-local gaze。

### Checkpoint projection variants

使用现有 `checkpoints/best_model.pth` 在 test split 上评估：

- 最优：`camera_no_rotation + zero_origin + dataset_median`
  - mean 216.2 px
  - median 150.0 px
  - P95 688.2 px
  - corr_x 0.663，corr_y 0.714
  - mono_x 0.825，mono_y 0.759
- 几乎等价：`camera_no_rotation + zero_origin + fixed_720mm`
  - mean 216.5 px
  - median 150.0 px
  - P95 688.8 px
  - topology 指标相同
- `head_space_with_rotation` 全部 valid ratio 为 0。
- `face_translation` 在当前离线近似下明显更差。

## 当前判断

1. 3D 路线仍有希望，但希望不在 head-local relabel，而在 camera-space gaze + 正确 screen distance / ray origin / runtime contract。
2. fixed 720 mm 或 dataset median 是当前最合理的实时候选；fixed 500 mm 不应继续作为默认 3D 链路。
3. 当前 checkpoint 的 topology 指标并没有完全崩，说明模型至少学到了一部分屏幕拓扑；实时失败更可能来自 runtime 契约和实时域差异共同导致。
4. 还不应马上重训。下一步应先用 500/720/900 三个 runtime 配置做真实 25 点校准，对比 raw topology 是否改善。

## 下一步手动验证

分别切换以下配置运行 `python main.py`，每次做 deep 25 点校准，并保存日志中的 25 个 raw/target：

```powershell
conda activate gaze-env
cd D:\Projects\Look2Act_worktrees\deep-3d-gaze-to-screen
python main.py --config configs\experiments\system_deep_camera_zero_500.yaml
python main.py --config configs\experiments\system_deep_camera_zero_720.yaml
python main.py --config configs\experiments\system_deep_camera_zero_900.yaml
```

建议先看 720 mm。验证时先关闭平滑，所以追踪会抖，但 raw topology 更真实。

2026-05-06 追加：720 mm 实时 25 点校准已能跑通，但 raw topology 显示 `raw_y` 比 `raw_x` 更稳定，横向仍有明显折叠。因此新增 `configs/experiments/system_deep_camera_zero_720_pose_zero.yaml`，保持 camera-space + zero origin + 720 mm 不变，只把模型输入的 `head_pose` 置零，用于隔离实时 PnP 姿态特征污染。

2026-05-06 继续追加：`pose_zero` 没有稳定修复横向拓扑，因此新增实时 eye-crop 契约消融：

- `configs/experiments/system_deep_camera_zero_720_pose_zero_swap.yaml`
- `configs/experiments/system_deep_camera_zero_720_pose_zero_flip.yaml`
- `configs/experiments/system_deep_camera_zero_720_pose_zero_swap_flip.yaml`

这三组只改变进入模型前的左右眼顺序/水平翻转，用于检查训练和实时的 left/right eye crop、镜像方向是否错位。

2026-05-06 结论更新：`swap` 是当前最优 runtime eye-crop 契约。一次实时 25 点校准中达到 residual 160.67 px，验证阶段光标已能随眼球方向正确变化。对应拓扑指标约为 `corr_x=0.773`、`corr_y=0.880`、`mono_x=0.85`、`mono_y=0.95`、`area_ratio=0.216`。`flip` 表现为横向反向，`swap_flip` 基本折叠。因此当前 3D runtime 首选契约是：

- `deep_gaze_space: camera`
- `deep_ray_origin: zero_origin`
- `screen_distance_mm: 720`
- `deep_pose_input: zero`
- `deep_eye_input_mode: swap`

新增两个后续验证配置：

- `configs/experiments/system_deep_camera_zero_720_pose_zero_swap_ema.yaml`：在最佳 raw 契约上加 EMA 平滑，用于验证演示观感。
- `configs/experiments/system_deep_camera_zero_720_swap_live.yaml`：保持 `swap`，恢复 live head pose 输入，用于判断头部移动是否能被模型补偿。

2026-05-06 再更新：`swap` 在 1920x1080 摄像头配置下进一步改善，拓扑约为 `corr_x=0.855`、`corr_y=0.909`、`mono_x=0.85`、`mono_y=0.95`。`swap_live` 在轻微转头时仍会压缩上下方向，说明当前 live head pose 输入会破坏 runtime topology，不宜作为默认演示配置。另发现 EMA 配置会在校准阶段污染采样，因此已修复为：校准模式永远采 `raw_point`，pipeline 在校准模式下跳过 smoother；平滑只用于验证/交互。

2026-05-06 最终阶段更新：修复校准采 raw 点后，`system_deep_camera_zero_720_pose_zero_swap_ema.yaml` 达到 residual 152.33 px，验证阶段可用。阶段总结见 `docs/research/3d_gaze_to_screen_stage_summary_2026-05-06.md`。此时可以认为 3D runtime 契约修复第一阶段完成：现有模型在正确 runtime 契约下能输出可校准的屏幕注视点。

如果 720 mm 的实时 raw topology 明显优于旧 deep 链路，再进入 10 epoch smoke 重训；否则先继续排查实时 domain gap、camera crop、screen distance 估计和输入归一化。
