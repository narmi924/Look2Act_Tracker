# Look2Act 恢复与研究状态

## 分支状态

- 当前工作分支：`feat/look2act-diagnose-classic-flow`
- 远端分支：`origin/feat/look2act-diagnose-classic-flow`
- PR 尚未由 Codex 自动创建，原因是本机 `gh` 未认证。
- 浏览器手动开 PR 链接：
  `https://github.com/narmi924/Look2Act_Tracker/pull/new/feat/look2act-diagnose-classic-flow`

## 已完成改动

- 默认演示路径切换为 `classic`。
- `deep` 研究路径保留，可继续用于论文实验和坐标系对照。
- 校准文件已经按后端分离：
  - `calibration_classic.json`
  - `calibration_deep.json`
- 追踪页诊断模式现在会显示 backend、模型版本、ONNX 输入、raw/calibrated/pre-clamp/clamped 坐标、阶段耗时、可用 head pose、校准方法、校准点数量和校准路径。
- Classic tracker 优先使用 MediaPipe iris offset；不可用时回退到眼部裁剪图的 pupil centroid。
- Classic 校准使用 5x5 polynomial fitting；deep 校准使用 3x3 affine fitting。
- 校准流程不会再因为某个点无有效 gaze 而无限卡住；弱采样点会跳过，最终拟合前会检查最少有效点数。
- 默认交互流程改为全屏验证和独立交互窗口，并加入可用 gaze dwell 操作的五子棋窗口。
- 设置页加入基础/高级分层，默认隐藏模型路径、屏幕物理尺寸等研究配置。
- deep 研究链路新增：
  - `deep_gaze_space=head/camera`
  - `deep_pose_input=live/zero`
  - `smoother=kalman/ema/none`
- 新增有限帧数诊断脚本和 CSV 分析脚本，便于比较 classic/deep 实时输出。

## 手动 Smoke Test

请在新终端运行：

```powershell
cd D:\Projects\Look2Act_Tracker_Project
conda activate gaze-env
python main.py
```

建议顺序：

1. 打开设置页，确认 backend 为 `classic`，smoother 为 `kalman`。
2. 启动追踪。
3. 执行 5x5 校准并保存。
4. 在追踪页加载校准。
5. 打开全屏验证窗口。
6. 打开全屏交互窗口，并进入五子棋。
7. 如果追踪看起来不对，打开诊断模式查看 raw/calibrated/pre-clamp/clamped 坐标和阶段耗时。

## 命令行实时诊断

Classic：

```powershell
conda activate gaze-env
python scripts/diagnose_tracker.py --backend classic --frames 300 --csv diagnostics_classic.csv
python scripts/analyze_tracker_diagnostics.py diagnostics_classic.csv
```

Deep camera-space 消融：

```powershell
conda activate gaze-env
python scripts/diagnose_tracker.py --backend deep --deep-space camera --deep-pose-input zero --smoother none --frames 300 --csv diagnostics_deep.csv
python scripts/analyze_tracker_diagnostics.py diagnostics_deep.csv
```

## 标签坐标审计

运行：

```powershell
conda activate gaze-env
python scripts/audit_gaze_labels.py
```

当前对 `dataset_processed` 的审计结论：

- train/val/test 中的 gaze label 都是单位向量。
- 归一化屏幕目标点都在 `[0, 1]` 范围内。
- target duplicate ratio 很高，这是因为每个校准目标点包含多帧重复采样。
- `head_pitch` 几乎总是 `abs(pitch) > 90°`。
- 如果把当前 gaze label 当成 camera-space，再用 `R^-1 @ gaze` 转成 head-local，几乎全部样本会得到负的 local z。这说明当前存储的 Euler 角不能直接用于 head-local label regeneration。
- 用固定 runtime screen plane 把现有 label 投影回屏幕时，平均误差已经很大；使用 runtime-style ray origin 会进一步放大误差。这说明问题不只是“是否乘 PnP rotation”，还包括 label generation geometry 与 runtime screen geometry 的不一致。

研究含义：在 head-pose 坐标约定、label/runtime 屏幕几何关系被修正或显式建模前，不要直接跑 deep label-regeneration/retraining。

## 短测试

Codex 已使用 `gaze-env` 跑过：

```powershell
conda run --no-capture-output -n gaze-env python -m pytest tests\test_calibration.py tests\test_smoother.py tests\test_tracker_pipeline.py tests\test_classic_tracker.py tests\test_classic_pipeline.py tests\test_calibration_flow_helpers.py tests\test_tracking_page_unit.py tests\test_settings_page.py tests\test_audit_gaze_labels.py tests\test_analyze_tracker_diagnostics.py -q
```

最新结果：`40 passed, 1 warning`。警告来自第三方 `qfluentwidgets/scipy` 弃用提示。

## 下一步决策

- 如果 classic 手动体验已经足够顺滑，下一步调 dwell timing、验证窗口和交互窗口视觉细节。
- 如果 classic 不稳定，先查看 `diagnostics_classic.csv`，重点看 feature jump、invalid frame、calibration residual。
- 如果 deep 输出仍不可用，先比较：
  - `deep_gaze_space=head` vs `camera`
  - `deep_pose_input=live` vs `zero`
  - `smoother=kalman/ema/none`
- 如果需要重新训练 deep，必须先解决 `audit_gaze_labels.py` 标出的 head-pose Euler 约定和 fixed-plane/distance mismatch。
- 如果要加入 synthetic/UnityEyes 数据，按 `docs/research/unityeyes_integration_plan.md` 执行；只把它作为预训练/消融实验，不要直接当成 live tracking 失败的补丁。
- preprocess/train/export/evaluate 这类长命令仍由你在独立终端手动运行。
