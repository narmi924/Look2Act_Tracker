# 重启进度：R1（待外部审阅、人工摄像头验证）

当前目标：普通 Windows + RGB 摄像头上的眼控优先交互；保留键鼠急停。
Classic / Deep / deep_pog 都是现有基线，不预先确定永久产品路线。
一次一个任务，PR 后停止，不自动合并。本轮不做精度、模型、训练、论文或新交互框架。

## 基线与已确认问题

- 分支：`codex/rebuild-r1-observation-safety`。
- fetch 后实际 origin/main：`ba3a8ef50f529d7782000269013ca47ecd8d579c`；开始时工作区干净。
- `pipeline.py::process_frame` 及三个后端失败分支曾将旧点标为 valid=True；run 异常分支亦如此。
- `pipeline.py::get_latest_result` 曾直接返回共享对象，没有观测身份、源时间和失效连续性标记。
- `tracking_page.py::_update_tracking_data` 每次 UI tick 都再次平滑/推进交互；无新结果时直接返回。
- 启动器、桌面点击、棋盘使用消费时钟；`gomoku_window.py::paintEvent` 也按当前时间增加旧点进度。
- `calibration_page.py::_on_sampling_tick` 同样重复采样 latest result。

## 实际修改

- 冻结的 Observation 保存 UUID 会话、递增序号、perf_counter 时间、连续性代号、时间来源。
  真实采集时间是 **VideoCapture.read 返回后的主机时间**；直接 process_frame 调用为处理入口时间。
  均不是传感器曝光时间，不能据此报告完整端到端显示延迟。
- 失败统一 invalid + held + 原因；保留旧点仅供显示。检查模型、姿态、几何、校准和平滑非有限值。
  Classic 仍不依赖姿态成功。未改 Deep 坐标、pose/origin/swap、几何顺序或平滑参数。
- 每次失败读取/估计增加连续性代号；即使失败结果被下一成功结果覆盖，UI 仍中断停留。
  发布有锁，消费者拿独立副本，源身份不可变；旧线程未结束时拒绝重启。
- ObservationGate 区分 NEW / DUPLICATE / INVALID：仅 NEW 更新滤波与动作；重复不推进、不重置。
  `tracker.max_observation_age_ms` 默认 250 ms，同时约束年龄与相邻样本间隔，必须正且有限；等于阈值允许，超过拒绝。
  这是工程初值，未做人体验证。长间隔的首帧拒绝，后续新帧重新开始。
- TrackingPage 定时轮询即使没有新结果也检查失效，三种动作入口共用此规则；停留用源时间。
  失效清除 hover/进度，取消待执行的返回启动器回调；棋盘 paint 只画已确认进度。
- 会话、全屏交互界面、后端配置、重新校准切换重置消费边界；只接受边界之后采集的结果。
  配置保存保留新超时字段；校准模式不允许桌面/启动器/棋盘操作，校准采样复用去重规则。
- 去重使 Classic 屏幕滤波按新观测频率更新；未调整历史窗口、滤波参数、停留阈值或冷却策略，未声称响应速度改善。

## 自动验证（2026-09-22）

用户确认 conda/gaze-env 已删除，并授权 uv。使用仓库外隔离 Python 3.11.9；未改依赖清单。
项目声明 OpenCV 4.12 与 NumPy 1.26 存在依赖约束冲突，测试环境使用 OpenCV 4.11.0.86 + NumPy 1.26.4。
未安装/升级 PyTorch、IPEX；新增核心测试不依赖它们、摄像头、权重、外网或人工操作。

可在自选隔离环境复现（Git Bash；已有环境可将 R1_PY 指向其 python.exe）：

```bash
uv venv --python 3.11 .venv
export R1_PY=.venv/Scripts/python.exe
uv pip install --python "$R1_PY" numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.16.2 scikit-learn==1.7.2 PyYAML==6.0.2 pytest==9.0.2 hypothesis==6.151.9 PyQt6==6.10.2 pyqt6-fluent-widgets==1.11.1 pandas==3.0.1
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_observation_safety.py tests/test_tracker_pipeline.py tests/test_classic_pipeline.py tests/test_classic_tracker.py tests/test_tracking_page_unit.py tests/test_calibration.py tests/test_calibration_flow_helpers.py tests/test_smoother.py tests/test_geometry.py tests/test_head_pose.py tests/test_analyze_tracker_diagnostics.py tests/test_config.py -k 'not test_error_callback and not test_process_frame_with_mock_frame' -q
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_settings_page.py -q
```

- 原有短测试：36 passed、2 deselected，无基线失败。
- 修复前新增 `test_no_face_fallback_is_not_actionable`：实际执行 **1 failed**，断言发现旧点 valid=True；未改测试接口绕过旧实现。
- 最终相关套件：**143 passed、2 deselected**；独立设置测试：**13 passed、14 第三方弃用 warnings**。
  总计 **156 passed，0 failed，0 skipped，2 主动不执行**；`uv pip check` 通过。
- 2 项不执行：`test_error_callback` 打开真实摄像头；`test_process_frame_with_mock_frame` 导入真实检测器和 torch/模型接口。
  未运行全库训练/模型/数据测试、实时诊断、摄像头、OS 操作或长 GUI。测试用假时钟/相机/ONNX，替换鼠标和应用执行函数。
- 过程中扩大测试曾因缺 pandas 收集失败，安装声明版本后解决；新增 Qt 夹具导致跨模块 QConfig 被销毁的 6 项失败，改会话级 QApplication 后解决。
- 安装 MediaPipe 后混合测试进程出现原生 access violation 输出（退出码仍为 0）；独立进程先导入 MediaPipe/ONNX/UI 成功，设置测试独立运行通过。
  保持独立测试命令，未以此修改依赖/打包/CI；真实摄像头仍待验。

## 最小人工验证：待执行

本机额外安装了 MediaPipe 0.10.21、opencv-contrib-python 4.11.0.86、ONNX Runtime 1.24.1、matplotlib 3.10.6、pillow 11.0.0。
其他机器需在上述隔离环境安装这些版本。启动入口与原运行方式相同：

```bash
"$R1_PY" main.py --config configs/classic.yaml
# Classic 验证后，在本机已有 ONNX 权重且配置匹配时再测 Deep：
"$R1_PY" main.py --config configs/deep.yaml
```

1. 在空白测试桌面/本地无敏感操作窗口中启动、校准、验证；仍需键鼠完成的步骤如实记录，不称纯眼控。
2. 桌面点击接近 1 秒、启动器接近 3 秒、棋盘接近 0.9 秒时遮挡摄像头/移出脸部：进度清零，无点击、启动、落子。
3. 拔掉摄像头或令来源停止更新；超过 250 ms 后旧点不得继续完成选择（清除可能再等待一个 33 ms UI tick）。
4. 恢复后第一帧不得触发；完整的新连续停留仍可点击、打开本地记事本、落子。用 Esc/鼠标停止后，切换界面与重启重复检查。
   内置摄像头不便断开时先只做遮挡；不要在含发送/删除按钮的真实页面试错。

## 限制与下一轮候选（不自动执行）

- 只保证已知无效/过期观测不继续触发；数值有效、新鲜不代表意图正确，也不能保证识别闭眼、反光或错误视线。
- 摄像头驱动缓冲可能使主机 read 返回时间晚于实际曝光；250 ms 可能中断低帧率设备上的选择，需人工评估。
- Qt 主线程阻塞时不能准时重绘失效状态；恢复执行后先做 freshness 检查，绘制不自行增长进度。
- 卡住的采集线程退出前不允许重启；这是保守保护，需要实机确认设备行为。
- 前置 clamp、旧滤波历史行为、校准精度、配置切换的完整运行时重建及全量依赖兼容性留待独立任务。
- 候选：实机失效/恢复与采样间隔观测、校准可靠性、头眼解耦实验；没有启动承诺。
- 回退：审阅后如需回退，revert 本 PR 提交；原有模型/数据/校准文件未修改。回退也会恢复旧的交互风险。
