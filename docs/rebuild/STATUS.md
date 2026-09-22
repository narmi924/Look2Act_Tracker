# 重启进度：R2（待外部审阅、人工摄像头验证）

当前目标：普通 Windows + RGB 摄像头上的眼控优先交互；保留键鼠急停。
Classic / Deep / deep_pog 都是现有基线，不预先确定永久产品路线。
一次一个任务，PR 后停止，不自动合并。本轮不做精度、模型、训练、论文或新交互框架。

## R2：统一原始输入与屏幕处理顺序

R1 PR #22 已合并，人工摄像头验证仍未执行。R2 从 fetch 后的
`origin/main=b01b8f1f6c860c3588f4c27dad703f13d930a1e2` 建立
`codex/rebuild-r2-screen-mapping`；开始时工作区干净。沿用 R1 隔离 uv 环境，未安装或升级依赖。

确认的问题与修改：

- `pipeline.py::_process_frame` Deep 的 world-to-screen 曾在追踪时 clamp，校准时不 clamp；
  `_process_deep_pog_output` 同样提前 clip。两者现均输出有限、未截断 raw，后台不再执行输出 EMA。
- 原 Deep/PoG 链：原始映射 → clamp → 后台 EMA → UI 校准 → display clamp。
  原 Classic 链：归一化特征 → UI 校准 → clamp → Kalman/历史稳定器。
  新链统一为：R1 门控 → **raw → 校准一次 → 屏幕平滑 → 显示边界** → R1 生产端分派前复核。
- `screen_mapping.py::ScreenMapper` 是无 Qt 的共用后处理；`tracking_page.py` 四个入口
  （桌面、启动器、棋盘、全屏验证）及合成对照调用同一实现。旧 UI 后处理分支已移除。
- `raw_point`：Classic 为相机归一化特征；Deep 为原几何映射的屏幕像素；deep_pog 保持模型输出乘 W/H。
  `calibrated_point` 和 `smoothed_point` 为未截断屏幕像素，`display_point` 仅供显示。
  `gaze_point` 保留为 raw 兼容别名，不能当最终屏幕点；采样及应用均不从该别名补缺失 raw。
  后处理副本保留源 Observation，不改 session/sequence/timestamp/continuity。
- 范围为当前单屏闭区间 `[0,W-1] × [0,H-1]`，不改 W/H 归一化或几何定义。
  raw 越界可被校准映回屏内；校准或平滑越界/非有限/异常则拒绝操作、清空选择和滤波。
  校准已越界时不再喂给滤波器，smoothed 字段及其范围状态为 None；可计算截断显示值，但 UI 不分派它。
- Classic 仍用原 Kalman + 60 历史；Deep/PoG 仍用原 EMA alpha（原配置写 kalman 时实际也是 EMA）；
  none 完全绕过输出平滑。Deep EMA 从生产端逐帧改为消费端逐个新观测，latest-result 跳帧会改变更新次数。
  未调参数补偿，也没有宣称响应速度改善。失效、过期、上下文/校准变化与恢复边界均清理历史。
- `calibration_page.py` 明确只采样 raw；`demo_tracker.py` 标注原始单位，`diagnose_tracker.py`
  使用共用后处理并分别记录四阶段值。诊断脚本按自己的采样间隔更新，不能据此推断 UI 平滑响应。

### R2 自动测试与合成对照（本机）

下列 `R1_PY` 指向已授权隔离环境的 Python（环境版本见后方 R1 记录），命令为 Git Bash 写法：

```bash
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_screen_mapping.py tests/test_observation_safety.py tests/test_tracker_pipeline.py tests/test_classic_pipeline.py tests/test_classic_tracker.py tests/test_tracking_page_unit.py tests/test_calibration.py tests/test_calibration_flow_helpers.py tests/test_smoother.py tests/test_geometry.py tests/test_head_pose.py tests/test_analyze_tracker_diagnostics.py tests/test_config.py -k 'not test_error_callback and not test_process_frame_with_mock_frame' -q
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_settings_page.py -q
"$R1_PY" scripts/compare_screen_mapping.py
```

- 修改前已有短套件：180 passed、2 deselected。先仅新增
  `test_backend_raw_is_identical_in_calibration_and_tracking`，实际运行 **2 failed**：
  Deep/PoG 同一输入在校准为 `(-200,100)`，追踪却为 `(0,100)`；修复后两项通过。
- 最终短套件：**233 passed、2 deselected**；独立设置测试 **13 passed、14 第三方 warnings**。
  合计 **246 passed、0 failed、0 skipped、2 主动不执行**。
  排除真实摄像头 `test_error_callback` 和真实检测器/模型 `test_process_frame_with_mock_frame`。
  全库训练/模型/数据测试、真实摄像头/系统动作均未执行。
- 新文件 `test_screen_mapping.py`：同源 raw、负值/超边界、校准一次、非线性顺序、原滤波参数、
  NaN/Inf/异常、显示边界、三类实际入口越界中断与恢复、EMA 去重/清理、验证窗口、合成旧文件。
  原 R1 的处理中生产端失效复核、停留恢复和迟到线程资源清理回归全部包含在通过套件内。
- 远端 CI 与本机测试分开记录；本轮未建设 CI。原生 access violation 的历史记录与未解决状态保留。

对照脚本用固定少量数据、无随机数，屏幕 1000×800，EMA alpha=0.3，基线为上方 SHA。
R1 对照逐步复制该基线 Deep 的真实旧顺序，仅在脚本内；R2 调用实际 ScreenMapper 并断言独立参考：

| 案例（坐标单位 px） | R1 关键阶段/参考偏差 | R2 关键阶段/参考偏差 |
|---|---|---|
| A：x=-200,-50；y=100；C(x)=x+300；none | raw 被截成 0,0 → 输出 300,300；误差 200,50 | raw 保留 → 校准/输出 100,250；误差 0,0 |
| B：x=2,4,6；y=100；C(x)=x²；EMA | 先平滑 2,2.6,3.62 → 校准 4,6.76,13.1044；误差 0,0.84,3.0156 | 校准 4,16,36 → 平滑 4,7.6,16.12；最大浮点误差约 3.6e-15 |
| C：几何交点(100,100,720) mm；屏幕500×400 mm；none | 原/新几何映射均(200,200)，平移校准(+10,+20)后(210,220)；误差0 | 同值；另核对 deep_pog (.2,.25)×(W,H)仍为(200,200) |

这仅证明软件链路与数学参考一致，不证明摄像头精度、人体成功率、模型角度误差或头眼解耦改善。

### 兼容性、人工验证与限制

- 用合成 version 1 校准文件覆盖三后端 × affine/polynomial 的保存、加载、应用；原校准采样空间未改变。
  相同后端、原始输入定义和几何配置下可沿用，未要求全部重校准。旧文件未充分记录后端/源配置，
  不能保证任意历史文件兼容；若来源不匹配，只为当前后端重新校准。未读写用户校准内容。
- 人工摄像头验证 **待执行**：沿用下方安全桌面步骤；在各后端匹配的校准/配置下查看四阶段诊断，
  检查已知越界时进度清零且不操作、恢复后完整停留可触发，再测遮挡/断流/停止重启。
  真实输出落在屏内不代表用户意图正确；Deep 几何仍是现有编码，未修正为真实眼球方向。
- 单屏/DPI 约定保持原状；边界更保守可能增加中断。原生 access violation 根因仍未解决。
- 回退本 R2 提交可恢复 R1 链路（会恢复提前截断/先平滑问题），不会撤回 R1 安全门控和生命周期补修。
  下一轮仅候选：人工验证与原生崩溃定位；本轮不启动。

## R1 历史记录：基线与已确认问题

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
- 首轮提交的相关套件：**143 passed、2 deselected**；独立设置测试：**13 passed、14 第三方弃用 warnings**。
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
- R1 当时将前置 clamp/旧滤波链留待后续，现由上方 R2 处理；校准精度、配置切换的完整运行时重建及全量依赖兼容性仍未处理。
- 候选：实机失效/恢复与采样间隔观测、校准可靠性、头眼解耦实验；没有启动承诺。
- 回退：审阅后如需回退，revert 本 PR 提交；原有模型/数据/校准文件未修改。回退也会恢复旧的交互风险。

## PR #22 审阅补修（基于 8b3b254fa700dd6844ce0d84d8b9c9904b5aa44a）

当时仅修复两项，保留首轮 R1；开始时本地/远端 HEAD 一致、工作区干净。补修提交时 PR 为 Open、非 Draft；现 PR #22 已合并。

- **分派前生产端复核**：TrackingPage 在校准/平滑完成后、所有视线入口分派之前调用 `TrackerPipeline.get_dispatch_rejection`。
  在生产端同一把 `_lock` 内检查运行/线程状态、校准状态、会话、连续性、源观测年龄和当前结果可用性；相关状态写入也使用此锁。
  更新的正常有效帧不要求序号相等；即使失效已被恢复结果覆盖，连续性改变仍拒绝旧观测。拒绝时不分派动作，清空选择，恢复后重新停留。
  这是**执行前的检查时点**，锁不跨越 UI/系统动作，不能预知检查返回后才发生的故障。
- **停止超时后补清理**：活线程仍禁止重启且不释放它的资源；之后 start 确认旧线程已退出，复用 `_cleanup_stopped_resources`，先 release/close，再 initialize。
  正常 stop 复用同一幂等清理路径，新会话/旧结果清除规则保留。

新增回归：

- `test_dispatch_rechecks_producer_after_processing`：桌面/启动器/棋盘 × 校准/平滑期间发布失效、失效再恢复、仅更新正常有效帧、停止、换会话、进入校准；时间仅推进 62.5 ms。
  失效用例还验证恢复首帧不触发、完整新停留能触发；复用实际 TrackingPage 分派、真实生产端锁/失败连续性与合成输入，替换所有系统操作。
- `test_late_worker_exit_cleans_resources_before_restart`：假线程模拟停止超时→活线程重启拒绝→迟到退出→释放旧资源→初始化新会话，验证 release/close 各一次且先于 initialize。

```bash
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_observation_safety.py -k 'dispatch_rechecks_producer_after_processing or late_worker_exit_cleans_resources_before_restart' -q --tb=short
```

实际修复前 **31 failed、6 passed、92 deselected**（仅新增测试和夹具，生产代码尚未修改）；修复后 **37 passed、92 deselected**。
沿用上方完整相关套件及独立设置测试命令，当前本机结果 **180 + 13 = 193 passed，0 failed，0 skipped，2 deselected**；设置测试仍有 14 条第三方弃用 warnings。
2 项未执行及原因与首轮相同；未访问摄像头、执行真实鼠标/点击/应用启动，未升级依赖。

**远端 CI**：检查 PR 未报告 checks，不能将本机测试称为 CI 通过。
**人工摄像头验证**：上方项目全部仍待执行；另需验证断流后停止超时、线程迟到退出与再次启动的设备释放/重新打开行为。
**原生 access violation**：首轮记录保留，根因尚未解决；本轮独立测试通过不代表该问题已消失。
