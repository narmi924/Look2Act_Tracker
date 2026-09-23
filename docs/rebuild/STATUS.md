# 重启进度：R5（待外部审阅；R4 已合并）

当前目标：普通 Windows + RGB 摄像头上的眼控优先交互；保留键鼠急停。
Classic / Deep / deep_pog 都是现有基线，不预先确定永久产品路线。
一次一个任务，PR 后停止，不自动合并。R5 只做独立会话复现工具和实际头动质检；不改在线行为、模型或交互。

## R5：独立 AB 会话上的 F2 复现

- PR #25/R4 已合并；从 `origin/main` 的 `cea8e135d8e5d5271f69ab43788f48bf3590faf5` 建立 `codex/rebuild-r5-f2-replication`，开始时工作区干净。固定方法、会话审计、质检定义和本地运行结果见 [R5_F2_REPLICATION.md](R5_F2_REPLICATION.md)。
- 已完成只读会话审计、R4 拟合复用、额外 F0–F2 共同支持配对、SO(3) 相对旋转质检、会话级聚合和预定判定。输出只进入 Git 忽略的 `experiment_sessions/r5_runs/`。每分组至少 80 个有效 measurement、R4 源快照覆盖至少 90%、B 每目标有 measurement；头动以 5 帧稳态基线、相对旋转向量和相对 natural 至少 3° 增量判断。这些是工程门槛，不是人体实验标定。
- 用户新增两份完整真实 Classic AB，会话目录现为 13 个（7 真实、6 合成）；五个真实 AB 中两个不完整、一个已用于 R4、两份满足预定资格。独立主复现会话 **2/2**，两份实际 B yaw/pitch 头动质检均通过，源 session/events/summary/calibration 前后 SHA-256 不变。会话级共同支持比较：F2 相对 F1、F0 在两份的 A 留出和三个 B 组均改善，预定判定均为 `support`；F3 相对 F2 为 `not_supported`。**支持下一轮仅将 F2 作为在线实验候选/shadow 验证**，尚不切换默认 backend 或声称在线性能已改善。完整脱敏数值见 R5 报告。
- 本机 R5/R4/R3 定向短套件 **83 passed、0 failed、0 skipped**，R3 合成回放 **20/20 一致**；两会话只读 `run-all` 成功、缺口 0，冻结清单和共同支持 ID 保留本地忽略目录。历史原生 access violation 根因仍未解决；R1/R2 完整人工交互与 R3 未绘制前跳过、长暂停、异常退出等实机流程仍待验证。

## R4：旧数据审计与离线特征对照

- PR #24/R3 已合并；从最新 `origin/main` 的 `ab967ffd61d48734a9917e4428296ce52ae5fe10` 建立 `codex/rebuild-r4-feature-baselines`，开始时工作区干净。详见 [R4_FEATURE_BASELINES.md](R4_FEATURE_BASELINES.md)。
- 旧 raw 7463 行、processed 7395 行，7395 个 processed 来源均能对应同一 raw 帧；31 个会话/用户标签、5 个设备标签，不推定自然人数。raw 缺的 68 图像全在 invalid 行；原有 split 的来源/会话/用户标签无交叉。固定抽样只读解码 raw 31 张、processed 32 对，未发现翻转代右眼的抽样证据；旧源无眼角、虹膜、ROI 原点，不能强造 R4 特征。
- 选择唯一完整真实 Classic AB 会话（30 次绘制、2154 条源观测），按 A 第一轮 9 段训练、A 第二轮 9 段和 B 12 段留出。F0 与录制 raw_point 完全一致；F0/F1/F2/F3 输入可用率均 2154/2154，离线 PnP 2148/2154。PR #25 审阅补修将每段等权**再归一化为总权重 1**，`λ=0.01` 不变；旧实现总权重 9、等价归一化 `λ=0.01/9`，其结果只作历史记录。固定样本及配对共同 ID 清单保留在 Git 忽略的运行目录；没有在线改动或后验调参。
- 修正实现单会话 A 留出平均偏差 F0/F1/F2/F3 为 354.4/302.0/202.8/179.3 px；B yaw 为 832.1/780.2/360.1/648.2 px。旧实现相应为 359.3/279.7/203.5/181.1 和 845.5/855.3/360.1/654.5 px，不与修正主表混用。匹配 B 位置的 A 留出配对原误报 0，现 131 个共同测试 ID；共同训练 391 个 ID。修正结果中 F2 在 A、B 均优于 F1；F3 在 A 全部目标略好，但匹配目标子集和 B 变差。原始欧拉角穿越 ±180° 导致 H0 与拼接 H 严重外推，不能据此断言头姿无用。结果是探索性单会话证据，不是跨用户、真实角精度或在线性能证明。
- PR #25 审阅补修还修复 `image_limit=1` 的零除，并覆盖空/0/1/默认 32/非法限制；实际旧数据抽样上限未增加。新增失败回归修复前定向实测 **5 failed、10 deselected**，修复后 R4 **15 passed**；R1/R2/R3 短套件 **293 passed、2 主动 deselected**（相机初始化/模型加原生检测器测试未运行）；独立设置页 **13 passed、14 第三方 warnings**，合计 **321 passed、0 failed、0 skipped、2 主动未执行**；R3 合成录制回放 **20/20 一致**。原会话 session/events 和旧 R4 聚合文件前后 SHA-256 相同；修正运行另写 Git 忽略的 `experiment_sessions/r4_runs`，记录 2154 条冻结样本、8 组模型与配对 ID。远端 CI 结果以 PR 实际状态为准，R4 无新增人工摄像头验证。R1/R2 人工交互、R3 未绘制前跳过/长暂停/异常退出仍待验；原生 access violation 根因未解决。
- 下一轮仅候选：在新会话预先固定 F2 对照与头动质量检查，验证单会话结果是否复现；尚未执行或承诺。

## R3：最小实验采集与确定性数值回放

- 已确认 R2 PR #23 合并；fetch 后基线为 `d3d42c4cfb545ea4cb244315dc7d39951b35f308`，
  与已审阅 R2 内容一致。工作区原先干净，新分支 `codex/rebuild-r3-experiment-replay`。
  R1/R2 人工摄像头验证仍未执行，未将其设为自动开发前置条件。
- 新增独立 `scripts/experiment.py collect` 窗口；用户点击开始才启用一个后端/相机并保存。
  A：3×3、2 s/目标、两轮 seed=924；B：三个目标，各 2/4/4/2 s 自然保持/左右/抬低头/保持。
  自动按时间推进，可暂停/继续/跳过/结束，不显示预测反馈、不执行系统动作。
- `pipeline.py` 实验开关内提取同帧已有眼角/眼睑、虹膜可用状态、ROI、Classic 暗色质心、PnP、
  在线头姿和模型实际姿态输入；每条生产发布（含失败）在锁外入有界写队列，不拼接另一个 latest frame。
  Classic 不新增在线姿态估计；原始数值单位/左右眼约定不变，不保存任何图像。
- `src/experiment/` 提供版本 1 JSONL 记录、共用 Consumer、协议状态机、回放和基础摘要。
  配置/计划/模型标识、校准参数及 SHA-256、相机/屏幕尺寸与近似 PnP 参数冻结到 session.json。
  输出在 Git 忽略的 experiment_sessions；数值和校准参数也视为敏感，不上传、不提交。
- 所有事件在 perf_counter 秒域；分别记录源 read 返回、发布、消费/处理、计划切换、paint 提交。
  源时间关联实际目标历史；未画出的目标记 skipped，过渡 ±50 ms 记时间不确定，前 500 ms 记 settling。
  两阈值都是工程初值。read/paint 时间不是曝光/像素亮起时间，没有硬件同步保证。
- Consumer 保存空/重复/无效事件、四阶段和范围、分派检查时刻及锁内状态。
  回放使用精确记录的门控/重置时刻、虚拟时钟、实际 ObservationGate/ScreenMapper/R1 复核函数；
  不复制期望输出，不绕过复核，不执行动作。离散状态精确，数值 atol=1e-8/rtol=1e-10，耗时不作确定性断言。
- 队列默认 512 条，满队列计丢失首末 event ID/数量；磁盘异常、尾行损坏、未正常结束保留 incomplete。
  回放报告缺口并清理状态，不跨缺口积累。每行 flush 支持完整前缀，不承诺断电持久性。
- R2 诊断补齐：实际校准/平滑调用处测时，不覆盖源 timings；缺测 null/未执行状态，UI 显示“—”。
  diagnose 使用同一 Consumer，约 33 ms 消费轮询独立于打印间隔；frames 改为轮询次数，CSV 每轮一行。
  分析器缺单位/门控写 unknown，Classic raw 位移不标 px，不跨失效连接位移。

### R3 本机验证

2026-09-23 采集启动修复（PR #24 补充）：

- 用户合成自检通过，但两次真实采集尝试均在相机打开后加载 MediaPipe
  `_framework_bindings` 时 DLL 初始化失败，尚未完成真实录制。
- 无相机独立进程复现：MediaPipe 单独加载成功；先导入 QtWidgets 即失败，
  无需创建 QApplication。提前导入检测器模块后同一入口通过。
  `scripts/experiment.py` 仅在 collect 分支、Qt 导入前加载检测器模块；
  不创建检测器、不提前打开相机，不升级依赖。底层 DLL 冲突根因未进一步确定。
- 新增 `tests/test_experiment_startup.py::test_collect_native_dependencies_before_qt`：
  实际执行 main/Qt 窗口初始化，替换展示和禁止 VideoCapture，随后检查原延迟导入；
  修复前实测 **1 failed**（同一 DLL 错误），修复后 **1 passed**。
  `test_offline_commands_do_not_load_native_detector_or_qt` 禁止检测器/Qt 导入，
  实际 selftest + replay 均通过。缺少 MediaPipe 的环境会明确跳过原生启动测试。
- 下方主套件命令加入 `tests/test_experiment_startup.py` 后实测 **275 passed、2 deselected**；
  独立 settings 测试 **13 passed、14 warnings**；合计 **288 passed、0 failed、0 skipped**，
  2 项主动排除的测试及全库长任务仍未执行。R2 合成对照再次通过。
- 远端 PR #24 为普通 OPEN PR，检查列表为空，不宣称 CI 通过。
  修复后用户已完成一次真实 AB 采集；后续 A 协议控件验证见下。
  此修复只验证采集入口的加载顺序，历史原生 access violation **仍未解决**。

2026-09-23 真实采集反馈与本地离线核验：

- 用户在 Windows/RGB 相机上运行 `collect --config configs/classic.yaml --protocol AB`；
  日志确认相机、检测器、Classic 管道启动及正常停止。用户本次未报告 DLL 错误。
- 检查 Git 忽略的最新本地会话，仅提取完成/完整性统计，未读取或上传原始观测值：
  `complete=true`；AB 计划/绘制目标 30/30，结束原因为 `protocol_complete`。
  2154 条生产观测，1533 次消费轮询，1525 条独立消费；`write_lost=0`，
  `write_unconfirmed=0`。重新运行 `src.experiment.session.replay`：1533 条比较一致，
  0 mismatch、0 integrity issue、`strictly_reproducible=true`。
- 这证实一次真实采集文件的数值回放可重现，不代表视线精度或长时稳定性。
  当时尚未验证暂停/继续/跳过；后续人工检查见下。开始前无文件写入、
  R1/R2 交互人工验证仍待明确执行。
  历史混合进程 access violation 根因仍未解决。

2026-09-23 PR #24 审阅补修（基于 `efc582aff25addff1915f8da58161626d438c6cd`）：

- `Protocol.skip/tick` 原先递增 epoch 后用新 epoch 检查旧目标，已绘制目标会被误记
  `not_painted_before_deadline`。现在目标关闭记录 segment+epoch 和结果：正常绘制完成、
  截止前未绘制、用户跳过已绘制/未绘制、手动提前结束。真正未绘制仍保留 skipped；
  最后目标手动跳过以 `user_skip_complete` 结束。摘要 skipped 按 segment 去重，
  user_skips/unpainted/outcome_counts 分别显示动作、缺画与关闭结果。
- 恢复同一目标会重新发出 target_request；target_painted 保存对应 requested_at、原 planned_at、
  实际 at。paint_delay_s 现在是请求到绘制提交的主机等待；计划偏差另列
  paint_plan_deviation_s。旧 schema_version=1 会话没有 requested_at 时标 unavailable，
  不用计划时刻伪造零等待。没有改动用户的原始 events.jsonl 或已有真实 AB 会话。
- 诊断 CSV 从该轮源结果取 face_detected/fps；无结果/身份留空。分析器对已知布尔值
  报 true/denominator/unknown_rows，实测 False 与 FPS=0 保留为零；缺列/全缺测为
  unknown。终端与 JSON 使用同一统计；重复 poll 按行计，不声称独立相机帧检出率。
- 本机先写失败回归：已绘制后 skip 实测多出 skipped；恢复后回归因缺 requested_at
  失败；旧 CSV 缺 face 字段实测为 0.0；诊断投影缺 face_detected 字段。修复后通过。
  审阅者独立环境的 15 项（13 通过、2 失败）是另一次记录，不与下述本机套件混计。
- 本机新定向测试 `tests/test_experiment_protocol_regressions.py`（12 项）、
  `tests/test_analyze_tracker_diagnostics.py` 中缺列/混合/全 False/真实零 FPS 回归，
  及 `tests/test_experiment_replay.py::test_diagnostic_csv_preserves_observed_face_and_fps_but_empty_poll_is_missing`
  均通过；诊断命令 JSON/终端一致性也通过。主相关套件 **293 passed、2 deselected**；
  独立 settings **13 passed、14 第三方 warnings**，合计 **306 passed、0 failed、0 skipped、2 主动未执行**。
  合成 selftest/replay 两个命令退出码均为 0，
  **20/20 比较一致、0 mismatch、0 完整性问题、0 写丢失**。远端 CI 无检查结果；
  未自动运行真实相机或系统动作。R3 控件实机结果见下，
  历史原生 access violation 根因未解决。

2026-09-23 用户协作的短时 R3 控件验证：

- 在安全桌面运行 Classic A；用户确认点击“开始”前终端无相机打开日志、指示灯未亮。
  开始后相机/检测器/管道初始化、黄色目标显示正常；暂停后目标消失并显示暂停；
  继续后目标重新出现，跳过后切换目标，结束后未报告异常。
- 只检查最新本地会话的事件类型、目标轮次和完成/回放计数；未输出、提交或上传眼部数值。
  会话 complete=true，`write_lost=0`、`write_unconfirmed=0`，自动回放 **143/143 一致**，
  0 mismatch、0 issue。事件有 1 次 pause/resume、2 次已绘制后的 skip；
  两次对应 `user_skipped_after_paint`，均没有错误的 skipped 事件。
  正常手动结束为 `user_end`，4 次 target_closed 中有 1 次自然完成、2 次已绘制跳过、
  1 次已绘制后手动结束。
- 记录到的暂停间隔约 1.43 s（没有达到原提示的 3 s），恢复请求到绘制提交约 2.1 ms；
  恢复绘制后 100 ms 的目标标签是 settling。后两者是主机时间/软件状态，
  不是物理显示延迟或用户注视真值。
- 已验证一次普通启动、短暂停/继续、已绘制目标跳过及结束保存；
  未绘制前跳过、长时间暂停、异常退出、断流，以及 R1/R2 眼控交互仍待人工验证。
  单次采集未出现 DLL 错误，历史原生 access violation 根因仍未解决。

沿用已授权 uv 隔离环境，未安装/升级依赖。`R1_PY` 同后方环境约定：

```bash
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_experiment_protocol_regressions.py tests/test_experiment_startup.py tests/test_experiment_replay.py tests/test_screen_mapping.py tests/test_observation_safety.py tests/test_tracker_pipeline.py tests/test_classic_pipeline.py tests/test_classic_tracker.py tests/test_tracking_page_unit.py tests/test_calibration.py tests/test_calibration_flow_helpers.py tests/test_smoother.py tests/test_geometry.py tests/test_head_pose.py tests/test_analyze_tracker_diagnostics.py tests/test_config.py -k 'not test_error_callback and not test_process_frame_with_mock_frame' -q
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_settings_page.py -q
"$R1_PY" scripts/compare_screen_mapping.py
"$R1_PY" scripts/experiment.py selftest --output experiment_sessions/synthetic-demo
"$R1_PY" scripts/experiment.py replay experiment_sessions/synthetic-demo
```

- 基线短测试 233 passed、2 deselected。最终相关套件 **273 passed、2 deselected**；独立设置套件
  **13 passed、14 条第三方弃用 warnings**，合计 **286 passed、0 failed、0 skipped、2 主动未执行**。
  新 R3 测试 40 项，含真实数值 Classic 滤波越界/恢复、精确时钟边界、Qt offscreen 实际实验窗口闭环及诊断主入口打印间隔独立性。
  开发中测试矩阵列顺序写错、将 CPU 耗时也纳入确定性比较，各造成一次测试失败；分别按现有多项式基底和明确的耗时排除规则修正，未改算法。
- 2 项未执行仍为真实摄像头 test_error_callback、真实检测器/模型 test_process_frame_with_mock_frame。
  全库模型/训练/数据测试、Codex 自动真实相机/系统操作未执行；原生 access violation **仍未解决**。
- R1 分派期间失效/恢复和迟到线程资源清理、R2 四阶段与 A/B/C 合成对照继续通过。
  新合成录制：2 个目标、19 条生产观测、20 次消费、18 个独立消费 ID、1 次重复；
  源失败 1/19（被 latest 覆盖但记录保留），消费过期/长间隔/校准越界各 1 次；写入丢失/未确认均 0。
  文件回放 **20/20 一致，0 mismatch，0 完整性问题**；不同速度一致、篡改输出可检出。
- 远端 CI 与以上本机结果分开，不宣称 CI 通过；本轮没有建设 CI。

### 使用、限制与后续边界

格式、三个入口和参数见 [EXPERIMENTS.md](EXPERIMENTS.md)。用户手动采集命令：
`"$R1_PY" scripts/experiment.py collect --config configs/classic.yaml --protocol AB`。
真实采集已完成一次 AB 全协议且回放通过；另一次 A 协议验证了开始前相机未打开、
短暂停/继续与已绘制目标跳过。其余人工核验 **待执行**：开始前无文件写入、
未绘制前跳过、长时间暂停、异常退出与断流场景。
采集无需通过旧验证页面；无校准只记录可用阶段，Classic 不计算屏幕误差。不要在本次评估数据上拟合再报告效果。

这支持眼角/虹膜/暗色质心/ROI/PnP 的后续数值特征研究及后处理回放，不支持重跑检测器、重裁眼图或外观网络训练。
instructed_target 不是真实注视真值；不声称精度/人体成功率改善。记录开销未做实机性能测量，回放索引在内存中，定位为短协议。
相机驱动、Qt 阻塞、单屏/DPI、时间近似及历史原生崩溃风险仍在；完整数值重算不等于完整摄像头视频/调度复现。
回退本 R3 提交会移除记录入口及遥测补充，不撤回 R1/R2 处理链。下一轮仅候选真实采集质量审查，不自动启动。

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
