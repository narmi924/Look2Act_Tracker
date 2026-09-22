# R3 本地数值实验与回放

这不是视频录制。只保存同帧眼部/PnP/姿态数值、校准快照及处理事件；
不保存图像、音频或桌面，不上传，不执行鼠标/点击/应用操作。
数值与校准参数仍可能敏感，真实输出固定在被 Git 忽略的 `experiment_sessions/`，不得提交。

## 三个入口

使用已授权的隔离 Python。以下为仓库根目录下的 Git Bash 命令，`R1_PY` 指向该环境解释器：

```bash
# 不访问相机：生成最小合成会话，写入文件，然后实际重算并核对
"$R1_PY" scripts/experiment.py selftest --output experiment_sessions/synthetic-demo

# 用户手动：打开独立实验窗口，点击“开始”才打开一个摄像头并保存
"$R1_PY" scripts/experiment.py collect --config configs/classic.yaml --protocol AB

# 不打开窗口/相机/模型：回放、比较并生成 summary.json
"$R1_PY" scripts/experiment.py replay experiment_sessions/synthetic-demo
```

新建会话目录必须不存在，避免覆盖历史数据；不指定 selftest 的 output 会生成唯一目录。
collect 的输出使用随机会话目录名，结束时显示名称。可选 `--protocol A` 或 `B`。
显式加载旧校准：增加 `--calibration <文件> --calibration-backend classic`（须与所选后端一致）。
旧文件缺乏后端来源信息，该参数是使用者对来源的确认，不是自动证明校准兼容。
开始时冻结参数及 SHA-256，回放绝不读取当前校准文件；本轮不拟合/在线更新。
无校准仍可采集；Classic 屏幕阶段不可计算，Deep 可保留原几何屏幕输出，但目标校准偏差指标记为 unavailable。

## 固定协议

- A：屏幕内部 3×3 目标，每目标 2 s；两轮不同顺序，seed=924。前 500 ms 为 settling，全部原始数据保留。
- B：中心和左右偏离中心三个目标；依次自然保持 2 s、缓慢左右转头 4 s、缓慢抬低头 4 s、自然保持 2 s。
  只要求舒服范围内的小幅动作，不判定角度达标。目标切换独立于预测是否命中。
- 可暂停、继续、跳过、结束；暂停期间没有有效任务标签，恢复后重新建立 settling 和滤波状态。
  默认无预测光标/轨迹/成绩反馈。开始、暂停等按钮仍需手动，不能称为纯眼控流程。
- `--protocol-config <JSON>` 可覆盖 `seed`、`dwell_s`、`settling_s`、`transition_guard_s`、`a_rounds`、
  `b_durations`（四项）。默认切换不确定带为 ±50 ms。500/50 ms 都是工程初值，未经人体验证。
  session.json 保存完整实际计划、参数、目标顺序。

## 一套记录格式（schema_version=1）

| 文件/事件 | 内容与语义 |
|---|---|
| `session.json` | 代码 commit/dirty、后端与生效配置、相机请求/实际尺寸、Qt 屏幕/窗口/缩放、计划、校准参数及哈希、模型名称/哈希/运行时版本、软件版本、PnP 通用模型/近似内参、完整性状态 |
| `producer` | 原 Observation 全部字段、published_at、source valid（字段 `result.valid`）、失败原因、raw/单位、同帧眼部/PnP/在线头姿/模型实际姿态输入、生产计算耗时 |
| `consume` | 读取 ID（可空）、read_at/processed_at/age、门控结果与 reset、四阶段数值、范围/拒绝、检查时刻与生产端一致快照、放行结果、实际处理耗时/执行状态 |
| `context` | 消费会话/重置原因/active；回放恢复相同边界 |
| `start/target_request/target_painted/skipped/pause/resume/skip/end` | 计划时刻、实际绘制提交时刻、未显示/用户跳过、暂停等历史；requested_motion 只是协议指令 |
| `summary.json` | 计数与分母、源失败/消费拒绝、采样间隔/年龄/耗时、阶段可计算/越界比例、按目标连续段的偏差/离散程度、在线姿态范围及回放差异 |

事件用有序 `event_id` 流式写入 `events.jsonl`；观察身份仍是原 session/sequence，不重分配。
monotonic_origin_s 记录统一 perf_counter 时间原点，各事件保留该时钟的秒值；日期仅说明创建时间。
read_at 是取得结果快照之后、开始门控前的主机时刻；published_at 在生产端发布锁内建立；
dispatch_state.checked_at 是同一生产端锁内复核快照的时刻。processed_at 表示处理完成，并非显示完成。
相机时间是 read 返回时间，不是曝光时间；target_painted 在 Qt 绘制提交附近，不是像素真正亮起时间。
目标按源 timestamp 关联**实际**绘制历史；延迟推理不会被贴到新目标，跳过未绘制目标不产生标签。
边界 ±guard 标 time_uncertain；没有实际目标、暂停、恢复未绘制、结束后均 unavailable。无硬件同步保证。

raw 单位：Classic 为 camera_normalized_feature；Deep/PoG 为 screen_px。后续屏幕阶段均为 screen_px，
闭区间 [0,W-1]×[0,H-1] 不改原 W/H 换算。calibrated 越界不再喂滤波器，smoothed 此时为空。
显式 `{"nonfinite":"nan"}`/`inf` 编码保留非有限失败数值；JSON 不使用非标准裸 NaN。

眼部快照只从同帧已有检测结果提取：带 MediaPipe 原索引的眼角/眼睑点、已有虹膜中心、原始 ROI 原点/尺寸，
Classic 另存暗色质心的 ROI/帧绝对位置；6 个已裁边 PnP 输入与约定也保存。
左右眼沿用原索引命名，不修正 swap/翻转。未启用/未得到虹膜结果标 unavailable，不以暗色质心替代。
模型零姿态输入单列，不能视为测得的零头动。Classic 不额外运行头姿估计，在线姿态缺失如实保留。
内参焦距近似图像宽度、人脸通用 3D 点、屏幕物理尺寸等记录来源，不是实测标定真值。

## 完整性、回放与摘要

记录器默认 512 条有界队列，生产锁外建立数值事件并非阻塞入队；写线程负责 JSON/磁盘。
队列满计数 write_lost 及首末丢失 event ID（中间是否连续丢失由 JSONL ID 缺口判别），会话 incomplete。
写失败记录错误类别和 write_unconfirmed；初始文件即 complete=false。正常结束等待落盘；写线程超时也不标完整。
每条 flush 支持读取进程异常后的完整前缀，但不承诺断电持久性。缺口/坏尾行不能严格复现，不插值补帧。
回放在缺口重置门控/滤波，缺少源结果显式报告。源采样间隔及目标离散程度不跨记录缺口；
目标统计不跨失效/暂停/恢复边界，重复 tick 不增独立样本。保留有限越界校准值的目标偏差，并列缺失数量。
少于两点不报告离散程度。instructed_target 不是独立测量的真实注视点。

回放复用 ObservationGate、ScreenMapper、R1 dispatch_rejection，使用虚拟时钟和录制的检查状态；
绝不把检查改成永远通过。离散状态精确比对，浮点 atol=1e-8、rtol=1e-10，计算耗时不要求相等。
`--speed 0` 默认立即计算；正数只影响等待，不改变虚拟时间。篡改期望输出会产生 mismatch。
strictly_reproducible 必须无记录问题且逐条一致。摘要写入本地，不覆盖原事件。

这支持保存后的屏幕校准/滤波数值回放，以及利用眼角/虹膜/暗色质心/ROI/PnP 做后续相对眼部特征实验。
PnP 参数足以用现有估计器离线生成辅助姿态，但 R3 没有自动生成或回注 Classic；后续如生成必须标 derived_offline。
缺少图像，不能重新检测/裁眼、训练外观网络，也不能复现硬件曝光、真实线程调度、新算法实际延迟或用户意图。

## 现有诊断命令的变化

`scripts/diagnose_tracker.py` 复用同一 Consumer，固定约 33 ms 轮询，不补造追赶 tick；`--interval` 只控制打印。
`--frames` 现在是消费轮询次数（含空/重复），CSV 每次轮询一行，不再按打印次数截取。
CSV 是事件字段投影，不能替代包含全部生产事件的规范会话；没有生产记录时无法发现所有被覆盖的数值。
CSV 增加身份/源时间来源/消费时间/年龄/状态/范围和执行耗时，源 valid 与消费接受分开。
分析器旧文件缺状态/单位写 unknown，Classic raw 位移不标 px；跨失效不连接位移，也不把位移叫静态抖动。

ScreenMapper 在实际校准/平滑调用处测时，独立 processing_timings 不覆盖源 timings；
未执行为 null，none 对应 disabled；UI 后处理后显示，缺测为“—”。计算耗时不等于滤波响应延迟，
各模块耗时之和不是完整端到端显示延迟。

真实摄像头采集和 R1/R2 人工验收仍待执行；原生 access violation 根因未解决。
