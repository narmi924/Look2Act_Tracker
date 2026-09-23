# R5：独立会话复现与实际头动质检

基线：已合并 R4/PR #25 的 `origin/main`，`cea8e135d8e5d5271f69ab43788f48bf3590faf5`。本轮没有改在线检测、校准、平滑、交互、特征定义或回归超参数。所有源数值、身份、样本 ID、模型系数和预测都留在被 Git 忽略的 `experiment_sessions/r5_runs/`。

## 固定方法与资格

复用 R4 的 F0/F1/F2/F3/C0/H0/F2+H/F3+H、A 第一轮训练、A 第二轮及 B 三种动作留出、measurement 统计、带截距的归一化段等权岭回归（`λ=0.01`、总权重 1）、训练集加权标准化、无平滑/裁剪/按误差剔除。保留 R4 原配对，另对 F0–F2 在共同训练和测试 ID 上重拟合比较；这不新增预测方法。每个本地运行目录记录源 session/events SHA-256、冻结样本清单及其 SHA、配对共同 ID、代码 SHA 和运行 commit。

主复现来源必须是 schema 1、`complete=true`、非合成、Classic、AB 标准 18 个 A + 12 个 B 目标、30/30 实际绘制、无写丢失/记录缺口；R4 源的暗色质心与眼角快照覆盖至少 90%；A_train、A_holdout、B_natural、B_yaw、B_pitch 各至少 **80 个**有效 measurement，且每个 B 展示有 measurement。固定阈值 80 是保守工程下限，并非人体实验验证。R4 已用源从本地旧运行清单的 SHA 确认；找不到旧来源标识则拒绝主选择。相同 events 哈希或 experiment ID 的目录视为复制，全部排除，不按预测误差挑选。

每个 B 目标与连续片段，以前 5 个有效稳态 measurement 的旋转矩阵作 SO(3) 基准；计算 `R_i R_baseᵀ` 的 Rodrigues 旋转向量。向量范数为总旋转角，绝对 x/y/z 分量分别为 camera-axis pitch/yaw/roll-like 变化，**不是**独立标定的解剖学角度。每片段报告总角中位/P95 和各轴绝对分量 P95；分组值为片段 P95 的中位。少于 5 个有效姿态的片段明确为 unavailable。主会话需 B_yaw 的总角与 yaw-like、B_pitch 的总角与 pitch-like，均比 B_natural 对应指标至少高 **3°**；不足为 `motion_quality_insufficient`，姿态不可用为 `motion_quality_unknown`。两者仍可只读分析已有眼部特征，但不进入主复现汇总。PnP 使用原录制六点、通用人脸模型与记录内参，沿用 R4 的 20 px 重投影有效性限制；头动质检不加入本轮预测特征。

预定决策以**会话**为单位。对 F2–F1、F2–F0、F3–F2 分别看共同支持集差值：严格多数主会话在 A_holdout 和至少两个 B split 改善，且这些 split 的会话级平均差值同向，才标 `support`；有部分但不足标 `mixed`，全无则 `not_supported`。至少要有两份新独立主会话才能作判断。只有 F2 相对 F1、F0 都得到 `support`，才建议进入下一轮在线候选验证；本轮不接入在线系统。

## 本机会话审计与只读试运行

用户手动完成两份新的真实 Classic AB 录制。本机现有 **13** 个会话目录：**7** 个真实、**6** 个合成。真实来源中两个短 A；五个 AB 中两个不完整、一个已用于 R4、两份满足本轮资格。新增独立主复现会话 **2/2**。两份来源有不同的 experiment ID、session/events SHA-256、录制启动，相隔约 40.6 分钟；不把它们称为不同用户或独立重新校准。`all_candidates.json`、`eligible_sessions.json`、`excluded_sessions.json` 及原因留在 Git 忽略的本地审计目录；最终只读批量结果在 `experiment_sessions/r5_runs/two-independent-verified/`。两份源 session/events/summary/calibration 前后 SHA-256 不变，R4 旧分析目录也未覆盖。

批量输出按候选目录排序匿名为 S01、S02（不是采集时间顺序）。两份均实际绘制 30/30，分别有源 producer 2154、2155 条。有效 measurement 与 F2 支持集一致：S01 的 A_train/A_holdout/B_natural/B_yaw/B_pitch 为 **390/390/259/310/312**，S02 为 **393/393/261/309/312**。下表是各方法自身支持集 mean，单位 px；训练与留出均使用冻结清单和 R4 固定拟合规则。

| 会话 | 方法 | A_holdout | B_natural | B_yaw | B_pitch |
|---|---|---:|---:|---:|---:|
| S01 | C0 | 341.3 | 256.3 | 256.2 | 255.8 |
| S01 | F0 | 235.4 | 391.5 | 806.8 | 796.2 |
| S01 | F1 | 249.4 | 419.2 | 781.8 | 798.4 |
| S01 | F2 | 174.4 | 223.8 | 306.3 | 295.1 |
| S01 | F3 | 176.1 | 346.2 | 478.2 | 441.6 |
| S01 | H0 | 390.1 | 623.0 | 1505.4 | 8154.8 |
| S01 | F2+H | 197.6 | 307.3 | 538.2 | 3123.7 |
| S01 | F3+H | 163.7 | 314.5 | 603.8 | 3705.5 |
| S02 | C0 | 339.3 | 258.8 | 255.8 | 255.8 |
| S02 | F0 | 355.1 | 690.9 | 1144.7 | 1419.2 |
| S02 | F1 | 347.9 | 751.8 | 1306.2 | 1516.9 |
| S02 | F2 | 231.3 | 275.6 | 326.3 | 254.5 |
| S02 | F3 | 192.1 | 283.0 | 392.4 | 311.6 |
| S02 | H0 | 360.9 | 742.3 | 1319.4 | 1577.5 |
| S02 | F2+H | 235.5 | 429.6 | 553.3 | 532.9 |
| S02 | F3+H | 244.4 | 662.4 | 1020.4 | 1127.2 |

共同训练与测试支持上的配对 mean 差（右项减左项；负数为右项改善）。F0–F2、F1–F2 在 S01/S02 各有共同训练 390/393 个 ID，测试同样按共同 ID 冻结；完整清单仅在本地忽略目录。

| 会话 | 配对 | A_holdout | B_natural | B_yaw | B_pitch |
|---|---|---:|---:|---:|---:|
| S01 | F2 − F1 | −75.0 | −195.4 | −475.5 | −503.3 |
| S01 | F2 − F0 | −61.0 | −167.8 | −500.6 | −501.1 |
| S01 | F3 − F2 | +1.7 | +122.4 | +171.9 | +146.4 |
| S02 | F2 − F1 | −116.6 | −476.3 | −979.9 | −1262.4 |
| S02 | F2 − F0 | −123.8 | −415.3 | −818.4 | −1164.7 |
| S02 | F3 − F2 | −39.2 | +7.4 | +66.1 | +57.2 |

两份 B 头动质检均为 `pass`，所有已测 B 连续片段都有足够有效姿态。数值为片段 P95 的分组中位，单位 °；只是近似 PnP 相对角，不是外部头动真值。

| 会话 | 总角 natural / yaw / pitch | yaw-like natural / yaw | pitch-like natural / pitch |
|---|---:|---:|---:|
| S01 | 7.32 / 50.33 / 26.30 | 5.21 / 49.97 | 2.64 / 15.01 |
| S02 | 6.12 / 48.70 / 12.30 | 5.69 / 32.44 | 1.38 / 5.61 |

旧 R4 来源曾单独只读复算，8 组误差与 [R4 修正结果](R4_FEATURE_BASELINES.md)一致，但不进入上述两会话主表。它的 B_pitch 总角 22.44°，低于预定 natural+3° 的 22.63°，质检为 `motion_quality_insufficient`；未因此改动门槛。R4 旧欧拉 H0 跨支问题仍存在，相对旋转质检不修复 H0。

## 多会话汇总与结论

下表每个会话只贡献一个方法×分组均值，**没有把帧当作独立实验对象**。两会话的中位数与均值数值相同（样本数恰为 2）；完整分布与支持数在本地 `replication_summary.json`。

| 方法 | A_holdout | B_natural | B_yaw | B_pitch |
|---|---:|---:|---:|---:|
| C0 | 340.3 | 257.5 | 256.0 | 255.8 |
| F0 | 295.3 | 541.2 | 975.8 | 1107.7 |
| F1 | 298.7 | 585.5 | 1044.0 | 1157.7 |
| F2 | 202.9 | 249.7 | 316.3 | 274.8 |
| F3 | 184.1 | 314.6 | 435.3 | 376.6 |
| H0 | 375.5 | 682.7 | 1412.4 | 4866.1 |
| F2+H | 216.6 | 368.5 | 545.8 | 1828.3 |
| F3+H | 204.1 | 488.4 | 812.1 | 2416.4 |

会话级共同支持配对差值的均值（px，负数为右项改善）：

| 配对 | A_holdout | B_natural | B_yaw | B_pitch | 预定判定 |
|---|---:|---:|---:|---:|---|
| F2 − F1 | −95.8 | −335.8 | −727.7 | −882.9 | `support`：2/2 会话 A 与三种 B 均改善 |
| F2 − F0 | −92.4 | −291.5 | −659.5 | −832.9 | `support`：2/2 会话 A 与三种 B 均改善 |
| F3 − F2 | −18.7 | +64.9 | +119.0 | +101.8 | `not_supported`：0/2 会话满足 A 加至少两种 B 改善 |

按预先固定的规则，**支持下一轮将 F2 作为在线实验候选或 shadow mode 验证**，不支持 F3 取代 F2。这里的“支持”只表示两次同类录制中的相对排序复现；不是已经改进在线操作。F2 留出均值仍为约 175–326 px，S02 的 B_natural 甚至略差于不区分目标的 C0；准确点击、长期稳定、不同用户或重新校准后表现都尚未验证。两个会话都未加载校准快照，不能由此推断在线校准后的表现。H0 原始欧拉姿态分支仍可能外推，不能把 H 组合的误差解释为头动信息无用。在线默认行为保持不变。

## 可执行复现命令

以下在仓库根目录的 Git Bash 运行。`audit` 列出无私人路径的候选编号及排除原因；当前两份新会话为 C01、C02。完整本地清单仅在忽略目录。`run` 显式分析指定来源，`run-all` 自动选择两份合格且未用于 R4 的主会话，输出两会话汇总。每次运行创建新目录，拒绝覆盖已有输出；若未来主会话不足两份仍写审计与缺口文件、以退出码 2 拒绝伪造结论。

```bash
R1_PY="E:/TMP/look2act-r1-env/Scripts/python.exe"
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py audit
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py selftest
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py run --candidate C01
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py run --candidate C02
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py run-all
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py instructions
```

## 验证与限制

本机已授权隔离 Python：

```bash
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_r5_replication.py tests/test_r4_feature_baselines.py tests/test_experiment_replay.py tests/test_experiment_protocol_regressions.py -q --tb=short
R3_SYNTH="experiment_sessions/r5_runs/r3-selftest-$(date +%s)"
"$R1_PY" scripts/experiment.py selftest --output "$R3_SYNTH"
"$R1_PY" scripts/experiment.py replay "$R3_SYNTH"
```

定向短套件再次运行 **83 passed、0 failed、0 skipped**。R5 合成自检通过；R3 合成录制/回放 **20/20 一致，0 mismatch、0 integrity issue、0 write_lost**（此前本机运行）。用户手动完成两份真实 AB 采集；本机只读 `run-all` **2/2、退出码 0、online_candidate=support**，R4 源未纳入，样本清单哈希与配对共同 ID 数量核对通过，原 session/events/summary/calibration 均未改写。Codex 未自行访问真实摄像头、执行系统操作、重处理旧数据、训练或运行长 GUI。R1/R2 全流程人工验收和 R3 未绘制前跳过/长暂停/异常退出仍待执行；历史原生 access violation 根因未解决。本机质检基于保存的近似 PnP，相机 read 主机时间不是曝光时间；目标指令不是独立测得的眼球真值。两份录制不代表跨用户泛化或在线眼控改善。
