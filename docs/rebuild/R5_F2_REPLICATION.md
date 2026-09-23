# R5：独立会话复现与实际头动质检

基线：已合并 R4/PR #25 的 `origin/main`，`cea8e135d8e5d5271f69ab43788f48bf3590faf5`。本轮没有改在线检测、校准、平滑、交互、特征定义或回归超参数。所有源数值、身份、样本 ID、模型系数和预测都留在被 Git 忽略的 `experiment_sessions/r5_runs/`。

## 固定方法与资格

复用 R4 的 F0/F1/F2/F3/C0/H0/F2+H/F3+H、A 第一轮训练、A 第二轮及 B 三种动作留出、measurement 统计、带截距的归一化段等权岭回归（`λ=0.01`、总权重 1）、训练集加权标准化、无平滑/裁剪/按误差剔除。保留 R4 原配对，另对 F0–F2 在共同训练和测试 ID 上重拟合比较；这不新增预测方法。每个本地运行目录记录源 session/events SHA-256、冻结样本清单及其 SHA、配对共同 ID、代码 SHA 和运行 commit。

主复现来源必须是 schema 1、`complete=true`、非合成、Classic、AB 标准 18 个 A + 12 个 B 目标、30/30 实际绘制、无写丢失/记录缺口；R4 源的暗色质心与眼角快照覆盖至少 90%；A_train、A_holdout、B_natural、B_yaw、B_pitch 各至少 **80 个**有效 measurement，且每个 B 展示有 measurement。固定阈值 80 是保守工程下限，并非人体实验验证。R4 已用源从本地旧运行清单的 SHA 确认；找不到旧来源标识则拒绝主选择。相同 events 哈希或 experiment ID 的目录视为复制，全部排除，不按预测误差挑选。

每个 B 目标与连续片段，以前 5 个有效稳态 measurement 的旋转矩阵作 SO(3) 基准；计算 `R_i R_baseᵀ` 的 Rodrigues 旋转向量。向量范数为总旋转角，绝对 x/y/z 分量分别为 camera-axis pitch/yaw/roll-like 变化，**不是**独立标定的解剖学角度。每片段报告总角中位/P95 和各轴绝对分量 P95；分组值为片段 P95 的中位。少于 5 个有效姿态的片段明确为 unavailable。主会话需 B_yaw 的总角与 yaw-like、B_pitch 的总角与 pitch-like，均比 B_natural 对应指标至少高 **3°**；不足为 `motion_quality_insufficient`，姿态不可用为 `motion_quality_unknown`。两者仍可只读分析已有眼部特征，但不进入主复现汇总。PnP 使用原录制六点、通用人脸模型与记录内参，沿用 R4 的 20 px 重投影有效性限制；头动质检不加入本轮预测特征。

预定决策以**会话**为单位。对 F2–F1、F2–F0、F3–F2 分别看共同支持集差值：严格多数主会话在 A_holdout 和至少两个 B split 改善，且这些 split 的会话级平均差值同向，才标 `support`；有部分但不足标 `mixed`，全无则 `not_supported`。至少要有两份新独立主会话才能作判断。只有 F2 相对 F1、F0 都得到 `support`，才建议进入下一轮在线候选验证；本轮不接入在线系统。

## 本机会话审计与只读试运行

用户随后手动完成一份新的真实 Classic AB 录制。本机现有 **12** 个会话目录：**6** 个真实、**6** 个合成。真实来源中两个短 A；四个 AB 中两个不完整、一个已用于 R4、新的一份满足本轮资格。新增独立主复现会话 **1/2，仍缺 1 份**。`all_candidates.json`、`eligible_sessions.json`、`excluded_sessions.json` 及原因留在 Git 忽略的本地审计目录；新会话的只读分析在 `experiment_sessions/r5_runs/independent-one-verified/`，批量缺口结果在 `batch-after-one/`。新旧两次试运行的源 session/events/summary/calibration 前后 SHA-256 均不变。

新独立会话（匿名 S01）实际绘制 30/30、源 producer 2155 条；有效 measurement 为 A_train **393**、A_holdout **393**、B_natural **261**、B_yaw **309**、B_pitch **312**。下表是这**一份新会话**的各方法自身支持集 mean，单位 px；尚不是多会话复现结论。

| 方法 | A_holdout | B_natural | B_yaw | B_pitch |
|---|---:|---:|---:|---:|
| C0 | 339.3 | 258.8 | 255.8 | 255.8 |
| F0 | 355.1 | 690.9 | 1144.7 | 1419.2 |
| F1 | 347.9 | 751.8 | 1306.2 | 1516.9 |
| F2 | 231.3 | 275.6 | 326.3 | 254.5 |
| F3 | 192.1 | 283.0 | 392.4 | 311.6 |
| H0 | 360.9 | 742.3 | 1319.4 | 1577.5 |
| F2+H | 235.5 | 429.6 | 553.3 | 532.9 |
| F3+H | 244.4 | 662.4 | 1020.4 | 1127.2 |

共同训练与测试支持上的配对 mean 差（右项减左项；负数为右项改善）：

| 配对 | A_holdout | B_natural | B_yaw | B_pitch |
|---|---:|---:|---:|---:|
| F2 − F1 | −116.6 | −476.3 | −979.9 | −1262.4 |
| F2 − F0 | −123.8 | −415.3 | −818.4 | −1164.7 |
| F3 − F2 | −39.2 | +7.4 | +66.1 | +57.2 |

新会话的实际头动质检 `pass`，各片段 P95 的分组中位为：总角 natural/yaw/pitch **6.12/48.70/12.30°**；yaw-like natural/yaw **5.69/32.44°**；pitch-like natural/pitch **1.38/5.61°**。所有已测 B 连续片段都有足够有效姿态。它们是近似 PnP 的相对角，不是外部头动真值。

下表是**旧 R4 来源的复算核对**，单位 px、各方法自身支持集 mean；不是 R5 独立复现结果，也不能并入多会话主表。数值与 R4 修正权重的聚合表一致。

| 方法 | A_holdout | B_natural | B_yaw | B_pitch |
|---|---:|---:|---:|---:|
| C0 | 338.4 | 254.4 | 256.2 | 255.8 |
| F0 | 354.4 | 550.1 | 832.1 | 1055.1 |
| F1 | 302.0 | 528.7 | 780.2 | 955.0 |
| F2 | 202.8 | 239.9 | 360.1 | 394.4 |
| F3 | 179.3 | 362.3 | 648.2 | 719.4 |
| H0 | 9035.0 | 9229.7 | 21747.9 | 16874.6 |
| F2+H | 2741.1 | 2879.1 | 6842.3 | 5815.9 |
| F3+H | 3056.2 | 3224.0 | 7672.2 | 6161.9 |

该旧来源各分组 measurement 为 A_train **391**、A_holdout **388**、B_natural **261**、B_yaw **313**、B_pitch **309**。实际头动质检的分组片段 P95 中位如下；这些是通用 PnP 的相对角，不是外部测量真值：

| 来源（仅旧 R4） | natural | yaw | pitch |
|---|---:|---:|---:|
| 总相对旋转 ° | 19.63 | 44.69 | 22.44 |
| yaw-like ° | 10.26 | 43.05 | — |
| pitch-like ° | 5.33 | — | 16.83 |

B_yaw 达到预定增量；B_pitch 的总角门槛为 `19.63+3=22.63°`，实测 **22.44°**，差约 0.19°，因此该旧来源的质检状态为 `motion_quality_insufficient`。未因这一结果调整门槛。R4 旧欧拉 H0 跨支问题仍存在；相对旋转质检不修复 H0。

## 多会话汇总与结论

主复现清单现有 S01 一份。它的 F2 在 A_holdout 与三个 B split 均优于 F1、F0；F3 只在 A_holdout 优于 F2，B 三组较差；本次 B 头动质检通过。但至少两份新独立主会话的预定条件尚未满足，不能计算有意义的会话级一致性、胜负分布或最终 go/no-go。F2 相对 F1、F2 相对 F0、F3 相对 F2 的 R5 判定仍为 `unavailable_insufficient_independent_sessions`，**不足以支持直接把 F2 接为在线实验候选**。主要阻碍是还缺一份独立合格 AB 会话；没有把单会话表现当作跨会话或跨用户结论。在线默认行为未变。

## 可执行命令与补采

以下在仓库根目录的 Git Bash 运行。`audit` 列出无私人路径的 `C01` 等候选编号和排除原因，完整本地清单仅在忽略目录。当前 `run --candidate C01` 是新独立会话的显式只读分析；新增会话后重新 `audit` 并使用新列出的编号，或用 `--session "experiment_sessions/<实际会话目录>"`。每次运行会创建新目录；重复指定已有输出会拒绝覆盖。`run-all` 在主会话不足两份时仍写审计与缺口文件，并按设计以退出码 2 报告不足，绝不产生伪造结论。

```bash
R1_PY="E:/TMP/look2act-r1-env/Scripts/python.exe"
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py audit
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py selftest
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py run --candidate C01
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py run-all
PYTHONUTF8=1 "$R1_PY" scripts/r5_replication.py instructions
```

仍需手动新采 **1 份**，再次运行下列命令，点击开始后完成 30 个目标。A 两轮正常注视；B 自然保持、舒适范围内左右缓慢转头、抬头/低头都实际执行。重新启动录制，让程序生成新的 `experiment_sessions/` 目录；不要复制或覆盖旧会话。重新校准不是本轮资格硬条件；如需使用已有校准，请按 R3 现有 `--calibration <文件> --calibration-backend classic` 显式加载并保留原文件。采集时保持可控光照与坐姿，在安全桌面操作，结束后原目录留在 `experiment_sessions/` 即会自动发现。

```bash
"$R1_PY" scripts/experiment.py collect --config configs/classic.yaml --protocol AB
```

## 验证与限制

本机已授权隔离 Python：

```bash
QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_r5_replication.py tests/test_r4_feature_baselines.py tests/test_experiment_replay.py tests/test_experiment_protocol_regressions.py -q --tb=short
R3_SYNTH="experiment_sessions/r5_runs/r3-selftest-$(date +%s)"
"$R1_PY" scripts/experiment.py selftest --output "$R3_SYNTH"
"$R1_PY" scripts/experiment.py replay "$R3_SYNTH"
```

定向短套件 **83 passed、0 failed、0 skipped**。R5 合成自检通过；R3 合成录制/回放 **20/20 一致，0 mismatch、0 integrity issue、0 write_lost**。用户已手动完成一份真实 AB 采集；本机只读审计、单会话分析与批量运行确认该会话合格且源哈希未变。`run-all` 因独立会话 1/2 按设计退出码 2；另一份真实补采、真实系统操作、旧数据集重处理、训练与长 GUI 主动未执行。R1/R2 全流程人工验收和 R3 未绘制前跳过/长暂停/异常退出亦未完成。历史原生 access violation 根因未解决。本机质检基于保存的近似 PnP，相机 read 主机时间不是曝光时间；目标指令不是独立测得的眼球真值。即使未来两份会话通过，也不能自动声称跨用户泛化或在线眼控改善。
