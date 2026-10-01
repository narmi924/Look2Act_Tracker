# R6：F2 实时个人校准与无操作权限 shadow

基线是已合并 R5/PR #26 的 `origin/main`：`4e10dc9aa47d7c1aab0f43f2a2e713b80554dc15`。本轮不改变 Classic/Deep 的默认在线交互、检测器、模型、旧校准、平滑或阈值。独立入口 `scripts/r6_shadow.py` 只使用 Classic 的一次摄像头/FaceMesh 感知，在明确开启的数值观测路径，从同帧眼角、暗色质心与 ROI 取得 R4 的四维 F2。关闭该开关时普通管道不提取 F2。历史原生 access violation 的根因仍未解决；此入口保留检测器模块先于 QtWidgets 加载的启动顺序。

## 流程和数据含义

用户点击“开始”后才打开摄像头并写入本地会话。自动显示 A 第一轮 9 个目标，每点 2 s；前 500 ms 是 settling，目标切换 ±50 ms 不作测量。生产线程通过 512 槽有界队列交付独立帧；按 `VideoCapture.read` 返回时的单调源时间和实际 `target_painted` 历史归属，重复身份、过期发布、未绘制、暂停与队列丢失均明确拒绝或记录。最后一个目标在计划时间截止，最多等 250 ms 收尾；仅纳入源时间不晚于截止的帧。九段都需实际绘制且各有至少 10 个不同 measurement 帧。队列丢失、记录失败、退化输入或缺点会使本次校准失败，不自动套用旧模型。

拟合只用第一轮校准样本：每次展示段权重相同，全部权重之和 1；`λ=0.01`，仅训练集加权标准化，截距不惩罚。小型拟合在线程中执行；停止或重新校准会使迟到结果的 token 失效。模型冻结后才启动独立验证时钟：A 第二轮 9 段及 B 12 段，仍只画指令目标，不画预测点、不用验证样本调模型。完成后进入自由观察，才在专用窗口画不平滑的候选圆点；暂停、失效、断流过期和越界都隐藏它。停止、Esc、重新校准为开发控制，仍需键鼠。

F2 是 `[left_rx,left_ry,right_rx,right_ry]` 四维眼部特征，绝不塞进旧二维 `raw_point`。冻结映射计算 `z=(F2-mean)/scale`、`predicted_norm=z@coefficients+intercept`，再**只乘一次**窗口逻辑宽高得未截断 `screen_point`；显示点只在有限且屏内时存在。旧 affine/polynomial 校准器不参与此路径。每次成功校准的映射 JSON 独立存于被忽略的 `experiment_sessions/r6_runs/<run_id>/f2_mapping-<attempt_id>-<model_id>.json`；版本、F2 顺序/角点/眼宽、形状/有限性、相机和窗口尺寸、检测配置、DPR、固定岭定义均须匹配。旧会话的 `f2_mapping.json` 保持原样，仍可显式加载。显式加载进入独立的“加载映射验证”模式；元数据匹配不能证明相机位置、用户或坐姿未变。

候选消费使用 R1 `ObservationGate` 和生产端锁内快照；F2 缺一眼时有自己的连续性标记，后续有效帧覆盖最新结果也不能让旧候选继续显示。核对发生在候选计算后、绘制提交前各一次，不能预知核对之后发生的故障。新有效帧只要连续性不变，不要求与正在消费的序号相等。窗口完全不连接桌面光标、点击、启动器或棋盘动作入口，也没有启用动作的参数；shadow 没有系统操作权限。

R6 会话类型为 `f2_shadow_r6`，只存本地数值/映射/事件，不存图片。事件记录生产源观测及快照、候选连续性、源/发布/消费/绘制提交时间、校准轮次与样本/截止、模型激活、F2/归一化与屏幕预测、边界/显示状态、队列和写入缺口。R6 回放以虚拟单调时钟按每次封存的样本重新拟合，再重算 F2、映射、门控和绘制资格并与记录比较；旧 R3 回放会拒绝 R6 类型。`shadow_summary.json` 区分 producer、UI 新消费和可见绘制数量，报告特征/映射耗时、年龄、读返回到绘制提交、缺特征/越界等带分母数量。多次校准的样本、截止和拟合耗时分别列在 `calibration_attempts`，激活来源列在 `model_activations`；兼容用的顶层样本总数标记为 `all_attempts`，多轮时顶层校准/拟合耗时为 `null`，避免把不同轮次拼接。没有测量的指标为 `null`。读返回不是传感器曝光，paint 提交不是物理屏幕发光，二者之差不能称硬件端到端延迟。

## 本机软件证据与限制

本机授权的 uv Python 3.11 环境（设置 `PYTHONUTF8=1`、Qt 离屏）运行 R6 合成测试 **33 passed**，包含实际窗口全流程、同帧开关、数学等价、采样时序、失效/断流/越界、模型保存加载、迟到拟合 token、零系统动作钩子和篡改检测，以及重复校准、结束后重启、加载后重校准完整周期。R1/R2/R3/R4/R5 相关合并短套件的最终数量见 `STATUS.md`。R6 合成 selftest 复算 5 次消费、零差异、零完整性问题；该小会话不等于真人校准。

对 R5 两份现有真实 AB 会话只读计算：分别在第一轮 A 的 **390/393** 条 F2 measurement 后才拟合并冻结，在其后的相同观测 ID 上与 R4/R5 批量数值比较 **1271/1275** 条，最大绝对屏幕差两份均为 `2.27e-13 px`，小于预定 `atol=1e-6 px, rtol=1e-10`。源 session/events 前后 SHA-256 不变。该历史计算使用全部合格 producer 帧，尚不是本轮真实 UI latest-result 消费支持集，也不证明实时精度或延迟。源目标是指令点而非独立测得的注视真值；R5 F2 在独立 A/B 上仍有约百至数百像素偏差，不能据此开放点击。

**指定的真实摄像头 R6 主流程已由用户执行**（PR 初始提交 `bcd251f095e8eef7a0a7232a448a8f3ae80d4c71`，`configs/classic.yaml`）。第一次在九点校准中途主动结束，摘要标记不完整。第二次摄像头启动、九点校准、独立 21 段验证自动推进，进入自由观察后用户手动停止；自由观察本来不自动结束。第二次本地摘要：九点各 42–44 个样本，共 390；队列和写入缺口均为 0；`complete=true`。独立运行 `scripts/r6_shadow.py replay` 核对真实会话 4167 项、0 mismatch/issue，原 session/events SHA-256 前后不变。producer 4593 项、UI 新消费 2803 项、自由观察可见绘制 1341 次；新消费中越界 391 项，该比例混合验证与自由观察阶段，不能直接当作自由观察可见率。

该次会话的 producer 为 29.98 Hz、UI 新消费为 20.76 Hz、自由观察可见绘制为 16.55 Hz；观测年龄中位数 23.4 ms、p95 38.9 ms，read 返回到绘制提交的主机时间中位数 25.5 ms、p95 41.8 ms。这些数值不是曝光到屏幕发光的延迟。用户目视确认青点大致随跨屏视线移动，但效果**非常差**，主要是位置偏差或跳动，未感觉到明显滞后；遮住眼睛或头离开画面时青点消失，恢复后重新出现（事件中也有两次无脸失效后重新出现可见候选）。暂停/继续正常，本地事件在自由观察阶段各记录一次 pause/resume。

对同一次本地会话做只读描述性距离核对：校准列使用全部 390 条 `calibration_sample` 与冻结映射；独立验证列仅使用 `gate_state=new`、有限预测点、源时间处于实际 `target_painted.at+0.55 s` 到下一目标 `requested_at−0.05 s`（最后一段取计划结束前 0.05 s）的观测，874 条，包含 47 条越界估计。距离是未截断屏幕预测到**指令目标**的欧氏像素距离，不是独立测得的用户注视真值；校准列还是训练内残差，不能与独立列混称精度。

| 目标口径 | 样本数 | 距离中位数 | p95 |
| --- | ---: | ---: | ---: |
| 校准训练 | 390 | 157.9 px | 320.1 px |
| 独立 A 第二轮 | 266 | 235.6 px | 647.3 px |
| 独立 B natural | 177 | 207.2 px | 550.6 px |
| 独立 B yaw | 216 | 193.4 px | 837.8 px |
| 独立 B pitch | 215 | 197.6 px | 642.4 px |

此结果与用户报告的差体验一致，**不支持将 F2 shadow 候选开放给系统操作**；不能从一次会话推断普遍性能或精确归因。重新校准、结束后重启、加载后重校准及其他断流情形已做合成回归，**尚未做新的人工实机重复周期验证**。自由观察没有目标真值，真实数值记录及映射只留在本地忽略目录。R1/R2 旧交互的完整实机验收、R3 某些异常流程及原生 access violation 根因仍待单独处理。

## PR #27 生命周期审阅补修

基于 `5697f05f7eaa6150f6016e78cb38e69b5abd49e3`：同一窗口第二次成功校准原先仍写固定 `f2_mapping.json`，`save_mapping` 因存在旧文件而拒绝；结束后再次开始残留旧 `seal_cutoff`，新会话样本被判在截止之外；先显式加载再重校准时，回放曾依据整个会话的 `loaded_mapping_validation` 模式错误拒绝新的个人校准来源。这些是生命周期/来源错误，不是感知精度问题。审阅者独立提取原方法执行的 10 项检查为 6 通过、4 失败；它不是完整 Qt 测试。我们在生产代码未改时增加完整周期回归，定向运行实得 **4 failed、2 passed、4 deselected**；另用原版窗口源码配合假时钟确认旧 cutoff=118 拒绝第二会话 source_time=500.6 的有效样本。

现在每轮校准分配独立 `attempt_id` 与轮次，拟合后先校验、保存并读回不可覆盖的 `f2_mapping-<attempt_id>-<model_id>.json`，然后才激活。保存/拟合失败不会恢复先前模型。首次开始、同窗重启与手动重校准共用状态重置，清空旧截止/封存/暂存/队列统计并使迟到拟合 token 失效；旧摄像头线程仍存活时继续拒绝重启。每次激活独立记录 `source_kind`（`explicit_load` 或 `personal_calibration`）、映射 ID、来源轮次与文件。回放对个人校准重拟合对应封存样本并核对保存的版本文件；对显式加载检查快照和兼容性。旧 schema 1 事件缺来源字段时，带校准 attempt 的激活按个人校准核对；加载模式且无 attempt 的首次激活按显式加载核对；无法确定的来源标为 unknown，不伪造训练证明。旧单文件映射和原实机会话均不改写。

完整 Qt 离屏周期覆盖：九点成功→自由观察→第二轮九点成功、结束后同窗重启并拟合新会话、显式加载→重校准→两次来源回放、第二轮拟合期间停止/再次重置、第二次保存失败、第二轮系数/样本/来源/保存文件篡改；合成摄像头与失败即报错的系统动作钩子确保零系统操作。修复后 R6 **33 passed**；合并相关短套件 **356 passed、2 主动 deselected**，独立设置页 **13 passed**（14 条第三方 warnings），合计 **369 passed、0 failed、0 skipped**。R6 合成回放 **5/5**、R3 合成录制回放 **20/20**，均零差异；两份 R5 历史数值检查仍为 390/393、1271/1275 条，最大绝对差均 `2.27e-13 px`。旧 R6 真实会话只读兼容回放 **4167 项、0 mismatch/issue**，输出仅写到新的本地忽略诊断目录；原 session/events/单文件映射前后 SHA-256 不变。自动验证未访问摄像头、模型训练或系统动作；重复周期实机测试仍待人工执行，且不能据此声称位置偏差/跳动或原生 access violation 已解决。

## 命令（Git Bash，仓库根目录）

```bash
R1_PY="E:/TMP/look2act-r1-env/Scripts/python.exe"

# 用户手动；窗口出现后点“开始”才访问一个摄像头
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py collect --config configs/classic.yaml

# 显式加载某次 R6 个人映射；实际文件名可在本机该会话的 model_activations.mapping_file 查到
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py validate-loaded --config configs/classic.yaml --mapping "experiment_sessions/r6_runs/<run_id>/f2_mapping-<attempt_id>-<model_id>.json"
# 旧 R6 会话若只有 f2_mapping.json，仍可用同一命令将 --mapping 指向它

# 完全合成，无摄像头/系统动作；目录必须尚不存在
SYNTH="experiment_sessions/r6_runs/synthetic-$(date +%s)"
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py selftest --output "$SYNTH"
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py replay "$SYNTH"

# 两份已存在的 R5 完整 AB 来源，只读、匿名汇总
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py historical --eligible-r5

# R6 合成/离屏自动测试（不触碰摄像头或系统输入）
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_r6_shadow.py tests/test_r6_shadow_window.py tests/test_r6_startup.py -q --tb=short
```

新实机会话结束后，使用 `scripts/r6_shadow.py replay "experiment_sessions/r6_runs/<run_id>"` 核对候选链路与完整性。记录及映射含眼部数值和个人系数，只留在被 Git 忽略的本机目录，不提交或上传。下一轮是否进入受控窗口的纯眼控交互验证，由人工结果和审阅决定；本轮不启用系统动作。
