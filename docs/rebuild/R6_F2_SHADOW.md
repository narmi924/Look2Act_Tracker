# R6：F2 实时个人校准与无操作权限 shadow

基线是已合并 R5/PR #26 的 `origin/main`：`4e10dc9aa47d7c1aab0f43f2a2e713b80554dc15`。本轮不改变 Classic/Deep 的默认在线交互、检测器、模型、旧校准、平滑或阈值。独立入口 `scripts/r6_shadow.py` 只使用 Classic 的一次摄像头/FaceMesh 感知，在明确开启的数值观测路径，从同帧眼角、暗色质心与 ROI 取得 R4 的四维 F2。关闭该开关时普通管道不提取 F2。历史原生 access violation 的根因仍未解决；此入口保留检测器模块先于 QtWidgets 加载的启动顺序。

## 流程和数据含义

用户点击“开始”后才打开摄像头并写入本地会话。自动显示 A 第一轮 9 个目标，每点 2 s；前 500 ms 是 settling，目标切换 ±50 ms 不作测量。生产线程通过 512 槽有界队列交付独立帧；按 `VideoCapture.read` 返回时的单调源时间和实际 `target_painted` 历史归属，重复身份、过期发布、未绘制、暂停与队列丢失均明确拒绝或记录。最后一个目标在计划时间截止，最多等 250 ms 收尾；仅纳入源时间不晚于截止的帧。九段都需实际绘制且各有至少 10 个不同 measurement 帧。队列丢失、记录失败、退化输入或缺点会使本次校准失败，不自动套用旧模型。

拟合只用第一轮校准样本：每次展示段权重相同，全部权重之和 1；`λ=0.01`，仅训练集加权标准化，截距不惩罚。小型拟合在线程中执行；停止或重新校准会使迟到结果的 token 失效。模型冻结后才启动独立验证时钟：A 第二轮 9 段及 B 12 段，仍只画指令目标，不画预测点、不用验证样本调模型。完成后进入自由观察，才在专用窗口画不平滑的候选圆点；暂停、失效、断流过期和越界都隐藏它。停止、Esc、重新校准为开发控制，仍需键鼠。

F2 是 `[left_rx,left_ry,right_rx,right_ry]` 四维眼部特征，绝不塞进旧二维 `raw_point`。冻结映射计算 `z=(F2-mean)/scale`、`predicted_norm=z@coefficients+intercept`，再**只乘一次**窗口逻辑宽高得未截断 `screen_point`；显示点只在有限且屏内时存在。旧 affine/polynomial 校准器不参与此路径。映射 JSON 独立存于本次被忽略的 `experiment_sessions/r6_runs/<id>/f2_mapping.json`；版本、F2 顺序/角点/眼宽、形状/有限性、相机和窗口尺寸、检测配置、DPR、固定岭定义均须匹配。显式加载进入独立的“加载映射验证”模式；元数据匹配不能证明相机位置、用户或坐姿未变。

候选消费使用 R1 `ObservationGate` 和生产端锁内快照；F2 缺一眼时有自己的连续性标记，后续有效帧覆盖最新结果也不能让旧候选继续显示。核对发生在候选计算后、绘制提交前各一次，不能预知核对之后发生的故障。新有效帧只要连续性不变，不要求与正在消费的序号相等。窗口完全不连接桌面光标、点击、启动器或棋盘动作入口，也没有启用动作的参数；shadow 没有系统操作权限。

R6 会话类型为 `f2_shadow_r6`，只存本地数值/映射/事件，不存图片。事件记录生产源观测及快照、候选连续性、源/发布/消费/绘制提交时间、校准轮次与样本/截止、模型激活、F2/归一化与屏幕预测、边界/显示状态、队列和写入缺口。R6 回放以虚拟单调时钟按封存样本重新拟合，再重算 F2、映射、门控和绘制资格并与记录比较；旧 R3 回放会拒绝 R6 类型。`shadow_summary.json` 区分 producer、UI 新消费和可见绘制数量，报告特征/映射耗时、年龄、读返回到绘制提交、缺特征/越界等带分母数量。没有测量的指标为 `null`。读返回不是传感器曝光，paint 提交不是物理屏幕发光，二者之差不能称硬件端到端延迟。

## 本机软件证据与限制

本机授权的 uv Python 3.11 环境（设置 `PYTHONUTF8=1`、Qt 离屏）运行 R6 合成测试 **26 passed**，包含实际窗口全流程、同帧开关、数学等价、采样时序、失效/断流/越界、模型保存加载、迟到拟合 token、零系统动作钩子和篡改检测。R1/R2/R3/R4/R5 相关合并短套件的最终数量见 `STATUS.md`。R6 合成 selftest 复算 5 次消费、零差异、零完整性问题；该小会话不等于真人校准。

对 R5 两份现有真实 AB 会话只读计算：分别在第一轮 A 的 **390/393** 条 F2 measurement 后才拟合并冻结，在其后的相同观测 ID 上与 R4/R5 批量数值比较 **1271/1275** 条，最大绝对屏幕差两份均为 `2.27e-13 px`，小于预定 `atol=1e-6 px, rtol=1e-10`。源 session/events 前后 SHA-256 不变。该历史计算使用全部合格 producer 帧，尚不是本轮真实 UI latest-result 消费支持集，也不证明实时精度或延迟。源目标是指令点而非独立测得的注视真值；R5 F2 在独立 A/B 上仍有约百至数百像素偏差，不能据此开放点击。

**真实摄像头 R6 部分流程已由用户执行**（PR 初始提交 `bcd251f095e8eef7a0a7232a448a8f3ae80d4c71`，`configs/classic.yaml`）。第一次在九点校准中途主动结束，摘要标记不完整。第二次摄像头启动、九点校准、独立 21 段验证自动推进，进入自由观察后用户手动停止；自由观察本来不自动结束。第二次本地摘要：九点各 42–44 个样本，共 390；队列和写入缺口均为 0；`complete=true`，回放核对 4167 项、0 mismatch/issue。全部 UI 新消费 2803 项，其中越界 391 项；该比例混合验证与自由观察阶段，不能直接当作自由观察可见率。

该次会话的 producer 为 29.98 Hz、UI 新消费为 20.76 Hz、自由观察可见绘制为 16.55 Hz；观测年龄中位数 23.4 ms、p95 38.9 ms，read 返回到绘制提交的主机时间中位数 25.5 ms、p95 41.8 ms。这些数值不是曝光到屏幕发光的延迟。用户报告青色候选点跳跃；遮住眼睛或头离开画面时青点消失，移开遮挡或回到画面后青点重新出现（用户目视确认，事件也有两次无脸失效后重新出现可见候选）。用户报告暂停/继续正常，本地事件在自由观察阶段各记录一次 pause/resume。尚未建立可用性或精度结论；仍需检查大范围目光跟随、重新校准及其他断流情形。训练残差、独立验证偏差与自由观察现象应区分，自由观察没有目标真值。真实数值记录及映射留在本地忽略目录，未提交。R1/R2 旧交互的完整实机验收、R3 某些异常流程及原生 access violation 根因仍待单独处理。

## 命令（Git Bash，仓库根目录）

```bash
R1_PY="E:/TMP/look2act-r1-env/Scripts/python.exe"

# 用户手动；窗口出现后点“开始”才访问一个摄像头
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py collect --config configs/classic.yaml

# 显式加载某次 R6 个人映射；先检查本机相机/显示配置，窗口仍需点“开始”
PYTHONUTF8=1 "$R1_PY" scripts/r6_shadow.py validate-loaded --config configs/classic.yaml --mapping "experiment_sessions/r6_runs/<run_id>/f2_mapping.json"

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
