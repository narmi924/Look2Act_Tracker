# R8：录制工具升级——移动刺激、同帧眼部裁剪图与全部关键点

基线 `8fbe5428bb729718e526572f26f350b1bd380e3e`（已合并 R6 的 `origin/main`），分支 `claude/rebuild-r8-recording-tool`。R7（PR #28）结论：几何模型需要比 6 点通用 PnP 更好的头部旋转来源，而现有录制没有保存图像和完整网格，任何更好的观测方法都无法离线评估。R8 只升级录制与离线检查，不改在线默认行为、不做系统动作、不显示预测反馈。

## 新增内容

- **移动刺激协议 `R8`**（`src/experiment/r8_protocol.py`，`make_plan(size, 'R8')`）：9 点静止注视（第 0 轮）→ 7 段平滑追随路径（3 条水平扫、3 条垂直扫、1 段 Lissajous，共 62 s）→ 20 次追随选择试次（4 或 6 个小点绕各自锚点做椭圆轨道，相位均分、方向交替，只有带圈高亮的一个要跟）→ 中心目标的自然/左右转头/抬低头 → 9 点静止注视（第 1 轮）。总长约 3 分 10 秒。段字段 `stimulus ∈ {fixation, path, choice}`；轨迹数学与窗口绘制和离线分析共用同一函数。
- **每次绘制记录实际位置**：`Protocol.moved()` 在已有 `target_painted` 之后，为 path 段写 `target_moved`（phase_s, x, y），为 choice 段写 `markers_moved`（全部小点位置和被指示下标）；时间是宿主绘制提交时间，不是屏幕点亮时间。`Protocol.phase()` 给出扣除暂停后的段内时间。
- **同帧图像与关键点**（`src/experiment/frames.py`）：`TrackerPipeline` 新增显式开关 `collect_full_landmarks` 与 `frame_sink`，默认关闭、不进入任何 JSON 事件。开启后检测器保留全部 478 个 MediaPipe 关键点（x px、y px、z×宽），管线在发布同一帧的数值快照之后把帧、结果和关键点交给 `FrameWriter`。采集线程只做两个小裁剪并入有界队列（64），PNG 编码与写文件在独立线程；队列满则丢帧并计数，绝不阻塞采集。文件：`frames/<序号>_L.png` / `_R.png`（眼角中点为中心、2.2 倍眼宽 × 1.2 倍眼宽、原始分辨率 BGR）、`landmarks.f32`（478×3 float32 流）、`frames.jsonl`（观测身份、源时间、裁剪框、文件名、关键点记录号）、`landmarks_meta.json`。
- **窗口** `src/ui/r8_record_window.py`：继承 R3 实验窗口，增加移动点、轨道小点与高亮圈、顶部说明与剩余秒数；开始后挂接 `FrameWriter`，结束时先停止采集线程、再关闭写入器，把写入统计写进 `session.json`（`session_type='r8_pursuit_v1'`、`frames`、`images`、`full_landmarks`）。写入器出错、线程未退出或关闭时队列仍有未写帧，会话标记为不完整。开启全关键点采集时无论哪个后端都使用 478 点精细网格；第一帧固定记录形状，形状不同的帧被计数跳过。
- **离线检查** `scripts/r8_record.py check <session>`（`src/experiment/r8_pursuit.py`）：完整性（事件数、有效 producer、带图像的 producer、缺失 PNG、关键点记录与索引一致、各段绘制、写入丢失）；只用 F2 给出两个可行性数字——(1) **追随选择准确率**：每个试次跳过前 0.75 s，取 F2 双眼均值信号与每个小点轨迹做相关（0.65 横 + 0.35 纵，方向符号由第 0 轮注视数据确定），取最高者，与被指示下标比较；(2) **密集追随校准 vs 9 点校准**：分别用第 0 轮注视、追随路径帧（标签为 t−lag 时刻的绘制位置）、两者合并训练 R4 岭回归，在第 1 轮注视上测试。lag 固定报告 0 与 150 ms 两档，不选择。结果写到会话目录 `r8_check.json`；退出码 0 要求记录完整、无读取问题且 `artifacts_ok`（无缺失 PNG、关键点记录与索引一致、写入器无错误且关闭时无未写帧）。丢帧数只报告，不作为失败。

## 验证

本机：`tests/test_r8_recording.py` 9 项（计划结构与边界、轨迹数学、`phase/moved` 随绘制与暂停、裁剪框与退化输入、`FrameWriter` 端到端与关闭、关键点数组、管线 `frame_sink` 接线且默认关闭、离屏窗口录制移动刺激与图像、合成会话的离线检查）；R3/R4/R5/R6/R7 相关短套件 **270 passed、2 主动 deselected**（真实相机初始化与原生检测器测试）；`scripts/r8_record.py selftest` 合成会话通过。合成会话里眼睛按线性关系跟随刺激，只验证数据链路和分析代码，不代表真人表现。未运行真实摄像头。

## 使用（Git Bash，仓库根目录）

```bash
R1_PY="E:/TMP/look2act-r1-env/Scripts/python.exe"
PYTHONUTF8=1 "$R1_PY" scripts/r8_record.py selftest
PYTHONUTF8=1 "$R1_PY" scripts/r8_record.py collect --config configs/classic.yaml     # 全屏窗口，点"开始"才开摄像头
PYTHONUTF8=1 "$R1_PY" scripts/r8_record.py check "experiment_sessions/<会话目录>"
PYTHONUTF8=1 QT_QPA_PLATFORM=offscreen "$R1_PY" -m pytest tests/test_r8_recording.py -q
```

录制建议：正面光、不逆光；坐姿与平时一致；除 B 段指令外头部自然保持；追随段眼睛跟点、不要预判；选择段只看带圈的高亮点。可随时暂停/跳过/结束，记录前缀仍可分析。

## 限制

图像和关键点只在本机 Git 忽略目录，含生物特征，请勿分享原目录。裁剪框依赖 FaceMesh 眼角，眼角偶发漂移时裁剪会抖动；`frame_sink` 在管线启动后挂接，前几帧可能没有图像。`check` 只用 F2，不分析图像；追随选择准确率只是"运动相关性能否区分 4–6 个小点"，不是在线交互成功率；追随路径的标签是指令点位置减固定 lag，不是测得的注视。头部旋转的新估计方法（用完整网格或个人化人脸形状）和基于图像的个人模型留给后续轮次，本轮只保证数据被完整保存。
