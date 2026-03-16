# Research Log

## 2026-03-16

### 今日目标
基于导师分析的 10 篇论文，提炼跨论文共识，制定优先级实验计划，并开始实施最高 ROI 的实验。

### 论文分析总结

导师分析了 10 篇论文，按与 Look2Act 的相关性和可操作性，提炼出以下跨论文共识：

**最高优先级（多篇论文交叉推荐）：**

1. **校准实验**（Paper6 CalibMe 8.7, Paper7 OpenGaze 9.0, Paper8 Few-Shot 8.3）
   - 0/1/3/5/9 点校准对比
   - affine vs polynomial 映射函数对比
   - hold-out 评估（不能只报 fit residual）
   - 校准覆盖率指标

2. **跨用户/跨 session 评估**（Paper1 MPIIGaze 8.5, Paper3 Gaze360 9.0, Paper5 RT-GENE 8.2, Paper9 Extended 9.1）
   - leave-one-user-out cross-validation
   - 跨设备泛化
   - 按 session 分组的误差分析

3. **条件分桶误差分析**（Paper1, Paper2 Full-Face 8.0, Paper5, Paper7）
   - 误差 vs 头部姿态（yaw/pitch 分桶）
   - 误差 vs session（跨 session 稳定性）
   - 屏幕空间误差热图（中心 vs 边角）

4. **头部姿态消融**（Paper1, Paper2）
   - 有 vs 无 PnP 几何补偿的系统级对比
   - 量化 head pose 在几何链路中的贡献

5. **平滑对比**（Paper3 Gaze360）
   - EMA α 参数扫描
   - 潜在引入 One Euro Filter 对比

### 今日完成

1. ✅ 完成 10 篇论文的跨论文分析和优先级排序
2. ✅ 写入 research_log、paper_notes、experiment_log
3. ✅ calibrator.py 扩展：支持 polynomial 校准 + 1 点平移修正
4. ✅ 校准对比实验（0/1/3/5/9 pt × affine/polynomial，hold-out 评估）
5. ✅ 条件分桶误差分析（head pose / session / user / 屏幕区域，5 张图）
6. ✅ 平滑对比实验（EMA α 扫描 + One Euro Filter，3 张图）
7. ✅ 论文笔记：title candidates、contribution、related work 分组、reviewer risks

### 当前系统基线

| 指标 | 值 |
|------|-----|
| Mean Angular Error | 7.28° |
| Median Angular Error | 7.90° |
| Mean Pixel Error | 369.4 px |
| Std Angular Error | 2.73° |
| Test Samples | 447 |
| 训练数据 | 1545 train / 201 val / 447 test |
| 用户数 | 3 |
| 设备数 | 3 |
| Session 数 | 10 |

### 当前问题

- 校准模块只支持 affine，缺少 polynomial 对比
- 评估只有 overall metrics，缺少条件分桶分析
- 没有 hold-out 校准评估（当前只报 fit residual）
- 缺少跨用户 leave-one-out 评估

### 下一步

1. 头部姿态消融实验（有 vs 无 PnP 几何补偿）— 需要修改 pipeline 代码
2. 跨用户 leave-one-out 评估 — 需要重新训练模型（告知用户命令）
3. 数据增强改进（分辨率退化、更强光照增强）
4. 考虑将 One Euro Filter 集成到系统中作为可选平滑器
