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
8. ✅ 头部姿态消融实验（full/no_pose/no_yaw/no_pitch，3 张图）
   - Head pose 贡献 27.7%（503→364 px），pitch 是主要维度
   - Yaw 在桌面场景下影响可忽略（范围仅 [-20°, +13°]）

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

1. ~~头部姿态消融实验~~ ✅ 已完成
2. ~~跨用户 leave-one-out 评估~~ ✅ 已完成
   - 跨用户平均 5.03° ±1.64°，像素误差 349.5 px
   - 最佳 User 14 (3.34°)，最差 User 5 (9.30°)
   - 用户间差异显著，支持 calibration 必要性论述
3. 数据增强改进（分辨率退化、更强光照增强）
4. 考虑将 One Euro Filter 集成到系统中作为可选平滑器
5. 系统消融总表（Table 3）：整合 head pose / calibration / smoothing 的 on/off 对比


## 2026-03-18

### 新数据集入库

`dataset_raw/` 新增 12 个用户的数据（User 88~99），来自新设备 LAPTOP-DHE14D8H，屏幕分辨率 1440×900，摄像头 1920×1080。每个 session 约 250 帧，5×5 网格采集，15 FPS 保存。

**数据集规模变化**：
- 原有：10 个用户（4/5/6/13/14/15/16/25/26/27），3 台设备，~2193 samples
- 新增：12 个用户（88~99），1 台新设备，预计 ~3000 samples
- 合计：22 个用户，4 台设备，预计 ~5000+ samples

**新数据特点**：
- 新设备、新屏幕分辨率（1440×900 vs 之前 1536×864）
- 大幅增加用户多样性（10→22 用户）
- 有助于改善跨用户/跨设备泛化能力

### ⚠️ 待办：模型重新训练

新数据集已就绪，下次开始工作前需要：
1. 运行 `scripts/preprocess.py` 预处理新数据
2. 用全量数据（22 用户）重新训练 GazeNet
3. 重新跑 leave-one-user-out 评估（22 折）
4. 对比新旧模型性能，预期跨用户泛化显著改善
5. 更新 evaluation_results/ 下的所有图表和指标

**预期收益**：
- 训练数据量翻倍以上，模型泛化能力应显著提升
- 新增设备维度，跨设备泛化评估更有说服力
- 用户数从 10→22，LOO 评估统计意义更强


## 2026-03-24

### 今日目标
完成 GazeNetV2 模型升级（双眼 + head pose 融合），为重新训练做好准备。

### 完成工作

1. ✅ 数据预处理：22 用户全量数据预处理完成（train=3350, val=604, test=868）
2. ✅ V1 基线训练：100 epochs，best val_angle=4.38°（epoch 56）
3. ✅ GazeNetV2 模型实现：双眼共享 CNN + head pose 融合，参数量 < 2.5M
4. ✅ 数据集适配：GazeDataset 支持 V1/V2 双模式
5. ✅ 训练脚本适配：train.py 支持 V1/V2 自动切换
6. ✅ 预处理脚本适配：同时保存左右眼裁剪图像
7. ✅ ONNX 导出适配：V2 三输入（left_eye, right_eye, head_pose）
8. ✅ 实时推理管道适配：pipeline.py 支持 V2 模型（PyTorch + ONNX）
9. ✅ 配置文件更新：train_config.yaml + system_config.yaml

### V2 架构决策记录

**为什么选择共享 backbone？**
- 左右眼结构对称，共享权重可减少参数量约 50%
- 防止小数据集下过拟合
- 文献中 RT-GENE、Full-Face 等论文均采用类似策略

**为什么融合 head pose？**
- 消融实验证明 head pose 几何补偿贡献 27.7%
- 但几何补偿是线性的（旋转矩阵），融入模型可学习非线性补偿
- 特别是对大角度头部偏转，非线性补偿可能更有效

**为什么加 Dropout？**
- V1 训练观察到 train_loss=0.017 vs val_loss=0.088，存在过拟合
- Dropout 0.3 是轻量级正则化，不影响推理速度

### 下一步

1. 重新运行 `scripts/preprocess.py --clean` 生成右眼图像
2. 用 V2 配置训练 100 epochs
3. 对比 V1 vs V2 性能
4. 如果 V2 显著优于 V1，重新跑 leave-one-user-out 评估
5. 更新 evaluation_results/ 下的图表
