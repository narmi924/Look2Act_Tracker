# Paper Notes

## Title Candidates

- Look2Act: A Lightweight Real-Time Gaze Interaction System for Consumer Laptops
- What Matters for Practical Webcam Gaze Tracking? A System-Level Analysis
- Look2Act: CPU-Deployable Gaze-Driven Desktop Interaction via Geometric Projection and Lightweight CNN
- From Gaze Estimation to Gaze Interaction: A Complete Pipeline for Webcam-Based Desktop Control

## Possible Contributions

1. **完整的 gaze interaction pipeline**：从 webcam 输入到光标控制的端到端系统，不是孤立的 gaze estimator
   - 学习模块（GazeNet）负责 3D gaze direction
   - 几何模块（PnP + ray-plane）负责 camera-to-screen projection
   - 校准模块负责 user-specific residual correction
   - 时序平滑负责交互稳定性

2. **CPU-only 实时部署**：面向消费级笔记本，无 NVIDIA GPU 依赖
   - ~2MB ONNX 模型，~30 FPS
   - 证明轻量 CNN + 几何投影在桌面场景下可行

3. **系统级消融与分析**：不只报模型精度，而是从部署相关维度系统考察
   - 校准点数 vs 精度 vs 用户负担
   - 头部姿态几何补偿的贡献
   - 跨用户/跨设备泛化
   - 屏幕空间误差分布

## Intro 逻辑草案

1. 视线追踪在 HCI 中的价值（辅助交互、注意力分析）
2. 专业眼动仪成本高、部署复杂 → 普通 webcam 方案的需求
3. 现有 appearance-based 方法多停留在 gaze estimation benchmark，缺少从估计到交互的完整系统
4. 本文提出 Look2Act：3D gaze regression + 几何投影 + 轻量校准 + 实时部署
5. 贡献点列表

## Related Work 分组

### Group 1: Appearance-Based Gaze Estimation
- MPIIGaze (Paper1): in-the-wild dataset, head-pose-aware modeling
- Full-Face (Paper2): 输入表示对比（eye-only vs full-face）
- Gaze360 (Paper3): 大规模 3D benchmark, temporal modeling
- RT-GENE (Paper5): 自然环境数据集

### Group 2: Data Strategy
- UnityEyes (Paper4): 合成数据增强
- MPIIGaze Extended (Paper9): 预训练策略, benchmark

### Group 3: Calibration & Personalization
- CalibMe (Paper6): 校准作为 HCI 子问题, 覆盖率, 质量控制
- Few-Shot Personalization (Paper8): 校准 → few-shot 问题

### Group 4: System & Deployment
- OpenGaze/Evaluation (Paper7): 系统级 pipeline, 部署动机, 应用边界
- EFE (Paper10): 模块化 vs 端到端对比

## 主表候选

- **Table 1**: 校准点数对比（0/1/3/5/9 × affine/polynomial），含 hold-out error
- **Table 2**: 跨用户 leave-one-out 结果
- **Table 3**: 系统消融（head pose on/off, calibration on/off, smoothing on/off）
- **Table 4**: 与相关工作的系统级对比（精度/FPS/硬件/部署）

## 主图候选

- **Fig 1**: 系统 pipeline 图（仿 OpenGaze 风格，但突出几何链路）
- **Fig 2**: 屏幕空间误差热图（校准前后对比）
- **Fig 3**: 校准点数 vs 误差曲线
- **Fig 4**: 头部姿态分桶误差分析
- **Fig 5**: 跨 session 误差柱状图

## Reviewer 可能质疑的点

1. **Novelty**: 系统集成 vs 方法创新 → 需要强调系统级分析的价值
2. **数据规模**: 3 用户 2193 样本偏少 → 需要 cross-user 实验证明泛化
3. **精度**: 7.28° 与 SOTA 有差距 → 强调 CPU-only 实时部署约束下的 trade-off
4. **User study**: 缺少正式用户实验 → 至少需要 Fitts' Law 或任务完成时间
5. **校准依赖**: 9 点校准用户负担 → 需要 few-shot (3点) 也能工作的证据

## 系统论文味的关键表述

- "本文实现了一个完整的 gaze-driven interaction pipeline，而非孤立的 gaze estimator。"
- "系统将学习式视线估计、几何投影与用户特定校准进行分层集成。"
- "校准学习的是从系统性几何残差到真实屏幕点的用户特定修正映射。"
- "本文并非仅评估独立 gaze 模型精度，而是从交互部署相关维度系统考察其性能边界。"

## 统一符号约定（参考 Paper7 OpenGaze）

| 符号 | 含义 |
|------|------|
| I | 输入 RGB 图像 |
| l | facial landmarks |
| R_h, t_h | 头部姿态旋转/平移 |
| o | gaze origin (eye centre) |
| g_c | camera-space 3D gaze direction |
| Π_s | 屏幕坐标映射算子 |
| p^raw | 未校准屏幕点 |
| p^cal | 校准后屏幕点 |
| f_θ | gaze regression network |
| h_φ | calibration mapping function |
