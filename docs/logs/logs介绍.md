[1]experiment_log.md:
这是实验事实库。

每条实验都尽量结构化，建议一条一个小节。

示例格式
## Exp-2026-03-20-3D-vs-2D

### 目标
比较 3D gaze regression 与 2D screen regression 在桌面场景下的效果差异。

### 配置
- dataset:
- split:
- model:
- calibration: 3x3
- head pose: off
- distance: off

### 指标
- pixel error
- angular error
- FPS

### 结果
- 3D:
- 2D:

### 结论
3D 在头部姿态变化较大时更稳定。

### 文件位置
- results/exp_3d_vs_2d/
- checkpoint: ...
适合写什么

它最适合成为你写论文实验部分时的原始依据。

[2]paper_notes.md:
这个建议专门放论文思考，不要和 general log 混。

记录：

标题候选

contribution 候选

intro 逻辑

related work 分组

哪个结果适合做主表

reviewer 可能会质疑什么

示例格式
## Title candidates
- What Matters for Practical Webcam Gaze Tracking? ...
- Look2Act: ...

## Possible contributions
1. 一个面向普通笔记本的实时 gaze interaction 系统
2. 系统分析 3D gaze、head pose、distance、calibration 的作用
3. 在真实桌面部署条件下验证实时性和可用性

## Reviewer risks
- novelty 是否足够
- user study 是否太弱
- cross-device 数据量是否不足
适合写什么

它最适合把“毕设系统”转成“论文叙事”。

[3]research_log.md:
这是总日志，按时间顺序写。

记录：

今天做了什么

为什么做

卡在哪里

做出了什么决定

下一步是什么

示例格式
## 2026-03-13
### 今日目标


### 今日完成


### 当前问题


### 下一步

适合写什么

它最适合写“研究过程的连续叙事”。