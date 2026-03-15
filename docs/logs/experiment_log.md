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