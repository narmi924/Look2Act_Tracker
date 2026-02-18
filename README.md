# Look2Act Tracker — 视线驱动交互系统

基于"三维视线方向回归 + 头部姿态估计 + 屏幕几何建模"的实时视线追踪系统。

## 环境要求

- conda 环境：`gaze-env`
- Python 3.11+
- Intel CPU（已适配 IPEX）

## 快速开始

```bash
conda activate gaze-env
pip install -r requirements.txt
```

## 项目结构

```
src/            # 源代码
  data/         # 数据处理管道
  models/       # GazeNet 模型定义
  vision/       # 人脸检测、头部姿态估计
  geometry/     # 坐标转换、射线-平面求交
  calibration/  # 校准模块
  tracker/      # 实时推理管道
  ui/           # PyQt6 GUI
scripts/        # 训练、评估、预处理脚本
tests/          # 测试
configs/        # YAML 配置文件
dataset_raw/    # 原始数据集（不入库）
```

## 开发阶段

- Phase 1：数据管道 + 模型训练
- Phase 2：实时推理 + 校准
- Phase 3：GUI 界面 + CPU 优化
