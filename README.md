# Look2Act Tracker — 视线驱动交互系统

<div align="center">

**用眼神控制电脑，让交互更自然**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](#english-version) | [中文](#中文版本)

</div>

---

## 中文版本

### 🎯 项目简介

Look2Act Tracker 是一个基于深度学习的实时视线追踪系统，让你可以用眼神控制电脑。无需昂贵的专业眼动仪，只需一个普通摄像头，就能实现精准的视线追踪和交互。

**核心特性：**
- 👁️ **实时追踪**：≥15 FPS，端到端延迟 <66ms
- 🎯 **高精度**：9 点校准后，屏幕注视点误差 <2cm
- 💻 **轻量化**：CPU 推理，无需 GPU，适配 Intel IPEX 加速
- 🔧 **易用性**：图形化界面，一键启动，开箱即用
- 🌐 **跨平台**：支持 Windows/Linux/macOS（ONNX Runtime）

### 🚀 快速开始

#### 环境准备

```bash
# 1. 克隆项目
git clone https://github.com/your-username/Look2Act_Tracker_Project.git
cd Look2Act_Tracker_Project

# 2. 创建 conda 环境
conda create -n gaze-env python=3.11
conda activate gaze-env

# 3. 安装依赖
pip install -r requirements.txt
```

#### 运行应用

```bash
# 启动图形界面
conda run -n gaze-env python main.py
```

#### 使用流程

1. **摄像头预览** 📷：检查摄像头是否正常工作，查看人脸检测效果
2. **视线校准** 🎯：注视屏幕上的 9 个校准点，优化追踪精度
3. **实时追踪** 👁️：启动视线追踪，屏幕上会显示你的注视点光标
4. **系统设置** ⚙️：调整摄像头、模型参数、平滑系数等

### 🏗️ 技术架构

```
输入：摄像头画面
  ↓
[人脸检测] → MediaPipe Face Mesh (468 关键点)
  ↓
[特征提取] → 双眼区域裁剪 + 归一化
  ↓
[视线回归] → GazeNet (CNN) → 三维视线方向向量
  ↓
[头部姿态] → PnP 算法 → 头部旋转矩阵 + 平移向量
  ↓
[几何建模] → 射线-平面求交 → 屏幕坐标系注视点
  ↓
[校准优化] → 9 点仿射变换 → 误差补偿
  ↓
输出：屏幕注视点坐标 (x, y)
```

**核心技术：**
- **三维视线方向回归**：不依赖特定屏幕尺寸，泛化性强
- **头部姿态估计**：PnP 算法计算头部在相机坐标系的位置和朝向
- **屏幕几何建模**：射线-平面求交，将视线向量映射到屏幕坐标
- **仿射变换校准**：9 点校准优化系统误差，提升精度

### 📁 项目结构

```
Look2Act_Tracker_Project/
├── src/                    # 源代码
│   ├── data/              # 数据处理管道（数据集加载、预处理、增强）
│   ├── models/            # GazeNet 模型定义（CNN 架构）
│   ├── vision/            # 计算机视觉模块
│   │   ├── face_detector.py    # MediaPipe 人脸检测
│   │   ├── headpose.py         # 头部姿态估计（PnP）
│   │   └── eye_extractor.py    # 双眼区域提取
│   ├── geometry/          # 几何计算模块
│   │   ├── coordinate_transform.py  # 坐标系转换
│   │   └── ray_plane_intersection.py  # 射线-平面求交
│   ├── calibration/       # 校准模块（9 点仿射变换）
│   ├── tracker/           # 实时推理管道
│   │   ├── tracker_pipeline.py  # 端到端推理流程
│   │   └── smoothing.py         # 卡尔曼滤波平滑
│   └── ui/                # PyQt6 图形界面
│       ├── main_window.py       # 主窗口
│       ├── home_page.py         # 主页
│       ├── camera_page.py       # 摄像头预览页
│       ├── calibration_page.py  # 校准页
│       ├── tracking_page.py     # 追踪页
│       └── settings_page.py     # 设置页
├── scripts/               # 训练、评估、预处理脚本
│   ├── train.py          # 模型训练
│   ├── evaluate.py       # 模型评估
│   ├── preprocess.py     # 数据预处理
│   └── export_onnx.py    # 导出 ONNX 模型
├── tests/                 # 单元测试和集成测试（pytest + hypothesis）
├── tools/                 # 开发工具和验证脚本
├── configs/               # YAML 配置文件
│   ├── system_config.yaml    # 系统运行配置
│   └── train_config.yaml     # 训练配置
├── docs/                  # 文档
│   ├── camera_page_usage.md      # 摄像头预览页使用说明
│   ├── tracking_page_usage.md    # 追踪页使用说明
│   ├── settings_page_usage.md    # 设置页使用说明
│   ├── cpu_optimization.md       # CPU 优化策略说明
│   └── logs/                     # 研究日志和实验记录
├── dataset_raw/           # 原始数据集（不入库）
├── dataset_processed/     # 预处理后的数据集
├── checkpoints/           # 模型权重文件
│   ├── best_model.pth    # PyTorch 模型（训练用）
│   └── gaze_net.onnx     # ONNX 模型（推理用）
├── evaluation_results/    # 评估结果（图表、指标）
├── logs/                  # 训练日志
├── main.py                # 应用入口
└── requirements.txt       # 依赖列表
```

### 🔬 技术细节

#### 模型架构

**GazeNet**：轻量级 CNN 网络，输入双眼图像，输出三维视线方向向量

```python
输入：左眼图像 (128×128) + 右眼图像 (128×128)
  ↓
[卷积层 × 4] → 特征提取
  ↓
[全连接层 × 2] → 视线方向回归
  ↓
输出：(gaze_x, gaze_y, gaze_z) 单位向量
```

- 参数量：~500K
- 推理速度：CPU 批处理 (batch=2) <10ms
- 训练数据：自采集数据集（使用 Gaze_Dataset_Collector 工具采集）

#### 训练与推理优化

**训练阶段**：使用 Intel XPU + IPEX 加速
- 默认配置：`device: xpu`, `use_ipex: true`
- 自动回退：XPU 不可用时自动切换到 CPU
- 训练速度：相比纯 CPU 提升 3-5 倍

**推理阶段**：使用 ONNX Runtime（跨平台 CPU 优化）
- 默认配置：`use_onnx: true`, `use_ipex: false`
- 跨平台兼容：无需 GPU，无需 IPEX，仅依赖 ONNX Runtime
- 批处理优化：左右眼合并为 batch=2，减少推理调用

详见：[CPU 优化策略说明](docs/cpu_optimization.md)

#### 坐标系转换

系统涉及 4 个坐标系：
1. **相机坐标系**：以摄像头为原点
2. **头部坐标系**：以头部中心为原点
3. **世界坐标系**：以屏幕中心为原点
4. **屏幕坐标系**：以屏幕左上角为原点（像素坐标）

转换流程：
```
头部坐标系 → (旋转矩阵 R + 平移向量 t) → 相机坐标系
相机坐标系 → (外参矩阵) → 世界坐标系
世界坐标系 → (射线-平面求交) → 屏幕坐标系
```

#### 校准算法

采用 9 点仿射变换校准：
1. 用户依次注视屏幕上的 9 个校准点
2. 记录每个点的预测坐标和真实坐标
3. 计算仿射变换矩阵（2×3）
4. 实时追踪时应用变换矩阵补偿系统误差

### 🛠️ 开发指南

#### 数据采集

推荐使用配套工具 [Gaze_Dataset_Collector](../Gaze_Dataset_Collector_Project) 采集训练数据：
- 自动记录眼部图像、头部姿态、注视点坐标
- 支持隐私保护（人脸模糊化）
- 双语界面，操作简单

#### 训练自己的模型

```bash
# 1. 准备数据集（放在 dataset_raw/ 目录）
# 2. 预处理数据
conda run -n gaze-env python scripts/preprocess.py

# 3. 训练模型（自动使用 XPU + IPEX 加速）
conda run -n gaze-env python scripts/train.py --config configs/train_config.yaml

# 4. 导出 ONNX 模型（用于跨平台推理）
conda run -n gaze-env python scripts/export_onnx.py \
  --checkpoint checkpoints/best_model.pth \
  --output checkpoints/gaze_net.onnx

# 5. 评估模型
conda run -n gaze-env python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

#### 运行测试

```bash
# 运行所有测试
conda run -n gaze-env pytest tests/

# 运行特定测试
conda run -n gaze-env pytest tests/test_tracker_pipeline.py -v

# 运行 Property-Based Testing
conda run -n gaze-env pytest tests/test_optimization.py -v
```

#### 详细文档

- [摄像头预览页使用说明](docs/camera_page_usage.md)
- [实时追踪页使用说明](docs/tracking_page_usage.md)
- [系统设置页使用说明](docs/settings_page_usage.md)
- [CPU 优化策略说明](docs/cpu_optimization.md)
- [追踪管道使用说明](docs/tracker_pipeline_usage.md)

### 📊 性能指标

| 指标 | 数值 |
|------|------|
| 推理帧率 | ~30 FPS (Intel Ultra 5 125H) |
| 端到端延迟 | ~30ms |
| 校准后精度 | <2cm (24 英寸显示器，60cm 距离) |
| 模型大小 | ~2MB (ONNX) |
| CPU 占用 | <15% (单核) |
| 内存占用 | <200MB |

**性能分解**（Intel Ultra 5 125H）：

| 模块 | 耗时 |
|------|------|
| 人脸检测 (MediaPipe) | ~15ms |
| 头部姿态估计 (PnP) | ~2ms |
| 视线回归 (ONNX batch=2) | ~10ms |
| 坐标转换 + 射线求交 | <1ms |
| 时序平滑 (卡尔曼滤波) | <1ms |
| **总计** | **~30ms** |

### 🎓 适用场景

- 🎮 **游戏交互**：眼控射击、视线选择
- ♿ **无障碍辅助**：帮助肢体障碍人士操作电脑
- 📊 **用户研究**：网页热力图、广告注意力分析
- 🎨 **创意应用**：眼控绘画、视线艺术
- 🔬 **科研教学**：视线追踪算法研究、人机交互实验

### 🔗 相关项目

- [Gaze_Dataset_Collector](../Gaze_Dataset_Collector_Project)：配套的视线数据集采集工具
- [Eye_Touch_Project](../Eye_Touch_Project)：早期视线交互系统原型

### 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改（推荐使用科研友好的提交信息）
   ```bash
   git commit -m 'exp: add 3D vs 2D comparison config'
   git commit -m 'log: update experiment notes for head pose ablation'
   git commit -m 'paper: draft abstract v1'
   git commit -m 'demo: add gaze cursor overlay prototype'
   ```
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

**提交信息前缀建议**：
- `exp:` 实验相关
- `log:` 日志和记录
- `paper:` 论文相关
- `demo:` 演示和原型
- `fix:` 修复 bug
- `feat:` 新功能
- `docs:` 文档更新
- `test:` 测试相关

### 📄 开源协议

本项目采用 MIT 协议开源，详见 [LICENSE](LICENSE) 文件。

### 📧 联系方式

- 项目主页：[GitHub](https://github.com/your-username/Look2Act_Tracker_Project)
- 问题反馈：[Issues](https://github.com/your-username/Look2Act_Tracker_Project/issues)

---

## English Version

### 🎯 Introduction

Look2Act Tracker is a real-time gaze tracking system based on deep learning, enabling you to control your computer with your eyes. No expensive professional eye trackers needed—just a regular webcam for precise gaze tracking and interaction.

**Key Features:**
- 👁️ **Real-time Tracking**: ≥15 FPS, end-to-end latency <66ms
- 🎯 **High Precision**: <2cm screen gaze error after 9-point calibration
- 💻 **Lightweight**: CPU inference, no GPU required, Intel IPEX accelerated
- 🔧 **User-friendly**: GUI interface, one-click launch, ready to use
- 🌐 **Cross-platform**: Windows/Linux/macOS support (ONNX Runtime)

### 🚀 Quick Start

#### Environment Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Look2Act_Tracker_Project.git
cd Look2Act_Tracker_Project

# 2. Create conda environment
conda create -n gaze-env python=3.11
conda activate gaze-env

# 3. Install dependencies
pip install -r requirements.txt
```

#### Run Application

```bash
# Launch GUI
conda run -n gaze-env python main.py
```

#### Usage Workflow

1. **Camera Preview** 📷: Check camera functionality and face detection
2. **Gaze Calibration** 🎯: Look at 9 calibration points to optimize accuracy
3. **Real-time Tracking** 👁️: Start gaze tracking with on-screen cursor
4. **System Settings** ⚙️: Adjust camera, model parameters, smoothing coefficients

### 🏗️ Technical Architecture

```
Input: Camera Frame
  ↓
[Face Detection] → MediaPipe Face Mesh (468 landmarks)
  ↓
[Feature Extraction] → Eye region cropping + normalization
  ↓
[Gaze Regression] → GazeNet (CNN) → 3D gaze direction vector
  ↓
[Head Pose] → PnP algorithm → Head rotation matrix + translation vector
  ↓
[Geometric Modeling] → Ray-plane intersection → Screen coordinate gaze point
  ↓
[Calibration] → 9-point affine transformation → Error compensation
  ↓
Output: Screen gaze coordinates (x, y)
```

**Core Technologies:**
- **3D Gaze Direction Regression**: Screen-size independent, strong generalization
- **Head Pose Estimation**: PnP algorithm for head position and orientation
- **Screen Geometric Modeling**: Ray-plane intersection for gaze-to-screen mapping
- **Affine Transformation Calibration**: 9-point calibration for error optimization

### 🔬 Technical Details

#### Model Architecture

**GazeNet**: Lightweight CNN network, inputs eye images, outputs 3D gaze direction vector

```python
Input: Left eye (64×64) + Right eye (64×64)
  ↓
[Conv Layers × 4] → Feature extraction
  ↓
[FC Layers × 2] → Gaze direction regression
  ↓
Output: (gaze_x, gaze_y, gaze_z) unit vector
```

- Parameters: ~500K
- Inference speed: CPU single frame <10ms
- Training data: Self-collected dataset + MPIIGaze augmentation

#### Coordinate Systems

The system involves 4 coordinate systems:
1. **Camera Coordinate System**: Origin at camera
2. **Head Coordinate System**: Origin at head center
3. **World Coordinate System**: Origin at screen center
4. **Screen Coordinate System**: Origin at screen top-left (pixel coordinates)

### 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Inference FPS | ≥15 FPS (Intel Ultra 5 125H) |
| End-to-end Latency | <66ms |
| Calibrated Accuracy | <2cm (24" display, 60cm distance) |
| Model Size | ~2MB (ONNX) |
| CPU Usage | <15% (single core) |
| Memory Usage | <200MB |

### 🎓 Use Cases

- 🎮 **Gaming**: Eye-controlled shooting, gaze selection
- ♿ **Accessibility**: Assist users with physical disabilities
- 📊 **User Research**: Webpage heatmaps, ad attention analysis
- 🎨 **Creative Apps**: Eye-controlled painting, gaze art
- 🔬 **Research & Education**: Gaze tracking algorithms, HCI experiments

### 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 📧 Contact

- Project Homepage: [GitHub](https://github.com/your-username/Look2Act_Tracker_Project)
- Issue Tracker: [Issues](https://github.com/your-username/Look2Act_Tracker_Project/issues)
