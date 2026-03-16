# Look2Act — Research System Overview

> 本文档面向研究人员和 AI 助手，用于快速理解 Look2Act 系统的研究结构、算法链路和实验设计。
> 不包含安装说明、UI 使用指南等工程内容。

---

## 1 Research Problem

### 1.1 问题定义

Look2Act 解决的核心问题是：**如何仅使用普通 webcam 实现实时、低成本的桌面视线追踪与交互**。

具体而言，系统需要在以下约束下工作：

- 输入设备：笔记本内置摄像头（720p/1080p，无红外光源）
- 计算平台：消费级 CPU（Intel Ultra 5 125H），无 NVIDIA GPU
- 延迟要求：端到端 < 66ms（≥15 FPS）
- 精度要求：校准后屏幕注视点误差 < 2cm（24 英寸显示器，60cm 距离）
- 部署要求：跨平台（Windows/Linux/macOS），无需专用硬件

### 1.2 为什么传统 Eye Tracker 不够

| 维度 | 专业眼动仪 | Look2Act |
|------|-----------|----------|
| 硬件成本 | $5,000–$30,000 | $0（使用已有摄像头） |
| 红外光源 | 需要 | 不需要 |
| 专用传感器 | 需要 | 不需要 |
| 部署复杂度 | 高（固定安装） | 低（软件即用） |
| 精度 | 0.5°–1.0° | ~7°（当前），校准后 <2cm |
| 适用场景 | 实验室 | 日常桌面 |

### 1.3 Appearance-Based Gaze Estimation 的挑战

基于外观的视线估计面临以下核心挑战：

1. **头部姿态变化**：用户头部的 yaw/pitch/roll 变化导致眼部外观剧烈变化
2. **光照变化**：室内光照不均匀，无红外光源辅助
3. **个体差异**：不同用户的眼部形态、瞳孔大小、眼镜佩戴等差异
4. **距离变化**：用户与屏幕的距离不固定，影响眼部图像分辨率
5. **实时性约束**：模型必须在 CPU 上实现低延迟推理
6. **标定漂移**：校准参数随时间和姿态变化而退化

---

## 2 System Overview

### 2.1 端到端 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Look2Act Tracker Pipeline                        │
│                                                                     │
│  Camera Frame (1280×720 BGR)                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────┐                                           │
│  │  Face Detection       │  MediaPipe FaceMesh (468 landmarks)      │
│  │  + Landmark Extract   │  → 68-point subset + 6 PnP points       │
│  └──────────┬───────────┘                                           │
│             │                                                       │
│       ┌─────┴─────┐                                                 │
│       ▼           ▼                                                 │
│  ┌─────────┐ ┌──────────────┐                                       │
│  │ Eye Crop │ │ Head Pose    │  solvePnP (6-point, ITERATIVE)       │
│  │ L+R      │ │ Estimation   │  → R (3×3), t (3×1)                  │
│  │ 128×128  │ │ yaw/pitch/   │  → yaw, pitch, roll (degrees)        │
│  └────┬─────┘ │ roll         │                                       │
│       │       └──────┬───────┘                                       │
│       ▼              │                                               │
│  ┌──────────────┐    │                                               │
│  │ GazeNet CNN  │    │                                               │
│  │ batch=2      │    │  gaze_local (3,) 单位向量                     │
│  │ → (gx,gy,gz) │    │                                               │
│  └──────┬───────┘    │                                               │
│         │            │                                               │
│         ▼            ▼                                               │
│  ┌─────────────────────────┐                                         │
│  │ Coordinate Transform    │  gaze_cam = R @ gaze_local              │
│  │ Eye-local → Camera      │  ray_origin = t                         │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────────┐                                         │
│  │ Ray–Plane Intersection  │  t = N·(P0-O) / (N·D)                  │
│  │ Screen Geometry Model   │  P = O + t·D → (x_mm, y_mm)            │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────────┐                                         │
│  │ Screen Projection       │  mm → pixel: px = x_mm/W_mm * W_px     │
│  │ Physical → Pixel        │                                         │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────────┐                                         │
│  │ Calibration (optional)  │  Affine: [px',py']^T = A·[px,py,1]^T   │
│  │ 9-point Affine          │  A: 2×3 matrix (least-squares fit)      │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────────┐                                         │
│  │ Temporal Smoothing      │  EMA: s_t = α·x_t + (1-α)·s_{t-1}     │
│  │ (EMA, α=0.3)           │                                         │
│  └──────────┬──────────────┘                                         │
│             │                                                        │
│             ▼                                                        │
│  Screen Gaze Point (px, py)                                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 各模块职责

| 模块 | 源码位置 | 职责 | 输入 → 输出 |
|------|---------|------|------------|
| Face Detection | `src/vision/face_detector.py` | 人脸检测、关键点提取、眼部裁剪 | BGR frame → FaceDetectionResult |
| Head Pose | `src/vision/head_pose.py` | 基于 PnP 的头部姿态估计 | 6 个 2D 关键点 → R, t, yaw/pitch/roll |
| GazeNet | `src/models/gaze_net.py` | 3D 视线方向回归 | (B,3,128,128) 眼部图像 → (B,3) 单位向量 |
| Coordinate Transform | `src/geometry/coordinate.py` | 视线方向从眼球坐标系转到相机坐标系 | gaze_local + R + t → ray |
| Ray–Plane Intersect | `src/geometry/ray_plane.py` | 射线与屏幕平面求交 | ray + plane → 3D 交点 |
| Screen Geometry | `src/geometry/screen_geometry.py` | 屏幕平面建模 + 3D→pixel 转换 | 3D 交点 → (px, py) |
| Calibration | `src/calibration/calibrator.py` | 仿射变换校准 | raw (px,py) → calibrated (px,py) |
| Smoother | `src/tracker/smoother.py` | EMA 时序平滑 | noisy point → smoothed point |
| Pipeline | `src/tracker/pipeline.py` | 端到端推理管道（线程化） | camera → gaze point stream |

### 2.3 推理耗时分解（实测，Intel Ultra 5 125H）

| 阶段 | 耗时 | 占比 |
|------|------|------|
| Face Detection (MediaPipe) | ~15ms | 50% |
| Head Pose (PnP) | ~2ms | 7% |
| Gaze Regression (ONNX, batch=2) | ~10ms | 33% |
| Coordinate Transform + Ray Intersect | <1ms | 3% |
| Smoothing | <1ms | 1% |
| **Total** | **~30ms** | **~30 FPS** |

---

## 3 Mathematical Formulation

### 3.1 Gaze Direction Representation

视线方向表示为三维单位向量 $\mathbf{g} = (g_x, g_y, g_z)^T$，满足 $\|\mathbf{g}\| = 1$。

模型输出经过 L2 归一化：

```
g = F.normalize(fc(features), p=2, dim=1)
```

即：

```
g_hat = f(x) / ||f(x)||_2
```

其中 `f(x)` 是 GazeNet 全连接层的原始输出，`x` 是 128×128 的眼部 RGB 图像。

坐标系约定（眼球局部坐标系）：
- X 轴：向右
- Y 轴：向下
- Z 轴：向前（远离眼球，朝向注视方向）

### 3.2 Head Pose Estimation via PnP

头部姿态通过 Perspective-n-Point (PnP) 算法估计。

**输入**：6 个面部关键点的 2D 像素坐标和对应的 3D 模型点。

6 个关键点：
```
nose_tip:        (0, 0, 0)           mm
chin:            (0, -330, -65)      mm
left_eye_outer:  (-225, 170, -135)   mm
right_eye_outer: (225, 170, -135)    mm
left_mouth:      (-150, -150, -125)  mm
right_mouth:     (150, -150, -125)   mm
```

**求解**：使用 `cv2.solvePnP(SOLVEPNP_ITERATIVE)` 求解旋转向量 `rvec` 和平移向量 `tvec`：

```
s · [u, v, 1]^T = K · [R | t] · [X, Y, Z, 1]^T
```

其中：
- `K` 是 3×3 相机内参矩阵（焦距近似为图像宽度）
- `R` 是 3×3 旋转矩阵（通过 Rodrigues 变换从 `rvec` 得到）
- `t` 是 3×1 平移向量（头部在相机坐标系中的位置）
- `(u, v)` 是 2D 像素坐标
- `(X, Y, Z)` 是 3D 模型点

**相机内参**（简化模型）：

```
K = [[f, 0, cx],
     [0, f, cy],
     [0, 0,  1]]
```

其中 `f = image_width`，`(cx, cy) = (width/2, height/2)`。

**欧拉角提取**：通过 `cv2.RQDecomp3x3(R)` 得到 yaw, pitch, roll（度）。

### 3.3 Coordinate Transformation

将视线方向从眼球局部坐标系转换到相机坐标系：

```
d_cam = R · g_local
```

射线定义：
```
ray_origin    = t          （眼球在相机坐标系中的位置）
ray_direction = R · g_local （视线在相机坐标系中的方向）
```

其中 `R` 和 `t` 来自 PnP 求解结果。

### 3.4 Ray–Plane Intersection

屏幕建模为一个平面，由平面上一点 `P_0` 和法向量 `N` 定义。

射线方程：
```
P(t) = O + t · D
```

平面方程：
```
N · (P - P_0) = 0
```

求交：
```
t = N · (P_0 - O) / (N · D)
```

约束条件：
- 若 `|N · D| < ε`：射线与屏幕平行，无交点
- 若 `t < 0`：交点在射线反方向，无效

交点：
```
P_intersect = O + t · D
```

### 3.5 Screen Projection (3D → Pixel)

将 3D 交点投影到屏幕像素坐标：

```
delta = P_intersect - P_origin
local_x_mm = delta · x_axis
local_y_mm = delta · y_axis

px = local_x_mm / W_mm * W_px
py = local_y_mm / H_mm * H_px
```

其中：
- `P_origin` 是屏幕左上角在相机坐标系中的 3D 位置
- `x_axis`, `y_axis` 是屏幕局部坐标系的基向量
- `W_mm`, `H_mm` 是屏幕物理尺寸（mm）
- `W_px`, `H_px` 是屏幕分辨率（pixel）

### 3.6 Calibration Mapping Function

校准使用 2D 仿射变换：

```
[px_cal]       [a11 a12 a13]   [px_raw]
[py_cal]   =   [a21 a22 a23] · [py_raw]
                                [  1   ]
```

即 `p_cal = A · [p_raw; 1]`，其中 `A` 是 2×3 仿射矩阵。

**求解**：最小二乘法

```
min_A  Σ_i || A · [x_i, y_i, 1]^T - [x_i_true, y_i_true]^T ||^2
```

通过 `numpy.linalg.lstsq` 求解。

### 3.7 Loss Function

训练使用角度损失（angular loss）：

```
L = mean( arccos( clamp( <g_pred, g_true>, -1+ε, 1-ε ) ) )
```

其中 `<·,·>` 表示向量点积，`g_pred` 和 `g_true` 均为单位向量。

损失值单位为弧度，评估时转换为度。

---

## 4 Model Architecture

### 4.1 GazeNet 概述

GazeNet 是一个轻量级 CNN，用于从单眼 RGB 图像回归 3D 视线方向向量。

设计目标：参数量 < 2M，CPU 推理 < 10ms（batch=2）。

### 4.2 网络结构

```
Input: (B, 3, 128, 128) — 眼部 RGB 图像

Layer 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
         输出: (B, 32, 64, 64)

Layer 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
         输出: (B, 64, 32, 32)

Layer 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
         输出: (B, 128, 16, 16)

Layer 4: Conv2d(128→256, 3×3, pad=1) → BN → ReLU → MaxPool(2)
         输出: (B, 256, 8, 8)

AdaptiveAvgPool2d(1):
         输出: (B, 256, 1, 1)

Flatten → Linear(256→3):
         输出: (B, 3)

L2 Normalize:
         输出: (B, 3) — 单位视线向量
```

### 4.3 设计选择的理由

| 设计决策 | 理由 |
|---------|------|
| 4 层 Conv + BN + ReLU | 足够提取眼部纹理特征，同时保持轻量 |
| 3×3 卷积核 | 标准选择，感受野逐层扩大 |
| BatchNorm | 加速收敛，稳定训练 |
| AdaptiveAvgPool | 消除空间维度，输出固定长度特征 |
| 单层 FC (256→3) | 直接回归 3D 方向，避免过拟合 |
| L2 归一化 | 确保输出为单位向量，与 angular loss 一致 |
| 通道数 [32,64,128,256] | 渐进式特征提取，参数量可控 |

### 4.4 训练配置

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 128×128 RGB |
| 输出维度 | 3（单位向量） |
| 通道数 | [32, 64, 128, 256] |
| 优化器 | Adam (lr=0.001, wd=0.0001) |
| 调度器 | CosineAnnealingLR (T_max=100) |
| Batch Size | 64 |
| Epochs | 100（实际 50 epoch 收敛） |
| Loss | Angular Loss (arccos-based) |
| 设备 | Intel XPU + IPEX / CPU fallback |

### 4.5 推理优化

- 训练阶段：PyTorch + Intel XPU + IPEX 加速
- 推理阶段：导出为 ONNX，使用 ONNX Runtime (CPUExecutionProvider)
- 左右眼合并为 batch=2，单次前向传播处理双眼
- ONNX 导出验证：PyTorch 与 ONNX 输出最大差异 < 1e-4

### 4.6 数据增强策略

训练时对眼部图像应用以下增强（概率触发）：

| 增强方式 | 概率 | 标签同步 |
|---------|------|---------|
| 水平翻转 | 50% | gaze_x 取反 |
| 亮度抖动 (±30) | 50% | 标签不变 |
| 轻微旋转 (±5°) | 30% | 标签不变（角度小，影响可忽略） |

---

## 5 Coordinate Systems

### 5.1 四个坐标系

Look2Act 系统涉及四个坐标系，理解它们之间的关系是理解整个 pipeline 的关键。

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Camera Coordinate System (相机坐标系)                      │
│   原点：相机光心                                             │
│   X: 向右  Y: 向下  Z: 向前（远离相机）                      │
│                                                             │
│        O ──────────── X                                     │
│        │\                                                   │
│        │ \                                                  │
│        Y  Z (向前)                                          │
│                                                             │
│   ┌─────────────────────────────────────────┐               │
│   │ Head Coordinate System (头部坐标系)      │               │
│   │ 原点：鼻尖（PnP 3D 模型原点）            │               │
│   │ 通过 R, t 与相机坐标系关联               │               │
│   │                                         │               │
│   │   Eye-local: 视线方向 g_local            │               │
│   │   → R @ g_local = g_camera               │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
│   ┌─────────────────────────────────────────┐               │
│   │ Screen Plane (屏幕平面)                  │               │
│   │ 定义：P_origin + N (法向量)              │               │
│   │ 局部坐标系：x_axis (向右), y_axis (向下) │               │
│   │                                         │               │
│   │   3D 交点 → 投影到局部坐标 → mm          │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
│   ┌─────────────────────────────────────────┐               │
│   │ Screen Pixel Coordinate (屏幕像素坐标系) │               │
│   │ 原点：屏幕左上角                         │               │
│   │ X: 向右 (0 → W_px)                      │               │
│   │ Y: 向下 (0 → H_px)                      │               │
│   │                                         │               │
│   │   mm → pixel: px = x_mm/W_mm * W_px     │               │
│   └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 坐标系变换链

```
Eye-local          Camera              Screen 3D           Screen Pixel
g_local  ──R──→  g_camera  ──ray──→  P_intersect  ──proj──→  (px, py)
              (旋转矩阵)      (射线-平面求交)       (物理→像素)
```

### 5.3 训练标签的坐标系

训练数据的视线标签在相机坐标系中定义：

```
target_3d_x = screen_x_mm - W_mm / 2
target_3d_y = cam_above_screen_mm + screen_y_mm
target_3d_z = distance_mm

g = normalize([target_3d_x, target_3d_y, target_3d_z])
```

假设：
- 相机在屏幕正上方中央
- 屏幕平面垂直于相机光轴
- `cam_above_screen_mm = 5mm`（相机在屏幕上边缘上方 5mm）

---

## 6 Calibration Strategy

### 6.1 方法：9-Point Affine Calibration

Look2Act 使用 9 点校准方案，通过仿射变换补偿系统误差。

**校准流程**：

```
1. 在屏幕上显示 9 个校准点（3×3 网格）
2. 用户依次注视每个点，系统记录：
   - raw_gaze: 未校准的视线映射坐标 (x, y)
   - target:   屏幕真实目标坐标 (x, y)
3. 收集 9 对 (raw, target) 数据
4. 最小二乘法拟合 2×3 仿射矩阵 A
5. 实时追踪时：p_calibrated = A · [p_raw; 1]
```

### 6.2 数学细节

构建超定方程组：

```
[x1_raw  y1_raw  1]       [x1_true  y1_true]
[x2_raw  y2_raw  1]       [x2_true  y2_true]
[  ...    ...   ..]  @ A^T =  [  ...     ...  ]
[x9_raw  y9_raw  1]       [x9_true  y9_true]
```

通过 `numpy.linalg.lstsq(src, dst)` 求解 `A^T`，得到 2×3 仿射矩阵 `A`。

### 6.3 质量评估

校准后计算平均残差：

```
residual_i = || A · [x_i, y_i, 1]^T - [x_i_true, y_i_true]^T ||
mean_residual = mean(residual_i)
```

若 `mean_residual > 50px`（可配置），系统提示用户重新校准。

### 6.4 校准的局限性

| 局限 | 说明 |
|------|------|
| 姿态敏感 | 校准时的头部姿态与使用时不同会导致精度下降 |
| 距离敏感 | 用户与屏幕距离变化后校准参数退化 |
| 仿射假设 | 仿射变换只能补偿线性误差，非线性畸变无法修正 |
| 时间漂移 | 长时间使用后校准参数可能不再准确 |
| 单用户 | 校准参数与特定用户绑定，换人需重新校准 |

### 6.5 校准数据持久化

校准参数序列化为 JSON 文件（`src/calibration/serializer.py`），包含：
- 2×3 仿射矩阵
- 原始校准点对
- 平均残差
- 时间戳和屏幕分辨率

---

## 7 System Design Constraints

### 7.1 设计目标

Look2Act 的系统设计受以下约束驱动：

| 约束 | 目标值 | 实现方式 |
|------|--------|---------|
| 实时推理 | ≥15 FPS (目标 30 FPS) | 轻量 CNN + ONNX Runtime + batch 推理 |
| CPU-only 部署 | 无 NVIDIA GPU 依赖 | ONNX CPUExecutionProvider |
| 跨平台 | Windows / Linux / macOS | PyQt6 + ONNX Runtime |
| 最小硬件 | 普通笔记本 + 内置摄像头 | 无红外、无深度相机 |
| 模型体积 | < 5MB | ~2MB ONNX 模型 |
| 内存占用 | < 200MB | 轻量模型 + 流式处理 |

### 7.2 关键设计决策

**为什么选择 3D 视线回归而非 2D 屏幕坐标回归？**

- 3D 视线向量与屏幕尺寸无关，泛化性更强
- 头部姿态变化时，3D 表示更稳定
- 可以通过几何模型适配不同屏幕配置
- 代价：需要额外的坐标转换和屏幕几何建模

**为什么选择 MediaPipe 而非 dlib/MTCNN？**

- MediaPipe FaceMesh 提供 468 个关键点（vs dlib 的 68 个）
- 内置 GPU 加速（移动端），CPU 性能也优秀
- 提供 refine_landmarks 选项，眼部关键点更精确
- 免费、跨平台、维护活跃

**为什么使用 EMA 而非 Kalman Filter？**

- EMA 实现简单，计算开销极低（< 0.01ms）
- 单参数 α 易于调节
- 对于视线追踪场景，EMA 已能有效减少抖动
- Kalman Filter 需要建模状态转移，复杂度更高

### 7.3 容错机制

Pipeline 实现了多层容错（`src/tracker/pipeline.py`）：

| 异常场景 | 处理策略 |
|---------|---------|
| 摄像头帧获取失败 | 重试 3 次，失败后标记断开并通知 UI |
| 未检测到人脸 | 返回上一帧有效结果 |
| 头部姿态估计失败 | 返回上一帧有效结果 |
| 眼部裁剪失败 | 返回上一帧有效结果 |
| 视线回归异常 | 返回上一帧有效结果 |
| 射线-平面无交点 | clamp 到屏幕边缘 |

---

## 8 Experimental Evaluation

### 8.1 当前评估结果

基于现有测试集（447 样本，2 个 session，跨设备）的评估结果：

| 指标 | 值 |
|------|-----|
| Mean Angular Error | 7.28° |
| Median Angular Error | 7.90° |
| Mean Pixel Error | 369.4 px |
| Std Angular Error | 2.73° |
| Test Samples | 447 |

数据集划分（按 session 划分，非按样本随机划分）：

| 划分 | Session 数 | 样本数 |
|------|-----------|--------|
| Train | 7 | 1545 |
| Val | 1 | 201 |
| Test | 2 | 447 |

训练收敛情况：50 epoch 后 val_angle_error 稳定在 ~5.9°–6.5°。

### 8.2 已有评估产物

```
evaluation_results/
├── metrics.json           # 量化指标
├── error_histogram.png    # 角度误差分布直方图
├── gaze_heatmap.png       # 注视点目标分布热力图
└── session_errors.png     # 各 Session 平均误差对比
```

### 8.3 计划实验矩阵

基于研究日志（`docs/logs/`）和系统架构，以下实验已规划或可执行：

#### Exp-1: 3D vs 2D Gaze Regression

| 维度 | 3D (当前) | 2D (对照) |
|------|----------|----------|
| 输出 | (gx, gy, gz) 单位向量 | (screen_x, screen_y) 归一化坐标 |
| Loss | Angular Loss | MSE Loss |
| 评估 | Angular Error + Pixel Error | Pixel Error |
| 假设 | 3D 在头部姿态变化时更稳定 | 2D 在固定姿态下可能更精确 |

#### Exp-2: Cross-User Generalization

- 训练集：用户 A, B, C 的数据
- 测试集：用户 D 的数据（未见过的用户）
- 评估：leave-one-user-out cross-validation
- 目标：量化个体差异对模型泛化的影响

#### Exp-3: Calibration Comparison

| 方案 | 校准点数 | 方法 |
|------|---------|------|
| No calibration | 0 | 直接使用模型输出 |
| 3-point | 3 | 仿射变换 |
| 5-point | 5 | 仿射变换 |
| 9-point (当前) | 9 | 仿射变换 |
| 9-point + polynomial | 9 | 二次多项式拟合 |

#### Exp-4: Latency Analysis

- 逐模块计时（已实现于 `TrackerResult.timings`）
- 不同硬件平台对比（Intel Ultra 5 vs 其他 CPU）
- ONNX vs PyTorch 推理速度对比
- batch=1 vs batch=2 的吞吐量差异

#### Exp-5: Ablation Studies

| 消融项 | 变体 |
|--------|------|
| Head Pose | 有 vs 无头部姿态估计 |
| Distance Estimation | 固定距离 vs ArUco 距离估计 |
| Eye Selection | 左眼 vs 右眼 vs 双眼平均 |
| Smoothing | α=0.1, 0.3, 0.5, 0.7, 1.0 |
| Input Resolution | 64×64 vs 128×128 vs 256×256 |
| Channel Width | [16,32,64,128] vs [32,64,128,256] |

---

## 9 Research Positioning

### 9.1 研究领域定位

Look2Act 位于以下研究领域的交叉点：

```
Appearance-Based Gaze Estimation
        │
        ├── Model-based (3D eye model fitting)
        │
        └── Learning-based (CNN regression) ◄── Look2Act
                │
                ├── Person-specific (需要个人数据微调)
                │
                └── Person-independent ◄── Look2Act (+ calibration)

Gaze Interaction Systems
        │
        ├── Research prototypes (高精度，实验室环境)
        │
        └── Deployable systems ◄── Look2Act (低成本，实时，跨平台)

Real-Time Computer Vision
        │
        ├── GPU-dependent (CUDA, TensorRT)
        │
        └── CPU-deployable ◄── Look2Act (ONNX Runtime, IPEX)
```

### 9.2 与相关工作的对比

| 系统/方法 | 输入 | 输出 | 硬件要求 | 精度 | 实时性 |
|----------|------|------|---------|------|--------|
| **MPIIGaze** (Zhang et al., 2015) | 单眼图像 | 2D gaze | GPU | ~4.5° | 离线 |
| **Gaze360** (Kellnhofer et al., 2019) | 全脸图像 | 3D gaze | GPU | ~11.4° | 离线 |
| **UnityEyes** (Wood et al., 2016) | 合成眼部图像 | 3D gaze | GPU | ~9.9° | 离线 |
| **OpenGaze** (Palmero et al., 2021) | 全脸 + 眼部 | 2D screen | GPU | — | ~15 FPS |
| **CalibMe** (Santini et al., 2017) | 眼部图像 | 2D screen | 专用硬件 | <1° | 实时 |
| **GazeCapture** (Krafka et al., 2016) | 全脸 + 眼部 | 2D screen | GPU | ~2.5cm | 离线 |
| **Look2Act** (本系统) | 双眼裁剪 | 3D gaze | CPU only | ~7.3° | ~30 FPS |

### 9.3 Look2Act 的差异化定位

1. **CPU-only 实时部署**：不依赖 NVIDIA GPU，面向消费级笔记本
2. **3D 视线 + 几何投影**：不直接回归屏幕坐标，通过几何模型适配不同屏幕
3. **端到端系统**：从摄像头输入到光标控制的完整链路，非单一算法
4. **自采集数据训练**：使用配套工具 Gaze_Dataset_Collector 采集数据
5. **校准补偿**：通过 9 点仿射校准弥补模型精度不足

### 9.4 相关研究方向

- **Appearance-based gaze estimation**: MPIIGaze, GazeCapture, ETH-XGaze, RT-Gene
- **Gaze interaction / HCI**: Tobii eye tracking, eye-typing, gaze-contingent displays
- **Head pose estimation**: 6DoF head pose, face alignment, 3DMM
- **Lightweight CNN deployment**: MobileNet, EfficientNet, ONNX Runtime, TFLite
- **Calibration methods**: polynomial mapping, SVR calibration, few-shot calibration
- **Temporal smoothing**: Kalman filter, One Euro filter, EMA

---

## 10 Future Research Directions

基于当前系统的设计和实验结果，以下方向值得进一步探索：

### 10.1 Cross-Device Generalization

当前模型在特定设备上训练和测试。未来需要：
- 多设备数据采集（不同摄像头、不同屏幕尺寸）
- Domain adaptation 技术（减少设备间的分布差异）
- 设备无关的特征表示学习

### 10.2 Few-Shot Calibration

当前 9 点校准对用户负担较大。可探索：
- 3 点甚至 1 点快速校准
- 基于 meta-learning 的 few-shot 个性化
- 隐式校准（用户正常使用过程中自动校准）

### 10.3 Head Pose Robustness

当前系统对大角度头部转动的鲁棒性有限。可探索：
- 头部姿态作为模型输入特征（multi-modal fusion）
- 头部姿态补偿机制
- 大角度头部转动时的 graceful degradation

### 10.4 Gaze Smoothing 优化

当前使用简单 EMA。可探索：
- One Euro Filter（自适应截止频率）
- Kalman Filter（状态空间建模）
- 基于注视行为的自适应平滑（fixation vs saccade 检测）

### 10.5 HCI Interaction Experiments

系统层面的用户实验：
- Fitts' Law 实验：量化视线选择的吞吐量
- 视线打字速度和准确率
- 视线控制 vs 鼠标控制的任务完成时间对比
- 长时间使用的疲劳度评估
- 无障碍辅助场景的可用性测试

### 10.6 模型架构改进

- 引入注意力机制（SE-Net, CBAM）
- 多尺度特征融合
- 知识蒸馏（从大模型蒸馏到轻量模型）
- 全脸 + 双眼的多输入融合架构

### 10.7 数据集扩展

- 增加用户多样性（年龄、种族、眼镜/无眼镜）
- 增加环境多样性（光照、背景）
- 合成数据增强（UnityEyes 风格的合成眼部图像）
- 跨数据集评估（在 MPIIGaze、GazeCapture 上测试）

---

## 附录 A: 数据集结构

### 原始数据（Gaze_Dataset_Collector 采集）

```
dataset_raw/
├── {session_id}/
│   ├── meta.json          # 采集元数据（屏幕尺寸、相机参数等）
│   ├── labels.csv         # 每帧标签（target_x/y, head_yaw/pitch/roll, distance_proxy）
│   └── images/
│       └── *.jpg          # 隐私合成图（4 列：左眼|右眼|ArUco|骨架）
```

合成图布局（640×128）：
```
[左眼 128×128] [右眼 128×128] [ArUco marker] [骨架图]
```

### 预处理后数据

```
dataset_processed/
├── split_info.json        # 划分信息
├── train/
│   ├── labels.csv         # 标签（gaze_x/y/z, norm_target_x/y, session_id, user_id）
│   └── images/
│       └── {session}_{frame}_L.jpg  # 左眼裁剪 128×128
├── val/
│   └── ...
└── test/
    └── ...
```

### 当前数据规模

| 划分 | Session 数 | 样本数 | 用户数 | 设备数 |
|------|-----------|--------|--------|--------|
| Train | 7 | 1545 | 3 | 3 |
| Val | 1 | 201 | 1 | 1 |
| Test | 2 | 447 | 2 | 2 |
| **Total** | **10** | **2193** | **3** | **3** |

---

## 附录 B: 关键配置参数

### 系统运行配置 (`configs/system_config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| camera.width | 1280 | 摄像头分辨率宽度 |
| camera.height | 720 | 摄像头分辨率高度 |
| face_detection.eye_crop_size | 128 | 眼部裁剪尺寸 |
| model.use_onnx | true | 使用 ONNX 推理 |
| geometry.screen_w_mm | 344.0 | 屏幕物理宽度 (mm) |
| geometry.screen_h_mm | 194.0 | 屏幕物理高度 (mm) |
| calibration.num_points | 9 | 校准点数量 |
| calibration.max_residual_px | 50.0 | 残差阈值 (px) |
| smoother.alpha | 0.3 | EMA 平滑系数 |
| tracker.target_fps | 30 | 目标帧率 |

### 训练配置 (`configs/train_config.yaml`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model.channels | [32,64,128,256] | 卷积通道数 |
| model.input_size | 128 | 输入图像尺寸 |
| model.output_dim | 3 | 输出维度 |
| training.batch_size | 64 | 批大小 |
| training.learning_rate | 0.001 | 学习率 |
| training.epochs | 100 | 训练轮数 |
| training.device | xpu | 训练设备 |
| data.train_split | 0.7 | 训练集比例 |
| data.val_split | 0.15 | 验证集比例 |
| data.test_split | 0.15 | 测试集比例 |
