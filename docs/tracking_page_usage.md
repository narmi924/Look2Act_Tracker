# 实时追踪页面使用说明

## 概述

实时追踪页面（`TrackingPage`）提供了视线追踪的核心功能，包括：
- 启动/停止实时追踪
- 在屏幕上显示半透明注视点光标
- 显示实时 FPS 和各阶段延迟信息
- 支持加载和应用校准参数

## 主要组件

### 1. TrackingPage（追踪控制页面）

主要功能：
- 控制追踪的启动和停止
- 显示性能监控信息（FPS、各阶段延迟）
- 显示追踪状态（人脸检测、视线有效性、校准状态）
- 加载校准参数

### 2. GazeCursorOverlay（注视点光标覆盖层）

主要功能：
- 全屏半透明窗口，显示注视点光标
- 鼠标穿透（不影响其他应用的交互）
- 实时更新光标位置

## 使用流程

### 基本使用

1. **启动追踪**
   - 点击"启动追踪"按钮
   - 系统会自动初始化 TrackerPipeline
   - 默认使用独立验证/交互窗口；系统级注视点覆盖层需要手动开启

2. **查看性能信息**
   - FPS：实时帧率
   - 各阶段延迟：人脸检测、头部姿态、视线回归等各模块的耗时

3. **查看追踪状态**
   - 人脸检测：是否检测到人脸
   - 视线有效性：视线数据是否有效
   - 校准状态：是否已加载校准参数

4. **停止追踪**
   - 点击"停止追踪"按钮
   - 注视点光标会隐藏
   - TrackerPipeline 保持运行（可快速重启）

### 使用校准

1. **加载校准参数**
   - 点击"加载校准"按钮
   - classic 后端会从 `calibration_classic.json` 加载校准参数
   - deep 后端会从 `calibration_deep.json` 加载校准参数
   - 校准状态会显示残差信息

2. **校准效果**
   - 加载校准后，注视点光标会自动应用校准变换
   - 提高注视点的精度

## 配置

### 系统配置文件

路径：`configs/system_config.yaml`

主要配置项：
```yaml
camera:
  index: 0                    # 摄像头索引
  width: 640                  # 摄像头分辨率宽度
  height: 480                 # 摄像头分辨率高度
  backend: "dshow"            # 摄像头后端（Windows 使用 dshow）

model:
  checkpoint_path: "checkpoints/best_model.pth"  # 模型权重路径
  use_ipex: false             # 是否使用 IPEX 优化
  use_onnx: false             # 是否使用 ONNX Runtime
  deep_gaze_space: "head"     # deep 原链路；camera 为坐标系实验

geometry:
  screen_w_mm: 344.0          # 屏幕物理宽度（毫米）
  screen_h_mm: 194.0          # 屏幕物理高度（毫米）

calibration:
  num_points: 25
  save_path: "calibration_classic.json"
  max_residual_px: 300.0

smoother:
  type: "kalman"              # kalman、ema 或 none
  alpha: 0.3                  # 平滑系数（0-1，越小越平滑）

tracker:
  backend: "classic"          # classic 体验模式；deep 研究模型
  target_fps: 30              # 目标帧率
```

### 光标样式配置

在 `GazeCursorOverlay` 类中可以修改：
```python
self.cursor_radius = 15                      # 光标半径（像素）
self.cursor_color = QColor(255, 0, 0, 150)   # 光标颜色（RGBA）
self.cursor_border_color = QColor(255, 255, 255, 200)  # 边框颜色
self.cursor_border_width = 2                 # 边框宽度
```

## 性能监控

### 诊断模式

- 追踪页点击"诊断"可查看 backend、模型版本、ONNX 输入、raw/calibrated/pre-clamp/clamped 坐标、校准方法、校准点数量和各阶段耗时。
- 不打开完整 GUI 时，可运行：

```powershell
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend classic --frames 300
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend deep --frames 300 --csv diagnostics_deep.csv
```

### FPS 显示

- 实时显示当前帧率
- 基于最近 30 帧的平均时间计算

### 各阶段延迟

显示以下模块的耗时（毫秒）：
1. **人脸检测**（Face Detection）：检测人脸和关键点
2. **头部姿态**（Head Pose）：估计头部姿态
3. **视线回归**（Gaze Regression）：CNN 模型推理
4. **坐标转换**（Coordinate Transform）：坐标系转换
5. **射线求交**（Ray-Plane Intersect）：计算视线与屏幕交点
6. **平滑滤波**（Smoothing）：时序平滑

## 错误处理

### 常见错误

1. **摄像头打开失败**
   - 检查摄像头是否被其他应用占用
   - 检查摄像头索引是否正确

2. **模型文件不存在**
   - 检查 `configs/system_config.yaml` 中的 `checkpoint_path`
   - 确保模型文件存在

3. **未检测到人脸**
   - 调整光照条件
   - 确保人脸在摄像头视野内
   - 保持适当的距离（30-60cm）

4. **校准文件不存在**
   - 先在校准页面进行校准并保存
   - 确保当前后端对应的 `calibration_classic.json` 或 `calibration_deep.json` 文件存在

## 集成示例

### 在主窗口中使用

```python
from PyQt6.QtWidgets import QMainWindow, QStackedWidget
from src.ui.tracking_page import TrackingPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 创建页面堆栈
        self.stack = QStackedWidget()
        
        # 创建追踪页面
        self.tracking_page = TrackingPage()
        self.stack.addWidget(self.tracking_page)
        
        self.setCentralWidget(self.stack)
```

### 独立使用

```python
from PyQt6.QtWidgets import QApplication
from src.ui.tracking_page import TrackingPage

app = QApplication(sys.argv)
page = TrackingPage()
page.show()
app.exec()
```

## 注意事项

1. **资源释放**
   - 关闭窗口时会自动停止追踪并释放资源
   - TrackerPipeline 会自动释放摄像头

2. **线程安全**
   - TrackerPipeline 在独立线程中运行
   - 通过定时器（约 30 FPS）从主线程获取结果

3. **性能优化**
   - 使用 IPEX 或 ONNX Runtime 可以提高推理速度
   - 调整 `update_interval_ms` 可以控制 UI 更新频率

4. **校准建议**
   - 建议在使用前进行校准以提高精度
   - 校准参数会自动应用到注视点光标
   - 更换摄像头或屏幕后需要重新校准
