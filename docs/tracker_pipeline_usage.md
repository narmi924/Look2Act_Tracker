# TrackerPipeline 使用指南

## 概述

`TrackerPipeline` 是 Look2Act Tracker 的核心实时推理管道，串联了完整的视线追踪流程：

```
摄像头 → 人脸检测 → 眼部裁剪 → 视线回归 → 头部姿态估计 
→ 坐标转换 → 屏幕几何建模 → 时序平滑 → 屏幕注视点
```

## 快速开始

### 1. 基本使用

```python
from tracker.pipeline import SystemConfig, TrackerPipeline

# 加载配置
config = SystemConfig.from_yaml("configs/system_config.yaml")

# 创建管道
pipeline = TrackerPipeline(
    model_path="checkpoints/best_model.pth",
    config=config,
)

# 启动推理线程
if pipeline.start():
    # 获取实时结果
    while True:
        result = pipeline.get_latest_result()
        if result and result.valid:
            px, py = result.gaze_point
            print(f"注视点: ({px:.0f}, {py:.0f})")
        
        # ... 你的业务逻辑 ...

# 停止管道
pipeline.stop()
```

### 2. 带错误回调

```python
def on_error(message: str):
    print(f"[错误] {message}")
    # 通知 UI 层显示错误提示

pipeline = TrackerPipeline(
    model_path="checkpoints/best_model.pth",
    config=config,
    error_callback=on_error,
)
```

### 3. 运行演示脚本

```bash
conda run -n gaze-env python scripts/demo_tracker.py
```

按 `q` 键退出演示。

## 配置说明

### SystemConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `camera_index` | int | 0 | 摄像头索引 |
| `camera_width` | int | 640 | 摄像头分辨率宽度 |
| `camera_height` | int | 480 | 摄像头分辨率高度 |
| `camera_backend` | str | "dshow" | Windows 推荐使用 DirectShow |
| `eye_crop_size` | int | 128 | 眼部裁剪尺寸 |
| `checkpoint_path` | str | - | 模型权重路径 |
| `use_ipex` | bool | False | 是否启用 IPEX 优化 |
| `screen_w_mm` | float | 344.0 | 屏幕物理宽度（毫米） |
| `screen_h_mm` | float | 194.0 | 屏幕物理高度（毫米） |
| `smoother_alpha` | float | 0.3 | EMA 平滑系数（越小越平滑） |

### 从 YAML 加载配置

```yaml
# configs/system_config.yaml
camera:
  index: 0
  width: 640
  height: 480

model:
  checkpoint_path: "checkpoints/best_model.pth"
  use_ipex: true

smoother:
  alpha: 0.3
```

```python
config = SystemConfig.from_yaml("configs/system_config.yaml")
```

## TrackerResult 结构

```python
@dataclass
class TrackerResult:
    gaze_point: Optional[tuple[float, float]]  # 屏幕像素坐标
    valid: bool                                # 结果是否有效
    fps: float                                 # 实时 FPS
    timings: dict[str, float]                  # 各阶段耗时（毫秒）
    error_message: Optional[str]               # 错误信息
    face_detected: bool                        # 是否检测到人脸
```

### 性能指标

`timings` 字典包含各阶段耗时：

- `face_detection`: 人脸检测耗时
- `head_pose`: 头部姿态估计耗时
- `gaze_regression`: 视线回归耗时
- `coordinate_transform`: 坐标转换耗时
- `ray_plane_intersect`: 射线-平面求交耗时
- `smoothing`: 时序平滑耗时

## 错误处理与容错

### 1. 摄像头错误

**场景**: 摄像头帧获取失败

**处理策略**:
- 自动重试最多 3 次
- 失败后标记摄像头断开
- 通过 `error_callback` 通知 UI 层

**示例**:
```python
def on_error(message: str):
    if "摄像头断开" in message:
        # 显示 UI 提示：请检查摄像头连接
        show_camera_error_dialog()
```

### 2. 未检测到人脸

**场景**: 当前帧未检测到人脸

**处理策略**:
- 返回上一帧的有效结果（如果存在）
- `result.face_detected = False`
- `result.error_message = "未检测到人脸"`

**UI 建议**: 显示"请保持人脸在画面中"提示

### 3. 视线无效

**场景**: 视线与屏幕平面无交点

**处理策略**:
- 自动 clamp 到屏幕边缘（`clamp_to_screen=True`）
- 如果仍无交点，返回上一帧结果

### 4. 模型推理异常

**场景**: 视线回归过程中发生异常

**处理策略**:
- 捕获异常并记录日志
- 返回上一帧有效结果
- 通过 `error_callback` 通知 UI

## 线程安全

`TrackerPipeline` 在独立线程中运行推理循环，通过以下方法线程安全地访问结果：

```python
# 获取最新推理结果（线程安全）
result = pipeline.get_latest_result()

# 获取最新摄像头帧（线程安全，返回副本）
frame = pipeline.get_latest_frame()

# 检查管道是否运行中
if pipeline.is_running():
    # ...
```

## 性能优化

### 1. 启用 IPEX 优化

```python
config = SystemConfig(use_ipex=True)
```

需要安装 `intel-extension-for-pytorch`。

### 2. 调整平滑参数

```python
config = SystemConfig(smoother_alpha=0.5)  # 更快响应
config = SystemConfig(smoother_alpha=0.1)  # 更平滑
```

### 3. 降低摄像头分辨率

```python
config = SystemConfig(
    camera_width=320,
    camera_height=240,
)
```

降低分辨率可减少人脸检测耗时，但可能影响精度。

## 常见问题

### Q: 如何获取实时 FPS？

```python
result = pipeline.get_latest_result()
if result:
    print(f"FPS: {result.fps:.1f}")
```

### Q: 如何查看各阶段耗时？

```python
result = pipeline.get_latest_result()
if result:
    for stage, time_ms in result.timings.items():
        print(f"{stage}: {time_ms:.2f}ms")
```

### Q: 如何处理摄像头断开？

```python
def on_error(message: str):
    if "摄像头断开" in message:
        # 尝试重新初始化
        pipeline.stop()
        time.sleep(1)
        pipeline.start()
```

### Q: 如何自定义屏幕几何参数？

修改 `configs/system_config.yaml` 中的 `geometry` 部分：

```yaml
geometry:
  screen_w_mm: 344.0  # 实际测量值
  screen_h_mm: 194.0  # 实际测量值
```

## 下一步

- 查看 `scripts/demo_tracker.py` 了解完整示例
- 阅读 `tests/test_tracker_pipeline.py` 了解测试用例
- 参考设计文档了解架构细节
