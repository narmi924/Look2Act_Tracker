# 设置页面使用说明

## 概述

设置页面（`src/ui/settings_page.py`）提供了 Look2Act Tracker 系统的配置界面，支持以下功能：

1. **摄像头设置**：配置摄像头索引、分辨率、后端
2. **模型设置**：配置模型路径、IPEX 优化、ONNX Runtime
3. **几何设置**：配置屏幕物理尺寸
4. **平滑设置**：调节视线平滑系数
5. **追踪设置**：配置目标帧率

## 主要功能

### 1. 摄像头设置

- **摄像头索引**：选择要使用的摄像头（0 表示默认摄像头）
- **分辨率**：选择摄像头分辨率（宽度和高度）
  - 常见选项：640x480, 1280x720, 1920x1080
- **后端**：选择摄像头后端
  - `dshow`：DirectShow（Windows 推荐）
  - `auto`：自动选择

### 2. 模型设置

- **模型权重路径**：PyTorch 模型文件路径（.pth 或 .pt）
  - 默认：`checkpoints/best_model.pth`
  - 可通过"浏览"按钮选择文件
- **IPEX 优化**：启用 Intel Extension for PyTorch 优化
  - 适用于 Intel CPU/GPU
  - 可能提升推理速度
- **ONNX Runtime**：使用 ONNX Runtime 进行推理
  - 推荐启用，通常比 PyTorch 更快
  - 需要先导出 ONNX 模型
- **ONNX 模型路径**：ONNX 模型文件路径（.onnx）
  - 默认：`checkpoints/gaze_net.onnx`

### 3. 几何设置

- **屏幕物理尺寸**：屏幕的实际物理尺寸（单位：毫米）
  - 需要使用尺子测量屏幕的宽度和高度（不含边框）
  - 准确的物理尺寸对视线映射精度至关重要
  - 示例：15.6 英寸笔记本屏幕约为 344mm × 194mm

### 4. 平滑设置

- **平滑系数 Alpha**：控制视线平滑程度（范围：0.01 ~ 1.00）
  - 较小的值（如 0.1）：更平滑，但响应较慢
  - 较大的值（如 0.8）：响应快，但可能抖动
  - 推荐值：0.3
  - 使用滑块实时调节

### 5. 追踪设置

- **目标帧率**：追踪系统的目标帧率（FPS）
  - 范围：10 ~ 60
  - 推荐值：30
  - 更高的帧率需要更强的计算性能

## 使用流程

### 基本使用

1. 打开设置页面
2. 根据需要修改各项设置
3. 点击"保存设置"按钮
4. 设置将保存到 `configs/system_config.yaml`
5. 如果 TrackerPipeline 正在运行，需要重启以应用新设置

### 恢复默认设置

1. 点击"恢复默认"按钮
2. 确认对话框中选择"是"
3. 所有设置将恢复为默认值（未保存）
4. 点击"保存设置"以持久化默认配置

### 配置文件位置

- 配置文件路径：`configs/system_config.yaml`
- 可以手动编辑此文件（YAML 格式）
- 设置页面会在启动时自动加载此文件

## 配置文件格式

```yaml
camera:
  index: 0
  width: 640
  height: 480
  backend: dshow

face_detection:
  eye_crop_size: 128
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5

model:
  checkpoint_path: checkpoints/best_model.pth
  use_ipex: false
  use_onnx: true
  onnx_path: checkpoints/gaze_net.onnx

geometry:
  screen_w_mm: 344.0
  screen_h_mm: 194.0

calibration:
  num_points: 9
  save_path: calibration.json
  max_residual_px: 50.0

smoother:
  alpha: 0.3

tracker:
  target_fps: 30
  timer_interval_ms: 33
```

## 注意事项

1. **屏幕尺寸测量**：
   - 使用尺子测量屏幕的实际显示区域（不含边框）
   - 单位为毫米（mm）
   - 测量精度直接影响视线估计精度

2. **模型文件路径**：
   - 确保模型文件存在于指定路径
   - PyTorch 模型和 ONNX 模型需要分别指定

3. **ONNX Runtime**：
   - 如果启用 ONNX Runtime，需要先导出 ONNX 模型
   - 导出命令：`python scripts/export_onnx.py`

4. **设置生效**：
   - 大部分设置需要重启 TrackerPipeline 才能生效
   - 平滑系数可以实时生效（如果 TrackerPipeline 支持动态更新）

## 集成到主窗口

在主窗口中集成设置页面：

```python
from src.ui.settings_page import SettingsPage

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 创建设置页面
        self.settings_page = SettingsPage()
        
        # 连接配置变更信号
        self.settings_page.config_changed.connect(self._on_config_changed)
        
        # 添加到导航栏
        self.addSubInterface(self.settings_page, "设置", "Settings")
    
    def _on_config_changed(self, config):
        """配置变更回调。"""
        print(f"配置已更新: {config}")
        # 可以在这里通知 TrackerPipeline 重新加载配置
```

## API 参考

### SettingsPage 类

#### 信号

- `config_changed(SystemConfig)`: 配置变更信号，携带新的配置对象

#### 方法

- `get_config() -> SystemConfig`: 获取当前配置对象

#### 示例

```python
# 创建设置页面
settings_page = SettingsPage()

# 获取当前配置
config = settings_page.get_config()
print(f"摄像头索引: {config.camera_index}")
print(f"平滑系数: {config.smoother_alpha}")

# 监听配置变更
def on_config_changed(config):
    print(f"配置已变更: {config}")

settings_page.config_changed.connect(on_config_changed)
```

## 测试

运行设置页面测试：

```bash
# 单独测试设置页面 UI
conda run -n gaze-env python test_settings_page.py

# 运行单元测试（配置管理逻辑）
conda run -n gaze-env pytest tests/test_settings_page.py -v
```

## 故障排除

### 问题：配置文件加载失败

**原因**：配置文件格式错误或不存在

**解决方案**：
1. 检查 `configs/system_config.yaml` 是否存在
2. 验证 YAML 格式是否正确
3. 点击"恢复默认"并保存以重新生成配置文件

### 问题：设置保存后不生效

**原因**：TrackerPipeline 未重启

**解决方案**：
1. 停止当前的追踪任务
2. 重新启动追踪任务
3. 新配置将在重启后生效

### 问题：ONNX 模型路径无效

**原因**：ONNX 模型文件不存在

**解决方案**：
1. 确保已导出 ONNX 模型：`python scripts/export_onnx.py`
2. 检查 ONNX 文件路径是否正确
3. 或者禁用 ONNX Runtime，使用 PyTorch 推理
