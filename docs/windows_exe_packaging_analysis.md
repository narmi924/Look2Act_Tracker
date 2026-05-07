# Windows EXE 打包分析

本文评估如何把 Look2Act Tracker 打包成可双击启动的 Windows 应用程序，也就是类似 EyeTouch 的 `.exe` 打开方式。

## 推荐方案

优先使用 PyInstaller 的 `onedir` 目录模式生成 `Look2Act.exe`，确认稳定后再用 Inno Setup 封装安装包。

不建议一开始直接做单文件 EXE。Look2Act 当前依赖 PyQt6、QFluentWidgets、OpenCV、MediaPipe、ONNX Runtime、模型文件、YAML 配置和运行时校准文件。单文件模式会把资源临时解压到系统目录，启动更慢，也更容易遇到 native DLL、相对路径、可写配置文件失效等问题。

## 建议流程

1. 先冻结 GUI 入口：

```bash
conda activate gaze-env
pyinstaller --clean --noconfirm Look2Act.spec
```

2. 把运行所需资源一起放进发行目录：

```text
configs/
checkpoints/gaze_net.onnx
checkpoints/gaze_pog_zero.onnx
README.md
readme-images/
```

3. 保证用户运行时需要写入的文件不要放进不可写的内部压缩区：

```text
calibration_classic.json
calibration_deep.json
calibration_deep_pog.json
configs/classic.yaml
configs/deep.yaml
```

4. 必须从最终发行目录测试，而不是从源码目录测试：

```bash
dist/Look2Act/Look2Act.exe
```

5. 只有当 `dist/Look2Act/Look2Act.exe` 能稳定打开、预览摄像头、保存校准、启动追踪后，再制作安装包。

## PyInstaller 配置要点

- 入口脚本：`main.py`
- 应用名称：`Look2Act`
- 窗口模式：稳定后使用 `--windowed`，调试期可以先保留 console
- 需要包含的数据：
  - `configs`
  - 必需的 ONNX checkpoints
  - `README.md`
  - `readme-images`
- 可能需要声明的 hidden imports：
  - PyQt6 相关模块
  - `qfluentwidgets`
  - `cv2`
  - `mediapipe`
  - `onnxruntime`
  - `yaml`
- 应尽量排除的内容：
  - 训练专用依赖
  - 测试、论文、日志、数据集等非运行时文件

## 依赖裁剪建议

当前演示目标是 Classic + Deep ONNX Runtime 推理，不在 EXE 中训练模型。因此依赖应按运行时链路分层处理。

### 必须打包

| 依赖 | 用途 | 说明 |
| --- | --- | --- |
| `PyQt6` | 主界面、全屏校准、验证、交互窗口 | 必须带，不能删 |
| `pyqt6-fluent-widgets` | Fluent UI 控件与导航 | 必须带，不能删 |
| `opencv-python` / `cv2` | 摄像头、图像处理、classic pupil、Kalman、眼部 crop | 必须带，不能删 |
| `mediapipe` | FaceMesh 人脸关键点和眼部区域 | 必须带，不能删 |
| `numpy` | 几何、校准、模型输入预处理 | 必须带，不能删 |
| `PyYAML` | 读取和写入 YAML 配置 | 必须带，不能删 |
| `onnxruntime` | Deep Demo 的 CPU 推理 | Deep 模式必须带 |

### 可以先不打包或显式排除

| 依赖/目录 | 理由 | 风险 |
| --- | --- | --- |
| `pytest` | 只用于测试 | 无运行时风险 |
| `hypothesis` | 只用于测试 | 无运行时风险 |
| `matplotlib` | 只用于实验图表，不在 GUI 演示链路中使用 | 不影响演示 |
| `pandas` | 主要用于数据处理、训练、诊断脚本 | 不打包 `scripts/` 和 `src/data/` 时可排除 |
| `scipy` | 当前 `src/` 运行时未直接使用 | 可先排除，若某个间接依赖要求再补 |
| `scikit-learn` | 当前 `src/` 运行时未直接使用 | 可先排除 |
| `onnx` | 只用于导出/检查模型，运行 ONNX 推理只需要 `onnxruntime` | 可先排除 |
| `torchvision` / `torchaudio` | 训练生态依赖，运行时未使用 | 可排除 |
| `intel_extension_for_pytorch` | 训练/IPEX 优化，EXE 演示不应启用 | 必须排除，否则体积很大 |
| `dataset_raw` / `dataset_processed` | 数据集，不属于应用运行时 | 必须排除 |
| `paper-acm` / `evaluation_results` / `logs` | 论文、结果和日志材料 | 必须排除 |
| `tests` / `tools` / `scripts` | 测试、维护、训练和诊断脚本 | 应从发行包排除 |

### 需要重点处理的体积风险

`torch` 是当前最大风险。虽然 demo YAML 使用 `use_onnx: true`，但 `src/tracker/pipeline.py` 顶层仍然 import `torch`，并且顶层 import `src.models.gaze_net`，后者也会 import `torch`。这会让 PyInstaller 很可能把 PyTorch 打进 EXE，体积会暴涨。

建议在正式打包前做一次小重构：

1. 移除 `src/tracker/pipeline.py` 顶层的 `import torch`。
2. 移除顶层 `from src.models.gaze_net import GazeNet, GazeNetPoG, GazeNetV2`。
3. 只在 `use_onnx == false` 的 PyTorch 分支里局部 import `torch` 和 GazeNet。
4. ONNX 分支中的输入 tensor 预处理改成纯 `numpy`，不要为了 `.permute()` / `.float()` 引入 torch。
5. 打包规格中显式 `excludes=['torch', 'torchvision', 'torchaudio', 'intel_extension_for_pytorch', ...]`，再用 Classic 和 Deep ONNX smoke test 验证。

这样功能不删减：源码仍可保留 PyTorch checkpoint 推理和训练能力，但演示 EXE 默认走 ONNX Runtime，不需要把 PyTorch 运行库打进去。

## 主要风险与处理方式

| 风险点 | 问题 | 建议处理 |
| --- | --- | --- |
| 工作目录 | 打包后相对路径可能指向错误位置 | 增加 runtime path helper，区分源码运行和 frozen 运行 |
| 配置写入 | 启动弹窗会写入语言和全屏模式 | 源码/发行目录中的 `configs/classic.yaml`、`configs/deep.yaml` 只作为模板，用户配置写入 `%APPDATA%/Look2Act/configs/` |
| 校准文件 | 校准结果需要持久保存 | 后续建议同样迁移到 `%APPDATA%/Look2Act`，避免 Program Files 等目录权限问题 |
| native DLL | OpenCV、MediaPipe、ONNX Runtime、Qt 都有大量 DLL | 先用 `onedir`，根据 PyInstaller warning log 补 hidden imports |
| 启动速度 | 大型依赖会拖慢启动 | 保持 `main.py` 的延迟导入；稳定前避免单文件模式 |
| 发行体积 | PyQt6 + OpenCV + MediaPipe + ONNX Runtime 体积较大 | 只打包运行时必要模型和依赖，排除训练/数据集文件 |
| 摄像头兼容 | 不同 Windows 机器的摄像头 backend 表现不同 | 默认保持 `dshow`，在干净 Windows 环境做实际测试 |
| 授权风险 | PyQt6 分发涉及 GPL/商业授权选择 | 对外发布前确认项目开源协议和依赖授权 |

## 建议里程碑

1. 新增 `Look2Act.spec` 和 `build_exe.bat`，先生成 `dist/Look2Act/Look2Act.exe`。
2. 增加运行时路径处理，让 configs、checkpoints、calibration 文件在源码运行和 EXE 运行时都能正确定位。
3. 做冻结版 smoke test：
   - 启动弹窗能打开
   - 关闭按钮能完全退出
   - Classic / Deep 模式能切换
   - 主窗口默认无任务栏全屏
   - 摄像头预览能启动
   - 校准文件能保存
   - 追踪页能加载所选 backend
4. `onedir` 版本稳定后，再写 Inno Setup 脚本生成安装包。
5. 最后再评估是否需要单文件 EXE。

## 当前结论

当前最稳妥的产品化路径是先复刻 EyeTouch 的目录式发行：`Look2Act.exe` 加 `_internal`、`configs`、`checkpoints` 和校准文件。这样最容易定位问题，也最适合当前演示阶段。单文件 EXE 可以作为后续优化目标，而不是第一阶段目标。
