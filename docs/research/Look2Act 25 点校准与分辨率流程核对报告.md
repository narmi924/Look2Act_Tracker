# Look2Act 25 点校准与分辨率流程核对报告

## 1. 最终校准配置核对

| 项目 | Classic | Deep | 来源文件/函数 |
|---|---|---|---|
| `calibration.num_points` | 25 | 25 | `configs/classic.yaml`；`configs/deep.yaml` |
| effective method | `polynomial` | `polynomial`，因为显式配置了 25 点；若 Deep 未配置点数，代码 fallback 才是 9 点 `affine` | `src/tracker/pipeline.py:SystemConfig.effective_calibration_num_points/effective_calibration_method` |
| grid size | 5×5 | 5×5 | `src/ui/calibration_page.py:_generate_calibration_points`，`num_points >= 25` 时 `grid_size = 5` |
| save path | 运行期解析到 `%APPDATA%\Look2Act\calibration\calibration_classic.json` | 运行期解析到 `%APPDATA%\Look2Act\calibration\calibration_deep.json` | `configs/*.yaml`；`SystemConfig.from_yaml`；`src/runtime_paths.py:calibration_path_for_backend` |

结论：当前 Classic 与 Deep 的最终配置均为 **25 点 polynomial**。论文不能把 9 点 affine 写成最终系统配置。9 点 affine 只应作为少点校准对比实验或 Deep 未显式配置点数时的 fallback 说明。

## 2. 25 点校准数据是否存在

当前检查结果：

| 文件/数据 | 结果 |
|---|---|
| `%APPDATA%\Look2Act\calibration\calibration_classic.json` | 存在，25 点，`polynomial`，平均拟合残差 73.58 px |
| `%APPDATA%\Look2Act\calibration\calibration_deep.json` | 未发现 |
| 项目根目录 `calibration_classic.json` / `calibration_deep.json` | 未发现 |
| `bin/calibration_classic.json` | 存在，25 点，`polynomial`，平均拟合残差 93.26 px；属于非默认运行期路径样本 |
| `calibration.json` | 存在，但为旧 9 点 `polynomial`，不应作为最终系统配置 |
| 25 点全屏验证误差记录 | 未发现可直接引用的持久化记录 |
| UI/log 中 residual | UI 显示平均残差；JSON 保存 `residual_mean_px`；未持久化每点有效样本数 |
| max residual | JSON 不直接保存，但可由保存的点和矩阵重新计算 |

可用于论文的真实数据表建议如下：

| 模式 | 屏幕分辨率 | 摄像头分辨率 | 校准点数 | 映射方式 | 有效校准点数 | 平均误差/残差 | 最大误差 | 备注 |
|---|---:|---:|---:|---|---:|---:|---:|---|
| Classic | JSON 未记录；目标点坐标符合 1440×900、10% 边距 | JSON 未记录；当前配置请求 1280×720 | 25 | polynomial | 25 | 73.58 px | 212.54 px | 运行期默认路径 `%APPDATA%` 中的 25 点校准拟合残差；不是 hold-out 泛化误差 |
| Deep | 无真实校准文件 | 当前配置请求 1280×720 | 25 | polynomial | 无记录 | 无记录 | 无记录 | 不能填写数值 |
| Classic（非默认路径样本） | JSON 未记录；目标点坐标符合 1440×900、10% 边距 | JSON 未记录 | 25 | polynomial | 25 | 93.26 px | 203.90 px | `bin/calibration_classic.json`，可作为历史样本说明，不建议作为主结果 |

可运行脚本建议，用于汇总现有校准 JSON：

```python
import json, math, os
from pathlib import Path

paths = [
    Path(os.environ["APPDATA"]) / "Look2Act/calibration/calibration_classic.json",
    Path(os.environ["APPDATA"]) / "Look2Act/calibration/calibration_deep.json",
    Path("bin/calibration_classic.json"),
    Path("calibration.json"),
]

for path in paths:
    if not path.exists():
        print(path, "MISSING")
        continue

    data = json.loads(path.read_text(encoding="utf-8"))
    matrix = data.get("transform_matrix") or data.get("affine_matrix")
    points = data.get("calibration_points", [])
    mean = data.get("norm_mean", [0, 0])
    std = data.get("norm_std", [1, 1])
    method = data.get("method")

    residuals = []
    if matrix:
        for pt in points:
            x, y = pt["raw"]
            tx, ty = pt["target"]
            xn = (x - mean[0]) / std[0]
            yn = (y - mean[1]) / std[1]
            feat = [xn, yn, xn * yn, xn * xn, yn * yn, 1.0] if method == "polynomial" else [xn, yn, 1.0]
            px = sum(matrix[0][i] * feat[i] for i in range(len(feat)))
            py = sum(matrix[1][i] * feat[i] for i in range(len(feat)))
            residuals.append(math.hypot(px - tx, py - ty))

    print({
        "path": str(path),
        "method": method,
        "points": len(points),
        "stored_mean_residual": data.get("residual_mean_px"),
        "computed_max_residual": max(residuals) if residuals else None,
        "screen_w": data.get("screen_w"),
        "screen_h": data.get("screen_h"),
    })
```

CSV 模板：

```csv
date,mode,screen_resolution,camera_requested_resolution,camera_actual_resolution,calibration_points,mapping_method,valid_calibration_points,mean_fit_residual_px,max_fit_residual_px,fullscreen_validation_mean_px,fullscreen_validation_max_px,fps,notes
2026-05-09,Classic,1440x900,1280x720,,25,polynomial,,,,,,, 
2026-05-09,Deep,1440x900,1280x720,,25,polynomial,,,,,,, 
```

论文安全占位表述：

> 当前系统代码已将 Classic 与 Deep 的实时校准配置统一为 25 点 polynomial，并能在完成校准后保存拟合残差和校准点对应关系。现有文件中仅保留了 Classic 模式的 25 点校准拟合残差记录，尚未发现 Deep 模式 25 点校准文件或独立全屏验证误差日志。因此，本文仅将该结果作为实时校准流程的实现验证，不将其表述为 25 点 polynomial 的 hold-out 泛化结果；Deep 模式和全屏验证误差需在人工实测后补充。

## 3. 9 点实验和 25 点最终系统如何区分

可直接替换第六章相关正文：

> 本节的校准映射实验用于分析少点校准条件下不同映射策略的稳定性，而不是替代最终软件中的实时校准配置。实验结果显示，在 9 点校准条件下，affine 映射能够将平均像素误差由 314.5 px 降低到 221.2 px，说明少量校准点已能修正部分系统性偏差；而 9 点 polynomial 的平均误差升高到 437.0 px，表明当校准点较少时，高阶映射容易受到采样噪声影响并出现过拟合。

> 最终 Look2Act 软件采用 5×5 共 25 点实时校准流程。与 9 点少点实验相比，25 点校准覆盖屏幕中心、边缘和角落区域，能够为 polynomial 映射提供更完整的空间约束。当前 Classic 与 Deep 配置文件均将校准点数设为 25 点，运行时有效校准方法为 polynomial；因此，9 点 affine 只能作为少点校准策略对比结果，不能作为最终系统配置描述。

> 需要说明的是，现有 25 点实时校准记录主要反映系统在实际运行中完成目标点采样、映射拟合和残差保存的能力。除已保存的校准拟合残差外，当前文件中尚未发现独立的 25 点 hold-out 泛化误差或完整全屏验证误差日志。因此，论文中应将 25 点结果表述为“最终实时校准配置与实现验证”，而不应写成已经完成 25 点 polynomial 的 hold-out 泛化实验。

## 4. 屏幕分辨率检测核对

| 代码位置 | 功能 | 是否影响论文写法 |
|---|---|---|
| `src/tracker/pipeline.py` | 初始化屏幕几何时通过 `QApplication.primaryScreen().geometry()` 获取主屏幕逻辑分辨率；无 Qt app 时用 tkinter 回退 | 可写“系统自动读取当前主屏幕逻辑分辨率用于屏幕几何映射” |
| `src/ui/calibration_page.py:_generate_calibration_points` | 通过 `QApplication.primaryScreen().geometry()` 生成校准目标点；失败时 fallback 为 1920×1080 | 可写“校准点基于主屏幕分辨率动态生成，失败时使用 1920×1080 回退值” |
| `src/ui/main_window.py:_apply_window_mode` | 全屏模式使用 `screen.geometry()` 设置主窗口大小；自适应模式使用 `availableGeometry()` | 可写“窗口可按主屏幕全屏或任务栏避让模式显示” |
| `src/ui/tracking_page.py` | 注视点 clamp 到 `primaryScreen().geometry()` 范围 | 可写“输出坐标最终限制在主屏幕像素范围内” |
| `src/ui/interaction_overlay.py` | 全屏验证参考点使用主屏幕 `geometry()`，失败时 1920×1080 | 可写“验证点按主屏幕生成；没有持久化验证误差日志” |
| 配置文件 | 没有把 1280×720 写成屏幕默认分辨率 | 论文不能写“屏幕默认 1280×720” |

结论：屏幕分辨率是自动读取主屏幕逻辑分辨率，不是固定 1280×720。代码主要使用主屏幕；多显示器场景下未见显式显示器选择逻辑。Qt 读取的是逻辑分辨率，Windows 缩放可能导致逻辑像素与物理像素不同，但该选择与 PyQt 绘制坐标一致。

## 5. 摄像头分辨率配置核对

| 项目 | 结论 |
|---|---|
| TrackerPipeline 摄像头输入 | 使用配置中的 `camera.width` / `camera.height` 调用 OpenCV `CAP_PROP_FRAME_WIDTH/HEIGHT`，随后读取实际帧宽高 |
| 当前 Classic/Deep 配置 | 均请求 1280×720 摄像头分辨率 |
| 缺省配置 | 若 YAML 未给出，`SystemConfig.from_yaml` 默认 640×480 |
| 摄像头预览页 | `CameraPage` 内部默认 1280×720，并用该分辨率启动预览流 |
| 设置页 | 提供常见分辨率列表，默认显示 1280×720；可检测摄像头实际支持分辨率并选择保存 |
| 影响 | 摄像头分辨率影响人脸检测、眼区裁剪输入质量、实际帧尺寸、头姿估计内参近似和 FPS |

安全结论：

- 屏幕分辨率：自动读取主屏幕逻辑分辨率；失败时部分模块回退 1920×1080；没有固定 1280×720 屏幕假设。
- 摄像头分辨率：配置文件和设置页可选择；当前 Classic/Deep 请求 1280×720；OpenCV 仍会返回设备实际打开的帧尺寸。
- 论文中应写：1280×720 是当前摄像头请求分辨率或预览默认分辨率，不是屏幕分辨率。

## 6. 使用流程补充建议

可直接放入第三章或第五章：

> 用户完成启动语言和运行模式选择后，首先进入摄像头预览与环境确认阶段。系统使用普通 RGB 摄像头采集实时画面，用户需要确认摄像头能够正常打开，人脸位于画面中央，人脸检测和眼区裁剪稳定，且当前光照、坐姿和屏幕观看距离适合进行校准。摄像头分辨率可由配置文件或设置页指定，系统会向摄像头请求相应分辨率，并在运行时读取实际帧尺寸用于后续检测和头姿估计。

> 在进入校准和追踪流程时，系统自动读取当前主屏幕的逻辑分辨率，并以该分辨率生成屏幕目标点、建立屏幕像素坐标范围和进行输出裁剪。最终校准阶段采用 5×5 共 25 个目标点，用户依次注视全屏目标点，系统采集原始注视输出与目标屏幕坐标之间的对应关系，并拟合 polynomial 校准映射。屏幕分辨率直接影响目标点坐标和像素误差计算，摄像头分辨率则主要影响图像检测质量、眼区裁剪稳定性和实时处理帧率，二者在论文中应分别说明。

## 7. 图表建议

| 图表 | 建议 |
|---|---|
| 使用流程图 | 加入“屏幕分辨率读取 / 摄像头分辨率确认”节点，位置放在摄像头预览之后、25 点校准之前 |
| 25 点校准表格 | 放在第六章 6.5.3；当前表6-5中的 Classic/Deep 数值应删除或改成“待实测”，除非后续补齐真实记录 |
| 设置页/摄像头预览截图 | 建议标注“摄像头请求分辨率/实际帧尺寸”，不要标注为屏幕分辨率 |
| 第六章校准对比图说明 | 图6-6应改为“少点校准策略对比”，明确 9 点 affine/polynomial 是离线 hold-out 对比，不是最终系统配置 |
| 全屏验证图 | 可新增一张验证界面图，但不要声称已有自动记录的全屏验证误差，除非后续增加日志或人工记录 |

## 8. 最终论文修改清单

| 类型 | 修改项 |
|---|---|
| 需补文字 | 第三章或第五章加入“分辨率检测与确认流程”；第六章 6.5.3 加入“25 点实时校准配置说明” |
| 需新增/修改表格 | 修改表6-5：删除当前无证据的 260/650/300/700 等数值；改为真实 Classic 25 点拟合残差，Deep 留空待测，或改为人工测试记录模板 |
| 需调整图题 | 图6-6改为“少点校准映射对比结果”；流程图加入分辨率确认节点 |
| 需删除旧说法 | 删除或改写“9 点 affine 是最终系统配置”的暗示；删除无真实来源的 25 点 Deep 误差和全屏验证误差数值 |
| 必须人工实测后填写 | Deep 25 点校准文件、Deep 平均/最大拟合残差、Classic/Deep 全屏验证平均误差、最大误差、实际摄像头帧尺寸、FPS |
| 可保留但需限定 | 9 点 affine / 9 点 polynomial 对比可保留，定位为“少点校准策略对比实验” |
| 必须区分 | 屏幕分辨率是主屏幕逻辑分辨率；1280×720 是当前摄像头请求/预览分辨率，不是屏幕默认分辨率 |