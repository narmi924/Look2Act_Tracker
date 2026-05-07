# 附录

## 附录 A Look2Act 系统核心程序清单

本附录仅列出 Look2Act Tracker 中与论文正文实现说明直接相关的核心程序片段。为便于论文阅读，程序清单对界面样式、日志输出、异常提示和重复性代码进行了适度省略。

### A.1 程序启动与模式选择

**程序清单 A-1 程序启动、配置校验与模式选择**

文件路径：`main.py`

功能说明：该片段展示程序启动时对配置文件进行解析和校验，并在 Qt 应用初始化后弹出语言与运行模式选择窗口，随后创建主窗口。

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Look2Act Tracker")
    parser.add_argument("--config", default="configs/classic.yaml")
    args, _ = parser.parse_known_args(argv)
    return args


def validate_config_path(config_path: Path) -> tuple[bool, str]:
    if not config_path.exists():
        return False, f"配置文件不存在：{config_path}"
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        return False, f"配置文件读取失败：{config_path}\n{e}"
    required_sections = {"camera", "model", "tracker"}
    missing = sorted(s for s in required_sections if s not in data)
    if missing:
        return False, f"配置文件不完整：{', '.join(missing)}"
    return True, ""


def main() -> int:
    args = parse_args(sys.argv[1:])
    config_path = resource_path(args.config)
    validate_config_path(config_path)
    app = setup_application()
    from src.ui.i18n import load_language
    from src.ui.language_dialog import LanguageSelectionDialog
    load_language(config_path)

    dialog = LanguageSelectionDialog(config_path=config_path)
    if dialog.exec() != dialog.DialogCode.Accepted:
        return 0

    selected_config = dialog.selected_config_path
    ok, _ = validate_config_path(selected_config)
    if not ok:
        return 2

    from src.ui.main_window import MainWindow

    main_window = MainWindow(config_path=selected_config)
    main_window.show()
    return app.exec()
```

该程序清单支撑论文第三章中“系统运行流程”的说明，体现程序并非直接进入追踪，而是先完成配置解析、语言选择和 Classic/Deep 模式选择。它也支撑第五章“软件页面与使用流程实现”，说明主窗口是在用户完成启动选择后创建的。

### A.2 用户配置与本地运行路径管理

**程序清单 A-2 用户可写目录、配置复制与启动设置保存**

文件路径：`src/runtime_paths.py`，`src/ui/language_dialog.py`

功能说明：该片段展示系统如何区分程序资源目录和用户数据目录，并将用户配置写入当前用户可写位置。

```python
APP_NAME = "Look2Act"


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def app_data_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / "AppData" / "Roaming" / APP_NAME


def user_config_path(mode: str) -> Path:
    normalized = mode if mode in {"classic", "deep"} else "classic"
    return app_data_dir() / "configs" / f"{normalized}.yaml"


def ensure_user_config(mode: str) -> Path:
    normalized = mode if mode in MODE_TEMPLATE_CONFIGS else "classic"
    target_path = user_config_path_for_mode(normalized)
    if not target_path.exists():
        template_path = template_config_path_for_mode(normalized)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template_path, target_path)
    return target_path


def _save_startup_config(language: str, mode: str) -> Path:
    config_path = ensure_user_config(mode)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    ui = data.setdefault("ui", {})
    ui["language"] = language
    ui["window_mode"] = "fullscreen"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    set_language(language)
    return config_path
```

该程序清单支撑论文第三章“部署与交付设计”，说明软件运行期配置和校准结果不依赖安装目录写入，而是保存到用户可写的数据目录中。该设计使本地 EXE 运行时也能保存语言、模式和后续校准相关配置。

### A.3 主窗口页面组织

**程序清单 A-3 主窗口页面创建与导航**

文件路径：`src/ui/main_window.py`

功能说明：该片段展示主窗口如何组织主页、摄像头预览、校准、追踪和设置页面，并在进入校准或追踪前初始化追踪管线。

```python
class MainWindow(FluentWindow):
    def __init__(self, config_path: Path | str | None = None) -> None:
        super().__init__()
        self.config_path = Path(config_path) if config_path is not None else Path("configs/system_config.yaml")
        self.tracker: Optional[TrackerPipeline] = None
        self.tracker_config: Optional[SystemConfig] = None

        self.page_home = HomePage()
        self.page_camera = CameraPage()
        self.page_calibration = CalibrationPage()
        self.page_tracking = TrackingPage(config_path=self.config_path)
        self.page_settings = SettingsPage(config_path=self.config_path)

        self.addSubInterface(self.page_home, FIF.HOME, tx("主页", "Home"))
        self.addSubInterface(self.page_camera, FIF.PHOTO, tx("预览", "Preview"))
        self.addSubInterface(self.page_calibration, FIF.EDIT, tx("校准", "Calibrate"))
        self.addSubInterface(self.page_tracking, FIF.VIEW, tx("追踪", "Track"))
        self.addSubInterface(self.page_settings, FIF.SETTING, tx("设置", "Settings"))

        self.page_home.navigate_to_camera.connect(self.go_camera)
        self.page_home.navigate_to_calibration.connect(self.go_calibration)
        self.page_home.navigate_to_tracking.connect(self.go_tracking)
        self.page_home.navigate_to_settings.connect(self.go_settings)
        self.page_calibration.calibration_ready.connect(self._on_calibration_ready)

    def go_calibration(self) -> bool:
        if not self._ensure_tracker_initialized():
            return False
        if not self.tracker.is_running() and not self.tracker.start():
            return False
        self.page_calibration.set_tracker(self.tracker)
        self.switchTo(self.page_calibration)
        return True

    def go_tracking(self) -> None:
        if not self._ensure_tracker_initialized():
            return
        self.page_tracking.tracker = self.tracker
        self.page_tracking.tracker_config = self.tracker_config
        if self.page_calibration.calibrator.is_calibrated:
            self.page_tracking.set_calibrator(self.page_calibration.calibrator)
        self.switchTo(self.page_tracking)
```

该程序清单支撑论文第三章“软件架构设计”，体现 UI 层以主窗口为中心管理多个功能页面。它也支撑第五章“软件页面实现”，说明用户从主页进入预览、校准、追踪和设置的实际页面流程。

### A.4 TrackerPipeline 单帧实时处理流程

**程序清单 A-4 单帧实时处理核心流程**

文件路径：`src/tracker/pipeline.py`

功能说明：该片段节选 `TrackerPipeline.process_frame` 的核心逻辑，展示人脸检测、检测失败容错、Classic/Deep 分支选择、眼区输入、模型推理入口以及屏幕映射和平滑输出。

```python
def process_frame(self, frame_bgr: np.ndarray) -> TrackerResult:
    timings = {}
    backend = self.config.normalized_backend

    face_result = self.face_detector.detect(frame_bgr)
    if not face_result.detected:
        fallback = self._last_valid_result
        if fallback is not None:
            return TrackerResult(gaze_point=fallback.gaze_point, valid=True, fps=self._calculate_fps(),
                                 timings=timings, error_message="未检测到人脸，使用上一帧结果",
                                 face_detected=False, backend=backend)
        return TrackerResult(gaze_point=None, valid=False, fps=self._calculate_fps(), timings=timings,
                             error_message="未检测到人脸", face_detected=False, backend=backend)

    if backend == "classic":
        return self._process_classic_result(face_result, timings)

    head_pose = self.head_pose_estimator.estimate(face_result.pnp_points_2d)
    if not head_pose.valid:
        return TrackerResult(gaze_point=None, valid=False, fps=self._calculate_fps(), timings=timings,
                             error_message="头部姿态估计失败", face_detected=True, backend=backend)

    left_eye, right_eye = face_result.left_eye_crop, face_result.right_eye_crop
    left_eye, right_eye = self._prepare_deep_eye_inputs(left_eye, right_eye)
    left_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
    left_chw = np.ascontiguousarray(left_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
    right_chw = np.ascontiguousarray(right_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0

    head_pose_vec = np.zeros(3, dtype=np.float32)
    inputs = {"left_eye": left_chw[None], "right_eye": right_chw[None], "head_pose": head_pose_vec[None]}
    gaze_vector = self.onnx_session.run([self.onnx_output_name], inputs)[0][0]

    d = gaze_vector.astype(np.float64)
    d = d / np.linalg.norm(d)
    ray_origin, ray_direction = self._compute_deep_ray(d, head_pose)
    intersection = self.screen_geometry.ray_plane_intersect(ray_origin, ray_direction)
    raw_point = self.screen_geometry.world_to_screen_px(intersection, clamp=not self._calibration_mode)
    smoothed_point = raw_point if self._calibration_mode else self.smoother.update(raw_point)
    result = TrackerResult(gaze_point=smoothed_point, valid=True, fps=self._calculate_fps(),
                           timings=timings, error_message=None, face_detected=True,
                           raw_point=raw_point, backend="deep")
    self._last_valid_result = result
    return result
```

该程序清单支撑论文第三章“实时处理流程”，说明系统以单帧图像为单位完成检测、分支处理和结果输出。它也支撑第五章“实时视觉处理流程”，体现 Deep 模式从眼区图像到三维视线方向，再到屏幕坐标和平滑输出的主要链路。

### A.5 Classic 传统视觉后端

**程序清单 A-5 眼区暗色区域质心与双眼特征融合**

文件路径：`src/tracker/classic.py`，`src/tracker/pipeline.py`

功能说明：该片段展示 Classic 后端通过灰度化、阈值分割、形态学处理、轮廓矩和暗像素质心提取眼区特征，并融合左右眼特征作为校准输入。

```python
def detect_pupil_centroid(eye_roi: np.ndarray) -> Optional[tuple[float, float]]:
    if eye_roi is None or eye_roi.size == 0:
        return None

    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    k = max(3, int(min(eye_roi.shape[:2]) / 8) | 1)
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    try:
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    except cv2.error:
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)

    mk = max(3, int(min(eye_roi.shape[:2]) / 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) >= 10:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    inv = 255.0 - gray.astype(np.float32)
    yy, xx = np.indices(gray.shape)
    return (float(np.sum(xx * inv) / (np.sum(inv) + 1e-6)),
            float(np.sum(yy * inv) / (np.sum(inv) + 1e-6)))


def _process_classic_result(self, face_result, timings: dict[str, float]) -> TrackerResult:
    left_pupil = detect_pupil_centroid(face_result.left_eye_roi)
    right_pupil = detect_pupil_centroid(face_result.right_eye_roi)
    left_abs = absolute_pupil_point(left_pupil, face_result.left_eye_origin)
    right_abs = absolute_pupil_point(right_pupil, face_result.right_eye_origin)
    feature = fuse_eye_features(left_abs, right_abs, *face_result.frame_size, method="classic_pupil")
    raw_point = feature.point
    return TrackerResult(gaze_point=raw_point, valid=True, fps=self._calculate_fps(),
                         timings=timings, error_message=None, face_detected=True,
                         raw_point=raw_point, backend="classic")
```

该程序清单支撑论文第五章“Classic 传统视觉路线实现”。Classic 后端不进行深度学习推理，而是使用传统图像处理得到相机归一化眼部特征，再交由校准和平滑模块用于交互输出。

### A.6 GazeNetV2 模型结构

**程序清单 A-6 双眼共享 CNN 与头姿融合模型**

文件路径：`src/models/gaze_net.py`

功能说明：该片段展示 GazeNetV2 的核心结构，包括左右眼共享卷积骨干、单眼 256 维特征提取、头姿融合、515 维融合特征、全连接回归和 L2 归一化输出。

```python
class GazeNetV2(nn.Module):
    def __init__(
        self,
        num_channels: list[int] | None = None,
        head_pose_dim: int = 3,
        fusion_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        if num_channels is None:
            num_channels = [32, 64, 128, 256]

        self.eye_features = nn.Sequential(
            nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[0]), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[1]), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(num_channels[1], num_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[2]), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(num_channels[2], num_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(num_channels[3]), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.eye_pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = num_channels[3] * 2 + head_pose_dim
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 3),
        )

    def _extract_eye_features(self, eye_img: torch.Tensor) -> torch.Tensor:
        x = self.eye_features(eye_img)
        x = self.eye_pool(x)
        return x.view(x.size(0), -1)

    def forward(self, left_eye: torch.Tensor, right_eye: torch.Tensor, head_pose: torch.Tensor) -> torch.Tensor:
        left_feat = self._extract_eye_features(left_eye)
        right_feat = self._extract_eye_features(right_eye)
        fused = torch.cat([left_feat, right_feat, head_pose], dim=1)
        out = self.fusion(fused)
        return F.normalize(out, p=2, dim=1)
```

该程序清单支撑论文第四章“GazeNetV2 模型结构设计”。模型通过共享 CNN 降低参数规模，同时融合左右眼外观特征和头姿向量，最终输出三维单位视线方向。

### A.7 Deep 模式 ONNX 实时推理与三维方向输出

**程序清单 A-7 Deep 模式眼区预处理与 ONNX 推理**

文件路径：`src/tracker/pipeline.py`

功能说明：该片段展示 Deep 模式在实时运行中对左右眼图像进行 BGR 到 RGB、CHW 转换、像素归一化和姿态输入构造，并调用 ONNX Runtime 输出三维 gaze vector。

```python
if self.config.use_onnx:
    import onnxruntime as ort
    onnx_path = resource_path(self.config.onnx_path)
    self.onnx_session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    inputs = self.onnx_session.get_inputs()
    self.onnx_input_names = [inp.name for inp in inputs]
    self.onnx_output_name = self.onnx_session.get_outputs()[0].name


left_eye, right_eye = self._prepare_deep_eye_inputs(left_eye, right_eye)
left_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
right_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
left_chw = np.ascontiguousarray(left_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
right_chw = np.ascontiguousarray(right_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0

if self.config.normalized_deep_pose_input == "zero":
    head_pose_vec = np.zeros(3, dtype=np.float32)
else:
    head_pose_vec = np.array([head_pose.yaw, head_pose.pitch, head_pose.roll], dtype=np.float32)

left_batch = np.expand_dims(left_chw, axis=0)
right_batch = np.expand_dims(right_chw, axis=0)
pose_batch = np.expand_dims(head_pose_vec, axis=0)
gaze_vector = self.onnx_session.run(
    [self.onnx_output_name],
    {
        "left_eye": left_batch,
        "right_eye": right_batch,
        "head_pose": pose_batch,
    },
)[0][0]
```

该程序清单支撑论文第四章“模型导出与实时推理接口”和第五章“Deep 深度学习路线实现”。模型参数和 ONNX 模型作为本地软件资源用于系统运行与演示，不作为公开发布资源。

### A.8 三维视线到屏幕坐标映射、校准和平滑

**程序清单 A-8 射线平面求交与屏幕像素坐标转换**

文件路径：`src/geometry/screen_geometry.py`，`src/tracker/pipeline.py`

功能说明：该片段展示 Deep 模式将三维视线方向归一化后构造视线射线，并通过屏幕几何模型完成射线平面求交和像素坐标转换。

```python
def ray_plane_intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
    if self._plane_normal is None or self._plane_origin is None:
        return None
    return ray_plane_intersect(
        ray_origin,
        ray_direction,
        self._plane_origin,
        self._plane_normal,
    )


def world_to_screen_px(self, point_3d: np.ndarray, clamp: bool = True) -> tuple[float, float]:
    if self._plane_origin is None:
        return 0.0, 0.0

    p = np.asarray(point_3d, dtype=np.float64).flatten()
    delta = p - self._plane_origin
    local_x_mm = np.dot(delta, self._screen_x_axis)
    local_y_mm = np.dot(delta, self._screen_y_axis)
    px = local_x_mm / self.screen_w_mm * self.screen_w_px
    py = local_y_mm / self.screen_h_mm * self.screen_h_px
    if clamp:
        px = float(np.clip(px, 0, self.screen_w_px - 1))
        py = float(np.clip(py, 0, self.screen_h_px - 1))
    return float(px), float(py)


d = gaze_vector.astype(np.float64)
d = d / np.linalg.norm(d)
ray_origin, ray_direction = self._compute_deep_ray(d, head_pose)
intersection = self.screen_geometry.ray_plane_intersect(ray_origin, ray_direction)
raw_point = self.screen_geometry.world_to_screen_px(
    intersection,
    clamp=not self._calibration_mode,
)
```

该程序清单支撑第五章“三维视线到屏幕坐标的几何映射”。其核心作用是把模型输出的三维方向向量转换为屏幕平面上的二维像素坐标，为后续校准和平滑提供原始注视点。

**程序清单 A-9 校准映射与指数滑动平均平滑**

文件路径：`src/calibration/calibrator.py`，`src/ui/tracking_page.py`，`src/tracker/smoother.py`

功能说明：该片段展示校准模块如何将原始注视点映射为校准后的屏幕坐标，并使用指数滑动平均降低实时输出抖动。

```python
def apply(self, raw_gaze: tuple[float, float]) -> tuple[float, float]:
    if not self._calibrated or self._transform_matrix is None:
        return raw_gaze
    normed_x = (raw_gaze[0] - self._norm_mean[0]) / self._norm_std[0]
    normed_y = (raw_gaze[1] - self._norm_mean[1]) / self._norm_std[1]
    feat = np.array(self._build_feature_row(normed_x, normed_y), dtype=np.float64)
    result = self._transform_matrix @ feat
    return (float(result[0]), float(result[1]))


def _apply_calibration_and_clamp_with_debug(self, gaze_x: float, gaze_y: float):
    if self.calibrator is not None and self.calibrator.is_calibrated:
        gaze_x, gaze_y = self.calibrator.apply((gaze_x, gaze_y))
    pre_clamp = (gaze_x, gaze_y)
    screen = QApplication.primaryScreen()
    if screen is not None:
        geo = screen.geometry()
        gaze_x = max(0.0, min(gaze_x, float(geo.width() - 1)))
        gaze_y = max(0.0, min(gaze_y, float(geo.height() - 1)))
    return (gaze_x, gaze_y), pre_clamp


class GazeSmoother:
    def __init__(self, alpha: float = 0.3):
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha
        self._prev: Optional[tuple[float, float]] = None

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        if self._prev is None:
            self._prev = point
            return point
        sx = self.alpha * point[0] + (1.0 - self.alpha) * self._prev[0]
        sy = self.alpha * point[1] + (1.0 - self.alpha) * self._prev[1]
        self._prev = (sx, sy)
        return (sx, sy)
```

该程序清单支撑第五章“用户校准模块实现”和“注视点平滑与稳定处理”。校准映射用于修正用户、设备和屏幕几何差异带来的系统偏差，EMA 平滑用于降低帧间抖动并改善注视光标显示稳定性。

## 附录 B Look2Act 软件运行说明

### B.1 运行环境

Look2Act 面向 Windows 桌面环境运行，使用普通 RGB 摄像头作为图像输入设备。最终软件可通过本地 EXE 启动，模型文件和配置文件作为本地软件资源加载，运行期配置和校准结果保存到当前用户可写目录中。

自建数据集仅用于本课题训练、评估和系统验证，出于隐私和授权原因不公开发布。模型参数和 ONNX 模型作为本地软件资源用于系统运行与演示，不作为公开发布资源。

### B.2 启动与模式选择

用户启动程序后，首先进入语言和模式选择界面。系统支持 Classic 与 Deep 两种运行模式：Classic 模式用于传统视觉路线演示，Deep 模式用于基于 GazeNetV2 的深度视线估计路线演示。用户选择后，系统加载对应配置并进入主窗口。

【图 B-1 Look2Act 启动与模式选择界面】

### B.3 摄像头预览

进入摄像头预览页面后，用户应检查摄像头画面是否正常、人脸是否能够被检测、眼区裁剪状态是否稳定。若画面过暗、过亮或人脸位置偏离画面中心，应先调整光照、摄像头角度和坐姿，再进入校准流程。

【图 B-2 摄像头预览界面】

### B.4 25 点校准

校准页面以 5×5 网格形式依次显示 25 个屏幕目标点。用户需要按提示注视每个目标点，系统同步记录原始注视输出和目标点坐标，并在完成后保存校准参数。校准完成后，系统可进入全屏验证或实时追踪阶段。

【图 B-3 25 点校准界面】

### B.5 实时追踪与全屏验证

完成校准后，用户可以进入实时追踪页面启动追踪管线，查看帧率、检测状态和注视点输出。进入正式交互前，建议先打开全屏验证页面，观察注视光标在屏幕主要区域内的移动情况；若验证效果较差，应返回预览或校准页面重新调整。

【图 B-4 实时追踪界面】

【图 B-5 全屏验证界面】

### B.6 注视交互演示

Look2Act 提供交互启动器、停留触发和五子棋演示，用于验证系统从注视点估计到注视行为触发的完整流程。当前系统适合注视点可视化、大目标选择和交互原型演示，不宣称替代高精度鼠标操作。

【图 B-6 交互启动器界面】

【图 B-7 五子棋注视交互界面】

### B.7 常见问题与处理方式

| 问题 | 处理建议 |
| --- | --- |
| 摄像头打不开 | 检查摄像头是否被其他程序占用，并在系统权限中允许应用访问摄像头。 |
| 检测不到人脸 | 调整坐姿、摄像头角度和光照，使完整面部位于画面中央。 |
| 眼镜反光明显 | 降低屏幕亮度或调整环境光方向，避免镜片上出现强反光区域。 |
| 校准效果差 | 重新进入摄像头预览确认检测稳定后，再按目标点顺序重新校准。 |
| Deep 模式注视点偏移 | 重新完成校准，并检查模式配置与所加载模型资源是否一致。 |
| 光标抖动 | 保持头部和坐姿稳定，必要时提高平滑强度或重新校准。 |

## 待人工补充清单

- 图 B-1 Look2Act 启动与模式选择界面
- 图 B-2 摄像头预览界面
- 图 B-3 25 点校准界面
- 图 B-4 实时追踪界面
- 图 B-5 全屏验证界面
- 图 B-6 交互启动器界面
- 图 B-7 五子棋注视交互界面
