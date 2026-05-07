# Look2Act 中文毕业论文代码证据文档

说明：本文只引用当前 `Look2Act_Tracker_Project` 工作区中真实存在的代码、配置、评估结果和打包文件。行号基于生成本文时的当前工作树。本文不粘贴隐私图片、密钥、个人绝对路径或逐 session 设备标识。

## 0. 仓库与版本概况

- 当前项目根目录名称：`Look2Act_Tracker_Project`。
- 当前主要入口文件：`main.py`。
- 当前主要配置文件：`configs/classic.yaml`、`configs/deep.yaml`，另有演示配置 `configs/experiments/system_classic_demo.yaml`、`configs/experiments/system_deep_demo.yaml`。
- 当前支持 Windows EXE 打包：存在 `Look2Act.spec`、`build_exe.bat` 和 `dist/Look2Act/Look2Act.exe`。当前 `dist/Look2Act` 目录总大小约 466.1 MB，单个 `Look2Act.exe` 约 14.4 MB。
- 当前 Classic 与 Deep 两种模式都存在：启动弹窗可选 `classic` / `deep`，配置中也分别存在 `tracker.backend: classic` 与 `tracker.backend: deep`。
- 当前 README 功能描述与真实代码基本一致：README 描述了启动配置、摄像头预览、校准、验证、交互、Classic/Deep、AppData 配置和 EXE 启动，这些在代码中均可找到。但 README 引用的 `readme-images/*.gif/png` 当前目录下只有 `.gitkeep`，截图资源缺失；数据目录当前为空，训练/评估数据规模只能从评估产物推断。

## 1. 最终事实核对表

| 事实项 | 当前建议写法 | 来源文件 |
| --- | --- | --- |
| 用户数量 | 31 用户。当前 `dataset_raw` / `dataset_processed` 为空，但评估 CSV 与 LOO summary 均显示 31 用户。 | `evaluation_results/error_analysis/per_sample_errors.csv`；`evaluation_results/leave_one_out_resume/summary.json` |
| 设备数量 | 当前可核查评估 CSV 可解析出 5 台设备口径；不要直接写 16 台，除非补充原始 `meta.json` 或采集平台统计截图。 | `evaluation_results/error_analysis/per_sample_errors.csv` 的 `session_id` 字段 |
| 样本数量 | 7395。 | `evaluation_results/error_analysis/summary.json`；`evaluation_results/leave_one_out_resume/summary.json` |
| train/val/test 样本数量 | 5018 / 987 / 1390。 | `evaluation_results/label_audit/split_summary.csv` |
| 是否存在 31 用户 LOO 结果 | 存在，`completed_users=31`。 | `evaluation_results/leave_one_out_resume/summary.json` |
| 固定测试集平均角度误差 | 2.8206°。 | `evaluation_results/metrics.json` |
| 固定测试集平均像素误差 | 314.5413 px。 | `evaluation_results/metrics.json` |
| LOO 平均角度误差 | 2.4788°。 | `evaluation_results/leave_one_out_resume/summary.json` |
| LOO 平均像素误差 | 321.3820 px。 | `evaluation_results/leave_one_out_resume/summary.json` |
| 当前 Classic 默认校准点数和映射方式 | 用户配置文件明确为 25 点，代码根据 backend 采用 polynomial。 | `configs/classic.yaml`；`src/tracker/pipeline.py` |
| 当前 Deep 默认校准点数和映射方式 | 用户配置文件明确为 25 点，因此采用 polynomial；代码 fallback 在未配置点数时 Deep 会变成 9 点 affine，需要单独说明。 | `configs/deep.yaml`；`src/tracker/pipeline.py` |
| 当前 Deep 运行时 gaze space / ray origin / pose input / eye input mode / smoother | `camera` / `zero_origin` / `zero` / `swap` / `ema alpha=0.22`。 | `configs/deep.yaml` |
| 当前模型权重路径 | `checkpoints/best_model.pth`，当前文件约 5.25 MB。 | `configs/deep.yaml`；`checkpoints/best_model.pth` |
| 当前 ONNX 模型路径 | `checkpoints/gaze_net.onnx`，当前文件约 1.74 MB。 | `configs/deep.yaml`；`checkpoints/gaze_net.onnx` |
| 当前 EXE 或 PyInstaller spec 是否存在 | 存在 `Look2Act.spec`、`build_exe.bat`、`dist/Look2Act/Look2Act.exe`。 | 项目根目录与 `dist/Look2Act` |

关于“16 设备”和“4 设备”：`docs/midterm/中期报告.md` 和 `docs/midterm/中期答辩PPT.md` 曾写 16 台设备；`docs/logs/research_log.md` 曾在 22 用户阶段写 4 台设备。当前仓库没有可核查的 `labels.csv` / `meta.json` 原始数据文件，只有评估 CSV 可解析设备口径，因此论文最终建议写“31 名用户、7395 条有效样本；评估产物可追溯到 5 台设备/31 个 session，若要写 16 台设备需补充原始 meta 统计证据”。

## 2. 软件入口与模式选择

**代码片段 2-1：`main.py`，函数 `parse_args` / `validate_config_path`，56-90 行**

```python
  56: def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  57:     """在 Qt 启动前解析命令行参数。"""
  58:     parser = argparse.ArgumentParser(description="Look2Act Tracker")
  59:     parser.add_argument(
  60:         "--config",
  61:         default="configs/classic.yaml",
  62:         help="Path to the system YAML config used by UI, tracker, and settings page.",
  63:     )
  64:     parser.add_argument(
  65:         "--smoke-test",
  66:         action="store_true",
  67:         help=argparse.SUPPRESS,
  68:     )
  69:     args, _ = parser.parse_known_args(argv)
  70:     return args
  71: 
  72: 
  73: def validate_config_path(config_path: Path) -> tuple[bool, str]:
  74:     """在界面写入配置前校验启动配置文件。"""
  75:     if not config_path.exists():
  76:         return False, (
  77:             f"配置文件不存在：{config_path}\n"
  78:             "如果你在 Git Bash 中运行，请使用正斜杠，例如：\n"
  79:             "python main.py --config configs/experiments/system_deep_camera_zero_720.yaml"
  80:         )
  81:     try:
  82:         with config_path.open("r", encoding="utf-8") as f:
  83:             data = yaml.safe_load(f) or {}
  84:     except Exception as e:
  85:         return False, f"配置文件读取失败：{config_path}\n{e}"
  86:     required_sections = {"camera", "model", "tracker"}
  87:     missing = sorted(section for section in required_sections if section not in data)
  88:     if missing:
  89:         return False, (
  90:             f"配置文件不完整：{config_path}\n"
```

这一片段适合论文第五章“软件系统实现”或“系统启动流程”小节。它证明程序通过 `--config` 加载 YAML，并在 UI 启动前做基本配置校验。默认入口配置是 `configs/classic.yaml`，因此论文不能写 Deep 是默认用户入口。它也说明命令行配置只是启动模板，后续还会被启动弹窗写入用户配置。

**代码片段 2-2：`main.py`，函数 `main`，237-284 行**

```python
 237: def main() -> int:
 238:     """应用主函数。
 239:     
 240:     返回:
 241:         退出代码（0 表示成功）
 242:     """
 243:     try:
 244:         args = parse_args(sys.argv[1:])
 245:         sys.argv = [sys.argv[0]]
 246:         if args.smoke_test:
 247:             return run_smoke_test()
 248:         config_path = resource_path(args.config)
 249:         ok, error_message = validate_config_path(config_path)
 250:         if not ok:
 251:             logger.warning(error_message)
 252:         logger.info("=" * 60)
 253:         logger.info("Look2Act Tracker 启动中...")
 254:         logger.info(f"系统配置路径: {config_path}")
 255:         logger.info("=" * 60)
 256:         
 257:         # 创建应用
 258:         app = setup_application()
 259:         logger.info("QApplication 已创建")
 260: 
 261:         from src.ui.i18n import load_language
 262:         from src.ui.language_dialog import LanguageSelectionDialog
 263: 
 264:         load_language(config_path)
 265:         dialog = LanguageSelectionDialog(config_path=config_path)
 266:         if dialog.exec() != dialog.DialogCode.Accepted:
 267:             return 0
 268:         language = dialog.selected_language
 269:         config_path = dialog.selected_config_path
 270:         ok, error_message = validate_config_path(config_path)
 271:         if not ok:
 272:             logger.error(error_message)
 273:             return 2
 274:         logger.info(f"界面语言已设置: {language}")
 275:         logger.info(f"演示模式已设置: {dialog.selected_mode}")
 276:         logger.info(f"系统配置路径: {config_path}")
 277:         
 278:         # 导入主窗口（延迟导入，避免在 QApplication 创建前导入 Qt 组件）
 279:         from src.ui.main_window import MainWindow
 280:         
 281:         # 创建主窗口
 282:         main_window = MainWindow(config_path=config_path)
 283:         logger.info("主窗口已创建")
 284:         
```

这一片段说明真实启动流程是：解析参数、创建 `QApplication`、弹出语言/模式选择框、再创建 `MainWindow`。论文中应写“启动阶段允许用户选择界面语言和 Classic/Deep 模式”，而不是写程序直接进入追踪。`load_language` 和 `LanguageSelectionDialog` 是 UI 启动链路的一部分。这里也体现了延迟导入主窗口以避免 Qt 初始化问题。

**代码片段 2-3：`src/runtime_paths.py`，函数 `app_data_dir` / `user_config_path` / `calibration_path_for_backend`，21-55 行**

```python
  21: def app_data_dir() -> Path:
  22:     """返回当前用户可写的应用数据目录。"""
  23:     appdata = os.environ.get("APPDATA")
  24:     if appdata:
  25:         return Path(appdata) / APP_NAME
  26:     return Path.home() / "AppData" / "Roaming" / APP_NAME
  27: 
  28: 
  29: def resource_path(path: str | Path) -> Path:
  30:     """解析源码或打包资源路径。"""
  31:     candidate = Path(path)
  32:     if candidate.is_absolute():
  33:         return candidate
  34:     return app_base_dir() / candidate
  35: 
  36: 
  37: def user_data_path(path: str | Path) -> Path:
  38:     """解析 AppData 下的当前用户可写路径。"""
  39:     candidate = Path(path)
  40:     if candidate.is_absolute():
  41:         return candidate
  42:     return app_data_dir() / candidate
  43: 
  44: 
  45: def user_config_path(mode: str) -> Path:
  46:     normalized = mode if mode in {"classic", "deep"} else "classic"
  47:     return user_data_path(Path("configs") / f"{normalized}.yaml")
  48: 
  49: 
  50: def calibration_path_for_backend(backend: str) -> Path:
  51:     if backend == "deep":
  52:         filename = "calibration_deep.json"
  53:     elif backend == "deep_pog":
  54:         filename = "calibration_deep_pog.json"
  55:     else:
```

这一片段适合支撑“用户可写配置与打包部署”描述。源码和 EXE 模式都通过 `resource_path` 解析资源，通过 `user_data_path` 把可变配置和校准文件放到用户 AppData 下。论文中可以写系统避免把运行期配置写入安装目录，从而适配普通 Windows 用户权限。不要写校准文件固定保存在项目根目录，当前代码会解析到用户数据目录。

**代码片段 2-4：`src/ui/language_dialog.py`，函数 `ensure_user_config` / `_save_startup_config`，42-49 与 234-249 行**

```python
  42: def ensure_user_config(mode: str) -> Path:
  43:     normalized = mode if mode in MODE_TEMPLATE_CONFIGS else "classic"
  44:     target_path = user_config_path_for_mode(normalized)
  45:     if not target_path.exists():
  46:         template_path = template_config_path_for_mode(normalized)
  47:         target_path.parent.mkdir(parents=True, exist_ok=True)
  48:         shutil.copyfile(template_path, target_path)
  49:     return target_path
 234:     @staticmethod
 235:     def _save_startup_config(language: str, mode: str) -> Path:
 236:         config_path = ensure_user_config(mode)
 237:         if config_path.exists():
 238:             with config_path.open("r", encoding="utf-8") as f:
 239:                 data = yaml.safe_load(f) or {}
 240:         else:
 241:             data = {}
 242:         ui = data.setdefault("ui", {})
 243:         ui["language"] = language
 244:         ui["window_mode"] = "fullscreen"
 245:         config_path.parent.mkdir(parents=True, exist_ok=True)
 246:         with config_path.open("w", encoding="utf-8") as f:
 247:             yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
 248:         set_language(language)
 249:         return config_path
```

这一片段说明模式选择后会从模板配置复制到当前用户目录，并写入语言与全屏模式。它适合论文“软件配置管理”小节。Classic/Deep 的配置不是只存在源码目录，而是运行时会产生用户副本。论文应避免把用户目录写成固定绝对路径，只需描述为 `%APPDATA%/Look2Act` 下的用户配置。

## 3. UI 页面与用户流程

**代码片段 3-1：`src/ui/main_window.py`，类 `MainWindow.__init__`，44-88 行**

```python
  44:     def __init__(self, config_path: Path | str | None = None) -> None:
  45:         super().__init__()
  46:         self.setWindowTitle(f"Look2Act Tracker - {tx('视线驱动交互系统', 'Gaze-Driven Interaction System')}")
  47:         self.config_path = Path(config_path) if config_path is not None else Path("configs/system_config.yaml")
  48:         
  49:         # TrackerPipeline 单例（延迟初始化）
  50:         self.tracker: Optional[TrackerPipeline] = None
  51:         self.tracker_config: Optional[SystemConfig] = None
  52:         
  53:         # 创建所有页面
  54:         self.page_home = HomePage()
  55:         self.page_camera = CameraPage()
  56:         self.page_calibration = CalibrationPage()
  57:         self.page_tracking = TrackingPage(config_path=self.config_path)
  58:         self.page_settings = SettingsPage(config_path=self.config_path)
  59: 
  60:         # 设置 objectName 供 FluentWindow 路由标识
  61:         self.page_home.setObjectName("HomePage")
  62:         self.page_camera.setObjectName("CameraPage")
  63:         self.page_calibration.setObjectName("CalibrationPage")
  64:         self.page_tracking.setObjectName("TrackingPage")
  65:         self.page_settings.setObjectName("SettingsPage")
  66:         
  67:         # 添加页面到侧边导航栏
  68:         self.addSubInterface(self.page_home, FIF.HOME, tx('主页', 'Home'))
  69:         self.addSubInterface(self.page_camera, FIF.PHOTO, tx('预览', 'Preview'))
  70:         self.addSubInterface(self.page_calibration, FIF.EDIT, tx('校准', 'Calibrate'))
  71:         self.addSubInterface(self.page_tracking, FIF.VIEW, tx('追踪', 'Track'))
  72:         self.addSubInterface(self.page_settings, FIF.SETTING, tx('设置', 'Settings'))
  73:         
  74:         # 连接主页导航信号
  75:         self.page_home.navigate_to_camera.connect(self.go_camera)
  76:         self.page_home.navigate_to_calibration.connect(self.go_calibration)
  77:         self.page_home.navigate_to_tracking.connect(self.go_tracking)
  78:         self.page_home.navigate_to_settings.connect(self.go_settings)
  79:         
  80:         # 连接设置页面配置变更信号
  81:         self.page_settings.config_changed.connect(self._on_config_changed)
  82:         self.page_calibration.calibration_ready.connect(self._on_calibration_ready)
  83:         self.page_calibration.return_home_requested.connect(self.go_home)
  84:         self.page_calibration.tracker_required.connect(self._on_calibration_tracker_required)
  85:         
  86:         self._apply_window_mode()
  87:         
  88:         # 禁用 QFluentWidgets 自定义标题栏的最大化/还原功能
```

这一片段适合第五章 UI 架构说明。它证明真实页面包括主页、摄像头预览、校准、追踪、设置，而不是单窗口脚本。页面由 QFluentWidgets 的 `FluentWindow` 路由管理。论文截图可以围绕这些真实页面组织：主页流程卡、预览页、人脸检测预览、校准页、追踪页、设置页。

**代码片段 3-2：`src/ui/main_window.py`，函数 `go_calibration` / `go_tracking`，280-326 行**

```python
 280:     def go_calibration(self) -> bool:
 281:         """导航到校准页面。"""
 282:         # 确保 TrackerPipeline 已初始化
 283:         if not self._ensure_tracker_initialized():
 284:             return False
 285:         
 286:         # 如果 TrackerPipeline 未运行，启动它
 287:         if not self.tracker.is_running():
 288:             print("[MAIN_WINDOW] 启动 TrackerPipeline 用于校准...")
 289:             success = self.tracker.start()
 290:             if not success:
 291:                 QMessageBox.critical(
 292:                     self,
 293:                     tx("启动失败", "Start Failed"),
 294:                     tx(
 295:                         "TrackerPipeline 启动失败，无法进行校准。\n\n请检查摄像头和模型文件。",
 296:                         "TrackerPipeline failed to start, so calibration cannot begin.\n\nPlease check the camera and model files.",
 297:                     )
 298:                 )
 299:                 return False
 300:         
 301:         # 将 TrackerPipeline 传递给校准页面
 302:         self.page_calibration.set_tracker(self.tracker)
 303:         
 304:         self.switchTo(self.page_calibration)
 305:         print("[MAIN_WINDOW] 导航到校准页面")
 306:         return True
 307: 
 308:     def _on_calibration_tracker_required(self) -> None:
 309:         """用户从侧边栏进入校准页时初始化追踪管道。"""
 310:         if self.go_calibration():
 311:             QTimer.singleShot(0, self.page_calibration._handle_start_calibration)
 312:     
 313:     def go_tracking(self) -> None:
 314:         """导航到实时追踪页面。"""
 315:         # 确保 TrackerPipeline 已初始化
 316:         if not self._ensure_tracker_initialized():
 317:             return
 318:         
 319:         # 将 TrackerPipeline 传递给追踪页面
 320:         if self.page_tracking.tracker is None:
 321:             self.page_tracking.tracker = self.tracker
 322:             self.page_tracking.tracker_config = self.tracker_config
 323:         else:
 324:             self.page_tracking.tracker = self.tracker
 325:             self.page_tracking.tracker_config = self.tracker_config
 326: 
```

这一片段证明校准页和追踪页共享同一个 `TrackerPipeline`。论文中可以把流程写成“预览检查摄像头，校准前启动追踪管道，校准结束后进入验证与交互”。它也说明校准不是离线脚本，而是在 UI 中实时读取 pipeline 输出。需要注意：摄像头预览页只是独立预览，进入校准时会启动真正的 tracker。

**代码片段 3-3：`src/ui/tracking_page.py`，函数 `_open_verification_window` / `_open_launcher_overlay` / `_open_gomoku_window`，558-683 行节选**

```python
 558:     def _open_verification_window(self) -> None:
 559:         if self.calibrator is None or not self.calibrator.is_calibrated:
 560:             QMessageBox.information(
 561:                 self,
 562:                 tx("需要校准", "Calibration Required"),
 563:                 tx("请先加载或完成校准，再进入验证阶段。", "Please load or complete calibration before verification.")
 564:             )
 565:             return
 566: 
 567:         if self.tracker is None or not self.tracker.is_running():
 568:             self._handle_start_tracking()
 569:             if self.tracker is None or not self.tracker.is_running():
 570:                 return
 571: 
 572:         self._close_launcher_overlay()
 573:         self._close_gomoku_window()
 574:         self._reset_dwell_state()
 575: 
 576:         if self.cursor_overlay is not None:
 577:             self.cursor_overlay.hide()
 578: 
 579:         if self.verification_window is None:
 580:             self.verification_window = GazeVerificationWindow()
 581:             self.verification_window.verified.connect(self._on_verification_passed)
 582:             self.verification_window.cancelled.connect(self._on_verification_cancelled)
 583: 
 584:         self.verification_window.show()
 585:         self.verify_status.setText(tx("验证中", "Verifying"))
 586:         self.verify_status.setStyleSheet("color: #2196F3; font-weight: 600;")
 587:         self._refresh_stage_controls()
 634:     def _open_launcher_overlay(self) -> None:
 635:         if not self._verification_passed:
 636:             QMessageBox.information(
 637:                 self,
 638:                 tx("先完成验证", "Verification Required"),
 639:                 tx("请先完成全屏验证，再进入交互阶段。", "Please complete fullscreen verification before entering interaction.")
 640:             )
 641:             return
 642: 
 643:         if self.tracker is None or not self.tracker.is_running():
 644:             self._handle_start_tracking()
 645:             if self.tracker is None or not self.tracker.is_running():
 646:                 return
 647: 
 648:         self._close_verification_window()
 649:         self._reset_dwell_state()
 650: 
 651:         if self.cursor_overlay is not None:
 652:             self.cursor_overlay.hide()
 653: 
 654:         self.interaction_overlay = InteractionLauncherOverlay()
 655:         self.interaction_overlay.request_toggle_dwell.connect(self._toggle_dwell_click)
 656:         self.interaction_overlay.request_open_gomoku.connect(self._open_gomoku_window)
 657:         self.interaction_overlay.closed.connect(self._on_launcher_closed)
 658:         self.interaction_overlay.show()
 659:         self._refresh_stage_controls()
```

这一片段适合论文第五章“用户流程与交互功能”。它证明全屏验证必须在校准完成后进入，交互窗口又要求验证通过。代码中真实存在交互启动器、五子棋窗口和停留点击开关。论文截图可放：全屏验证窗口、九宫格交互窗口、五子棋交互窗口；但不要声称它已经达到精确鼠标替代，只能写“提供注视驱动交互演示”。

## 4. TrackerPipeline 实时处理主流程

**代码片段 4-1：`src/tracker/pipeline.py`，类 `TrackerPipeline.process_frame`，526-590 行**

```python
 526:     def process_frame(self, frame_bgr: np.ndarray) -> TrackerResult:
 527:         """处理单帧，返回注视点和各阶段耗时。
 528:         
 529:         容错策略：
 530:         - 未检测到人脸：返回上一帧有效结果
 531:         - 视线无效（无交点）：已通过 clamp_to_screen=True 处理
 532:         
 533:         参数:
 534:             frame_bgr: BGR 格式输入图像
 535:             
 536:         返回:
 537:             TrackerResult 包含注视点和性能指标
 538:         """
 539:         timings = {}
 540:         backend = self.config.normalized_backend
 541:         
 542:         # 1. 人脸检测
 543:         t0 = time.perf_counter()
 544:         face_result = self.face_detector.detect(frame_bgr)
 545:         timings['face_detection'] = (time.perf_counter() - t0) * 1000
 546:         
 547:         if not face_result.detected:
 548:             self._no_face_count += 1
 549: 
 550:             if getattr(self.face_detector, "unavailable", False):
 551:                 return TrackerResult(
 552:                     gaze_point=None,
 553:                     valid=False,
 554:                     fps=self._calculate_fps(),
 555:                     timings=timings,
 556:                     error_message=None,
 557:                     face_detected=False,
 558:                     backend=backend,
 559:                 )
 560:             
 561:             # 容错：返回上一帧有效结果
 562:             if self._last_valid_result is not None:
 563:                 logger.debug(f"未检测到人脸（连续 {self._no_face_count} 帧），使用上一帧结果")
 564:                 return TrackerResult(
 565:                     gaze_point=self._last_valid_result.gaze_point,
 566:                     valid=True,
 567:                     fps=self._calculate_fps(),
 568:                     timings=timings,
 569:                     error_message="未检测到人脸，使用上一帧结果",
 570:                     face_detected=False,
 571:                     backend=backend,
 572:                 )
 585:         # 检测到人脸，重置计数器
 586:         self._no_face_count = 0
 587: 
 588:         if self.config.normalized_backend == "classic":
 589:             return self._process_classic_result(face_result, timings)
 590:         
```

这一片段支撑“实时逐帧处理链路”的起点。每帧先经过 FaceDetector，未检测到人脸时有容错逻辑。Classic 后端在 FaceMesh 完成后直接走 `_process_classic_result`，不进入深度模型和三维几何投影。论文画流程图时应把 Classic 和 Deep 分叉画在 FaceMesh 之后。

**代码片段 4-2：`src/tracker/pipeline.py`，类 `TrackerPipeline.process_frame`，591-677 行节选**

```python
 591:         # 2. 头部姿态估计
 592:         t0 = time.perf_counter()
 593:         head_pose = self.head_pose_estimator.estimate(face_result.pnp_points_2d)
 594:         timings['head_pose'] = (time.perf_counter() - t0) * 1000
 595:         
 596:         if not head_pose.valid:
 597:             # 容错：返回上一帧有效结果
 598:             if self._last_valid_result is not None:
 599:                 logger.debug("头部姿态估计失败，使用上一帧结果")
 600:                 return TrackerResult(
 620:         # 3. 视线回归（CNN 模型推理）
 621:         t0 = time.perf_counter()
 622:         
 623:         left_eye = face_result.left_eye_crop
 624:         right_eye = face_result.right_eye_crop
 625:         
 626:         if left_eye is None or right_eye is None:
 627:             if self._last_valid_result is not None:
 628:                 return TrackerResult(
 647:         try:
 648:             left_eye, right_eye = self._prepare_deep_eye_inputs(left_eye, right_eye)
 649:             # BGR → RGB（与训练时 _img_to_tensor 一致）
 650:             left_rgb = cv2.cvtColor(left_eye, cv2.COLOR_BGR2RGB)
 651:             right_rgb = cv2.cvtColor(right_eye, cv2.COLOR_BGR2RGB)
 652:             
 653:             # 转换为 CHW float32 并归一化。ONNX 路径保持纯 numpy，避免打包 PyTorch。
 654:             left_chw = np.ascontiguousarray(left_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
 655:             right_chw = np.ascontiguousarray(right_rgb.transpose(2, 0, 1), dtype=np.float32) / 255.0
 656:             
 657:             # 构建 head pose 向量 (yaw, pitch, roll)，单位：度
 658:             if self.config.normalized_deep_pose_input == "zero":
 659:                 head_pose_vec = np.zeros(3, dtype=np.float32)
 660:             else:
 661:                 head_pose_vec = np.array([
 662:                     head_pose.yaw, head_pose.pitch, head_pose.roll
 663:                 ], dtype=np.float32)
 664:             
 665:             if self.model_version in {"v2", "pog_v1"}:
 666:                 if self.config.use_onnx:
 667:                     left_batch = np.expand_dims(left_chw, axis=0)
 668:                     right_batch = np.expand_dims(right_chw, axis=0)
 669:                     pose_batch = np.expand_dims(head_pose_vec, axis=0)
 670:                     gaze_vector = self.onnx_session.run(
 671:                         [self.onnx_output_name],
 672:                         {
 673:                             'left_eye': left_batch,
 674:                             'right_eye': right_batch,
 675:                             'head_pose': pose_batch,
 676:                         }
 677:                     )[0][0]
```

这一片段适合算法实现章节。它证明 Deep 链路仍会实时求 `head_pose`，但当前配置可将输入给模型的 pose 置零。眼图输入由 BGR 转 RGB、CHW、归一化到 `[0,1]` 后进入 ONNX Runtime。论文应写“当前 Deep runtime 实际检测 live head pose，但默认 demo 配置不把 live pose 输入模型，而使用零向量消融配置”。

**代码片段 4-3：`src/tracker/pipeline.py`，类 `TrackerPipeline.process_frame` / `_compute_deep_ray`，763-853 行**

```python
 763:         ray_origin, ray_direction = self._compute_deep_ray(d, head_pose)
 764:         timings['coordinate_transform'] = (time.perf_counter() - t0) * 1000
 765: 
 766:         t0 = time.perf_counter()
 767:         clamp_to_screen = not self._calibration_mode
 768:         intersection = self.screen_geometry.ray_plane_intersect(ray_origin, ray_direction)
 769:         timings['ray_plane_intersect'] = (time.perf_counter() - t0) * 1000
 770: 
 771:         if intersection is None:
 772:             if self._last_valid_result is not None:
 773:                 return TrackerResult(
 792:         raw_point = self.screen_geometry.world_to_screen_px(intersection, clamp=clamp_to_screen)
 793:         
 794:         # 5. 时序平滑
 795:         t0 = time.perf_counter()
 796:         if self._calibration_mode or self.config.normalized_smoother_type == "none" or self.smoother is None:
 797:             smoothed_point = raw_point
 798:         else:
 799:             smoothed_point = self.smoother.update(raw_point)
 800:         timings['smoothing'] = (time.perf_counter() - t0) * 1000
 801:         
 802:         # 保存为有效结果（用于后续容错）
 803:         result = TrackerResult(
 804:             gaze_point=smoothed_point,
 805:             valid=True,
 806:             fps=self._calculate_fps(),
 807:             timings=timings,
 808:             error_message=None,
 809:             face_detected=True,
 810:             raw_point=raw_point,
 811:             backend="deep",
 834:     def _select_deep_ray_origin(self, face_translation: np.ndarray) -> np.ndarray:
 835:         """选择 Deep 链路的 3D 视线射线原点。"""
 836:         if self.config.normalized_deep_ray_origin == "zero_origin":
 837:             return np.zeros(3, dtype=np.float64)
 838:         return np.asarray(face_translation, dtype=np.float64).flatten()
 839: 
 840:     def _compute_deep_ray(self, gaze_direction: np.ndarray, head_pose) -> tuple[np.ndarray, np.ndarray]:
 841:         """根据当前坐标空间配置计算 3D 视线射线。"""
 842:         d = np.asarray(gaze_direction, dtype=np.float64).flatten()
 843:         norm = np.linalg.norm(d)
 844:         if norm > 1e-12:
 845:             d = d / norm
 846:         if self.config.deep_gaze_space == "camera":
 847:             return self._select_deep_ray_origin(head_pose.translation_vec), d
 848:         ray_origin, ray_direction = transform_gaze_to_camera(
 849:             d,
 850:             head_pose.rotation_matrix,
 851:             head_pose.translation_vec,
 852:         )
 853:         return self._select_deep_ray_origin(ray_origin), ray_direction
```

这一片段支撑三维视线到屏幕点映射。当前 Deep 配置 `deep_gaze_space: camera` 时会直接使用模型输出方向作为相机空间方向；`deep_ray_origin: zero_origin` 时 `_select_deep_ray_origin` 返回零向量。校准模式下禁用 clamp 和 smoother，以保留原始采样点。论文中应把 raw point、smoothed point、calibrated point 区分开。

适合论文的实时链路伪代码：

```text
输入：摄像头帧 frame
1. face = FaceDetector.detect(frame)
2. if no face: 返回上一帧有效 gaze 或 invalid
3. if backend == classic:
      从左右眼 ROI 检测暗瞳孔质心
      融合左右眼绝对坐标并归一化为 raw point
      返回 raw point，后续由 UI 应用校准和平滑
4. if backend == deep:
      head_pose = solvePnP(face.pnp_points)
      crop left/right eye -> RGB -> CHW -> [0,1]
      根据配置选择 live pose 或 zero pose
      ONNX/PyTorch 推理得到 3D gaze vector
      根据 camera/head space 与 ray origin 构造射线
      与屏幕平面求交得到 intersection
      intersection 转为屏幕 raw pixel
      非校准模式下进行 EMA 平滑
      返回 gaze_point 与 raw_point
5. TrackingPage 加载校准矩阵后 apply(raw/smoothed point)，再 clamp 到屏幕范围
```

## 5. Classic 传统视觉后端

**代码片段 5-1：`src/tracker/classic.py`，函数 `detect_pupil_centroid`，30-71 行**

```python
  30: def detect_pupil_centroid(eye_roi: np.ndarray) -> Optional[tuple[float, float]]:
  31:     """返回眼部 ROI 内的暗色瞳孔质心坐标。
  32: 
  33:     先使用 Otsu 反向阈值和轮廓矩定位；当轮廓不可用时，使用暗像素加权质心作为
  34:     备用估计。
  35:     """
  36:     if eye_roi is None or eye_roi.size == 0:
  37:         return None
  38: 
  39:     gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
  40:     k = max(3, int(min(eye_roi.shape[:2]) / 8) | 1)
  41:     gray = cv2.GaussianBlur(gray, (k, k), 0)
  42: 
  43:     try:
  44:         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
  45:     except cv2.error:
  46:         _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)
  47: 
  48:     mk = max(3, int(min(eye_roi.shape[:2]) / 20) | 1)
  49:     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
  50:     thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
  51:     thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
  52: 
  53:     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  54:     if contours:
  55:         contour = max(contours, key=cv2.contourArea)
  56:         if cv2.contourArea(contour) >= 10:
  57:             moments = cv2.moments(contour)
  58:             if moments["m00"] != 0:
  59:                 cx = float(moments["m10"] / moments["m00"])
  60:                 cy = float(moments["m01"] / moments["m00"])
  61:                 return (cx, cy)
  62: 
  63:     inv = 255.0 - gray.astype(np.float32)
  64:     weight_sum = float(np.sum(inv)) + 1e-6
  65:     yy, xx = np.indices(gray.shape)
  66:     cx = float(np.sum(xx * inv) / weight_sum)
  67:     cy = float(np.sum(yy * inv) / weight_sum)
  68:     h, w = gray.shape
  69:     if 0.0 <= cx < w and 0.0 <= cy < h:
  70:         return (cx, cy)
  71:     return None
```

这一片段适合论文“传统视觉后端”小节。它证明 Classic 使用灰度、模糊、Otsu 反阈值、形态学、轮廓矩和暗像素加权质心，不是深度学习模型。论文应把 Classic 描述为传统计算机视觉路线或稳定演示 fallback。不要写 Classic 使用 CNN、训练权重或 GazeNet。

**代码片段 5-2：`src/tracker/classic.py`，函数 `absolute_pupil_point` / `fuse_eye_features`，88-118 行**

```python
  88: def absolute_pupil_point(
  89:     pupil: Optional[tuple[float, float]],
  90:     roi_origin: Optional[tuple[int, int]],
  91: ) -> Optional[tuple[float, float]]:
  92:     """将 ROI 内的瞳孔坐标转换为原始相机坐标。"""
  93:     if pupil is None or roi_origin is None:
  94:         return None
  95:     return (float(roi_origin[0] + pupil[0]), float(roi_origin[1] + pupil[1]))
  96: 
  97: 
  98: def fuse_eye_features(
  99:     left_point: Optional[tuple[float, float]],
 100:     right_point: Optional[tuple[float, float]],
 101:     camera_width: int = 1,
 102:     camera_height: int = 1,
 103:     method: str = "classic_pupil",
 104: ) -> Optional[ClassicGazeFeature]:
 105:     """融合可用的左右眼绝对坐标，并按相机尺寸归一化。"""
 106:     points = [p for p in (left_point, right_point) if p is not None]
 107:     if not points:
 108:         return None
 109: 
 110:     arr = np.array(points, dtype=np.float64)
 111:     mean = arr.mean(axis=0)
 112:     norm_x, norm_y = normalize_camera_point(
 113:         (float(mean[0]), float(mean[1])),
 114:         camera_width,
 115:         camera_height,
 116:     )
 117:     confidence = 0.65 if len(points) == 1 else 1.0
 118:     return ClassicGazeFeature(norm_x, norm_y, confidence, method)
```

这一片段说明 Classic 将 ROI 内瞳孔点还原到整帧相机坐标，再融合左右眼。输出的 `ClassicGazeFeature.point` 是归一化相机点，不是直接屏幕坐标。论文中可写“Classic 原始特征点通过用户校准映射到屏幕坐标”。如果只检测到一只眼，代码降低 confidence，但仍可输出。

**代码片段 5-3：`src/tracker/pipeline.py`，函数 `_process_classic_result`，946-1018 行节选**

```python
 946:     def _process_classic_result(
 947:         self,
 948:         face_result,
 949:         timings: dict[str, float],
 950:     ) -> TrackerResult:
 951:         """使用 Classic 后端处理已检测到的人脸结果。"""
 952:         t0 = time.perf_counter()
 953:         left_pupil = detect_pupil_centroid(getattr(face_result, "left_eye_roi", None))
 954:         right_pupil = detect_pupil_centroid(getattr(face_result, "right_eye_roi", None))
 955:         left_abs = absolute_pupil_point(left_pupil, getattr(face_result, "left_eye_origin", None))
 956:         right_abs = absolute_pupil_point(right_pupil, getattr(face_result, "right_eye_origin", None))
 957:         frame_size = getattr(face_result, "frame_size", None) or (
 958:             self.config.camera_width,
 959:             self.config.camera_height,
 960:         )
 961:         feature = fuse_eye_features(
 962:             left_abs,
 963:             right_abs,
 964:             camera_width=int(frame_size[0]),
 965:             camera_height=int(frame_size[1]),
 966:             method="classic_pupil",
 967:         )
 968:         timings["classic_feature"] = (time.perf_counter() - t0) * 1000
 969: 
 970:         if feature is None:
 971:             if self._last_valid_result is not None:
 972:                 return TrackerResult(
 993:         raw_point = feature.point
 994: 
 995:         result = TrackerResult(
 996:             gaze_point=raw_point,
 997:             valid=True,
 998:             fps=self._calculate_fps(),
 999:             timings=timings,
1000:             error_message=None,
1001:             face_detected=True,
1002:             raw_point=raw_point,
1003:             backend="classic",
1004:             debug={
1005:                 "feature_method": feature.method,
1006:                 "feature_confidence": feature.confidence,
1007:                 "smoother_type": self.config.normalized_smoother_type,
1008:                 "left_pupil": left_pupil,
1009:                 "right_pupil": right_pupil,
1010:                 "left_abs": left_abs,
1011:                 "right_abs": right_abs,
1012:                 "frame_size": frame_size,
1013:                 "calibration_mode": self._calibration_mode,
1014:             },
1015:         )
1016:         self._last_valid_result = result
1017:         return result
1018:     
```

这一片段说明 Classic 的实时输出是 `feature.point`，后续校准和平滑主要在 `TrackingPage` 中处理。代码中创建了 `classic_smoother`，但 `_process_classic_result` 并未在 pipeline 内调用它；真实屏幕稳定由 UI 的 `ScreenGazeStabilizer` 处理。论文应避免写“Classic 在 pipeline 中完成 Kalman 输出”，更准确是“Classic raw point 在 UI 层经过校准和屏幕稳定处理”。

## 6. Deep / GazeNetV2 深度学习后端

**代码片段 6-1：`src/models/gaze_net.py`，类 `GazeNetV2.__init__`，87-143 行**

```python
  87: class GazeNetV2(nn.Module):
  88:     """V2：双眼共享 CNN + head pose 融合的视线回归模型。
  89: 
  90:     架构：
  91:     - 共享 CNN backbone 分别提取左右眼特征 (各 256 维)
  92:     - 拼接双眼特征 + 3 维 head pose → 515 维
  93:     - FC 融合层 → Dropout → FC 输出 → L2 归一化
  94: 
  95:     参数:
  96:         num_channels: 四层卷积的通道数列表，默认 [32, 64, 128, 256]
  97:         head_pose_dim: 头部姿态维度，默认 3 (yaw, pitch, roll)
  98:         fusion_dim: 融合层隐藏维度，默认 128
  99:         dropout: Dropout 概率，默认 0.3
 100:     """
 101: 
 102:     def __init__(
 103:         self,
 104:         num_channels: list[int] | None = None,
 105:         head_pose_dim: int = 3,
 106:         fusion_dim: int = 128,
 107:         dropout: float = 0.3,
 108:     ):
 109:         super().__init__()
 110:         if num_channels is None:
 111:             num_channels = [32, 64, 128, 256]
 112: 
 113:         assert len(num_channels) == 4, "需要恰好 4 层卷积通道配置"
 114: 
 115:         # 共享 CNN backbone（左右眼共用权重）
 116:         self.eye_features = nn.Sequential(
 117:             nn.Conv2d(3, num_channels[0], kernel_size=3, padding=1),
 118:             nn.BatchNorm2d(num_channels[0]),
 119:             nn.ReLU(inplace=True),
 120:             nn.MaxPool2d(2),
 121:             nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1),
 122:             nn.BatchNorm2d(num_channels[1]),
 123:             nn.ReLU(inplace=True),
 124:             nn.MaxPool2d(2),
 125:             nn.Conv2d(num_channels[1], num_channels[2], kernel_size=3, padding=1),
 126:             nn.BatchNorm2d(num_channels[2]),
 127:             nn.ReLU(inplace=True),
 128:             nn.MaxPool2d(2),
 129:             nn.Conv2d(num_channels[2], num_channels[3], kernel_size=3, padding=1),
 130:             nn.BatchNorm2d(num_channels[3]),
 131:             nn.ReLU(inplace=True),
 132:             nn.MaxPool2d(2),
 133:         )
 134:         self.eye_pool = nn.AdaptiveAvgPool2d(1)
 136:         # 融合层：左眼(256) + 右眼(256) + head_pose(3) → fusion_dim → 3
 137:         feat_dim = num_channels[3] * 2 + head_pose_dim  # 515
 138:         self.fusion = nn.Sequential(
 139:             nn.Linear(feat_dim, fusion_dim),
 140:             nn.ReLU(inplace=True),
 141:             nn.Dropout(dropout),
 142:             nn.Linear(fusion_dim, 3),
 143:         )
```

这一片段适合论文第四章“模型结构设计”。它证明 GazeNetV2 输入左右眼 128×128 RGB 张量和 3 维头姿，左右眼共享 CNN backbone，融合后输出 3 维向量。论文可写为监督式深度学习模型，训练标签来自屏幕点构造的 3D gaze vector。项目旧文档记录 GazeNetV2 参数量约 455,811，源自 `docs/logs/experiment_log.md` 和 `docs/midterm/模型链路与创新点梳理.md`；当前代码没有自动生成参数量文件。

**代码片段 6-2：`src/models/gaze_net.py`，类 `GazeNetV2._extract_eye_features` / `forward`，145-186 行**

```python
 145:     def _extract_eye_features(self, eye_img: torch.Tensor) -> torch.Tensor:
 146:         """提取单眼 CNN 特征。
 147: 
 148:         参数:
 149:             eye_img: (B, 3, 128, 128)
 150: 
 151:         返回:
 152:             (B, 256) 特征向量
 153:         """
 154:         x = self.eye_features(eye_img)
 155:         x = self.eye_pool(x)
 156:         return x.view(x.size(0), -1)
 157: 
 158:     def forward(
 159:         self,
 160:         left_eye: torch.Tensor,
 161:         right_eye: torch.Tensor,
 162:         head_pose: torch.Tensor,
 163:     ) -> torch.Tensor:
 164:         """前向传播。
 165: 
 166:         参数:
 167:             left_eye: 左眼图像 (B, 3, 128, 128)
 168:             right_eye: 右眼图像 (B, 3, 128, 128)
 169:             head_pose: 头部姿态 (B, 3)，包含 yaw/pitch/roll（度）
 170: 
 171:         返回:
 172:             (B, 3) 单位视线向量
 173:         """
 174:         # 共享 backbone 提取双眼特征
 175:         left_feat = self._extract_eye_features(left_eye)   # (B, 256)
 176:         right_feat = self._extract_eye_features(right_eye)  # (B, 256)
 177: 
 178:         # 拼接双眼特征 + head pose
 179:         fused = torch.cat([left_feat, right_feat, head_pose], dim=1)  # (B, 515)
 180: 
 181:         # 融合回归
 182:         out = self.fusion(fused)
 183: 
 184:         # L2 归一化
 185:         out = F.normalize(out, p=2, dim=1)
 186:         return out
```

这一片段证明输出为 3D 单位视线向量，并显式做 L2 normalization。论文训练章节可写损失在单位向量上计算角度误差。它也说明 head pose 是直接拼接到融合层，而不是用复杂 Transformer 或注意力机制。ONNX 模型当前 `checkpoints/gaze_net.onnx` 约 1.74 MB，README/研究文档中“约 2 MB ONNX”属于可接受近似。

## 7. 数据采集与预处理代码

**代码片段 7-1：`src/data/preprocessing.py`，函数 `extract_eyes_from_mosaic`，225-271 行**

```python
 225: def extract_eyes_from_mosaic(
 226:     mosaic_bgr: np.ndarray,
 227:     eye_crop_size: int = DEFAULT_EYE_CROP_SIZE,
 228: ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
 229:     """从 Collector 生成的隐私合成图中提取左右眼裁剪区域。
 230: 
 231:     Collector 的合成图布局（4 列）：
 232:     - 第 1 列 [0:128, 0:128]：左眼裁剪
 233:     - 第 2 列 [0:128, 128:256]：右眼裁剪
 234:     - 第 3 列：ArUco marker
 235:     - 第 4 列：骨架图
 236: 
 237:     合成图高度可能为 128（标准）或 240（旧版），宽度固定 640。
 238:     眼部区域始终在前 128 行。
 239: 
 240:     Args:
 241:         mosaic_bgr: Collector 生成的合成图，shape (H, 640, 3)，H 为 128 或 240
 242:         eye_crop_size: 眼部裁剪尺寸，默认 128
 243: 
 244:     Returns:
 245:         (left_eye, right_eye)，各为 (128, 128, 3) BGR 图像。
 246:         如果合成图尺寸不符合预期，返回 (None, None)。
 247:     """
 248:     if mosaic_bgr is None or mosaic_bgr.size == 0:
 249:         return None, None
 250: 
 251:     h, w = mosaic_bgr.shape[:2]
 252: 
 253:     # 验证合成图尺寸
 254:     if w < 2 * eye_crop_size or h < eye_crop_size:
 255:         logger.warning(
 256:             f"合成图尺寸不符合预期: {h}x{w}，"
 257:             f"需要至少 {eye_crop_size}x{2 * eye_crop_size}"
 258:         )
 259:         return None, None
 260: 
 261:     # 提取左右眼区域（始终在前 eye_crop_size 行）
 262:     left_eye = mosaic_bgr[:eye_crop_size, :eye_crop_size].copy()
 263:     right_eye = mosaic_bgr[:eye_crop_size, eye_crop_size:2 * eye_crop_size].copy()
 264: 
 265:     # 检查是否为全黑（表示 Collector 未检测到眼部）
 266:     if np.mean(left_eye) < 1.0:
 267:         left_eye = None
 268:     if np.mean(right_eye) < 1.0:
 269:         right_eye = None
 270: 
 271:     return left_eye, right_eye
```

这一片段适合论文“数据采集与隐私处理”小节。它说明训练数据不是保存整张真实人脸，而是从 Collector 合成图中取左右眼 128×128 crop。数据采集系统与 Tracker 主系统是两个项目：采集系统生成合成图、标签和 meta，Tracker 负责预处理、训练、评估与实时推理。当前仓库数据目录为空，因此论文中如果展示采集图片，应使用已脱敏合成图或论文已有截图，不应放真实人脸图。

**代码片段 7-2：`src/data/pipeline.py`，类 `DataPipeline.load_session`，91-151 行节选**

```python
  91:     def load_session(self, session_dir: str | Path) -> SessionData:
  92:         """加载单个 Session 目录，解析 labels.csv 和 meta.json。
  93: 
  94:         Args:
  95:             session_dir: Session 目录路径（如 dataset_raw/4/）
  96: 
  97:         Returns:
  98:             SessionData 对象
  99: 
 100:         Raises:
 101:             FileNotFoundError: labels.csv 或 meta.json 不存在
 102:             ValueError: 数据完整性验证失败
 103:         """
 104:         session_dir = Path(session_dir)
 105: 
 106:         # 检查必需文件
 107:         labels_path = session_dir / "labels.csv"
 108:         meta_path = session_dir / "meta.json"
 109: 
 110:         if not labels_path.exists():
 111:             raise FileNotFoundError(f"labels.csv 不存在: {labels_path}")
 112:         if not meta_path.exists():
 113:             raise FileNotFoundError(f"meta.json 不存在: {meta_path}")
 114: 
 115:         # 解析 meta.json
 116:         meta = self._parse_meta(meta_path)
 117: 
 118:         # 解析 labels.csv
 119:         labels = self._parse_labels(labels_path)
 120: 
 121:         # 验证数据完整性
 122:         self._validate_labels(labels, session_dir)
 123: 
 124:         # 过滤 valid=0 的样本
 125:         valid_mask = labels["valid"] == 1
 126:         skipped = labels[~valid_mask]
 127:         skip_reasons = []
 128: 
 129:         for idx, row in skipped.iterrows():
 130:             reason = f"frame_idx={row['frame_idx']}, valid=0"
 131:             if row.get("face_score", 1.0) < 0.5:
 132:                 reason += ", face_score 过低"
 133:             if row.get("lm_score", 1.0) < 0.5:
 134:                 reason += ", lm_score 过低"
 135:             skip_reasons.append(reason)
 145:         return SessionData(
 146:             meta=meta,
 147:             labels=labels,
 148:             session_dir=session_dir,
 149:             valid_count=int(valid_mask.sum()),
 150:             skipped_count=skipped_count,
 151:             skip_reasons=skip_reasons,
```

这一片段支撑论文“数据清洗与有效样本筛选”。它证明原始 session 需要同时包含 `labels.csv` 与 `meta.json`，并按 `valid` 字段过滤无效样本。当前仓库没有这些文件，因此不能声称当前目录仍保留完整原始数据。论文可写“预处理流程设计支持读取 labels/meta；当前论文证据以评估产物统计为准”。

**代码片段 7-3：`src/data/preprocessing.py`，函数 `compute_gaze_vector`，346-415 行节选**

```python
 346: def compute_gaze_vector(
 347:     target_x: float,
 348:     target_y: float,
 349:     screen_w: int,
 350:     screen_h: int,
 351:     geometry: ScreenCameraGeometry = DEFAULT_GEOMETRY,
 352:     distance_mm: Optional[float] = None,
 353: ) -> tuple[float, float, float]:
 354:     """计算从相机（近似眼球位置）到屏幕目标点的 3D 视线方向向量。
 355: 
 356:     几何模型：
 357:     1. 将屏幕像素坐标转换为屏幕物理坐标（mm）
 358:     2. 将屏幕物理坐标转换为摄像头坐标系中的 3D 点
 359:     3. 计算从相机原点到目标点的归一化方向向量
 386:     dist = distance_mm if distance_mm is not None else geometry.screen_distance_mm
 387: 
 388:     # 步骤 1：屏幕像素 → 屏幕物理坐标（mm）
 389:     # 屏幕左上角为 (0, 0)，右下角为 (screen_w, screen_h)
 390:     screen_x_mm = (target_x / screen_w) * geometry.screen_w_mm
 391:     screen_y_mm = (target_y / screen_h) * geometry.screen_h_mm
 403:     target_3d_x = screen_x_mm - geometry.screen_w_mm / 2.0
 404:     target_3d_y = geometry.cam_above_screen_mm + screen_y_mm
 405:     target_3d_z = dist
 406: 
 407:     # 步骤 3：计算从相机原点 (0,0,0) 到目标点的方向向量并归一化
 408:     vec = np.array([target_3d_x, target_3d_y, target_3d_z], dtype=np.float64)
 409:     norm = np.linalg.norm(vec)
 410:     if norm < 1e-10:
 411:         # 目标点与相机重合（不应发生），返回正前方
 412:         return 0.0, 0.0, 1.0
 413: 
 414:     unit_vec = vec / norm
 415:     return float(unit_vec[0]), float(unit_vec[1]), float(unit_vec[2])
```

这一片段适合论文“标签构造”小节。它证明 3D gaze label 是由屏幕像素目标、屏幕物理尺寸、相机到屏幕距离和相机位置假设构造的单位向量。它不是人工直接标注的 3D 方向，而是由 2D 屏幕注视点和几何模型转换得到。论文中需强调几何假设会影响实时映射效果，这也是后续 3D contract 诊断的来源。

**代码片段 7-4：`src/data/dataset.py`，类 `GazeDataset._load_online` / `_img_to_tensor` / `_head_pose_tensor`，112-176 与 252-265 行节选**

```python
 112:     def _load_online(self, row: pd.Series) -> dict:
 113:         """从 dataset_raw 的合成图中在线加载和预处理。"""
 114:         img_path = row["img_path"]
 115:         if self.image_root is not None:
 116:             full_path = self.image_root / img_path
 117:         else:
 118:             full_path = Path(img_path)
 120:         mosaic = cv2.imread(str(full_path))
 121:         if mosaic is None:
 122:             logger.warning(f"无法读取图像: {full_path}")
 123:             return self._empty_sample()
 125:         left_eye, right_eye = extract_eyes_from_mosaic(mosaic)
 126:         if left_eye is None:
 127:             return self._empty_sample()
 129:         labels = compute_gaze_labels_for_row(
 130:             float(row["target_x"]),
 131:             float(row["target_y"]),
 132:             int(row["screen_w"]),
 133:             int(row["screen_h"]),
 134:             distance_proxy=float(row.get("distance_proxy", -1)),
 135:             geometry=self.geometry,
 136:         )
 145:         gaze_tensor = torch.tensor([gx, gy, gz], dtype=torch.float32)
 157:         if self.model_version in {"v2", "pog_v1"}:
 158:             if right_eye is None:
 159:                 right_eye = cv2.flip(left_eye, 1)
 160:             head_pose = self._head_pose_tensor(row)
 161:             return {
 162:                 "left_eye": self._img_to_tensor(left_eye),
 163:                 "right_eye": self._img_to_tensor(right_eye),
 164:                 "head_pose": head_pose,
 165:                 "gaze": gaze_tensor,
 166:                 "pog": pog_tensor,
 167:                 "meta": meta,
 168:             }
 252:     def _img_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
 253:         """BGR uint8 图像 → (3, H, W) float32 张量，归一化到 [0, 1]。"""
 254:         img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
 255:         tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
 256:         return tensor
 258:     def _head_pose_tensor(self, row: pd.Series) -> torch.Tensor:
 259:         if self.head_pose_mode == "zero":
 260:             return torch.zeros(3, dtype=torch.float32)
 261:         return torch.tensor([
 262:             float(row.get("head_yaw", 0)),
 263:             float(row.get("head_pitch", 0)),
 264:             float(row.get("head_roll", 0)),
 265:         ], dtype=torch.float32)
```

这一片段适合论文“训练样本组织”小节。它证明 V2 样本包含 `left_eye`、`right_eye`、`head_pose`、`gaze` 和 `pog`。图像标准化与实时 pipeline 的 BGR→RGB、CHW、除以 255 一致。代码还支持 `head_pose_mode="zero"`，这与当前 Deep runtime 的 `pose zero` 思路一致。

## 8. FaceMesh 与头姿估计

**代码片段 8-1：`src/vision/face_detector.py`，FaceMesh 初始化与关键点定义，71-83 与 135-150 行**

```python
  71: # PnP 头姿估计用的 6 个关键点索引
  72: _PNP_INDICES = {
  73:     "nose_tip": 1,
  74:     "chin": 152,
  75:     "left_eye_outer": 33,
  76:     "right_eye_outer": 263,
  77:     "left_mouth": 61,
  78:     "right_mouth": 291,
  79: }
  80: 
  81: # 左右眼轮廓关键点索引（用于计算眼部裁剪区域）
  82: _LEFT_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 246, 161, 163, 144, 145, 153, 154, 155]
  83: _RIGHT_EYE_INDICES = [263, 362, 387, 386, 385, 384, 398, 466, 388, 390, 373, 374, 380, 381, 382]
 135:     def __init__(
 136:         self,
 137:         eye_crop_size: int = 128,
 138:         max_num_faces: int = 1,
 139:         min_detection_confidence: float = 0.5,
 140:         min_tracking_confidence: float = 0.5,
 141:         refine_landmarks: bool = False,
 142:     ):
 143:         self.eye_crop_size = eye_crop_size
 144:         self._mesh = mp_face_mesh.FaceMesh(
 145:             static_image_mode=False,
 146:             max_num_faces=max_num_faces,
 147:             refine_landmarks=refine_landmarks,
 148:             min_detection_confidence=min_detection_confidence,
 149:             min_tracking_confidence=min_tracking_confidence,
 150:         )
```

这一片段适合论文“人脸关键点检测”小节。它证明项目使用 MediaPipe FaceMesh，并指定 6 个 PnP 点与左右眼轮廓点。Classic 运行时 `refine_landmarks` 会根据 backend 决定，Deep 主要使用标准关键点和眼区 crop。论文中不要写使用 dlib 68 点模型作为当前实现；代码只是从 MediaPipe 点集中派生 68 点数组。

**代码片段 8-2：`src/vision/face_detector.py`，函数 `_extract_pnp_points` / `_crop_eye`，326-407 行节选**

```python
 326:     def _extract_pnp_points(self, lms, w: int, h: int) -> dict[str, tuple[float, float]]:
 327:         """提取 6 个 PnP 关键点像素坐标。"""
 328:         pnp_points = {}
 329:         for name, idx in _PNP_INDICES.items():
 330:             if idx < len(lms):
 331:                 px = float(np.clip(lms[idx].x * w, 0, w - 1))
 332:                 py = float(np.clip(lms[idx].y * h, 0, h - 1))
 333:                 pnp_points[name] = (px, py)
 334:         return pnp_points
 336:     def _crop_eye(
 337:         self,
 338:         frame_bgr: np.ndarray,
 339:         lms,
 340:         eye_indices: list[int],
 341:         w: int,
 342:         h: int,
 343:     ) -> np.ndarray:
 344:         """裁剪眼部区域并 resize 到固定尺寸。
 345: 
 346:         使用眼部关键点计算中心和范围，加 padding 后裁剪，
 347:         最终 resize 到 eye_crop_size × eye_crop_size。
 352:         # 收集眼部关键点像素坐标
 353:         pts_x = []
 354:         pts_y = []
 355:         for idx in eye_indices:
 356:             if idx < len(lms):
 357:                 pts_x.append(lms[idx].x * w)
 358:                 pts_y.append(lms[idx].y * h)
 363:         # 计算中心和范围
 364:         cx = sum(pts_x) / len(pts_x)
 365:         cy = sum(pts_y) / len(pts_y)
 366:         eye_w = max(pts_x) - min(pts_x)
 367:         eye_h = max(pts_y) - min(pts_y)
 369:         # 与离线眼部裁剪规则保持一致：四周各留 50% 边距，总边长为眼框的 2 倍。
 370:         side = max(eye_w, eye_h) * 2.0
 371:         side = max(side, 40.0)  # 最小边长
 402:         # resize 到固定尺寸
 403:         eye_img = cv2.resize(
 404:             crop,
 405:             (self.eye_crop_size, self.eye_crop_size),
 406:             interpolation=cv2.INTER_AREA,
 407:         )
```

这一片段支撑“实时眼区裁剪与头姿输入点提取”。它证明眼图 crop 是根据 FaceMesh 眼部轮廓点计算中心和边长，并 resize 到 128×128。PnP 点坐标会 clamp 到图像范围内。论文中可以把这个模块画为 FaceMesh 后的两个分支：眼区 crop 和 PnP 关键点。

**代码片段 8-3：`src/vision/head_pose.py`，类 `HeadPoseEstimator.estimate`，31-119 行节选**

```python
  31: _REQUIRED_KEYS = (
  32:     "nose_tip", "chin",
  33:     "left_eye_outer", "right_eye_outer",
  34:     "left_mouth", "right_mouth",
  35: )
  37: # 简化 3D 人脸模型点（单位：mm），鼻尖为原点
  38: _MODEL_POINTS_3D = np.array([
  39:     (0.0, 0.0, 0.0),          # nose_tip
  40:     (0.0, -330.0, -65.0),     # chin
  41:     (-225.0, 170.0, -135.0),  # left_eye_outer
  42:     (225.0, 170.0, -135.0),   # right_eye_outer
  43:     (-150.0, -150.0, -125.0), # left_mouth
  44:     (150.0, -150.0, -125.0),  # right_mouth
  45: ], dtype=np.float64)
  63:     def __init__(self, frame_size: tuple[int, int]):
  64:         w, h = frame_size
  65:         # 相机内参：焦距近似为图像宽度
  66:         focal_length = float(w)
  67:         center = (w / 2.0, h / 2.0)
  68:         self.camera_matrix = np.array([
  69:             [focal_length, 0.0, center[0]],
  70:             [0.0, focal_length, center[1]],
  71:             [0.0, 0.0, 1.0],
  72:         ], dtype=np.float64)
  73:         self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
  75:     def estimate(self, pnp_points_2d: dict[str, tuple[float, float]]) -> HeadPoseResult:
  84:         # 检查关键点完整性
  85:         if any(k not in pnp_points_2d for k in _REQUIRED_KEYS):
  86:             return _invalid_result()
  88:         image_points = np.array(
  89:             [pnp_points_2d[k] for k in _REQUIRED_KEYS],
  90:             dtype=np.float64,
  91:         )
  93:         # PnP 求解
  94:         ok, rvec, tvec = cv2.solvePnP(
  95:             _MODEL_POINTS_3D,
  96:             image_points,
  97:             self.camera_matrix,
  98:             self.dist_coeffs,
  99:             flags=cv2.SOLVEPNP_ITERATIVE,
 100:         )
 104:         # 旋转向量 → 旋转矩阵
 105:         rotation_matrix, _ = cv2.Rodrigues(rvec)
 107:         # 旋转矩阵 → 欧拉角（度）
 108:         angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
 109:         pitch = float(angles[0])
 110:         yaw = float(angles[1])
 111:         roll = float(angles[2])
 113:         return HeadPoseResult(
 114:             valid=True,
 115:             yaw=yaw,
 116:             pitch=pitch,
 117:             roll=roll,
 118:             rotation_matrix=rotation_matrix,
 119:             translation_vec=tvec,
```

这一片段适合论文“头姿估计方法”小节。它证明头姿来自 6 个 2D-3D 对应点和 OpenCV `solvePnP`，再由旋转矩阵分解得到 yaw/pitch/roll。当前 Deep demo 配置虽然实时估计 head pose，但 `deep_pose_input: zero` 使模型输入姿态为零；同时 `deep_gaze_space: camera` 跳过 head-space 旋转。论文不要套用旧“head-space + live PnP 旋转是最终方案”的说法。

## 9. 三维视线到屏幕点的几何映射

**代码片段 9-1：`src/tracker/pipeline.py`，屏幕平面初始化，488-512 行**

```python
 488:             self.screen_geometry = ScreenGeometry(
 489:                 screen_w_px=screen_w_px,
 490:                 screen_h_px=screen_h_px,
 491:                 screen_w_mm=self.config.screen_w_mm,
 492:                 screen_h_mm=self.config.screen_h_mm,
 493:                 camera_matrix=self.head_pose_estimator.camera_matrix,
 494:             )
 495:             
 496:             # 设置屏幕平面：左上角在摄像头坐标系中的位置
 497:             screen_origin = np.array([
 498:                 -self.config.screen_w_mm / 2.0,
 499:                 self.config.cam_above_screen_mm,
 500:                 self.config.screen_distance_mm,
 501:             ])
 502:             screen_normal = np.array([0.0, 0.0, -1.0])  # 屏幕法向量指向摄像头
 503:             screen_x_axis = np.array([1.0, 0.0, 0.0])
 504:             screen_y_axis = np.array([0.0, 1.0, 0.0])
 505:             
 506:             self.screen_geometry.setup_plane(
 507:                 screen_origin_mm=screen_origin,
 508:                 screen_normal=screen_normal,
 509:                 screen_x_axis=screen_x_axis,
 510:                 screen_y_axis=screen_y_axis,
 511:             )
 512:             print(f"屏幕几何模型已初始化：屏幕 {screen_w_px}x{screen_h_px} px")
```

这一片段支撑“屏幕平面建模”。屏幕左上角位于 `(-screen_w_mm/2, cam_above_screen_mm, screen_distance_mm)`，屏幕 X 轴向右，Y 轴向下，法向量指向摄像头。当前 Deep 配置使用 `screen_distance_mm=720.0`，Classic 配置使用 `500.0`。论文公式应明确几何参数来自配置，而不是自动标定出来。

**代码片段 9-2：`src/geometry/ray_plane.py` 与 `src/geometry/screen_geometry.py`，射线求交与像素转换，16-52 与 93-128 行节选**

```python
  16: def ray_plane_intersect(
  17:     ray_origin: np.ndarray,
  18:     ray_direction: np.ndarray,
  19:     plane_point: np.ndarray,
  20:     plane_normal: np.ndarray,
  21:     epsilon: float = 1e-8,
  22: ) -> Optional[np.ndarray]:
  35:     O = np.asarray(ray_origin, dtype=np.float64).flatten()
  36:     D = np.asarray(ray_direction, dtype=np.float64).flatten()
  37:     P0 = np.asarray(plane_point, dtype=np.float64).flatten()
  38:     N = np.asarray(plane_normal, dtype=np.float64).flatten()
  40:     denom = np.dot(N, D)
  42:     # 射线与平面平行
  43:     if abs(denom) < epsilon:
  44:         return None
  46:     t = np.dot(N, P0 - O) / denom
  48:     # 交点在射线反方向
  49:     if t < 0:
  50:         return None
  52:     return O + t * D
  93:     def world_to_screen_px(
  94:         self,
  95:         point_3d: np.ndarray,
  96:         clamp: bool = True,
  97:     ) -> tuple[float, float]:
 113:         p = np.asarray(point_3d, dtype=np.float64).flatten()
 114:         delta = p - self._plane_origin
 116:         # 投影到屏幕局部坐标系
 117:         local_x_mm = np.dot(delta, self._screen_x_axis)
 118:         local_y_mm = np.dot(delta, self._screen_y_axis)
 120:         # 物理坐标 → 像素坐标
 121:         px = local_x_mm / self.screen_w_mm * self.screen_w_px
 122:         py = local_y_mm / self.screen_h_mm * self.screen_h_px
 124:         if clamp:
 125:             px = float(np.clip(px, 0, self.screen_w_px - 1))
 126:             py = float(np.clip(py, 0, self.screen_h_px - 1))
 128:         return float(px), float(py)
```

这一片段适合论文“三维几何映射”公式说明。射线表示为 `R(t)=O+tD`，屏幕平面为 `(X-P0)·N=0`，交点参数 `t=((P0-O)·N)/(D·N)`。若 `D·N` 接近 0 或 `t<0`，视线无有效交点。交点转像素时先投影到屏幕局部坐标，再按物理尺寸与像素分辨率缩放；非校准模式会 clamp，校准模式不 clamp。

**代码片段 9-3：`src/geometry/coordinate.py`，函数 `transform_gaze_to_camera`，10-53 行**

```python
  10: def transform_gaze_to_camera(
  11:     gaze_vector: np.ndarray,
  12:     rotation_matrix: np.ndarray,
  13:     translation_vec: np.ndarray,
  14: ) -> tuple[np.ndarray, np.ndarray]:
  15:     """将视线方向从眼球局部坐标系转换到摄像头坐标系。
  16: 
  17:     使用旋转矩阵将视线方向旋转到摄像头坐标系，
  18:     使用平移向量作为射线起点（眼球在摄像头坐标系中的位置）。
  19: 
  20:     参数:
  21:         gaze_vector: 眼球局部坐标系中的视线方向向量 (3,)
  22:         rotation_matrix: 3×3 旋转矩阵（从眼球坐标系到摄像头坐标系）
  23:         translation_vec: 3×1 或 (3,) 平移向量（眼球在摄像头坐标系中的位置）
  24: 
  25:     返回:
  26:         (ray_origin, ray_direction) 在摄像头坐标系中的射线
  27:         ray_origin: (3,) 射线起点
  28:         ray_direction: (3,) 射线方向（单位向量）
  29:     """
  30:     gaze = np.asarray(gaze_vector, dtype=np.float64).flatten()
  31:     R = np.asarray(rotation_matrix, dtype=np.float64)
  32:     t = np.asarray(translation_vec, dtype=np.float64).flatten()
  33: 
  34:     # 旋转视线方向到摄像头坐标系
  35:     ray_direction = R @ gaze
  36: 
  37:     # 归一化为单位向量
  38:     norm = np.linalg.norm(ray_direction)
  39:     if norm > 1e-12:
  40:         ray_direction = ray_direction / norm
  41: 
  42:     # 射线起点为平移向量（眼球位置）
  43:     ray_origin = t.copy()
  44: 
  45:     return ray_origin, ray_direction
```

这一片段属于历史/可选 head-space 逻辑，当前 Deep demo 配置 `deep_gaze_space: camera` 不进入该旋转分支。论文可以把它写成“系统保留 head-space 到 camera-space 的转换实现”，但最终运行配置使用 camera-space 直接投影。旧文档中把该分支当作最终实时链路的描述需要修正。

公式建议：设模型输出单位方向 `d`，射线原点 `O`，屏幕左上角 `P0`，法向量 `n`，屏幕局部轴 `ex, ey`。交点为 `t=((P0-O)·n)/(d·n)`，`P=O+td`；局部物理坐标为 `u=(P-P0)·ex`、`v=(P-P0)·ey`；像素坐标为 `x_px=u/W_mm*W_px`、`y_px=v/H_mm*H_px`。

## 10. 校准模块

**代码片段 10-1：`src/tracker/pipeline.py`，类 `SystemConfig.effective_calibration_*`，197-208 行**

```text
src/tracker/pipeline.py:
 197:     def effective_calibration_num_points(self) -> int:
 198:         if self.calibration_num_points > 0:
 199:             return self.calibration_num_points
 200:         if self.normalized_backend in {"classic", "deep_pog"}:
 201:             return 25
 202:         return 9
 203: 
 204:     @property
 205:     def effective_calibration_method(self) -> str:
 206:         if self.normalized_backend in {"classic", "deep_pog"}:
 207:             return "polynomial"
 208:         return "polynomial" if self.effective_calibration_num_points >= 25 else "affine"

configs/classic.yaml:
 26: calibration:
 27:   num_points: 25
 28:   save_path: calibration_classic.json
 29:   max_residual_px: 300.0

configs/deep.yaml:
 27: calibration:
 28:   num_points: 25
 29:   save_path: calibration_deep.json
 30:   max_residual_px: 300.0
```

这一片段揭示了一个需要论文中特别说明的事实：代码 fallback 对 Deep 是 9 点 affine，但当前 `configs/deep.yaml` 明确配置了 25 点，因此实际用户配置为 25 点 polynomial。Classic 当前也是 25 点 polynomial。论文主线建议写“当前演示配置采用 25 点 polynomial 校准”，同时在冲突表中说明“Deep fallback 9 点 affine 是未显式配置时的代码默认/旧实验口径”。

**代码片段 10-2：`src/ui/calibration_page.py`，函数 `_generate_calibration_points` / `_on_sampling_tick`，134-168 与 193-207 行**

```python
 134:     def _generate_calibration_points(self) -> None:
 135:         """生成 3x3 网格的 9 个校准点。
 136:         
 137:         校准点分布在屏幕的 9 个位置：
 138:         - 边距：距离屏幕边缘 10%
 139:         - 网格：3 行 × 3 列
 140:         """
 141:         from PyQt6.QtWidgets import QApplication
 142:         screen = QApplication.primaryScreen()
 143:         if screen is None:
 144:             # 默认分辨率
 145:             screen_w, screen_h = 1920, 1080
 146:         else:
 147:             geometry = screen.geometry()
 148:             screen_w, screen_h = geometry.width(), geometry.height()
 150:         # 边距比例
 151:         margin_ratio = 0.1
 152:         margin_x = screen_w * margin_ratio
 153:         margin_y = screen_h * margin_ratio
 155:         # 有效区域
 156:         effective_w = screen_w - 2 * margin_x
 157:         effective_h = screen_h - 2 * margin_y
 159:         grid_size = 5 if self.calibrator.num_points >= 25 else 3
 161:         # 生成网格
 162:         index = 0
 163:         for row in range(grid_size):
 164:             for col in range(grid_size):
 165:                 denom = max(grid_size - 1, 1)
 166:                 x = margin_x + col * effective_w / denom
 167:                 y = margin_y + row * effective_h / denom
 168:                 self.calibration_points.append(CalibrationPoint(x=x, y=y, index=index))
 193:     def _on_sampling_tick(self) -> None:
 194:         """采样定时器回调。"""
 195:         # 从 TrackerPipeline 获取最新的视线数据
 196:         result = self.tracker.get_latest_result()
 197:         self.sampling_ticks += 1
 199:         sample_point = (result.raw_point or result.gaze_point) if result is not None else None
 201:         if (
 202:             self.sampling_ticks > self.discard_initial_frames
 203:             and result is not None
 204:             and result.valid
 205:             and sample_point is not None
 206:         ):
 207:             self.current_samples.append(sample_point)
```

这一片段支撑“校准点采样策略”。当前 25 点校准会生成 5×5 网格，9 点配置会生成 3×3 网格，均留 10% 屏幕边距。采样使用 `result.raw_point` 优先，因此不会用已经校准后的点再次拟合。代码注释仍写“生成 3x3 网格的 9 个校准点”，但实际 `grid_size` 会根据 `num_points>=25` 变成 5，这属于注释与代码不完全一致，应按代码事实写 25 点为 5×5。

**代码片段 10-3：`src/calibration/calibrator.py`，类 `CalibrationModule.calibrate`，108-180 行节选**

```python
 108:     def calibrate(self) -> float:
 109:         """执行校准，拟合映射函数。
 110: 
 111:         affine:     min_A Σ ||A × [xi, yi, 1]^T - target_i||²
 112:         polynomial: min_W Σ ||W × [xi, yi, xi*yi, xi², yi², 1]^T - target_i||²
 113: 
 114:         对 raw 数据进行归一化预处理，避免 polynomial 特征的数值不稳定。
 125:         n = len(self._raw_points)
 126:         if n < 1:
 127:             raise ValueError("校准点不足：需要至少 1 个")
 129:         # 特殊情况：1-2 个点只做平移修正
 130:         if n < 3 and self.method == CalibrationMethod.AFFINE:
 131:             raw_arr = np.array(self._raw_points, dtype=np.float64)
 132:             tgt_arr = np.array(self._target_points, dtype=np.float64)
 133:             offset = np.mean(tgt_arr - raw_arr, axis=0)  # (2,)
 134:             # 构造平移仿射矩阵: [[1, 0, dx], [0, 1, dy]]
 135:             self._transform_matrix = np.array([
 136:                 [1.0, 0.0, offset[0]],
 137:                 [0.0, 1.0, offset[1]],
 138:             ], dtype=np.float64)
 148:         min_points = 6 if self.method == CalibrationMethod.POLYNOMIAL else 3
 149:         if n < min_points:
 150:             raise ValueError(
 151:                 f"校准点不足：{self.method.value} 方法需要至少 {min_points} 个，当前 {n} 个"
 152:             )
 154:         # 归一化 raw 数据（避免 polynomial 特征数值爆炸）
 155:         raw_arr = np.array(self._raw_points, dtype=np.float64)
 156:         self._norm_mean = raw_arr.mean(axis=0)
 157:         self._norm_std = raw_arr.std(axis=0)
 159:         self._norm_std[self._norm_std < 1e-8] = 1.0
 160:         raw_normed = (raw_arr - self._norm_mean) / self._norm_std
 162:         # 构建特征矩阵（使用归一化后的坐标）
 163:         src = np.array(
 164:             [self._build_feature_row(p[0], p[1]) for p in raw_normed],
 165:             dtype=np.float64,
 166:         )
 168:         dst = np.array(self._target_points, dtype=np.float64)  # (N, 2)
 170:         # 最小二乘求解
 171:         result, _, _, _ = np.linalg.lstsq(src, dst, rcond=None)
 172:         self._transform_matrix = result.T  # affine: (2,3), polynomial: (2,6)
 174:         # 计算残差
 175:         predicted = src @ self._transform_matrix.T
 176:         residuals = np.sqrt(np.sum((predicted - dst) ** 2, axis=1))
 177:         self._residual_mean = float(np.mean(residuals))
 178:         self._calibrated = True
 180:         return self._residual_mean
```

这一片段适合论文“校准映射模型”。Affine 特征为 `[x,y,1]`，Polynomial 特征为 `[x,y,xy,x²,y²,1]`，用最小二乘拟合。当前 polynomial 会先对 raw 点做均值方差归一化，避免二次项数值不稳定。论文校准损失可写为 `min_W Σ_i ||W φ(p_i) - q_i||²`，其中 `p_i` 为 raw point，`q_i` 为目标屏幕点。

**代码片段 10-4：`src/ui/calibration_page.py` 与 `src/tracker/pipeline.py`，校准模式禁用 clamp/smoother，171-191 与 1179-1186 行**

```python
 171:     def start_calibration(self) -> None:
 172:         """开始校准流程。"""
 173:         # 重置状态
 174:         self.current_point_index = 0
 175:         self.calibrator.clear_points()
 176:         
 177:         # 启用校准模式（禁用坐标 clamp）
 178:         if self.tracker is not None:
 179:             self.tracker.set_calibration_mode(True)
 180:         
 181:         # 显示全屏
 182:         self.showFullScreen()
 183:         
 184:         # 校准点出现后立即采样，避免每点 3 秒等待影响效率。
 185:         self._start_sampling()
 187:     def _start_sampling(self) -> None:
 188:         """开始采样当前校准点的视线数据。"""
 189:         self.current_samples.clear()
 190:         self.sampling_ticks = 0
 191:         self.sampling_timer.start(33)  # 约 30 FPS
1179:     def set_calibration_mode(self, enabled: bool) -> None:
1180:         """设置校准模式。校准模式下禁用坐标 clamp，保留原始值。"""
1181:         self._calibration_mode = enabled
1182:         if self.smoother is not None:
1183:             self.smoother.reset()
1184:         if self.classic_smoother is not None:
1185:             self.classic_smoother.reset()
1186: 
```

这一片段支撑“校准阶段避免污染 raw point”的论点。校准开始时进入 calibration mode，pipeline 会禁用 clamp，并重置 smoother 状态。论文可写“校准拟合使用未裁剪、未平滑的原始注视点”，这比直接用显示光标点拟合更准确。需要注意 Classic 在 UI 层还有屏幕稳定器，校准采样优先 `raw_point`。

## 11. 平滑模块

**代码片段 11-1：`src/tracker/smoother.py`，类 `GazeSmoother`，10-38 行**

```python
  10: class GazeSmoother:
  11:     """指数移动平均（EMA）时序平滑滤波器。
  12: 
  13:     参数:
  14:         alpha: 平滑系数 (0, 1]，越小越平滑。
  15:                alpha=1 表示不平滑（直接使用新值）。
  16:     """
  17: 
  18:     def __init__(self, alpha: float = 0.3):
  19:         assert 0.0 < alpha <= 1.0, f"alpha 必须在 (0, 1] 范围内，当前为 {alpha}"
  20:         self.alpha = alpha
  21:         self._prev: Optional[tuple[float, float]] = None
  22: 
  23:     def update(self, point: tuple[float, float]) -> tuple[float, float]:
  24:         """输入新的注视点，返回平滑后的注视点。
  25: 
  26:         EMA 公式：smoothed = alpha * new + (1 - alpha) * prev
  27:         """
  28:         if self._prev is None:
  29:             self._prev = point
  30:             return point
  31: 
  32:         sx = self.alpha * point[0] + (1.0 - self.alpha) * self._prev[0]
  33:         sy = self.alpha * point[1] + (1.0 - self.alpha) * self._prev[1]
  34:         self._prev = (sx, sy)
  35:         return (sx, sy)
  36: 
  37:     def reset(self) -> None:
  38:         """重置滤波器状态。"""
```

这一片段适合论文“时序平滑”小节。当前 Deep 配置使用 EMA，`configs/deep.yaml` 中 `type: ema`、`alpha: 0.22`。EMA 公式可写为 `s_t = α x_t + (1-α)s_{t-1}`。较小 `α` 降低抖动但增加响应滞后，项目的 `evaluation_results/smoothing_compare/results.json` 记录了不同 α 的 jitter/RMSE/latency_proxy 对比。

**代码片段 11-2：`src/tracker/classic.py` 与 `src/ui/tracking_page.py`，Classic 屏幕稳定，188-207 与 37-54 行**

```python
 188: class ClassicScreenSmoother:
 189:     """Classic 后端的屏幕空间平滑器：Kalman 预测后叠加历史均值。"""
 190: 
 191:     def __init__(self, history_len: int = 60):
 192:         self.kalman = ClassicKalmanSmoother()
 193:         self.history: deque[tuple[float, float]] = deque(maxlen=history_len)
 194:         self.current: Optional[tuple[float, float]] = None
 195: 
 196:     def reset(self) -> None:
 197:         self.kalman.reset()
 198:         self.history.clear()
 199:         self.current = None
 200: 
 201:     def update(self, point: tuple[float, float]) -> tuple[float, float]:
 202:         predicted = self.kalman.update(point)
 203:         self.history.append(predicted)
 204:         avg_x = sum(p[0] for p in self.history) / len(self.history)
 205:         avg_y = sum(p[1] for p in self.history) / len(self.history)
 206:         self.current = (float(avg_x), float(avg_y))
 207:         return self.current
  37: class ScreenGazeStabilizer:
  38:     """Classic 后端使用的屏幕空间平滑器。"""
  40:     def __init__(
  41:         self,
  42:         median_window: int = 5,
  43:         alpha: float = 0.18,
  44:         fast_alpha: float = 0.32,
  45:         fast_threshold_px: float = 140.0,
  46:         max_step_px: float = 75.0,
  47:     ):
  48:         self._smoother = ClassicScreenSmoother(history_len=60)
  50:     def reset(self) -> None:
  51:         self._smoother.reset()
  53:     def update(self, point: tuple[float, float]) -> tuple[float, float]:
  54:         return self._smoother.update(point)
```

这一片段适合说明 Classic 平滑方式。Classic 配置文件写 `smoother.type: kalman`，UI 层 `ScreenGazeStabilizer` 调用 `ClassicScreenSmoother`，后者使用 Kalman 后再做历史均值。需要注意构造函数里的 `median_window/alpha/fast_alpha` 参数当前没有被实际使用，真实实现只委托给 `ClassicScreenSmoother(history_len=60)`。论文应写“Classic 使用 Kalman + 历史均值稳定”，不要写 OneEuro 是当前默认。

## 12. 训练、导出与评估脚本

**代码片段 12-1：`scripts/train.py`，函数 `load_split_data` / `compute_training_loss` / 训练输入选择，107-177 行节选**

```python
 107: def load_split_data(
 108:     processed_dir: Path, split: str, augment: bool = False,
 109:     model_version: str = "v2",
 110:     target_mode: str = "gaze3d",
 111:     head_pose_mode: str = "stored",
 112: ) -> GazeDataset | None:
 113:     """加载指定划分的数据集。"""
 114:     split_dir = processed_dir / split
 115:     labels_path = split_dir / "labels.csv"
 117:     if not labels_path.exists():
 118:         logger.warning(f"未找到 {split} 数据: {labels_path}")
 119:         return None
 121:     df = pd.read_csv(labels_path)
 122:     logger.info(f"{split} 数据: {len(df)} 样本")
 123:     return GazeDataset(
 124:         df, image_root=split_dir, augment=augment,
 125:         model_version="v2" if model_version == "pog_v1" else model_version,
 126:         target_mode=target_mode,
 127:         head_pose_mode=head_pose_mode,
 128:     )
 140: def compute_training_loss(
 141:     preds: torch.Tensor,
 142:     targets: torch.Tensor,
 143:     target_mode: str,
 144: ) -> torch.Tensor:
 145:     if target_mode == "pog2d":
 146:         return nn.functional.smooth_l1_loss(preds, targets)
 147:     return angular_loss(preds, targets)
 169:             left_eye = batch["left_eye"].to(device)
 170:             right_eye = batch["right_eye"].to(device)
 171:             head_pose = batch["head_pose"].to(device)
 172:             preds = model(left_eye, right_eye, head_pose)
 173:         else:
 174:             eye_imgs = batch["eye_img"].to(device)
 175:             preds = model(eye_imgs)
 177:         loss = compute_training_loss(preds, targets, target_mode)
```

这一片段适合论文“训练流程”。训练脚本从 `dataset_processed/{split}/labels.csv` 读取数据，但当前这些目录为空，因此不能从当前工作区复现实训数据加载。3D gaze 模式使用 `angular_loss`，PoG baseline 使用 Smooth L1。V2 训练输入为左右眼和 head pose。

**代码片段 12-2：`scripts/export_onnx.py`，函数 `export_to_onnx`，24-97 行节选**

```python
  24: def export_to_onnx(
  25:     checkpoint_path: str,
  26:     output_path: str,
  27:     opset_version: int = 11,
  28:     verify: bool = True,
  29: ) -> None:
  30:     """将 PyTorch 模型导出为 ONNX 格式。"""
  31:     print(f"加载 PyTorch 模型: {checkpoint_path}")
  33:     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  35:     # 检测模型版本
  36:     model_version = "v1"
  37:     config = {}
  38:     if isinstance(checkpoint, dict):
  39:         model_version = checkpoint.get("model_version", "v1")
  40:         config = checkpoint.get("config", {})
  45:     # 构建模型
  46:     model_cfg = config.get("model", {})
  47:     channels = model_cfg.get("channels", [32, 64, 128, 256])
  56:     elif model_version == "v2":
  57:         model = GazeNetV2(
  58:             num_channels=channels,
  59:             head_pose_dim=model_cfg.get("head_pose_dim", 3),
  60:             fusion_dim=model_cfg.get("fusion_dim", 128),
  61:             dropout=model_cfg.get("dropout", 0.3),
  62:         )
  66:     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
  67:         model.load_state_dict(checkpoint['model_state_dict'])
  71:     model.eval()
  76:     if model_version in {"v2", "pog_v1"}:
  77:         # V2/PoG：三个输入
  78:         dummy_left = torch.randn(1, 3, 128, 128)
  79:         dummy_right = torch.randn(1, 3, 128, 128)
  80:         dummy_pose = torch.randn(1, 3)
  82:         torch.onnx.export(
  83:             model,
  84:             (dummy_left, dummy_right, dummy_pose),
  85:             output_path,
  86:             export_params=True,
  87:             opset_version=opset_version,
  88:             do_constant_folding=True,
  89:             input_names=['left_eye', 'right_eye', 'head_pose'],
  90:             output_names=['output'],
  91:             dynamic_axes={
  92:                 'left_eye': {0: 'batch_size'},
  93:                 'right_eye': {0: 'batch_size'},
  94:                 'head_pose': {0: 'batch_size'},
  95:                 'output': {0: 'batch_size'},
  96:             },
  97:         )
```

这一片段适合“模型导出与部署优化”。它证明 ONNX V2 模型有三个输入：`left_eye`、`right_eye`、`head_pose`。当前运行配置 `use_onnx: true`，因此 Windows 演示默认走 ONNX Runtime，而不是 PyTorch。论文中可写训练与部署解耦：训练使用 PyTorch，部署使用 ONNX Runtime CPU。

**代码片段 12-3：`scripts/evaluate.py`，函数 `evaluate_on_test`，121-168 行**

```python
 121:         # 逐样本计算角度误差
 122:         for i in range(preds.shape[0]):
 123:             pred_vec = preds[i].numpy()
 124:             true_vec = gaze_targets[i].numpy()
 125: 
 126:             # 角度误差（度）
 127:             cos_sim = np.clip(np.dot(pred_vec, true_vec), -1.0, 1.0)
 128:             angle_deg = np.degrees(np.arccos(cos_sim))
 129:             all_angle_errors.append(angle_deg)
 131:             # 屏幕像素误差（基于归一化目标坐标的近似计算）
 132:             true_nx = float(meta["norm_target_x"][i])
 133:             true_ny = float(meta["norm_target_y"][i])
 135:             # 从视线向量近似推算归一化屏幕坐标
 136:             # 简化：使用 gaze_x/gaze_z 和 gaze_y/gaze_z 的比值
 137:             if abs(pred_vec[2]) > 1e-6:
 138:                 pred_nx = 0.5 + pred_vec[0] / pred_vec[2] * 0.5
 139:                 pred_ny = 0.5 + pred_vec[1] / pred_vec[2] * 0.5
 140:             else:
 141:                 pred_nx, pred_ny = 0.5, 0.5
 143:             pred_nx = np.clip(pred_nx, 0, 1)
 144:             pred_ny = np.clip(pred_ny, 0, 1)
 146:             px_err = np.sqrt(
 147:                 ((pred_nx - true_nx) * screen_w) ** 2
 148:                 + ((pred_ny - true_ny) * screen_h) ** 2
 149:             )
 150:             all_pixel_errors.append(px_err)
 158:     angle_errors = np.array(all_angle_errors)
 159:     pixel_errors = np.array(all_pixel_errors)
 161:     metrics = {
 162:         "mean_angle_error": float(np.mean(angle_errors)),
 163:         "median_angle_error": float(np.median(angle_errors)),
 164:         "mean_pixel_error": float(np.mean(pixel_errors)),
 165:         "std_angle_error": float(np.std(angle_errors)),
 166:         "num_samples": len(angle_errors),
 167:     }
 168: 
```

这一片段适合“固定测试集评估指标”。它证明角度误差按预测向量与真实向量夹角计算，像素误差是从 gaze 比值近似投到归一化屏幕坐标后计算。固定测试集结果保存在 `evaluation_results/metrics.json`，当前为 1390 样本、2.8206°、314.5413 px。论文应说明像素误差是评估脚本的近似投影口径，不等同于实时系统校准后的点击精度。

**代码片段 12-4：`scripts/exp_leave_one_out.py`，函数 `generate_loo_plots`，340-393 行**

```python
 340: def generate_loo_plots(results: list[dict], output_dir: Path) -> None:
 341:     """生成 leave-one-out 可视化图表。"""
 342:     import matplotlib
 343:     matplotlib.use("Agg")
 344:     import matplotlib.pyplot as plt
 346:     output_dir.mkdir(parents=True, exist_ok=True)
 348:     user_ids = [r["left_out_user"] for r in results]
 349:     angle_errors = [r["test_metrics"]["mean_angle_error"] for r in results]
 350:     pixel_errors = [r["test_metrics"]["mean_pixel_error"] for r in results]
 351:     train_sizes = [r["train_samples"] for r in results]
 352:     test_sizes = [r["test_samples"] for r in results]
 354:     # --- 图 1：各用户角度误差柱状图 ---
 355:     fig, ax = plt.subplots(figsize=(10, 5))
 356:     x = np.arange(len(user_ids))
 357:     bars = ax.bar(x, angle_errors, alpha=0.8, color="#2196F3")
 358:     ax.set_xticks(x)
 359:     ax.set_xticklabels([f"User {u}" for u in user_ids], rotation=45, ha="right")
 360:     ax.set_ylabel("Mean Angle Error (degrees)")
 361:     ax.set_title("Leave-One-User-Out: Per-User Angle Error")
 363:     # 标注数值和样本数
 364:     for bar, err, n in zip(bars, angle_errors, test_sizes):
 365:         ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
 366:                 f"{err:.1f}°\n(n={n})", ha="center", va="bottom", fontsize=7)
 368:     # 画均值线
 369:     mean_err = np.mean(angle_errors)
 370:     ax.axhline(mean_err, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_err:.2f}°")
 371:     ax.legend()
 372:     fig.tight_layout()
 373:     fig.savefig(output_dir / "loo_angle_error_per_user.png", dpi=150)
 374:     plt.close(fig)
 376:     # --- 图 2：各用户像素误差柱状图 ---
 377:     fig, ax = plt.subplots(figsize=(10, 5))
 378:     bars = ax.bar(x, pixel_errors, alpha=0.8, color="#FF5722")
 379:     ax.set_xticks(x)
 380:     ax.set_xticklabels([f"User {u}" for u in user_ids], rotation=45, ha="right")
 381:     ax.set_ylabel("Mean Pixel Error (px)")
 382:     ax.set_title("Leave-One-User-Out: Per-User Pixel Error")
 388:     mean_px = np.mean(pixel_errors)
 389:     ax.axhline(mean_px, color="red", linestyle="--", alpha=0.7, label=f"Mean: {mean_px:.0f}px")
 390:     ax.legend()
 391:     fig.tight_layout()
 392:     fig.savefig(output_dir / "loo_pixel_error_per_user.png", dpi=150)
 393:     plt.close(fig)
```

这一片段适合论文第六章“跨用户泛化评估”。当前最终 LOO 结果应使用 `evaluation_results/leave_one_out_resume/summary.json` 和同目录下图表，不应使用旧的 `evaluation_results/leave_one_out/summary.json` 作为主结果。最适合放入第六章的图表包括：`evaluation_results/leave_one_out_resume/loo_angle_error_per_user.png`、`evaluation_results/leave_one_out_resume/loo_pixel_error_per_user.png`、`evaluation_results/error_analysis/error_vs_user.png`、`evaluation_results/calibration_compare/calib_points_vs_error.png`、`evaluation_results/smoothing_compare/jitter_vs_latency.png`、`evaluation_results/3d_contract_runtime/variant_summary.csv`。

## 13. 打包与部署

**代码片段 13-1：`Look2Act.spec`，资源与模型包含，16-70 行**

```python
  16: project_datas = [
  17:     existing(PROJECT_ROOT / "configs", "configs"),
  18:     existing(PROJECT_ROOT / "README.md", "."),
  19:     existing(PROJECT_ROOT / "readme-images", "readme-images"),
  20:     existing(APP_ICON, "."),
  21: ]
  22: 
  23: checkpoint_datas = []
  24: for name in ("gaze_net.onnx", "gaze_pog_zero.onnx", "gaze_pog.onnx"):
  25:     item = existing(PROJECT_ROOT / "checkpoints" / name, "checkpoints")
  26:     if item:
  27:         checkpoint_datas.append(item)
  28: 
  29: mp_datas, mp_binaries, mp_hidden = collect_all("mediapipe")
  30: mp_datas = [
  31:     item for item in mp_datas
  32:     if not (
  33:         Path(item[0]).name == "__init__.py"
  34:         and item[1].replace("\\", "/") in {"mediapipe", "mediapipe/python/solutions"}
  35:     )
  36: ]
  37: qfw_binaries = collect_dynamic_libs("qfluentwidgets")
  38: ort_binaries = collect_dynamic_libs("onnxruntime")
  39: 
  40: cv2_binaries = collect_dynamic_libs("cv2")
  42: hiddenimports = [
  43:     "cv2",
  44:     "numpy",
  45:     "yaml",
  46:     "PyQt6.QtCore",
  47:     "PyQt6.QtGui",
  48:     "PyQt6.QtWidgets",
  49:     "PyQt6.sip",
  50:     "qframelesswindow",
  51:     "mediapipe.python.solutions.face_mesh",
  52:     "onnxruntime",
  53:     "onnxruntime.capi._pybind_state",
  54:     "onnxruntime.capi.onnxruntime_inference_collection",
  55:     "onnxruntime.capi.onnxruntime_pybind11_state",
  56:     *mp_hidden,
  57: ]
  59: datas = [
  60:     *(item for item in project_datas if item),
  61:     *checkpoint_datas,
  62:     *mp_datas,
  63: ]
  65: binaries = [
  66:     *mp_binaries,
  67:     *qfw_binaries,
  68:     *ort_binaries,
  69:     *cv2_binaries,
  70: ]
```

这一片段适合论文“软件打包与部署”。它证明 PyInstaller 会包含配置目录、README、图标、ONNX 模型、MediaPipe、QFluentWidgets、ONNX Runtime 和 OpenCV 动态库。注意 spec 只包含 ONNX，不包含 PyTorch 权重作为运行资源；训练权重仍在仓库中，但 EXE 演示目标是 ONNX Runtime 推理。普通 Windows 用户运行的是 `dist/Look2Act/Look2Act.exe` 目录式发行。

**代码片段 13-2：`Look2Act.spec` 与 `build_exe.bat`，排除训练依赖与构建命令，72-124 与 21-34 行节选**

```python
  72: excludes = [
  73:     "torch",
  74:     "torchvision",
  75:     "torchaudio",
  76:     "torio",
  77:     "torchgen",
  78:     "functorch",
  79:     "intel_extension_for_pytorch",
  80:     "jax",
  81:     "jaxlib",
  82:     "ml_dtypes",
  83:     "onnx",
  84:     "onnxruntime.backend",
  85:     "onnxruntime.datasets",
  86:     "onnxruntime.quantization",
  87:     "onnxruntime.tools",
  88:     "onnxruntime.transformers",
  89:     "pandas",
  90:     "matplotlib",
  91:     "matplotlib.pyplot",
  92:     "matplotlib.backends",
  93:     "sklearn",
 123:     runtime_hooks=[str(PROJECT_ROOT / "tools" / "pyinstaller_rth_mediapipe.py")],
 124:     excludes=excludes,
```

```bat
   1: @echo off
   2: setlocal
   3: 
   4: cd /d "%~dp0"
   5: 
   6: set CONDA_EXE=C:\ProgramData\anaconda3\Scripts\conda.exe
   7: set ENV_NAME=gaze-env
   8: set SPEC_FILE=Look2Act.spec
   9: set OUTPUT_EXE=dist\Look2Act\Look2Act.exe
  10: 
  11: if not exist "%CONDA_EXE%" (
  12:     echo Conda not found: %CONDA_EXE%
  13:     exit /b 1
  14: )
  15: 
  16: if not exist "%SPEC_FILE%" (
  17:     echo Spec file not found: %SPEC_FILE%
  18:     exit /b 1
  19: )
  21: echo Project root: %CD%
  22: echo Input: %SPEC_FILE%, main.py, configs\, checkpoints\, Look2Act.ico
  23: echo Output: %OUTPUT_EXE%
  24: echo.
  25: echo [1/2] Building Look2Act folder EXE...
  26: "%CONDA_EXE%" run --no-capture-output -n %ENV_NAME% python -m PyInstaller --clean --noconfirm "%SPEC_FILE%"
  27: if errorlevel 1 exit /b 1
  29: if not exist "%OUTPUT_EXE%" (
  30:     echo Build finished but output EXE was not found: %OUTPUT_EXE%
  31:     exit /b 1
  32: )
  34: echo [2/2] Build complete: %OUTPUT_EXE%
```

这一片段说明打包脚本使用 conda 环境 `gaze-env` 执行 PyInstaller，并显式排除训练/分析相关重量依赖。论文成果形式可写“目录式 Windows 可执行程序，包含运行时依赖、配置模板和 ONNX 模型”。当前可以确认 `dist/Look2Act` 目录约 466.1 MB；如果论文中写“467 MB EXE”，应改为“467 MB 左右的发行目录”，因为单个 EXE 文件不是 467 MB。

## 14. 论文可用代码片段清单

| 代码片段名称 | 文件路径 | 正文/附录 | 支撑章节 | 建议图号/表号 | 是否必须人工复核 |
| --- | --- | --- | --- | --- | --- |
| 启动参数与 YAML 校验 | `main.py` | 正文 | 第五章 软件启动流程 | 图5-1 | 否 |
| AppData 用户配置路径 | `src/runtime_paths.py` | 正文 | 第五章 配置管理 | 图5-2 | 否 |
| 启动弹窗保存 Classic/Deep 模式 | `src/ui/language_dialog.py` | 正文 | 第五章 模式选择 | 图5-1 | 否 |
| 主窗口页面组织 | `src/ui/main_window.py` | 正文 | 第五章 UI 架构 | 图5-3 | 否 |
| 校准和追踪页面跳转 | `src/ui/main_window.py` | 正文 | 第五章 用户流程 | 图5-4 | 否 |
| 验证、交互、五子棋流程 | `src/ui/tracking_page.py` | 正文/附录 | 第五章 交互演示 | 图5-5 | 否 |
| TrackerPipeline 分叉流程 | `src/tracker/pipeline.py` | 正文 | 第四章/第五章 实时管线 | 图4-1 | 否 |
| Deep ONNX 输入预处理 | `src/tracker/pipeline.py` | 正文 | 第四章 推理实现 | 图4-2 | 否 |
| Classic 瞳孔质心提取 | `src/tracker/classic.py` | 正文 | 第四章 传统视觉后端 | 算法4-1 | 否 |
| GazeNetV2 模型结构 | `src/models/gaze_net.py` | 正文 | 第四章 深度模型 | 图4-3 | 否 |
| Collector 合成图眼区提取 | `src/data/preprocessing.py` | 正文 | 第三章 数据处理 | 图3-2 | 否 |
| labels/meta 读取与 valid 过滤 | `src/data/pipeline.py` | 正文/附录 | 第三章 数据清洗 | 表3-1 | 是，因当前数据目录为空 |
| FaceMesh 与 PnP 点 | `src/vision/face_detector.py` | 正文 | 第四章 视觉前端 | 图4-1 | 否 |
| solvePnP 头姿估计 | `src/vision/head_pose.py` | 正文 | 第四章 头姿估计 | 公式4-1 | 否 |
| 射线-屏幕平面求交 | `src/geometry/ray_plane.py` | 正文 | 第四章 几何映射 | 公式4-2 | 否 |
| 校准最小二乘拟合 | `src/calibration/calibrator.py` | 正文 | 第四章 校准映射 | 公式4-3 | 否 |
| EMA 平滑 | `src/tracker/smoother.py` | 正文 | 第四章 平滑模块 | 公式4-4 | 否 |
| 训练损失与输入选择 | `scripts/train.py` | 附录/正文 | 第六章 训练设置 | 表6-1 | 是，因训练数据目录为空 |
| ONNX 导出 | `scripts/export_onnx.py` | 正文 | 第五章/第六章 部署 | 图5-6 | 否 |
| 固定测试集评估 | `scripts/evaluate.py` | 正文 | 第六章 实验结果 | 表6-2 | 否 |
| LOO 图表生成 | `scripts/exp_leave_one_out.py` | 正文 | 第六章 泛化实验 | 图6-1/图6-2 | 否 |
| PyInstaller 打包 | `Look2Act.spec` / `build_exe.bat` | 正文 | 第五章 软件成果 | 图5-7 | 否 |

## 15. 论文不可乱写清单

| 不可直接声称 | 原因 | 更安全论文写法 |
| --- | --- | --- |
| 系统能精确替代鼠标 | 当前代码提供停留点击与交互演示，但没有系统级 Fitts law、点击成功率或长期可用性评估。 | “实现了注视点显示、停留触发和应用启动等交互原型，可作为鼠标替代方向的探索。” |
| 已完成鲁棒头动补偿 | 当前 Deep demo 使用 `pose zero` 和 `camera` gaze space，不是 live head pose 补偿最终方案。 | “系统实现了 PnP 头姿估计，并保留 head-space 转换；当前最佳演示配置采用 pose zero 以降低实时几何不一致。” |
| 所有用户都高精度可用 | LOO 有长尾用户，像素误差仍较大。 | “31 用户 LOO 平均角度误差为 2.48°，多数用户表现较好，但仍存在长尾用户和像素映射误差。” |
| Deep 在所有交互场景都优于 Classic | Classic 是稳定产品 fallback，Deep 是研究链路；实时交互可靠性未证明全面优于 Classic。 | “Classic 用于稳定演示，Deep 用于模型与几何研究，两者面向不同目标。” |
| 当前数据目录仍包含完整数据 | `dataset_raw` 和 `dataset_processed` 当前为空目录。 | “当前仓库保留了评估产物和处理流程代码；完整原始数据需另行归档或补充。” |
| 9 点 affine 是最终实时系统配置 | 当前 `configs/deep.yaml` 和 `configs/classic.yaml` 均写 25 点；9 点 affine 主要出现在 fallback 或校准对比实验。 | “9 点 affine 是离线对比/未显式配置时的 fallback；当前演示配置采用 25 点 polynomial。” |
| 16 台设备是当前可核查最终统计 | 当前评估 CSV 可解析出 5 台设备；旧中期文档写 16 台但缺少当前 meta 文件支撑。 | “当前可核查评估产物覆盖 31 用户、7395 样本，设备口径需以补充 meta 统计为准。” |
| 当前 README 图片全部可用 | `readme-images` 当前只有 `.gitkeep`，README 引用图片缺失。 | “README 描述的功能与代码基本一致，但展示图片资源需补齐。” |
| 467 MB 是单个 EXE 文件大小 | 当前 `Look2Act.exe` 约 14.4 MB，`dist/Look2Act` 发行目录约 466.1 MB。 | “目录式发行包约 466 MB，包含 EXE、依赖库、配置和 ONNX 模型。” |

## 当前代码与旧文档冲突或需人工确认

1. 设备数量冲突：中期文档写 16 台设备，研究日志曾写 4 台设备，当前评估 CSV 可核查为 5 台设备口径。论文最终建议不要写 16 台，除非补充 `meta.json` 汇总表或采集系统统计截图。
2. 数据目录冲突：旧文档可能按完整数据目录叙述，但当前 `dataset_raw` / `dataset_processed` 为空。论文应引用 `evaluation_results`，并说明完整原始数据不在当前工作区。
3. Deep 校准冲突：`SystemConfig` fallback 是 Deep 9 点 affine，但当前 `configs/deep.yaml` 是 25 点 polynomial。论文主线应按当前 YAML 写，附注 fallback/旧实验口径。
4. Deep 几何口径冲突：旧 head-space/live-pose 描述与当前 `configs/deep.yaml` 的 `camera + zero_origin + pose zero + swap + EMA` 不一致。论文实时系统描述必须按当前配置。
5. LOO 结果冲突：`evaluation_results/leave_one_out/summary.json` 是旧 10 用户结果；`evaluation_results/leave_one_out_resume/summary.json` 是当前 31 用户结果。论文第六章主结果用后者。
6. README 资源冲突：README 引用 `readme-images` 下多张图，但当前目录没有这些图。若论文截图来自软件，需要重新截图或使用 `docs/midterm/images` 中已存在的非隐私图片。
7. EXE 体积口径冲突：可以确认发行目录约 466.1 MB，不能确认单个 EXE 为 467 MB。论文写“软件发行目录/打包产物约 466 MB”更准确。
