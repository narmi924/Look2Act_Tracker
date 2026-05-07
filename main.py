"""Look2Act Tracker 应用入口。

串联 GUI 与 TrackerPipeline，处理应用生命周期。

主要功能：
1. 创建 QApplication
2. 创建主窗口
3. 显示主窗口
4. 处理应用退出（释放资源）
5. 异常处理和日志记录

使用方法：
    conda run -n gaze-env python main.py
"""
import sys
import logging
import argparse
import os
from pathlib import Path

# Windows 下先导入 PyQt6 可能影响 MediaPipe DLL 加载，先预热依赖路径。
if sys.platform == "win32":
    _dll_directory_handles = []
    _bundle_dir = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    _mediapipe_dll_dir = _bundle_dir / "mediapipe" / "python"
    if _mediapipe_dll_dir.exists():
        try:
            _dll_directory_handles.append(os.add_dll_directory(str(_mediapipe_dll_dir)))
        except (AttributeError, OSError):
            pass
        os.environ["PATH"] = f"{_mediapipe_dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    try:
        from mediapipe.python.solutions import face_mesh as _mp_face_mesh  # noqa: F401
    except Exception:
        pass

import yaml
from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt

from qfluentwidgets import setTheme, Theme, setThemeColor

from src.runtime_paths import resource_path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """在 Qt 启动前解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Look2Act Tracker")
    parser.add_argument(
        "--config",
        default="configs/classic.yaml",
        help="Path to the system YAML config used by UI, tracker, and settings page.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args, _ = parser.parse_known_args(argv)
    return args


def validate_config_path(config_path: Path) -> tuple[bool, str]:
    """在界面写入配置前校验启动配置文件。"""
    if not config_path.exists():
        return False, (
            f"配置文件不存在：{config_path}\n"
            "如果你在 Git Bash 中运行，请使用正斜杠，例如：\n"
            "python main.py --config configs/experiments/system_deep_camera_zero_720.yaml"
        )
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        return False, f"配置文件读取失败：{config_path}\n{e}"
    required_sections = {"camera", "model", "tracker"}
    missing = sorted(section for section in required_sections if section not in data)
    if missing:
        return False, (
            f"配置文件不完整：{config_path}\n"
            f"缺少配置节：{', '.join(missing)}\n"
            "这通常是 Git Bash 反斜杠路径被转义后产生的错误文件。"
        )
    return True, ""


def setup_application() -> QApplication:
    """设置 QApplication。
    
    配置应用属性、样式等。
    
    返回:
        QApplication 实例
    """
    # 创建应用
    app = QApplication(sys.argv)
    app.setApplicationName("Look2Act Tracker")
    app.setOrganizationName("Look2Act")
    app.setApplicationVersion("1.0.0")
    
    # 应用 Look2Act 的品牌色和自定义样式表。
    from src.ui.fluent_theme import apply_fluent_theme
    apply_fluent_theme(app)
    
    return app


def _patch_modal_dialogs_for_smoke() -> None:
    """避免自动化冒烟检查被模态对话框阻塞。"""
    def _log_box(*args, **kwargs):
        title = args[1] if len(args) > 1 else ""
        text = args[2] if len(args) > 2 else ""
        logger.info("[SMOKE] suppressed message box: %s %s", title, text)
        return QMessageBox.StandardButton.Ok

    QMessageBox.critical = _log_box  # type: ignore[method-assign]
    QMessageBox.warning = _log_box  # type: ignore[method-assign]
    QMessageBox.information = _log_box  # type: ignore[method-assign]
    QMessageBox.question = lambda *args, **kwargs: QMessageBox.StandardButton.No  # type: ignore[method-assign]


def run_smoke_test() -> int:
    """执行非交互式启动、页面切换和按钮冒烟检查。"""
    _patch_modal_dialogs_for_smoke()

    try:
        from src.runtime_paths import app_data_dir
        smoke_log_path = app_data_dir() / "smoke-test.log"
        smoke_log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(smoke_log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(file_handler)
        logger.info("[SMOKE] file log: %s", smoke_log_path)
    except Exception as e:
        logger.warning("[SMOKE] could not enable file logging: %s", e)

    app = setup_application()
    logger.info("[SMOKE] QApplication ready")

    from src.ui.i18n import load_language
    from src.ui.language_dialog import LanguageSelectionDialog
    from src.ui.main_window import MainWindow
    import src.ui.calibration_page as calibration_page
    import src.ui.settings_page as settings_page

    class _SmokeMessageBox:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self) -> int:
            return 0

    calibration_page.MessageBox = _SmokeMessageBox  # type: ignore[assignment]
    settings_page.detect_supported_camera_resolutions = lambda *args, **kwargs: []  # type: ignore[assignment]

    for mode in ("classic", "deep"):
        config_path = LanguageSelectionDialog._save_startup_config("bilingual", mode)
        ok, error_message = validate_config_path(config_path)
        if not ok:
            logger.error("[SMOKE] %s config invalid: %s", mode, error_message)
            return 2

        load_language(config_path)
        dialog = LanguageSelectionDialog(config_path=config_path)
        for language in ("zh", "en", "bilingual"):
            dialog._select_language(language)
        dialog._set_mode(mode)
        dialog.close()

        window = MainWindow(config_path=config_path)
        window.show()
        app.processEvents()
        if not window.isFullScreen():
            logger.error("[SMOKE] %s window is not fullscreen", mode)
            window.close()
            return 3

        window.go_home()
        window.go_camera()
        app.processEvents()
        window.page_camera._handle_start()
        app.processEvents()
        window.page_camera._handle_stop()

        window.go_settings()
        app.processEvents()
        window.page_settings._toggle_advanced_settings()
        window.page_settings._handle_detect_camera_resolutions()
        window.page_settings._handle_save()
        window.page_settings._handle_reset()

        window.go_tracking()
        app.processEvents()
        window.page_tracking._handle_load_calibration()
        window.page_tracking._toggle_diagnostics()
        window.page_tracking._handle_start_tracking()
        app.processEvents()
        tracking_error = window.page_tracking.error_label.text().strip()
        if window.page_tracking.start_btn.isEnabled() or tracking_error.startswith(("错误", "Error", "TrackerPipeline")):
            logger.error("[SMOKE] %s tracking start failed: %s", mode, tracking_error)
            window.close()
            return 4
        tracker = window.page_tracking.tracker
        if tracker is None or getattr(tracker.face_detector, "unavailable", False):
            logger.error("[SMOKE] %s tracking entered protection mode instead of real FaceDetector", mode)
            window.close()
            return 5
        window.page_tracking._handle_stop_tracking()

        window.switchTo(window.page_calibration)
        app.processEvents()
        window.page_calibration._handle_save_calibration()
        window.page_calibration._handle_load_calibration()

        window.close()
        app.processEvents()
        logger.info("[SMOKE] %s passed", mode)

    app.quit()
    logger.info("[SMOKE] completed")
    return 0


def main() -> int:
    """应用主函数。
    
    返回:
        退出代码（0 表示成功）
    """
    try:
        args = parse_args(sys.argv[1:])
        sys.argv = [sys.argv[0]]
        if args.smoke_test:
            return run_smoke_test()
        config_path = resource_path(args.config)
        ok, error_message = validate_config_path(config_path)
        if not ok:
            logger.warning(error_message)
        logger.info("=" * 60)
        logger.info("Look2Act Tracker 启动中...")
        logger.info(f"系统配置路径: {config_path}")
        logger.info("=" * 60)
        
        # 创建应用
        app = setup_application()
        logger.info("QApplication 已创建")

        from src.ui.i18n import load_language
        from src.ui.language_dialog import LanguageSelectionDialog

        load_language(config_path)
        dialog = LanguageSelectionDialog(config_path=config_path)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return 0
        language = dialog.selected_language
        config_path = dialog.selected_config_path
        ok, error_message = validate_config_path(config_path)
        if not ok:
            logger.error(error_message)
            return 2
        logger.info(f"界面语言已设置: {language}")
        logger.info(f"演示模式已设置: {dialog.selected_mode}")
        logger.info(f"系统配置路径: {config_path}")
        
        # 导入主窗口（延迟导入，避免在 QApplication 创建前导入 Qt 组件）
        from src.ui.main_window import MainWindow
        
        # 创建主窗口
        main_window = MainWindow(config_path=config_path)
        logger.info("主窗口已创建")
        
        # 显示主窗口
        main_window.show()
        logger.info("主窗口已显示")
        
        logger.info("=" * 60)
        logger.info("Look2Act Tracker 已启动")
        logger.info("按 ESC 键返回主页或退出应用")
        logger.info("=" * 60)
        
        # 运行应用事件循环
        exit_code = app.exec()
        
        logger.info("=" * 60)
        logger.info("Look2Act Tracker 已退出")
        logger.info("=" * 60)
        
        return exit_code
        
    except Exception as e:
        logger.exception(f"应用启动失败：{e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

