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
from pathlib import Path

import yaml
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from qfluentwidgets import setTheme, Theme, setThemeColor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse Look2Act CLI arguments before Qt starts."""
    parser = argparse.ArgumentParser(description="Look2Act Tracker")
    parser.add_argument(
        "--config",
        default="configs/system_config.yaml",
        help="Path to the system YAML config used by UI, tracker, and settings page.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def validate_config_path(config_path: Path) -> tuple[bool, str]:
    """Validate the startup config before any UI writes to it."""
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
    
    # 应用 Gaze_Dataset_Collector 中的品牌色和自定义样式表
    from src.ui.fluent_theme import apply_fluent_theme
    apply_fluent_theme(app)
    
    return app


def main() -> int:
    """应用主函数。
    
    返回:
        退出代码（0 表示成功）
    """
    try:
        args = parse_args(sys.argv[1:])
        sys.argv = [sys.argv[0]]
        config_path = Path(args.config)
        ok, error_message = validate_config_path(config_path)
        if not ok:
            logger.error(error_message)
            return 2
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
        logger.info(f"界面语言已设置: {language}")
        
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

