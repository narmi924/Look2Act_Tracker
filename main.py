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
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


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
    
    return app


def main() -> int:
    """应用主函数。
    
    返回:
        退出代码（0 表示成功）
    """
    try:
        logger.info("=" * 60)
        logger.info("Look2Act Tracker 启动中...")
        logger.info("=" * 60)
        
        # 创建应用
        app = setup_application()
        logger.info("QApplication 已创建")
        
        # 导入主窗口（延迟导入，避免在 QApplication 创建前导入 Qt 组件）
        from src.ui.main_window import MainWindow
        
        # 创建主窗口
        main_window = MainWindow()
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

