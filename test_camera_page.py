"""测试摄像头预览页面。

运行此脚本以测试摄像头预览页面的功能。
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication

from src.ui.camera_page import CameraPage


def main():
    """主函数。"""
    app = QApplication(sys.argv)
    
    # 创建摄像头预览页面
    page = CameraPage()
    page.setWindowTitle("Look2Act Tracker - 摄像头预览测试")
    page.resize(1400, 900)
    page.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
