"""测试实时追踪页面。

简单的测试脚本，用于验证 TrackingPage 的基本功能。
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication
from src.ui.tracking_page import TrackingPage


def main():
    """主函数。"""
    app = QApplication(sys.argv)
    
    # 创建追踪页面
    page = TrackingPage()
    page.setWindowTitle("实时追踪测试 / Tracking Page Test")
    page.resize(1000, 800)
    page.show()
    
    print("[TEST] TrackingPage 已创建并显示")
    print("[TEST] 请点击\"启动追踪\"按钮测试功能")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
