"""测试校准页面的基本功能。"""
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication
from src.ui.calibration_page import CalibrationPage


def test_calibration_page_creation():
    """测试校准页面可以正常创建。"""
    app = QApplication(sys.argv)
    
    # 创建校准页面
    page = CalibrationPage()
    
    # 验证基本属性
    assert page.calibrator is not None
    assert page.calibrator.num_points == 9
    assert page.calibrator.max_residual_px == 50.0
    
    # 验证 UI 组件存在
    assert page.start_btn is not None
    assert page.save_btn is not None
    assert page.load_btn is not None
    assert page.status_label is not None
    assert page.residual_label is not None
    
    # 验证初始状态
    assert not page.calibration_success
    assert page.calibration_residual == 0.0
    assert not page.save_btn.isEnabled()
    
    print("✓ 校准页面创建测试通过")
    
    # 显示窗口（可选，用于手动检查）
    # page.show()
    # sys.exit(app.exec())


if __name__ == "__main__":
    test_calibration_page_creation()
    print("\n所有测试通过！")
