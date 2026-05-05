"""实时追踪页面单元测试。

验证 TrackingPage 和 GazeCursorOverlay 的基本功能。
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PyQt6.QtWidgets import QApplication

from src.ui.tracking_page import TrackingPage, GazeCursorOverlay


@pytest.fixture(scope="module")
def qapp():
    """创建 QApplication 实例（整个测试模块共享）。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_gaze_cursor_overlay_creation(qapp):
    """测试 GazeCursorOverlay 创建。"""
    overlay = GazeCursorOverlay()
    
    # 验证初始状态
    assert overlay.gaze_x is None
    assert overlay.gaze_y is None
    assert overlay.cursor_radius == 15
    
    # 验证窗口属性
    from PyQt6.QtCore import Qt
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
    overlay.close()


def test_gaze_cursor_overlay_update(qapp):
    """测试 GazeCursorOverlay 更新注视点。"""
    overlay = GazeCursorOverlay()
    
    # 更新注视点
    overlay.update_gaze_point(100.0, 200.0)
    assert overlay.gaze_x == 100.0
    assert overlay.gaze_y == 200.0
    
    # 清除注视点
    overlay.clear_gaze_point()
    assert overlay.gaze_x is None
    assert overlay.gaze_y is None
    
    overlay.close()


def test_tracking_page_creation(qapp):
    """测试 TrackingPage 创建。"""
    page = TrackingPage()
    
    # 验证初始状态
    assert page.tracker is None
    assert page.calibrator is None
    assert page.cursor_overlay is None
    
    # 验证 UI 组件存在
    assert page.start_btn is not None
    assert page.stop_btn is not None
    assert page.load_calib_btn is not None
    assert page.fps_value is not None
    assert page.face_status is not None
    assert page.gaze_status is not None
    assert page.calib_status is not None
    
    # 验证按钮初始状态
    assert page.start_btn.isEnabled()
    assert not page.stop_btn.isEnabled()
    
    page.close()


def test_tracking_page_timing_labels(qapp):
    """测试 TrackingPage 各阶段延迟标签。"""
    page = TrackingPage()
    
    # 验证所有阶段的标签都已创建
    expected_stages = [
        "face_detection",
        "head_pose",
        "gaze_regression",
        "coordinate_transform",
        "ray_plane_intersect",
        "smoothing",
    ]
    
    for stage in expected_stages:
        assert stage in page.timing_labels
        assert page.timing_labels[stage] is not None
    
    page.close()


def test_tracking_page_load_calibration_no_file(qapp):
    """测试加载不存在的校准文件。"""
    page = TrackingPage()
    
    # 确保校准文件不存在
    calib_path = Path("calibration_classic.json")
    if calib_path.exists():
        pytest.skip("校准文件已存在，跳过此测试")
    
    # 尝试加载校准（应该显示警告对话框，但不会崩溃）
    # 注意：这里不实际调用 _handle_load_calibration，因为它会弹出对话框
    # 只验证初始状态
    assert page.calibrator is None
    
    page.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
