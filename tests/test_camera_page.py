"""测试摄像头预览页面。

验证 CameraPage 的核心功能。
"""
import sys
from pathlib import Path

import pytest

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication

from src.ui.camera_page import CameraPage
from src.ui.camera_stream import Resolution


@pytest.fixture(scope="module")
def qapp():
    """创建 QApplication 实例（整个测试模块共享）。"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_camera_page_creation(qapp):
    """测试 CameraPage 可以正常创建。"""
    page = CameraPage()
    assert page is not None
    assert page.windowTitle() == ""  # 默认无标题
    
    # 验证关键组件存在
    assert page.preview_label is not None
    assert page.start_btn is not None
    assert page.stop_btn is not None
    assert page.fps_value is not None
    assert page.detect_value is not None
    assert page.error_label is not None


def test_camera_page_initial_state(qapp):
    """测试 CameraPage 的初始状态。"""
    page = CameraPage()
    
    # 初始状态：启动按钮启用，停止按钮禁用
    assert page.start_btn.isEnabled() is True
    assert page.stop_btn.isEnabled() is False
    
    # FPS 初始值为 0
    assert page.fps_value.text() == "0.0"
    
    # 检测状态初始值
    assert "未启动" in page.detect_value.text() or "Not Started" in page.detect_value.text()


def test_resolution_class():
    """测试 Resolution 类。"""
    res = Resolution(w=1920, h=1080)
    assert res.w == 1920
    assert res.h == 1080
    assert res.label() == "1920x1080"
    
    res2 = Resolution(w=1280, h=720)
    assert res2.label() == "1280x720"


def test_camera_page_cleanup(qapp):
    """测试 CameraPage 可以正常清理资源。"""
    page = CameraPage()
    
    # 模拟关闭事件
    from PyQt6.QtGui import QCloseEvent
    event = QCloseEvent()
    page.closeEvent(event)
    
    # 验证摄像头流已停止
    assert page._stream.is_running() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
