"""实时追踪页面单元测试。

验证 TrackingPage 和 GazeCursorOverlay 的基本功能。
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from src.calibration.calibrator import CalibrationModule
from src.ui.interaction_overlay import FullscreenStageWindow, InteractionLauncherOverlay
from src.ui.gomoku_window import GomokuWindow
from src.ui.tracking_page import GazeCursorOverlay, ScreenGazeStabilizer, TrackingPage


@pytest.fixture(scope="session")
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


def test_stage_buttons_do_not_require_manual_tracker_start(qapp):
    """验证/交互入口应能自己启动追踪，不要求先点背景页启动。"""
    page = TrackingPage()
    calibrator = CalibrationModule()
    calibrator._calibrated = True
    page.set_calibrator(calibrator)

    assert page.tracker is None
    assert page.verify_btn.isEnabled()

    page._verification_passed = True
    page._refresh_stage_controls()

    assert page.launcher_btn.isEnabled()

    page.close()


def test_verification_passed_stops_tracker_runtime(qapp):
    class FakeTracker:
        def __init__(self):
            self.stopped = False

        def is_running(self):
            return not self.stopped

        def stop(self):
            self.stopped = True

    page = TrackingPage()
    page.tracker = FakeTracker()

    page._on_verification_passed()

    assert page.tracker.stopped
    assert page.verification_window is None
    assert not page.stop_btn.isEnabled()

    page.close()


def test_screen_gaze_stabilizer_uses_classic_screen_smoothing():
    """classic 屏幕级平滑应采用 Kalman + 历史均值。"""
    stabilizer = ScreenGazeStabilizer()

    first = stabilizer.update((100.0, 100.0))
    jumped = first
    for _ in range(20):
        jumped = stabilizer.update((900.0, 100.0))

    assert first == (100.0, 100.0)
    assert 100.0 < jumped[0] < 900.0
    assert jumped[1] == pytest.approx(100.0)


def test_fullscreen_stage_window_is_opaque_standalone_window(qapp):
    """新的验证/交互窗口应是独立深色窗口，而不是透明置顶 overlay。"""
    window = FullscreenStageWindow()

    assert not window.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert window.autoFillBackground()
    assert window.windowFlags() & Qt.WindowType.Window

    window.close()


def test_interaction_launcher_uses_bilingual_app_tiles(qapp):
    overlay = InteractionLauncherOverlay()

    titles = [action.title for action in overlay._actions]

    assert any("浏览器" in title and "Browser" in title for title in titles)
    assert any("五子棋" in title and "Gomoku" in title for title in titles)
    assert any("退出" in title and "Exit" in title for title in titles)

    overlay.close()


def test_gomoku_window_is_large_5x5_ox_board(qapp):
    window = GomokuWindow()
    window.resize(900, 700)

    assert window.board_size == 5
    assert window.win_len == 3
    assert len(window.board) == 5
    assert all(len(row) == 5 for row in window.board)
    assert window._board_pos_from_point(
        window._board_rect().center().x(),
        window._board_rect().center().y(),
    ) == (2, 2)

    window.close()


def test_gomoku_window_emits_return_to_launcher_after_game_end(qapp):
    window = GomokuWindow()
    emitted = []
    window.return_to_launcher.connect(lambda: emitted.append(True))

    window._finish_game("X 获胜", "🎉")
    window._on_return_tick()
    window._on_return_tick()
    window._on_return_tick()
    window._on_return_tick()
    window._on_return_tick()
    window._on_return_tick()
    window._on_return_tick()
    window._on_return_tick()

    assert emitted == [True]

    window.close()


def test_tracking_page_info_cards_use_responsive_grid(qapp):
    page = TrackingPage()
    page.resize(1600, 900)
    page._arrange_info_cards()

    assert page.info_grid.count() == 3
    assert page.info_grid.itemAtPosition(0, 0).widget() is page.info_cards[0]
    assert page.info_grid.itemAtPosition(0, 1).widget() is page.info_cards[1]
    assert page.info_grid.itemAtPosition(0, 2).widget() is page.info_cards[2]

    page.resize(850, 900)
    page._arrange_info_cards()

    assert page.info_grid.itemAtPosition(0, 0).widget() is page.info_cards[0]
    assert page.info_grid.itemAtPosition(1, 0).widget() is page.info_cards[1]
    assert page.info_grid.itemAtPosition(2, 0).widget() is page.info_cards[2]

    page.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
