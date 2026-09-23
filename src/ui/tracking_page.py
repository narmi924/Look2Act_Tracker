from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import BodyLabel, CardWidget, PrimaryPushButton, PushButton

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import load_calibration
from src.tracker.classic import ClassicScreenSmoother
from src.tracker.screen_mapping import ScreenMapper
from src.tracker.pipeline import SystemConfig, TrackerPipeline
from src.tracker.observation import ObservationGate, ObservationState, validate_max_age
from src.ui.calibration_page import calibration_module_for_config
from src.ui.fluent_theme import PALETTE
from src.ui.i18n import tx, tx_button
from src.ui.interaction_overlay import (
    GazeVerificationWindow,
    InteractionLauncherOverlay,
    perform_left_click,
)
from src.ui.gomoku_window import GomokuWindow


class ScreenGazeStabilizer:
    """Classic 后端使用的屏幕空间平滑器。"""

    def __init__(
        self,
        median_window: int = 5,
        alpha: float = 0.18,
        fast_alpha: float = 0.32,
        fast_threshold_px: float = 140.0,
        max_step_px: float = 75.0,
    ):
        self._smoother = ClassicScreenSmoother(history_len=60)

    def reset(self) -> None:
        self._smoother.reset()

    def update(self, point: tuple[float, float]) -> tuple[float, float]:
        return self._smoother.update(point)


class GazeCursorOverlay(QWidget):
    """绘制当前注视点的全屏透明浮层。"""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.cursor_radius = 15
        self.cursor_color = QColor(255, 0, 0, 150)
        self.cursor_border_color = QColor(255, 255, 255, 200)
        self.cursor_border_width = 2

        self.gaze_x: Optional[float] = None
        self.gaze_y: Optional[float] = None
        self.dwell_enabled = False
        self.dwell_progress = 0.0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        screen = QApplication.primaryScreen()
        if screen is not None:
            self.setGeometry(screen.geometry())

    def update_gaze_point(self, x: float, y: float) -> None:
        self.gaze_x = x
        self.gaze_y = y
        self.update()

    def set_dwell_progress(self, progress: float, enabled: bool) -> None:
        self.dwell_enabled = enabled
        self.dwell_progress = max(0.0, min(progress, 1.0))
        self.update()

    def clear_gaze_point(self) -> None:
        self.gaze_x = None
        self.gaze_y = None
        self.dwell_enabled = False
        self.dwell_progress = 0.0
        self.update()

    def paintEvent(self, event) -> None:
        if self.gaze_x is None or self.gaze_y is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QPen(self.cursor_border_color, self.cursor_border_width))
        painter.setBrush(QBrush(self.cursor_color))
        painter.drawEllipse(
            int(self.gaze_x - self.cursor_radius),
            int(self.gaze_y - self.cursor_radius),
            self.cursor_radius * 2,
            self.cursor_radius * 2,
        )

        center_radius = 3
        painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(self.gaze_x - center_radius),
            int(self.gaze_y - center_radius),
            center_radius * 2,
            center_radius * 2,
        )

        if self.dwell_enabled:
            ring_radius = self.cursor_radius + 10
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 255, 70), 4))
            painter.drawEllipse(
                int(self.gaze_x - ring_radius),
                int(self.gaze_y - ring_radius),
                ring_radius * 2,
                ring_radius * 2,
            )

            if self.dwell_progress > 0.0:
                painter.setPen(QPen(QColor(255, 214, 10, 230), 4))
                painter.drawArc(
                    int(self.gaze_x - ring_radius),
                    int(self.gaze_y - ring_radius),
                    ring_radius * 2,
                    ring_radius * 2,
                    90 * 16,
                    -int(self.dwell_progress * 360 * 16),
                )


class TrackingPage(QWidget):
    def __init__(self, parent: Optional[QWidget] = None, config_path: Path | str | None = None):
        super().__init__(parent)

        self.tracker: Optional[TrackerPipeline] = None
        self.tracker_config: Optional[SystemConfig] = None
        self.config_path = Path(config_path) if config_path is not None else Path("configs/system_config.yaml")
        self.calibrator: Optional[CalibrationModule] = None

        self.cursor_overlay: Optional[GazeCursorOverlay] = None
        self.verification_window: Optional[GazeVerificationWindow] = None
        self.interaction_overlay: Optional[InteractionLauncherOverlay] = None
        self.gomoku_window: Optional[GomokuWindow] = None

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_tracking_data)
        self.update_interval_ms = 33

        self.dwell_enabled = False
        self.dwell_ms = 1000.0
        self.dwell_radius_px = 42.0
        self._dwell_anchor: Optional[tuple[float, float]] = None
        self._dwell_started_at = 0.0
        self._dwell_cooldown_until = 0.0
        self._reopen_launcher_after_gomoku = False
        self._launcher_return_timer = QTimer(self)
        self._launcher_return_timer.setSingleShot(True)
        self._launcher_return_timer.timeout.connect(self._return_to_launcher_if_running)

        self._verification_passed = False
        self.diagnostics_enabled = False
        self.show_cursor_overlay = False
        self.screen_stabilizer = ScreenGazeStabilizer()
        self.screen_mapper = ScreenMapper(self.screen_stabilizer)
        self.observation_gate = ObservationGate()
        self._observation_context = None

        self._init_ui()
        self._refresh_stage_controls()

    def set_calibrator(self, calibrator: CalibrationModule) -> None:
        self._reset_observation_context()
        self.calibrator = calibrator
        self.screen_mapper.reset()
        if self.calibrator.is_calibrated:
            residual = self.calibrator.residual_mean
            self.calib_status.setText(tx(f"已加载（残差：{residual:.2f} px）", f"Loaded (Residual: {residual:.2f} px)"))
            self.calib_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self._mark_verification_required(tx("待验证", "Verification Required"))
        else:
            self.calib_status.setText(tx("未加载", "Not Loaded"))
            self.calib_status.setStyleSheet("color: #888;")
            self._mark_verification_required(tx("待校准", "Pending Calibration"))

    def start_verification_flow(self) -> None:
        self._handle_start_tracking()
        if self.calibrator is not None and self.calibrator.is_calibrated:
            self._open_verification_window()

    def _init_ui(self) -> None:
        title = QLabel(tx("实时追踪", "Real-time Tracking"))
        title.setStyleSheet("font-size: 32px; font-weight: 900;")

        self.start_btn = PrimaryPushButton(tx_button("启动追踪", "Start Tracking"))
        self.start_btn.setFixedSize(160, 60)
        self.start_btn.clicked.connect(self._handle_start_tracking)

        self.stop_btn = PushButton(tx_button("停止追踪", "Stop Tracking"))
        self.stop_btn.setFixedSize(160, 60)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._handle_stop_tracking)

        self.load_calib_btn = PushButton(tx_button("加载校准", "Load Calibration"))
        self.load_calib_btn.setFixedSize(160, 60)
        self.load_calib_btn.clicked.connect(self._handle_load_calibration)

        self.verify_btn = PushButton(tx_button("验证追踪", "Verify Tracking"))
        self.verify_btn.setFixedSize(160, 60)
        self.verify_btn.setEnabled(False)
        self.verify_btn.clicked.connect(self._open_verification_window)

        self.dwell_btn = PushButton(tx_button("启用点击", "Enable Click"))
        self.dwell_btn.setFixedSize(160, 60)
        self.dwell_btn.setEnabled(False)
        self.dwell_btn.clicked.connect(self._toggle_dwell_click)

        self.launcher_btn = PushButton(tx_button("交互窗口", "Open Interaction"))
        self.launcher_btn.setFixedSize(160, 60)
        self.launcher_btn.setEnabled(False)
        self.launcher_btn.clicked.connect(self._toggle_launcher_overlay)

        self.diagnostics_btn = PushButton(tx_button("诊断", "Diagnostics"))
        self.diagnostics_btn.setFixedSize(120, 60)
        self.diagnostics_btn.clicked.connect(self._toggle_diagnostics)

        title_row = QHBoxLayout()
        title_row.addWidget(title)
        title_row.addStretch(1)

        button_bar = QHBoxLayout()
        button_bar.setSpacing(10)
        button_bar.addWidget(self.load_calib_btn)
        button_bar.addWidget(self.verify_btn)
        button_bar.addWidget(self.dwell_btn)
        button_bar.addWidget(self.launcher_btn)
        button_bar.addWidget(self.diagnostics_btn)
        button_bar.addWidget(self.start_btn)
        button_bar.addWidget(self.stop_btn)
        button_bar.addStretch(1)

        perf_card = CardWidget()
        perf_layout = QVBoxLayout(perf_card)
        perf_layout.setContentsMargins(20, 20, 20, 20)
        perf_layout.setSpacing(12)

        perf_title = BodyLabel(tx("性能监控", "Performance Monitor"))
        perf_title.setStyleSheet("font-size: 18px; font-weight: 600;")

        fps_row = QHBoxLayout()
        fps_label = BodyLabel("FPS:")
        fps_label.setStyleSheet("font-weight: 600; font-size: 16px;")
        self.fps_value = BodyLabel("0.0")
        self.fps_value.setStyleSheet(
            f"font-weight: 900; color: {PALETTE['accent']}; font-size: 24px;"
        )
        fps_row.addWidget(fps_label)
        fps_row.addSpacing(10)
        fps_row.addWidget(self.fps_value)
        fps_row.addStretch(1)

        timings_title = BodyLabel(tx("各阶段延迟（ms）：", "Stage Timings (ms):"))
        timings_title.setStyleSheet("font-weight: 600; margin-top: 10px;")

        self.timing_labels: dict[str, BodyLabel] = {}
        timing_stages = [
            ("face_detection", tx("人脸检测", "Face Detection")),
            ("head_pose", tx("头部姿态", "Head Pose")),
            ("gaze_regression", tx("视线回归", "Gaze Regression")),
            ("coordinate_transform", tx("坐标转换", "Coordinate Transform")),
            ("ray_plane_intersect", tx("射线求交", "Ray-Plane Intersect")),
            ("smoothing", tx("平滑滤波", "Smoothing")),
            ("calibration", tx("校准计算", "Calibration")),
        ]

        timings_grid = QGridLayout()
        timings_grid.setHorizontalSpacing(18)
        timings_grid.setVerticalSpacing(6)

        for index, (stage_key, stage_name) in enumerate(timing_stages):
            grid_row = index // 2
            grid_col = (index % 2) * 2
            label = BodyLabel(f"{stage_name}:")
            label.setStyleSheet("font-size: 13px;")
            value = BodyLabel("—")
            value.setStyleSheet("font-size: 13px; color: #666; font-family: 'Consolas', monospace;")
            self.timing_labels[stage_key] = value
            timings_grid.addWidget(label, grid_row, grid_col)
            timings_grid.addWidget(value, grid_row, grid_col + 1)
        timings_grid.setColumnStretch(0, 1)
        timings_grid.setColumnStretch(1, 0)
        timings_grid.setColumnStretch(2, 1)
        timings_grid.setColumnStretch(3, 0)

        perf_layout.addWidget(perf_title)
        perf_layout.addLayout(fps_row)
        perf_layout.addWidget(timings_title)
        perf_layout.addLayout(timings_grid)

        status_card = CardWidget()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 20, 20, 20)
        status_layout.setSpacing(12)

        status_title = BodyLabel(tx("追踪状态", "Tracking Status"))
        status_title.setStyleSheet("font-size: 18px; font-weight: 600;")

        face_row = QHBoxLayout()
        face_label = BodyLabel(tx("人脸检测：", "Face Detection:"))
        face_label.setStyleSheet("font-weight: 600;")
        self.face_status = BodyLabel(tx("未启动", "Not Started"))
        self.face_status.setStyleSheet("color: #888;")
        face_row.addWidget(face_label)
        face_row.addSpacing(10)
        face_row.addWidget(self.face_status)
        face_row.addStretch(1)

        gaze_row = QHBoxLayout()
        gaze_label = BodyLabel(tx("视线有效性：", "Gaze Valid:"))
        gaze_label.setStyleSheet("font-weight: 600;")
        self.gaze_status = BodyLabel(tx("未启动", "Not Started"))
        self.gaze_status.setStyleSheet("color: #888;")
        gaze_row.addWidget(gaze_label)
        gaze_row.addSpacing(10)
        gaze_row.addWidget(self.gaze_status)
        gaze_row.addStretch(1)

        calib_row = QHBoxLayout()
        calib_label = BodyLabel(tx("校准状态：", "Calibration:"))
        calib_label.setStyleSheet("font-weight: 600;")
        self.calib_status = BodyLabel(tx("未加载", "Not Loaded"))
        self.calib_status.setStyleSheet("color: #888;")
        calib_row.addWidget(calib_label)
        calib_row.addSpacing(10)
        calib_row.addWidget(self.calib_status)
        calib_row.addStretch(1)

        self.error_label = BodyLabel("")
        self.error_label.setStyleSheet("color: #D32F2F; font-weight: 600; margin-top: 10px;")
        self.error_label.setWordWrap(True)

        self.diagnostics_label = BodyLabel("")
        self.diagnostics_label.setWordWrap(True)
        self.diagnostics_label.setStyleSheet(
            "color: #444; font-family: 'Consolas', monospace; font-size: 12px;"
        )
        self.diagnostics_label.hide()

        status_layout.addWidget(status_title)
        status_layout.addLayout(face_row)
        status_layout.addLayout(gaze_row)
        status_layout.addLayout(calib_row)
        status_layout.addWidget(self.error_label)
        status_layout.addWidget(self.diagnostics_label)

        stage_card = CardWidget()
        stage_layout = QVBoxLayout(stage_card)
        stage_layout.setContentsMargins(20, 20, 20, 20)
        stage_layout.setSpacing(12)

        stage_title = BodyLabel(tx("交互流程", "Interaction Flow"))
        stage_title.setStyleSheet("font-size: 18px; font-weight: 600;")

        verify_row = QHBoxLayout()
        verify_label = BodyLabel(tx("验证：", "Verification:"))
        verify_label.setStyleSheet("font-weight: 600;")
        self.verify_status = BodyLabel(tx("待校准", "Pending Calibration"))
        verify_row.addWidget(verify_label)
        verify_row.addSpacing(10)
        verify_row.addWidget(self.verify_status)
        verify_row.addStretch(1)

        dwell_row = QHBoxLayout()
        dwell_label = BodyLabel(tx("停留点击：", "Dwell Click:"))
        dwell_label.setStyleSheet("font-weight: 600;")
        self.dwell_status = BodyLabel(tx("关闭", "Off"))
        dwell_row.addWidget(dwell_label)
        dwell_row.addSpacing(10)
        dwell_row.addWidget(self.dwell_status)
        dwell_row.addStretch(1)

        launcher_row = QHBoxLayout()
        launcher_label = BodyLabel(tx("交互窗口：", "Interaction Stage:"))
        launcher_label.setStyleSheet("font-weight: 600;")
        self.launcher_status = BodyLabel(tx("未进入", "Not Open"))
        launcher_row.addWidget(launcher_label)
        launcher_row.addSpacing(10)
        launcher_row.addWidget(self.launcher_status)
        launcher_row.addStretch(1)

        progress_row = QHBoxLayout()
        progress_label = BodyLabel(tx("停留进度：", "Dwell Progress:"))
        progress_label.setStyleSheet("font-weight: 600;")
        self.dwell_progress_label = BodyLabel("0%")
        progress_row.addWidget(progress_label)
        progress_row.addSpacing(10)
        progress_row.addWidget(self.dwell_progress_label)
        progress_row.addStretch(1)

        self.interaction_hint = BodyLabel(
            tx(
                "推荐流程：加载校准，进入全屏验证，确认注视光标后进入全屏交互。",
                "Recommended workflow: load calibration, open fullscreen verification, confirm the calibrated cursor, then enter fullscreen interaction.",
            )
        )
        self.interaction_hint.setWordWrap(True)
        self.interaction_hint.setStyleSheet("color: #666;")

        stage_layout.addWidget(stage_title)
        stage_layout.addLayout(verify_row)
        stage_layout.addLayout(dwell_row)
        stage_layout.addLayout(launcher_row)
        stage_layout.addLayout(progress_row)
        stage_layout.addWidget(self.interaction_hint)

        self.info_grid = QGridLayout()
        self.info_grid.setHorizontalSpacing(16)
        self.info_grid.setVerticalSpacing(16)
        self.info_cards = [perf_card, status_card, stage_card]

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)
        main_layout.addLayout(title_row)
        main_layout.addLayout(button_bar)
        main_layout.addLayout(self.info_grid)
        main_layout.addStretch(1)
        self._arrange_info_cards()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "info_grid"):
            self._arrange_info_cards()

    def _arrange_info_cards(self) -> None:
        if not hasattr(self, "info_grid"):
            return

        while self.info_grid.count():
            item = self.info_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self)

        width = max(self.width(), 0)
        if width >= 1450:
            placements = [(0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1)]
            column_stretches = [2, 1, 1]
        elif width >= 980:
            placements = [(0, 0, 1, 2), (1, 0, 1, 1), (1, 1, 1, 1)]
            column_stretches = [1, 1]
        else:
            placements = [(0, 0, 1, 1), (1, 0, 1, 1), (2, 0, 1, 1)]
            column_stretches = [1]

        for card, (row, col, row_span, col_span) in zip(self.info_cards, placements):
            self.info_grid.addWidget(card, row, col, row_span, col_span)

        for col in range(3):
            stretch = column_stretches[col] if col < len(column_stretches) else 0
            self.info_grid.setColumnStretch(col, stretch)

    def _handle_load_calibration(self) -> None:
        if self.tracker_config is None:
            self.tracker_config = SystemConfig.from_yaml(str(self.config_path)) if self.config_path.exists() else SystemConfig()

        load_path = Path(self.tracker_config.calibration_path)

        if not load_path.exists():
            self.calib_status.setText(tx(f"未加载（{load_path.name}）", f"Not Loaded ({load_path.name})"))
            self.calib_status.setStyleSheet("color: #FF9800; font-weight: 600;")
            self._mark_verification_required(tx("待校准", "Pending Calibration"))
            return

        try:
            if self.calibrator is None:
                self.calibrator = calibration_module_for_config(self.tracker_config)

            load_calibration(self.calibrator, str(load_path))
            self.screen_mapper.reset()
            residual = self.calibrator.residual_mean
            self.calib_status.setText(tx(f"已加载（残差：{residual:.2f} px）", f"Loaded (Residual: {residual:.2f} px)"))
            self.calib_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self._mark_verification_required(tx("待验证", "Verification Required"))
        except Exception as e:
            QMessageBox.critical(self, tx("加载失败", "Load Failed"), tx(f"加载校准参数失败：{e}", f"Failed to load calibration parameters: {e}"))

    def _handle_start_tracking(self) -> None:
        try:
            self.error_label.setText("")

            if self.tracker is None:
                self.tracker_config = SystemConfig.from_yaml(str(self.config_path)) if self.config_path.exists() else SystemConfig()
                self.tracker = TrackerPipeline(
                    model_path=self.tracker_config.checkpoint_path,
                    config=self.tracker_config,
                    error_callback=self._on_tracker_error,
                )

            if not self.tracker.is_running():
                success = self.tracker.start()
                if not success:
                    self.error_label.setText(tx("TrackerPipeline 启动失败，请检查摄像头和模型文件。", "TrackerPipeline failed to start. Check the camera and model files."))
                    return

            if self.cursor_overlay is None:
                self.cursor_overlay = GazeCursorOverlay()

            self._reset_observation_context()
            self.update_timer.start(self.update_interval_ms)
            self.screen_mapper.reset()

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._refresh_stage_controls()
        except Exception as e:
            self.error_label.setText(tx(f"启动失败：{e}", f"Start failed: {e}"))

    def _handle_stop_tracking(self) -> None:
        self._reset_observation_context()
        self.update_timer.stop()
        self.screen_mapper.reset()
        self._close_verification_window()
        self._close_launcher_overlay()
        self._close_gomoku_window()
        self._reset_dwell_state()
        self._stop_tracker_runtime()

        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()
            self.cursor_overlay.clear_gaze_point()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_value.setText("0.0")
        self.dwell_progress_label.setText("0%")

        for label in self.timing_labels.values():
            label.setText("—")

        self.face_status.setText(tx("未启动", "Not Started"))
        self.face_status.setStyleSheet("color: #888;")
        self.gaze_status.setText(tx("未启动", "Not Started"))
        self.gaze_status.setStyleSheet("color: #888;")

        self._refresh_stage_controls()

    def _open_verification_window(self) -> None:
        if self.calibrator is None or not self.calibrator.is_calibrated:
            QMessageBox.information(
                self,
                tx("需要校准", "Calibration Required"),
                tx("请先加载或完成校准，再进入验证阶段。", "Please load or complete calibration before verification.")
            )
            return

        if self.tracker is None or not self.tracker.is_running():
            self._handle_start_tracking()
            if self.tracker is None or not self.tracker.is_running():
                return

        self._close_launcher_overlay()
        self._close_gomoku_window()
        self._reset_dwell_state()

        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()

        if self.verification_window is None:
            self.verification_window = GazeVerificationWindow()
            self.verification_window.verified.connect(self._on_verification_passed)
            self.verification_window.cancelled.connect(self._on_verification_cancelled)

        self.verification_window.show()
        self.verify_status.setText(tx("验证中", "Verifying"))
        self.verify_status.setStyleSheet("color: #2196F3; font-weight: 600;")
        self._refresh_stage_controls()

    def _close_verification_window(self) -> None:
        if self.verification_window is not None:
            window = self.verification_window
            self.verification_window = None
            window.close()

    def _on_verification_passed(self) -> None:
        self.verification_window = None
        self._verification_passed = True
        self.verify_status.setText(tx("已通过", "Passed"))
        self.verify_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        self._stop_tracker_runtime()
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _on_verification_cancelled(self) -> None:
        self.verification_window = None
        if self._verification_passed:
            self.verify_status.setText(tx("已通过", "Passed"))
            self.verify_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        else:
            self.verify_status.setText(tx("待验证", "Verification Required"))
            self.verify_status.setStyleSheet("color: #FF9800; font-weight: 600;")
        self._stop_tracker_runtime()
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _toggle_dwell_click(self) -> None:
        if not self._verification_passed:
            QMessageBox.information(
                self,
                tx("先完成验证", "Verification Required"),
                tx("请先完成全屏验证，再启用点击交互。", "Please complete fullscreen verification before enabling dwell click.")
            )
            return
        self.dwell_enabled = not self.dwell_enabled
        self._reset_dwell_state()
        self._refresh_stage_controls()

    def _toggle_launcher_overlay(self) -> None:
        if self.interaction_overlay is None:
            self._open_launcher_overlay()
        else:
            self._close_launcher_overlay()

    def _open_launcher_overlay(self) -> None:
        if not self._verification_passed:
            QMessageBox.information(
                self,
                tx("先完成验证", "Verification Required"),
                tx("请先完成全屏验证，再进入交互阶段。", "Please complete fullscreen verification before entering interaction.")
            )
            return

        if self.tracker is None or not self.tracker.is_running():
            self._handle_start_tracking()
            if self.tracker is None or not self.tracker.is_running():
                return

        self._close_verification_window()
        self._reset_dwell_state()

        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()

        self._reset_observation_context()
        self.interaction_overlay = InteractionLauncherOverlay()
        self.interaction_overlay.request_toggle_dwell.connect(self._toggle_dwell_click)
        self.interaction_overlay.request_open_gomoku.connect(self._open_gomoku_window)
        self.interaction_overlay.closed.connect(self._on_launcher_closed)
        self.interaction_overlay.show()
        self._refresh_stage_controls()

    def _close_launcher_overlay(self) -> None:
        if self.interaction_overlay is not None:
            overlay = self.interaction_overlay
            self.interaction_overlay = None
            overlay.close()
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _on_launcher_closed(self) -> None:
        self._reset_observation_context()
        self.interaction_overlay = None
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _open_gomoku_window(self) -> None:
        self._reset_observation_context()
        self._close_launcher_overlay()
        self._reset_dwell_state()
        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()
        if self.gomoku_window is None:
            self.gomoku_window = GomokuWindow()
            self.gomoku_window.closed.connect(self._on_gomoku_closed)
            self.gomoku_window.return_to_launcher.connect(self._on_gomoku_return_to_launcher)
        self.gomoku_window.show()
        self._refresh_stage_controls()

    def _close_gomoku_window(self) -> None:
        if self.gomoku_window is not None:
            window = self.gomoku_window
            self.gomoku_window = None
            window.close()
        self._refresh_stage_controls()

    def _on_gomoku_closed(self) -> None:
        self._reset_observation_context()
        self.gomoku_window = None
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()
        if self._reopen_launcher_after_gomoku:
            self._reopen_launcher_after_gomoku = False
            self._launcher_return_timer.start(0)

    def _return_to_launcher_if_running(self) -> None:
        if self.tracker is not None and self.tracker.is_running():
            self._open_launcher_overlay()

    def _on_gomoku_return_to_launcher(self) -> None:
        self._reopen_launcher_after_gomoku = True

    def _stop_tracker_runtime(self) -> None:
        self._reset_observation_context()
        self.update_timer.stop()
        self.screen_mapper.reset()
        if self.tracker is not None:
            self.tracker.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_value.setText("0.0")
        for label in self.timing_labels.values():
            label.setText("—")
        self.face_status.setText(tx("未启动", "Not Started"))
        self.face_status.setStyleSheet("color: #888;")
        self.gaze_status.setText(tx("未启动", "Not Started"))
        self.gaze_status.setStyleSheet("color: #888;")

    def _interrupt_gaze(self) -> None:
        self._launcher_return_timer.stop()
        # A retained cursor is display-only. All unfinished selections are cleared.
        cooldown = self._dwell_cooldown_until
        self._reset_dwell_state()
        self._dwell_cooldown_until = cooldown
        self.screen_mapper.reset()
        self.dwell_progress_label.setText("0%")
        if self.cursor_overlay is not None:
            self.cursor_overlay.set_dwell_progress(0.0, False)
        if self.interaction_overlay is not None:
            self.interaction_overlay.reset_progress()
        if self.gomoku_window is not None:
            self.gomoku_window.reset_gaze_progress()

    def _reset_observation_context(self) -> None:
        self._observation_context = None
        self._launcher_return_timer.stop()
        self._interrupt_gaze()

    def _update_tracking_data(self) -> None:
        if (self.tracker is None or not self.tracker.is_running()
                or getattr(self.tracker, "_calibration_mode", False)):
            self._reset_observation_context()
            return

        context = (self.tracker, getattr(self.tracker, "session_id", None),
                   self.verification_window, self.interaction_overlay,
                   self.gomoku_window, self.calibrator, self.dwell_enabled,
                   self.tracker_config.normalized_backend if self.tracker_config else None)
        if context != self._observation_context:
            self._interrupt_gaze()
            self._observation_context = context
            self.observation_gate.max_age = validate_max_age(
                self.tracker_config.max_observation_age_ms if self.tracker_config else 250.)
            self.observation_gate.reset(context[1])
        result = self.tracker.get_latest_result()
        state = self.observation_gate.consume(result)
        if state is ObservationState.INVALID:
            self._update_timing_labels(None)
            self._interrupt_gaze()
            self.gaze_status.setText(tx("观测不可操作", "Observation unavailable"))
            self.error_label.setText(getattr(result, "error_message", None) or self.observation_gate.reason)
            return
        if state is ObservationState.DUPLICATE:
            return
        if self.observation_gate.reset_required:
            self._interrupt_gaze()
        observed_ms = result.observation.timestamp * 1000.0

        self.fps_value.setText(f"{result.fps:.1f}")

        if result.face_detected:
            self.face_status.setText(tx("检测到", "Detected"))
            self.face_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        else:
            self.face_status.setText(tx("未检测到", "Not Detected"))
            self.face_status.setStyleSheet("color: #FF9800; font-weight: 600;")

        if result.valid and result.gaze_point is not None:
            self.gaze_status.setText(tx("有效", "Valid"))
            self.gaze_status.setStyleSheet("color: #4CAF50; font-weight: 600;")

            screen = QApplication.primaryScreen()
            size = (screen.geometry().width(), screen.geometry().height()) if screen else (0, 0)
            result = self.screen_mapper.process(
                result, self.tracker_config or SystemConfig(), self.calibrator, size)
            self._update_timing_labels(result)
            if result.screen_rejection is not None:
                self.observation_gate.reject(result.screen_rejection)
                self._interrupt_gaze()
                self.gaze_status.setText(tx("估计不可操作", "Estimate unavailable"))
                self.error_label.setText(result.screen_rejection)
                if self.diagnostics_enabled:
                    self._update_diagnostics_label(result)
                return
            gaze_x, gaze_y = result.display_point

            # Recheck after calibration/smoothing, immediately before any gaze
            # dispatch. New valid frames are fine; a producer interruption is not.
            rejection = self.tracker.get_dispatch_rejection(
                result.observation, self.observation_gate.max_age,
                clock=self.observation_gate.clock,
            )
            if rejection is not None:
                self.observation_gate.reject(rejection)
                self._interrupt_gaze()
                return
            if self.verification_window is not None:
                self.verification_window.update_gaze_point(gaze_x, gaze_y)
                self.dwell_progress_label.setText("0%")
                if self.cursor_overlay is not None:
                    self.cursor_overlay.set_dwell_progress(0.0, False)
            elif self.gomoku_window is not None:
                self.gomoku_window.update_gaze_point(gaze_x, gaze_y, observed_ms=observed_ms)
                self.dwell_progress_label.setText("0%")
                if self.cursor_overlay is not None:
                    self.cursor_overlay.set_dwell_progress(0.0, False)
            elif self.interaction_overlay is not None:
                self.interaction_overlay.update_gaze_point(gaze_x, gaze_y, observed_ms=observed_ms)
                self.dwell_progress_label.setText("0%")
                if self.cursor_overlay is not None:
                    self.cursor_overlay.set_dwell_progress(0.0, False)
            else:
                if self.cursor_overlay is not None:
                    self.cursor_overlay.update_gaze_point(gaze_x, gaze_y)

                if self.dwell_enabled:
                    QCursor.setPos(int(gaze_x), int(gaze_y))
                    progress = self._update_dwell_click(gaze_x, gaze_y, observed_ms=observed_ms)
                    self.dwell_progress_label.setText(f"{int(progress * 100)}%")
                    if self.cursor_overlay is not None:
                        self.cursor_overlay.set_dwell_progress(progress, True)
                else:
                    self._reset_dwell_state()
                    self.dwell_progress_label.setText("0%")
                    if self.cursor_overlay is not None:
                        self.cursor_overlay.set_dwell_progress(0.0, False)
        else:
            self.gaze_status.setText(tx("无效", "Invalid"))
            self.gaze_status.setStyleSheet("color: #FF9800; font-weight: 600;")
            self._reset_dwell_state()
            self.dwell_progress_label.setText("0%")
            if self.cursor_overlay is not None:
                self.cursor_overlay.clear_gaze_point()

        if result.error_message:
            self.error_label.setText(tx(f"警告：{result.error_message}", f"Warning: {result.error_message}"))
        elif not self.dwell_enabled:
            self.error_label.setText("")

        if self.diagnostics_enabled:
            self._update_diagnostics_label(result)

    def _update_timing_labels(self, result) -> None:
        for key, label in self.timing_labels.items():
            values = (result.processing_timings if key in ('calibration', 'smoothing') else result.timings) if result else {}
            value = values.get(key)
            label.setText('—' if value is None else f'{value:.2f} ms')

    def _toggle_diagnostics(self) -> None:
        self.diagnostics_enabled = not self.diagnostics_enabled
        self.diagnostics_label.setVisible(self.diagnostics_enabled)
        self.diagnostics_btn.setText(tx_button("隐藏诊断", "Hide Diagnostics") if self.diagnostics_enabled else tx_button("诊断", "Diagnostics"))
        if not self.diagnostics_enabled:
            self.diagnostics_label.setText("")

    def _update_diagnostics_label(self, result) -> None:
        tracker_diag = self.tracker.get_diagnostics() if self.tracker is not None else {}
        raw = result.raw_point
        calibrated = result.calibrated_point
        calib_method = getattr(getattr(self.calibrator, "method", None), "value", "none")
        calib_points = len(getattr(self.calibrator, "_raw_points", [])) if self.calibrator is not None else 0
        lines = [
            f"backend={result.backend} model={tracker_diag.get('model_version')} fps={result.fps:.1f}",
            f"raw={self._fmt_point(raw)} [{result.raw_units}] calibrated={self._fmt_point(calibrated)}",
            f"calib={calib_method} points={calib_points} path={tracker_diag.get('calibration_path')}",
            f"onnx_inputs={tracker_diag.get('onnx_inputs', [])}",
        ]
        lines.append(f"smoothed={self._fmt_point(result.smoothed_point)} display={self._fmt_point(result.display_point)}")
        lines.append(f"in_bounds=({result.calibrated_in_bounds}, {result.smoothed_in_bounds}) rejected={result.screen_rejection}")
        head_pose = result.debug.get("head_pose") if isinstance(result.debug, dict) else None
        if isinstance(head_pose, dict):
            lines.append(
                "head=({yaw:.1f},{pitch:.1f},{roll:.1f})".format(
                    yaw=head_pose.get("yaw", 0.0),
                    pitch=head_pose.get("pitch", 0.0),
                    roll=head_pose.get("roll", 0.0),
                )
            )
        if result.timings:
            timing_text = " ".join(f"{k}={v:.1f}" for k, v in result.timings.items())
            lines.append(timing_text)
        self.diagnostics_label.setText("\n".join(lines))

    @staticmethod
    def _fmt_point(point: Optional[tuple[float, float]]) -> str:
        if point is None:
            return "None"
        return f"({point[0]:.3f},{point[1]:.3f})"

    def _update_dwell_click(self, gaze_x: float, gaze_y: float, *, observed_ms: float) -> float:
        now = observed_ms
        if now < self._dwell_cooldown_until:
            return 0.0

        current_point = (gaze_x, gaze_y)
        if self._dwell_anchor is None:
            self._dwell_anchor = current_point
            self._dwell_started_at = now
            return 0.0

        if math.dist(current_point, self._dwell_anchor) > self.dwell_radius_px:
            self._dwell_anchor = current_point
            self._dwell_started_at = now
            return 0.0

        progress = min((now - self._dwell_started_at) / self.dwell_ms, 1.0)
        if progress >= 1.0:
            if perform_left_click():
                self.error_label.setText(tx("信息：已触发停留左键点击。", "Info: Dwell left click triggered."))
            else:
                self.error_label.setText(tx("警告：当前平台不支持系统级左键点击。", "Warning: System-level left click is not supported on this platform."))
            self._dwell_anchor = None
            self._dwell_started_at = 0.0
            self._dwell_cooldown_until = now + 800.0
            return 0.0

        return progress

    def _reset_dwell_state(self) -> None:
        self._dwell_anchor = None
        self._dwell_started_at = 0.0
        self._dwell_cooldown_until = 0.0
        if self.cursor_overlay is not None:
            self.cursor_overlay.set_dwell_progress(0.0, self.dwell_enabled and self._no_fullscreen_stage_open())

    def _mark_verification_required(self, label: str) -> None:
        self._reset_observation_context()
        self._verification_passed = False
        self.dwell_enabled = False
        self.verify_status.setText(label)
        self.verify_status.setStyleSheet("color: #FF9800; font-weight: 600;")
        self._reset_dwell_state()
        self._close_launcher_overlay()
        self._refresh_stage_controls()

    def _restore_cursor_overlay_if_needed(self) -> None:
        if (
            self.cursor_overlay is not None
            and self.show_cursor_overlay
            and self.tracker is not None
            and self.tracker.is_running()
            and self._no_fullscreen_stage_open()
        ):
            self.cursor_overlay.show()

    def _no_fullscreen_stage_open(self) -> bool:
        return self.verification_window is None and self.interaction_overlay is None and self.gomoku_window is None

    def _refresh_stage_controls(self) -> None:
        tracker_running = self.tracker is not None and self.tracker.is_running()
        calibration_ready = self.calibrator is not None and self.calibrator.is_calibrated
        fullscreen_stage_open = not self._no_fullscreen_stage_open()

        self.verify_btn.setEnabled(calibration_ready and self.interaction_overlay is None)
        self.dwell_btn.setEnabled(tracker_running and self._verification_passed and self.interaction_overlay is None and self.verification_window is None)
        self.launcher_btn.setEnabled(self._verification_passed and self.verification_window is None)

        if self.verification_window is not None:
            self.verify_status.setText(tx("验证中", "Verifying"))
            self.verify_status.setStyleSheet("color: #2196F3; font-weight: 600;")
        elif not calibration_ready:
            self.verify_status.setText(tx("待校准", "Pending Calibration"))
            self.verify_status.setStyleSheet("color: #888;")
        elif self._verification_passed:
            self.verify_status.setText(tx("已通过", "Passed"))
            self.verify_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        elif self.verify_status.text() not in {tx("验证中", "Verifying")}:
            self.verify_status.setText(tx("待验证", "Verification Required"))
            self.verify_status.setStyleSheet("color: #FF9800; font-weight: 600;")

        if self.dwell_enabled:
            self.dwell_status.setText(tx("开启", "On"))
            self.dwell_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self.dwell_btn.setText(tx_button("关闭点击", "Disable Click"))
        else:
            self.dwell_status.setText(tx("关闭", "Off"))
            self.dwell_status.setStyleSheet("color: #888;")
            self.dwell_btn.setText(tx_button("启用点击", "Enable Click"))

        if self.interaction_overlay is not None:
            self.launcher_status.setText(tx("运行中", "Open"))
            self.launcher_status.setStyleSheet("color: #2196F3; font-weight: 600;")
            self.launcher_btn.setText(tx_button("关闭交互", "Close Interaction"))
        elif self.gomoku_window is not None:
            self.launcher_status.setText(tx("五子棋", "Gomoku"))
            self.launcher_status.setStyleSheet("color: #2196F3; font-weight: 600;")
            self.launcher_btn.setText(tx_button("交互窗口", "Open Interaction"))
        elif fullscreen_stage_open:
            self.launcher_status.setText(tx("等待中", "Stage Active"))
            self.launcher_status.setStyleSheet("color: #888;")
            self.launcher_btn.setText(tx_button("交互窗口", "Open Interaction"))
        else:
            self.launcher_status.setText(tx("未进入", "Not Open"))
            self.launcher_status.setStyleSheet("color: #888;")
            self.launcher_btn.setText(tx_button("交互窗口", "Open Interaction"))

    def _on_tracker_error(self, message: str) -> None:
        self.error_label.setText(tx(f"错误：{message}", f"Error: {message}"))

    def closeEvent(self, event) -> None:
        self._handle_stop_tracking()

        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None

        if self.cursor_overlay is not None:
            self.cursor_overlay.close()
            self.cursor_overlay = None

        self._close_verification_window()
        self._close_launcher_overlay()
        self._close_gomoku_window()
        super().closeEvent(event)
