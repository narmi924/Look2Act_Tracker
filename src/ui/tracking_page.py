from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QBrush, QColor, QCursor, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import BodyLabel, CardWidget, PrimaryPushButton, PushButton

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import load_calibration
from src.tracker.pipeline import SystemConfig, TrackerPipeline
from src.ui.fluent_theme import PALETTE
from src.ui.interaction_overlay import (
    GazeVerificationWindow,
    InteractionLauncherOverlay,
    perform_left_click,
)
from src.ui.gomoku_window import GomokuWindow


class GazeCursorOverlay(QWidget):
    """Fullscreen transparent overlay that draws the current gaze cursor."""

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
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.tracker: Optional[TrackerPipeline] = None
        self.tracker_config: Optional[SystemConfig] = None
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

        self._verification_passed = False
        self.diagnostics_enabled = False
        self.show_cursor_overlay = False

        self._init_ui()
        self._refresh_stage_controls()

    def set_calibrator(self, calibrator: CalibrationModule) -> None:
        self.calibrator = calibrator
        if self.calibrator.is_calibrated:
            residual = self.calibrator.residual_mean
            self.calib_status.setText(f"已加载 / Loaded (残差: {residual:.2f} px)")
            self.calib_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self._mark_verification_required("待验证 / Verification Required")
        else:
            self.calib_status.setText("未加载 / Not Loaded")
            self.calib_status.setStyleSheet("color: #888;")
            self._mark_verification_required("待校准 / Pending Calibration")

    def start_verification_flow(self) -> None:
        self._handle_start_tracking()
        if self.calibrator is not None and self.calibrator.is_calibrated:
            self._open_verification_window()

    def _init_ui(self) -> None:
        title = QLabel("实时追踪 / Real-time Tracking")
        title.setStyleSheet("font-size: 32px; font-weight: 900;")

        self.start_btn = PrimaryPushButton("启动追踪\nStart Tracking")
        self.start_btn.setFixedSize(160, 60)
        self.start_btn.clicked.connect(self._handle_start_tracking)

        self.stop_btn = PushButton("停止追踪\nStop Tracking")
        self.stop_btn.setFixedSize(160, 60)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._handle_stop_tracking)

        self.load_calib_btn = PushButton("加载校准\nLoad Calibration")
        self.load_calib_btn.setFixedSize(160, 60)
        self.load_calib_btn.clicked.connect(self._handle_load_calibration)

        self.verify_btn = PushButton("验证追踪\nVerify Tracking")
        self.verify_btn.setFixedSize(160, 60)
        self.verify_btn.setEnabled(False)
        self.verify_btn.clicked.connect(self._open_verification_window)

        self.dwell_btn = PushButton("启用点击\nDwell Click")
        self.dwell_btn.setFixedSize(160, 60)
        self.dwell_btn.setEnabled(False)
        self.dwell_btn.clicked.connect(self._toggle_dwell_click)

        self.launcher_btn = PushButton("交互窗口\nOpen Interaction")
        self.launcher_btn.setFixedSize(160, 60)
        self.launcher_btn.setEnabled(False)
        self.launcher_btn.clicked.connect(self._toggle_launcher_overlay)

        self.diagnostics_btn = PushButton("诊断\nDiagnostics")
        self.diagnostics_btn.setFixedSize(120, 60)
        self.diagnostics_btn.clicked.connect(self._toggle_diagnostics)

        top_bar = QHBoxLayout()
        top_bar.addWidget(title)
        top_bar.addStretch(1)
        top_bar.addWidget(self.load_calib_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.verify_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.dwell_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.launcher_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.diagnostics_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.start_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.stop_btn)

        perf_card = CardWidget()
        perf_layout = QVBoxLayout(perf_card)
        perf_layout.setContentsMargins(20, 20, 20, 20)
        perf_layout.setSpacing(12)

        perf_title = BodyLabel("性能监控 / Performance Monitor")
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

        timings_title = BodyLabel("各阶段延迟 / Stage Timings (ms):")
        timings_title.setStyleSheet("font-weight: 600; margin-top: 10px;")

        self.timing_labels: dict[str, BodyLabel] = {}
        timing_stages = [
            ("face_detection", "人脸检测 / Face Detection"),
            ("head_pose", "头部姿态 / Head Pose"),
            ("gaze_regression", "视线回归 / Gaze Regression"),
            ("coordinate_transform", "坐标转换 / Coordinate Transform"),
            ("ray_plane_intersect", "射线求交 / Ray-Plane Intersect"),
            ("smoothing", "平滑滤波 / Smoothing"),
        ]

        timings_grid = QVBoxLayout()
        timings_grid.setSpacing(6)

        for stage_key, stage_name in timing_stages:
            row = QHBoxLayout()
            label = BodyLabel(f"{stage_name}:")
            label.setStyleSheet("font-size: 13px;")
            value = BodyLabel("0.00 ms")
            value.setStyleSheet("font-size: 13px; color: #666; font-family: 'Consolas', monospace;")
            self.timing_labels[stage_key] = value
            row.addWidget(label)
            row.addStretch(1)
            row.addWidget(value)
            timings_grid.addLayout(row)

        perf_layout.addWidget(perf_title)
        perf_layout.addLayout(fps_row)
        perf_layout.addWidget(timings_title)
        perf_layout.addLayout(timings_grid)

        status_card = CardWidget()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 20, 20, 20)
        status_layout.setSpacing(12)

        status_title = BodyLabel("追踪状态 / Tracking Status")
        status_title.setStyleSheet("font-size: 18px; font-weight: 600;")

        face_row = QHBoxLayout()
        face_label = BodyLabel("人脸检测 / Face Detection:")
        face_label.setStyleSheet("font-weight: 600;")
        self.face_status = BodyLabel("未启动 / Not Started")
        self.face_status.setStyleSheet("color: #888;")
        face_row.addWidget(face_label)
        face_row.addSpacing(10)
        face_row.addWidget(self.face_status)
        face_row.addStretch(1)

        gaze_row = QHBoxLayout()
        gaze_label = BodyLabel("视线有效性 / Gaze Valid:")
        gaze_label.setStyleSheet("font-weight: 600;")
        self.gaze_status = BodyLabel("未启动 / Not Started")
        self.gaze_status.setStyleSheet("color: #888;")
        gaze_row.addWidget(gaze_label)
        gaze_row.addSpacing(10)
        gaze_row.addWidget(self.gaze_status)
        gaze_row.addStretch(1)

        calib_row = QHBoxLayout()
        calib_label = BodyLabel("校准状态 / Calibration:")
        calib_label.setStyleSheet("font-weight: 600;")
        self.calib_status = BodyLabel("未加载 / Not Loaded")
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

        stage_title = BodyLabel("交互流程 / Interaction Flow")
        stage_title.setStyleSheet("font-size: 18px; font-weight: 600;")

        verify_row = QHBoxLayout()
        verify_label = BodyLabel("Verification:")
        verify_label.setStyleSheet("font-weight: 600;")
        self.verify_status = BodyLabel("待校准 / Pending Calibration")
        verify_row.addWidget(verify_label)
        verify_row.addSpacing(10)
        verify_row.addWidget(self.verify_status)
        verify_row.addStretch(1)

        dwell_row = QHBoxLayout()
        dwell_label = BodyLabel("Dwell Click:")
        dwell_label.setStyleSheet("font-weight: 600;")
        self.dwell_status = BodyLabel("关闭 / Off")
        dwell_row.addWidget(dwell_label)
        dwell_row.addSpacing(10)
        dwell_row.addWidget(self.dwell_status)
        dwell_row.addStretch(1)

        launcher_row = QHBoxLayout()
        launcher_label = BodyLabel("Interaction Stage:")
        launcher_label.setStyleSheet("font-weight: 600;")
        self.launcher_status = BodyLabel("未进入 / Not Open")
        launcher_row.addWidget(launcher_label)
        launcher_row.addSpacing(10)
        launcher_row.addWidget(self.launcher_status)
        launcher_row.addStretch(1)

        progress_row = QHBoxLayout()
        progress_label = BodyLabel("Dwell Progress:")
        progress_label.setStyleSheet("font-weight: 600;")
        self.dwell_progress_label = BodyLabel("0%")
        progress_row.addWidget(progress_label)
        progress_row.addSpacing(10)
        progress_row.addWidget(self.dwell_progress_label)
        progress_row.addStretch(1)

        self.interaction_hint = BodyLabel(
            "Recommended workflow: load calibration, open the fullscreen verification stage, confirm the calibrated cursor, then enter the fullscreen interaction stage."
        )
        self.interaction_hint.setWordWrap(True)
        self.interaction_hint.setStyleSheet("color: #666;")

        stage_layout.addWidget(stage_title)
        stage_layout.addLayout(verify_row)
        stage_layout.addLayout(dwell_row)
        stage_layout.addLayout(launcher_row)
        stage_layout.addLayout(progress_row)
        stage_layout.addWidget(self.interaction_hint)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)
        main_layout.addLayout(top_bar)
        main_layout.addWidget(perf_card)
        main_layout.addWidget(status_card)
        main_layout.addWidget(stage_card)
        main_layout.addStretch(1)

    def _handle_load_calibration(self) -> None:
        if self.tracker_config is None:
            config_path = Path("configs/system_config.yaml")
            self.tracker_config = SystemConfig.from_yaml(str(config_path)) if config_path.exists() else SystemConfig()

        load_path = Path(self.tracker_config.calibration_path)

        if not load_path.exists():
            self.calib_status.setText(f"未加载 / Not Loaded ({load_path.name})")
            self.calib_status.setStyleSheet("color: #FF9800; font-weight: 600;")
            self._mark_verification_required("待校准 / Pending Calibration")
            return

        try:
            if self.calibrator is None:
                if self.tracker_config.normalized_backend == "classic":
                    self.calibrator = CalibrationModule(num_points=25, max_residual_px=300.0, method="polynomial")
                else:
                    self.calibrator = CalibrationModule(num_points=9, max_residual_px=300.0, method="affine")

            load_calibration(self.calibrator, str(load_path))
            residual = self.calibrator.residual_mean
            self.calib_status.setText(f"已加载 / Loaded (残差: {residual:.2f} px)")
            self.calib_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self._mark_verification_required("待验证 / Verification Required")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载校准参数失败：{e}")

    def _handle_start_tracking(self) -> None:
        try:
            self.error_label.setText("")

            if self.tracker is None:
                config_path = Path("configs/system_config.yaml")
                self.tracker_config = SystemConfig.from_yaml(str(config_path)) if config_path.exists() else SystemConfig()
                self.tracker = TrackerPipeline(
                    model_path=self.tracker_config.checkpoint_path,
                    config=self.tracker_config,
                    error_callback=self._on_tracker_error,
                )

            if not self.tracker.is_running():
                success = self.tracker.start()
                if not success:
                    self.error_label.setText("TrackerPipeline 启动失败，请检查摄像头和模型文件。")
                    return

            if self.cursor_overlay is None:
                self.cursor_overlay = GazeCursorOverlay()

            self.update_timer.start(self.update_interval_ms)

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._refresh_stage_controls()
        except Exception as e:
            self.error_label.setText(f"启动失败：{e}")

    def _handle_stop_tracking(self) -> None:
        self.update_timer.stop()
        self._close_verification_window()
        self._close_launcher_overlay()
        self._close_gomoku_window()
        self._reset_dwell_state()

        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()
            self.cursor_overlay.clear_gaze_point()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.fps_value.setText("0.0")
        self.dwell_progress_label.setText("0%")

        for label in self.timing_labels.values():
            label.setText("0.00 ms")

        self.face_status.setText("未启动 / Not Started")
        self.face_status.setStyleSheet("color: #888;")
        self.gaze_status.setText("未启动 / Not Started")
        self.gaze_status.setStyleSheet("color: #888;")

        self._refresh_stage_controls()

    def _open_verification_window(self) -> None:
        if self.calibrator is None or not self.calibrator.is_calibrated:
            QMessageBox.information(self, "需要校准", "请先加载或完成校准，再进入验证阶段。")
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
        self.verify_status.setText("验证中 / Verifying")
        self.verify_status.setStyleSheet("color: #2196F3; font-weight: 600;")
        self._refresh_stage_controls()

    def _close_verification_window(self) -> None:
        if self.verification_window is not None:
            window = self.verification_window
            self.verification_window = None
            window.close()

    def _on_verification_passed(self) -> None:
        self._verification_passed = True
        self.verify_status.setText("已通过 / Passed")
        self.verify_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _on_verification_cancelled(self) -> None:
        if self._verification_passed:
            self.verify_status.setText("已通过 / Passed")
            self.verify_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        else:
            self.verify_status.setText("待验证 / Verification Required")
            self.verify_status.setStyleSheet("color: #FF9800; font-weight: 600;")
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _toggle_dwell_click(self) -> None:
        if not self._verification_passed:
            QMessageBox.information(self, "先完成验证", "请先完成全屏验证，再启用点击交互。")
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
            QMessageBox.information(self, "先完成验证", "请先完成全屏验证，再进入交互阶段。")
            return

        if self.tracker is None or not self.tracker.is_running():
            self._handle_start_tracking()
            if self.tracker is None or not self.tracker.is_running():
                return

        self._close_verification_window()
        self._reset_dwell_state()

        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()

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
        self.interaction_overlay = None
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _open_gomoku_window(self) -> None:
        self._close_launcher_overlay()
        self._reset_dwell_state()
        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()
        if self.gomoku_window is None:
            self.gomoku_window = GomokuWindow()
            self.gomoku_window.closed.connect(self._on_gomoku_closed)
        self.gomoku_window.show()
        self._refresh_stage_controls()

    def _close_gomoku_window(self) -> None:
        if self.gomoku_window is not None:
            window = self.gomoku_window
            self.gomoku_window = None
            window.close()
        self._refresh_stage_controls()

    def _on_gomoku_closed(self) -> None:
        self.gomoku_window = None
        self._restore_cursor_overlay_if_needed()
        self._refresh_stage_controls()

    def _update_tracking_data(self) -> None:
        if self.tracker is None or not self.tracker.is_running():
            return

        result = self.tracker.get_latest_result()
        if result is None:
            return

        self.fps_value.setText(f"{result.fps:.1f}")

        for stage_key, label in self.timing_labels.items():
            label.setText(f"{result.timings.get(stage_key, 0.0):.2f} ms")

        if result.face_detected:
            self.face_status.setText("检测到 / Detected")
            self.face_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        else:
            self.face_status.setText("未检测到 / Not Detected")
            self.face_status.setStyleSheet("color: #FF9800; font-weight: 600;")

        if result.valid and result.gaze_point is not None:
            self.gaze_status.setText("有效 / Valid")
            self.gaze_status.setStyleSheet("color: #4CAF50; font-weight: 600;")

            (gaze_x, gaze_y), pre_clamp = self._apply_calibration_and_clamp_with_debug(*result.gaze_point)
            result.calibrated_point = (gaze_x, gaze_y)
            if isinstance(result.debug, dict):
                result.debug["pre_clamp_point"] = pre_clamp
                result.debug["clamped_point"] = result.calibrated_point

            if self.verification_window is not None:
                self.verification_window.update_gaze_point(gaze_x, gaze_y)
                self.dwell_progress_label.setText("0%")
                if self.cursor_overlay is not None:
                    self.cursor_overlay.set_dwell_progress(0.0, False)
            elif self.gomoku_window is not None:
                self.gomoku_window.update_gaze_point(gaze_x, gaze_y)
                self.dwell_progress_label.setText("0%")
                if self.cursor_overlay is not None:
                    self.cursor_overlay.set_dwell_progress(0.0, False)
            elif self.interaction_overlay is not None:
                self.interaction_overlay.update_gaze_point(gaze_x, gaze_y)
                self.dwell_progress_label.setText("0%")
                if self.cursor_overlay is not None:
                    self.cursor_overlay.set_dwell_progress(0.0, False)
            else:
                if self.cursor_overlay is not None:
                    self.cursor_overlay.update_gaze_point(gaze_x, gaze_y)

                if self.dwell_enabled:
                    QCursor.setPos(int(gaze_x), int(gaze_y))
                    progress = self._update_dwell_click(gaze_x, gaze_y)
                    self.dwell_progress_label.setText(f"{int(progress * 100)}%")
                    if self.cursor_overlay is not None:
                        self.cursor_overlay.set_dwell_progress(progress, True)
                else:
                    self._reset_dwell_state()
                    self.dwell_progress_label.setText("0%")
                    if self.cursor_overlay is not None:
                        self.cursor_overlay.set_dwell_progress(0.0, False)
        else:
            self.gaze_status.setText("无效 / Invalid")
            self.gaze_status.setStyleSheet("color: #FF9800; font-weight: 600;")
            self._reset_dwell_state()
            self.dwell_progress_label.setText("0%")
            if self.cursor_overlay is not None:
                self.cursor_overlay.clear_gaze_point()

        if result.error_message:
            self.error_label.setText(f"警告: {result.error_message}")
        elif not self.dwell_enabled:
            self.error_label.setText("")

        if self.diagnostics_enabled:
            self._update_diagnostics_label(result)

    def _apply_calibration_and_clamp(self, gaze_x: float, gaze_y: float) -> tuple[float, float]:
        clamped, _ = self._apply_calibration_and_clamp_with_debug(gaze_x, gaze_y)
        return clamped

    def _apply_calibration_and_clamp_with_debug(
        self,
        gaze_x: float,
        gaze_y: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        if self.calibrator is not None and self.calibrator.is_calibrated:
            gaze_x, gaze_y = self.calibrator.apply((gaze_x, gaze_y))
        pre_clamp = (gaze_x, gaze_y)

        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.geometry()
            gaze_x = max(0.0, min(gaze_x, float(geo.width() - 1)))
            gaze_y = max(0.0, min(gaze_y, float(geo.height() - 1)))
        return (gaze_x, gaze_y), pre_clamp

    def _toggle_diagnostics(self) -> None:
        self.diagnostics_enabled = not self.diagnostics_enabled
        self.diagnostics_label.setVisible(self.diagnostics_enabled)
        self.diagnostics_btn.setText("隐藏诊断\nDiagnostics" if self.diagnostics_enabled else "诊断\nDiagnostics")
        if not self.diagnostics_enabled:
            self.diagnostics_label.setText("")

    def _update_diagnostics_label(self, result) -> None:
        tracker_diag = self.tracker.get_diagnostics() if self.tracker is not None else {}
        raw = result.raw_point or result.gaze_point
        calibrated = result.calibrated_point
        calib_method = getattr(getattr(self.calibrator, "method", None), "value", "none")
        calib_points = len(getattr(self.calibrator, "_raw_points", [])) if self.calibrator is not None else 0
        lines = [
            f"backend={result.backend} model={tracker_diag.get('model_version')} fps={result.fps:.1f}",
            f"raw={self._fmt_point(raw)} calibrated={self._fmt_point(calibrated)}",
            f"calib={calib_method} points={calib_points} path={tracker_diag.get('calibration_path')}",
            f"onnx_inputs={tracker_diag.get('onnx_inputs', [])}",
        ]
        if isinstance(result.debug, dict):
            lines.append(
                "pre_clamp={} clamped={}".format(
                    self._fmt_point(result.debug.get("pre_clamp_point")),
                    self._fmt_point(result.debug.get("clamped_point")),
                )
            )
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

    def _update_dwell_click(self, gaze_x: float, gaze_y: float) -> float:
        now = time.perf_counter() * 1000.0
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
                self.error_label.setText("信息: 已触发 dwell left click。")
            else:
                self.error_label.setText("警告: 当前平台不支持系统级左键点击。")
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

        self.verify_btn.setEnabled(tracker_running and calibration_ready and self.interaction_overlay is None)
        self.dwell_btn.setEnabled(tracker_running and self._verification_passed and self.interaction_overlay is None and self.verification_window is None)
        self.launcher_btn.setEnabled(tracker_running and self._verification_passed and self.verification_window is None)

        if self.verification_window is not None:
            self.verify_status.setText("验证中 / Verifying")
            self.verify_status.setStyleSheet("color: #2196F3; font-weight: 600;")
        elif not calibration_ready:
            self.verify_status.setText("待校准 / Pending Calibration")
            self.verify_status.setStyleSheet("color: #888;")
        elif self._verification_passed:
            self.verify_status.setText("已通过 / Passed")
            self.verify_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        elif self.verify_status.text() not in {"验证中 / Verifying"}:
            self.verify_status.setText("待验证 / Verification Required")
            self.verify_status.setStyleSheet("color: #FF9800; font-weight: 600;")

        if self.dwell_enabled:
            self.dwell_status.setText("开启 / On")
            self.dwell_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self.dwell_btn.setText("关闭点击\nDwell Click")
        else:
            self.dwell_status.setText("关闭 / Off")
            self.dwell_status.setStyleSheet("color: #888;")
            self.dwell_btn.setText("启用点击\nDwell Click")

        if self.interaction_overlay is not None:
            self.launcher_status.setText("运行中 / Open")
            self.launcher_status.setStyleSheet("color: #2196F3; font-weight: 600;")
            self.launcher_btn.setText("关闭交互\nClose Interaction")
        elif self.gomoku_window is not None:
            self.launcher_status.setText("五子棋 / Gomoku")
            self.launcher_status.setStyleSheet("color: #2196F3; font-weight: 600;")
            self.launcher_btn.setText("交互窗口\nOpen Interaction")
        elif fullscreen_stage_open:
            self.launcher_status.setText("等待中 / Stage Active")
            self.launcher_status.setStyleSheet("color: #888;")
            self.launcher_btn.setText("交互窗口\nOpen Interaction")
        else:
            self.launcher_status.setText("未进入 / Not Open")
            self.launcher_status.setStyleSheet("color: #888;")
            self.launcher_btn.setText("交互窗口\nOpen Interaction")

    def _on_tracker_error(self, message: str) -> None:
        self.error_label.setText(f"错误: {message}")

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
