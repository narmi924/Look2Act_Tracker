"""实时追踪页面。

显示半透明注视点光标和实时性能监控信息（FPS、各阶段延迟）。

主要功能：
1. 启动/停止实时追踪
2. 在屏幕上显示半透明注视点光标（GazeCursorOverlay）
3. 显示实时 FPS 和各阶段延迟信息
4. 支持应用校准参数（如果已校准）
5. 显示追踪状态（人脸检测、视线有效性等）

追踪流程：
1. 用户点击"启动追踪"按钮
2. 初始化 TrackerPipeline（如果未初始化）
3. 启动 TrackerPipeline 推理线程
4. 创建全屏 GazeCursorOverlay 窗口
5. 定时从 TrackerPipeline 获取最新结果并更新光标位置
6. 在页面上显示性能监控信息
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
)

from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    PrimaryPushButton,
    PushButton,
)

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import load_calibration
from src.tracker.pipeline import TrackerPipeline, SystemConfig


class GazeCursorOverlay(QWidget):
    """全屏半透明注视点光标覆盖层。
    
    在屏幕上显示半透明圆点，表示当前注视位置。
    窗口为全屏、无边框、置顶、鼠标穿透。
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # 光标配置
        self.cursor_radius = 15  # 光标半径（像素）
        self.cursor_color = QColor(255, 0, 0, 150)  # 红色，半透明
        self.cursor_border_color = QColor(255, 255, 255, 200)  # 白色边框
        self.cursor_border_width = 2
        
        # 当前注视点位置（屏幕像素坐标）
        self.gaze_x: Optional[float] = None
        self.gaze_y: Optional[float] = None
        
        # 窗口配置
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowTransparentForInput  # 鼠标穿透
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        # 全屏显示
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen is not None:
            geometry = screen.geometry()
            self.setGeometry(geometry)
    
    def update_gaze_point(self, x: float, y: float) -> None:
        """更新注视点位置并刷新绘制。
        
        参数:
            x: 屏幕像素坐标 X
            y: 屏幕像素坐标 Y
        """
        self.gaze_x = x
        self.gaze_y = y
        self.update()  # 触发重绘
    
    def clear_gaze_point(self) -> None:
        """清除注视点（不显示光标）。"""
        self.gaze_x = None
        self.gaze_y = None
        self.update()
    
    def paintEvent(self, event) -> None:
        """绘制注视点光标。"""
        if self.gaze_x is None or self.gaze_y is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制光标圆圈
        painter.setPen(QPen(self.cursor_border_color, self.cursor_border_width))
        painter.setBrush(QBrush(self.cursor_color))
        painter.drawEllipse(
            int(self.gaze_x - self.cursor_radius),
            int(self.gaze_y - self.cursor_radius),
            self.cursor_radius * 2,
            self.cursor_radius * 2
        )
        
        # 绘制中心点
        center_radius = 3
        painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(self.gaze_x - center_radius),
            int(self.gaze_y - center_radius),
            center_radius * 2,
            center_radius * 2
        )


class TrackingPage(QWidget):
    """实时追踪页面。
    
    提供追踪控制界面：
    - 启动/停止追踪按钮
    - 显示实时 FPS 和各阶段延迟
    - 显示追踪状态（人脸检测、视线有效性）
    - 支持应用校准参数
    
    页面布局：
    - 顶部：标题和控制按钮
    - 中间：性能监控卡片（FPS、各阶段延迟）
    - 底部：状态信息卡片（人脸检测、视线有效性、错误信息）
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        # TrackerPipeline 实例
        self.tracker: Optional[TrackerPipeline] = None
        self.tracker_config: Optional[SystemConfig] = None
        
        # 校准模块（可选）
        self.calibrator: Optional[CalibrationModule] = None
        
        # 注视点光标覆盖层
        self.cursor_overlay: Optional[GazeCursorOverlay] = None
        
        # 更新定时器（用于从 TrackerPipeline 获取最新结果）
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_tracking_data)
        self.update_interval_ms = 33  # 约 30 FPS
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化 UI 布局。"""
        # 标题
        title = QLabel("实时追踪 / Real-time Tracking")
        title.setStyleSheet("font-size: 32px; font-weight: 900;")
        
        # 控制按钮
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
        
        # 顶部栏
        top_bar = QHBoxLayout()
        top_bar.addWidget(title)
        top_bar.addStretch(1)
        top_bar.addWidget(self.load_calib_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.start_btn)
        top_bar.addSpacing(10)
        top_bar.addWidget(self.stop_btn)
        
        # 性能监控卡片
        perf_card = CardWidget()
        perf_layout = QVBoxLayout(perf_card)
        perf_layout.setContentsMargins(20, 20, 20, 20)
        perf_layout.setSpacing(12)
        
        perf_title = BodyLabel("性能监控 / Performance Monitor")
        perf_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        
        # FPS 显示
        fps_row = QHBoxLayout()
        fps_label = BodyLabel("FPS:")
        fps_label.setStyleSheet("font-weight: 600; font-size: 16px;")
        self.fps_value = BodyLabel("0.0")
        self.fps_value.setStyleSheet("font-weight: 900; color: #0078D4; font-size: 24px;")
        fps_row.addWidget(fps_label)
        fps_row.addSpacing(10)
        fps_row.addWidget(self.fps_value)
        fps_row.addStretch(1)
        
        # 各阶段延迟显示
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
        
        # 状态信息卡片
        status_card = CardWidget()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 20, 20, 20)
        status_layout.setSpacing(12)
        
        status_title = BodyLabel("追踪状态 / Tracking Status")
        status_title.setStyleSheet("font-size: 18px; font-weight: 600;")
        
        # 人脸检测状态
        face_row = QHBoxLayout()
        face_label = BodyLabel("人脸检测 / Face Detection:")
        face_label.setStyleSheet("font-weight: 600;")
        self.face_status = BodyLabel("未启动 / Not Started")
        self.face_status.setStyleSheet("color: #888;")
        face_row.addWidget(face_label)
        face_row.addSpacing(10)
        face_row.addWidget(self.face_status)
        face_row.addStretch(1)
        
        # 视线有效性状态
        gaze_row = QHBoxLayout()
        gaze_label = BodyLabel("视线有效性 / Gaze Valid:")
        gaze_label.setStyleSheet("font-weight: 600;")
        self.gaze_status = BodyLabel("未启动 / Not Started")
        self.gaze_status.setStyleSheet("color: #888;")
        gaze_row.addWidget(gaze_label)
        gaze_row.addSpacing(10)
        gaze_row.addWidget(self.gaze_status)
        gaze_row.addStretch(1)
        
        # 校准状态
        calib_row = QHBoxLayout()
        calib_label = BodyLabel("校准状态 / Calibration:")
        calib_label.setStyleSheet("font-weight: 600;")
        self.calib_status = BodyLabel("未加载 / Not Loaded")
        self.calib_status.setStyleSheet("color: #888;")
        calib_row.addWidget(calib_label)
        calib_row.addSpacing(10)
        calib_row.addWidget(self.calib_status)
        calib_row.addStretch(1)
        
        # 错误信息
        self.error_label = BodyLabel("")
        self.error_label.setStyleSheet("color: #D32F2F; font-weight: 600; margin-top: 10px;")
        self.error_label.setWordWrap(True)
        
        status_layout.addWidget(status_title)
        status_layout.addLayout(face_row)
        status_layout.addLayout(gaze_row)
        status_layout.addLayout(calib_row)
        status_layout.addWidget(self.error_label)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)
        main_layout.addLayout(top_bar)
        main_layout.addWidget(perf_card)
        main_layout.addWidget(status_card)
        main_layout.addStretch(1)
    
    def _handle_load_calibration(self) -> None:
        """加载校准参数。"""
        load_path = Path("calibration.json")
        
        if not load_path.exists():
            self.calib_status.setText("未加载 / Not Loaded (使用原始预测)")
            self.calib_status.setStyleSheet("color: #FF9800; font-weight: 600;")
            print(f"[TRACKING_PAGE] 校准文件不存在：{load_path.absolute()}，将使用原始预测")
            return
        
        try:
            # 创建校准模块（如果未创建）
            if self.calibrator is None:
                self.calibrator = CalibrationModule(num_points=9, max_residual_px=50.0)
            
            # 加载校准参数
            load_calibration(self.calibrator, str(load_path))
            
            # 更新 UI
            residual = self.calibrator.residual_mean
            self.calib_status.setText(f"已加载 / Loaded (残差: {residual:.2f} px)")
            self.calib_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            
            print(f"[TRACKING_PAGE] 校准参数已加载：{load_path}, 残差={residual:.2f} px")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载校准参数失败：{e}")
            print(f"[TRACKING_PAGE] 加载校准失败：{e}")
    
    def _handle_start_tracking(self) -> None:
        """启动实时追踪。"""
        try:
            self.error_label.setText("")
            
            # 初始化 TrackerPipeline（如果未初始化）
            if self.tracker is None:
                # 加载系统配置
                config_path = Path("configs/system_config.yaml")
                if config_path.exists():
                    self.tracker_config = SystemConfig.from_yaml(str(config_path))
                    print(f"[TRACKING_PAGE] 系统配置已加载：{config_path}")
                else:
                    # 使用默认配置
                    self.tracker_config = SystemConfig()
                    print("[TRACKING_PAGE] 使用默认系统配置")
                
                # 创建 TrackerPipeline
                self.tracker = TrackerPipeline(
                    model_path=self.tracker_config.checkpoint_path,
                    config=self.tracker_config,
                    error_callback=self._on_tracker_error
                )
                print("[TRACKING_PAGE] TrackerPipeline 已创建")
            
            # 启动 TrackerPipeline
            if not self.tracker.is_running():
                success = self.tracker.start()
                if not success:
                    self.error_label.setText("TrackerPipeline 启动失败，请检查摄像头和模型文件")
                    return
                print("[TRACKING_PAGE] TrackerPipeline 已启动")
            
            # 创建注视点光标覆盖层
            if self.cursor_overlay is None:
                self.cursor_overlay = GazeCursorOverlay()
            
            self.cursor_overlay.show()
            print("[TRACKING_PAGE] 注视点光标覆盖层已显示")
            
            # 启动更新定时器
            self.update_timer.start(self.update_interval_ms)
            
            # 更新按钮状态
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            print("[TRACKING_PAGE] 实时追踪已启动")
            
        except Exception as e:
            error_msg = f"启动失败：{e}"
            self.error_label.setText(error_msg)
            print(f"[TRACKING_PAGE] 启动错误：{e}")
            import traceback
            traceback.print_exc()
    
    def _handle_stop_tracking(self) -> None:
        """停止实时追踪。"""
        # 停止更新定时器
        self.update_timer.stop()
        
        # 隐藏注视点光标
        if self.cursor_overlay is not None:
            self.cursor_overlay.hide()
            self.cursor_overlay.clear_gaze_point()
        
        # 停止 TrackerPipeline（可选，保持运行以便快速重启）
        # if self.tracker is not None:
        #     self.tracker.stop()
        #     self.tracker = None
        
        # 更新按钮状态
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 重置显示
        self.fps_value.setText("0.0")
        for label in self.timing_labels.values():
            label.setText("0.00 ms")
        self.face_status.setText("未启动 / Not Started")
        self.face_status.setStyleSheet("color: #888;")
        self.gaze_status.setText("未启动 / Not Started")
        self.gaze_status.setStyleSheet("color: #888;")
        
        print("[TRACKING_PAGE] 实时追踪已停止")
    
    def _update_tracking_data(self) -> None:
        """从 TrackerPipeline 获取最新结果并更新 UI。
        
        定时器回调函数，约 30 FPS 调用。
        """
        if self.tracker is None or not self.tracker.is_running():
            return
        
        # 获取最新结果
        result = self.tracker.get_latest_result()
        
        if result is None:
            return
        
        # 更新 FPS
        self.fps_value.setText(f"{result.fps:.1f}")
        
        # 更新各阶段延迟
        for stage_key, label in self.timing_labels.items():
            timing_ms = result.timings.get(stage_key, 0.0)
            label.setText(f"{timing_ms:.2f} ms")
        
        # 更新人脸检测状态
        if result.face_detected:
            self.face_status.setText("检测到 / Detected")
            self.face_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        else:
            self.face_status.setText("未检测到 / Not Detected")
            self.face_status.setStyleSheet("color: #FF9800; font-weight: 600;")
        
        # 更新视线有效性状态
        if result.valid and result.gaze_point is not None:
            self.gaze_status.setText("有效 / Valid")
            self.gaze_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            
            # 应用校准（如果已加载）
            gaze_x, gaze_y = result.gaze_point
            if self.calibrator is not None and self.calibrator.is_calibrated:
                gaze_x, gaze_y = self.calibrator.apply((gaze_x, gaze_y))
            
            # 更新注视点光标位置
            if self.cursor_overlay is not None:
                self.cursor_overlay.update_gaze_point(gaze_x, gaze_y)
        else:
            self.gaze_status.setText("无效 / Invalid")
            self.gaze_status.setStyleSheet("color: #FF9800; font-weight: 600;")
            
            # 清除注视点光标
            if self.cursor_overlay is not None:
                self.cursor_overlay.clear_gaze_point()
        
        # 显示错误信息（如果有）
        if result.error_message:
            self.error_label.setText(f"警告: {result.error_message}")
        else:
            self.error_label.setText("")
    
    def _on_tracker_error(self, message: str) -> None:
        """TrackerPipeline 错误回调。
        
        参数:
            message: 错误消息
        """
        self.error_label.setText(f"错误: {message}")
        print(f"[TRACKING_PAGE] TrackerPipeline 错误: {message}")
    
    def closeEvent(self, event) -> None:
        """窗口关闭时停止追踪并释放资源。"""
        self._handle_stop_tracking()
        
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
        
        if self.cursor_overlay is not None:
            self.cursor_overlay.close()
            self.cursor_overlay = None
        
        super().closeEvent(event)
