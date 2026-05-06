"""校准页面。

全屏校准界面，依次显示校准点并引导用户注视，采集视线数据并拟合校准参数。

主要功能：
1. 全屏显示校准点（3x3 网格，共 9 个点）
2. 每个校准点显示倒计时和进度提示
3. 在后台运行 TrackerPipeline 采集视线数据
4. 调用 CalibrationModule 进行仿射变换拟合
5. 显示校准结果（残差、成功/失败状态）
6. 提供重新校准和保存校准的选项

校准流程：
1. 用户点击"开始校准"按钮
2. 进入全屏模式，显示第一个校准点
3. 倒计时 3 秒，引导用户注视
4. 采集该点的视线数据（采样 30 帧，取平均）
5. 移动到下一个校准点，重复步骤 3-4
6. 所有点采集完成后，拟合仿射变换矩阵
7. 显示校准结果（残差、成功/失败）
8. 用户可选择保存校准或重新校准
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QColor, QPen
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
)

from qfluentwidgets import (
    BodyLabel,
    CardWidget,
    PrimaryPushButton,
    PushButton,
    MessageBox,
)

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import save_calibration, load_calibration
from src.tracker.pipeline import TrackerPipeline, SystemConfig


def calibration_path_for_backend(backend: str) -> Path:
    return Path("calibration_deep.json" if backend == "deep" else "calibration_classic.json")


def min_valid_points_for_calibration(num_points: int, method: str) -> int:
    """Return the minimum usable point count before fitting calibration."""
    if method == "polynomial":
        return 4 if num_points >= 25 else 6
    return 3


def min_samples_per_calibration_point(sampling_frames: int) -> int:
    """Return the minimum successful samples needed to accept one target."""
    return max(6, sampling_frames // 3)


@dataclass
class CalibrationPoint:
    """校准点数据结构。"""
    x: float  # 屏幕像素坐标 X
    y: float  # 屏幕像素坐标 Y
    index: int  # 点序号（0-8）


class CalibrationFullscreenWidget(QWidget):
    """全屏校准界面。
    
    显示校准点、倒计时、进度提示。
    
    Signals:
        calibration_finished: 校准完成信号，发送 (success: bool, residual: float)
        calibration_cancelled: 用户取消校准信号
    """
    calibration_finished = pyqtSignal(bool, float)  # (success, residual)
    calibration_cancelled = pyqtSignal()
    
    def __init__(
        self,
        tracker: TrackerPipeline,
        calibrator: CalibrationModule,
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        
        self.tracker = tracker
        self.calibrator = calibrator
        
        # 校准点配置（3x3 或 5x5 网格）
        self.calibration_points: list[CalibrationPoint] = []
        self._generate_calibration_points()
        
        # 校准状态
        self.current_point_index = 0
        self.sampling_frames = 55 if calibrator.num_points >= 25 else 30
        self.max_sampling_ticks = self.sampling_frames * 3
        self.min_samples_per_point = min_samples_per_calibration_point(self.sampling_frames)
        self.current_samples: list[tuple[float, float]] = []  # 当前点的采样数据
        self.sampling_ticks = 0
        
        self.sampling_timer = QTimer()
        self.sampling_timer.timeout.connect(self._on_sampling_tick)
        
        # UI 配置
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: #1E1E1E;")
        self.setCursor(Qt.CursorShape.BlankCursor)  # 隐藏鼠标光标
        
    def _generate_calibration_points(self) -> None:
        """生成 3x3 网格的 9 个校准点。
        
        校准点分布在屏幕的 9 个位置：
        - 边距：距离屏幕边缘 10%
        - 网格：3 行 × 3 列
        """
        from PyQt6.QtWidgets import QApplication
        screen = QApplication.primaryScreen()
        if screen is None:
            # 默认分辨率
            screen_w, screen_h = 1920, 1080
        else:
            geometry = screen.geometry()
            screen_w, screen_h = geometry.width(), geometry.height()
        
        # 边距比例
        margin_ratio = 0.1
        margin_x = screen_w * margin_ratio
        margin_y = screen_h * margin_ratio
        
        # 有效区域
        effective_w = screen_w - 2 * margin_x
        effective_h = screen_h - 2 * margin_y
        
        grid_size = 5 if self.calibrator.num_points >= 25 else 3

        # 生成网格
        index = 0
        for row in range(grid_size):
            for col in range(grid_size):
                denom = max(grid_size - 1, 1)
                x = margin_x + col * effective_w / denom
                y = margin_y + row * effective_h / denom
                self.calibration_points.append(CalibrationPoint(x=x, y=y, index=index))
                index += 1
    
    def start_calibration(self) -> None:
        """开始校准流程。"""
        # 重置状态
        self.current_point_index = 0
        self.calibrator.clear_points()
        
        # 启用校准模式（禁用坐标 clamp）
        if self.tracker is not None:
            self.tracker.set_calibration_mode(True)
        
        # 显示全屏
        self.showFullScreen()
        
        # 校准点出现后立即采样，避免每点 3 秒等待影响效率。
        self._start_sampling()
    
    def _start_sampling(self) -> None:
        """开始采样当前校准点的视线数据。"""
        self.current_samples.clear()
        self.sampling_ticks = 0
        self.sampling_timer.start(33)  # 约 30 FPS
    
    def _on_sampling_tick(self) -> None:
        """采样定时器回调。"""
        # 从 TrackerPipeline 获取最新的视线数据
        result = self.tracker.get_latest_result()
        self.sampling_ticks += 1
        
        if result is not None and result.valid and result.gaze_point is not None:
            self.current_samples.append(result.gaze_point)
        
        # 检查是否采样完成
        if (
            len(self.current_samples) >= self.sampling_frames
            or self.sampling_ticks >= self.max_sampling_ticks
        ):
            self.sampling_timer.stop()
            self._finish_current_point()
        
        self.update()  # 刷新绘制（显示采样进度）
    
    def _finish_current_point(self) -> None:
        """完成当前校准点的采集。"""
        if len(self.current_samples) < self.min_samples_per_point:
            # 采样失败，跳过该点
            print(
                "[CALIBRATION] 警告：校准点 "
                f"{self.current_point_index} 有效样本不足 "
                f"({len(self.current_samples)}/{self.min_samples_per_point})，跳过"
            )
            self._move_to_next_point()
            return
        
        # 计算平均视线坐标
        avg_x = sum(p[0] for p in self.current_samples) / len(self.current_samples)
        avg_y = sum(p[1] for p in self.current_samples) / len(self.current_samples)
        
        # 获取当前校准点的真实屏幕坐标
        current_point = self.calibration_points[self.current_point_index]
        target_x, target_y = current_point.x, current_point.y
        
        # 添加到校准模块
        self.calibrator.add_calibration_point(
            raw_gaze=(avg_x, avg_y),
            screen_target=(target_x, target_y)
        )
        
        print(f"[CALIBRATION] 校准点 {self.current_point_index}: raw=({avg_x:.1f}, {avg_y:.1f}), target=({target_x:.1f}, {target_y:.1f})")
        
        # 移动到下一个点
        self._move_to_next_point()
    
    def _move_to_next_point(self) -> None:
        """移动到下一个校准点。"""
        self.current_point_index += 1
        
        if self.current_point_index >= len(self.calibration_points):
            # 所有点采集完成，执行校准
            self._perform_calibration()
        else:
            # 继续下一个点，立即开始采样
            self._start_sampling()
    
    def _perform_calibration(self) -> None:
        """执行校准拟合。"""
        # 关闭校准模式（恢复坐标 clamp）
        if self.tracker is not None:
            self.tracker.set_calibration_mode(False)
        
        success = False
        residual = 0.0
        try:
            point_count = len(self.calibrator._raw_points)
            min_points = min_valid_points_for_calibration(
                self.calibrator.num_points,
                self.calibrator.method.value,
            )
            if point_count < min_points:
                raise ValueError(f"有效校准点不足：需要至少 {min_points} 个，当前 {point_count} 个")
            residual = self.calibrator.calibrate()
            success = not self.calibrator.needs_recalibration()
            print(f"[CALIBRATION] 校准完成：残差={residual:.2f} px, 成功={success}")
        except Exception as e:
            print(f"[CALIBRATION] 校准失败：{e}")
        
        # 先退出全屏，再发送信号（避免模态 MessageBox 被全屏窗口遮挡）
        self.close()
        self.calibration_finished.emit(success, residual)
    
    def paintEvent(self, event) -> None:
        """绘制校准界面。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制当前校准点
        if self.current_point_index < len(self.calibration_points):
            point = self.calibration_points[self.current_point_index]
            
            # 绘制校准点（红色圆圈）
            painter.setPen(QPen(QColor(255, 0, 0), 3))
            painter.setBrush(QColor(255, 0, 0, 200))
            painter.drawEllipse(int(point.x - 15), int(point.y - 15), 30, 30)
            
            # 绘制中心点
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(int(point.x - 5), int(point.y - 5), 10, 10)
            
            # 绘制采样进度
            painter.setPen(QColor(255, 255, 255))
            font = QFont("Arial", 24, QFont.Weight.Bold)
            painter.setFont(font)
            
            if self.sampling_timer.isActive():
                # 显示采样进度
                progress = len(self.current_samples)
                text = f"{progress}/{self.sampling_frames} ({self.sampling_ticks}/{self.max_sampling_ticks})"
                painter.drawText(int(point.x - 40), int(point.y + 60), text)
        
        # 绘制进度信息（顶部中央）
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Arial", 18)
        painter.setFont(font)
        valid_points = len(getattr(self.calibrator, "_raw_points", []))
        progress_text = (
            f"校准进度：{self.current_point_index + 1} / {len(self.calibration_points)}"
            f"  有效点：{valid_points}"
        )
        painter.drawText(self.width() // 2 - 100, 50, progress_text)
        
        # 绘制提示信息（底部中央）
        painter.setPen(QColor(150, 150, 150))
        font = QFont("Arial", 14)
        painter.setFont(font)
        hint_text = "请注视红色圆点，保持头部稳定 | 按 ESC 取消校准"
        painter.drawText(self.width() // 2 - 200, self.height() - 50, hint_text)
    
    def keyPressEvent(self, event) -> None:
        """处理键盘事件。"""
        if event.key() == Qt.Key.Key_Escape:
            # 用户取消校准，关闭校准模式
            self.sampling_timer.stop()
            if self.tracker is not None:
                self.tracker.set_calibration_mode(False)
            self.calibration_cancelled.emit()
            self.close()


class CalibrationPage(QWidget):
    """校准页面。
    
    提供校准控制界面：
    - 启动校准按钮
    - 显示校准状态和结果
    - 保存/加载校准参数
    - 重新校准选项
    """
    calibration_ready = pyqtSignal()
    return_home_requested = pyqtSignal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.backend = "classic"
        self.calibration_path = calibration_path_for_backend(self.backend)
        self.calibrator = CalibrationModule(num_points=25, max_residual_px=300.0, method="polynomial")
        
        # TrackerPipeline（需要外部传入或初始化）
        self.tracker: Optional[TrackerPipeline] = None
        
        # 全屏校准窗口
        self.fullscreen_widget: Optional[CalibrationFullscreenWidget] = None
        
        # 校准结果
        self.calibration_success = False
        self.calibration_residual = 0.0
        
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化 UI 布局。"""
        # 标题
        title = QLabel("视线校准 / Gaze Calibration")
        title.setStyleSheet("font-size: 32px; font-weight: 900;")
        
        # 说明文本
        description = BodyLabel(
            "校准可以提高视线追踪的精度。请按照屏幕提示注视校准点，保持头部稳定。\n"
            "Calibration improves gaze tracking accuracy. Follow on-screen instructions and keep your head stable."
        )
        description.setWordWrap(True)
        description.setStyleSheet("font-size: 14px; color: #666;")
        
        # 校准状态卡片
        status_card = CardWidget()
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(20, 20, 20, 20)
        status_layout.setSpacing(12)
        
        status_title = BodyLabel("校准状态 / Calibration Status")
        status_title.setStyleSheet("font-size: 16px; font-weight: 600;")
        
        self.status_label = BodyLabel("未校准 / Not Calibrated")
        self.status_label.setStyleSheet("font-size: 14px; color: #888;")
        
        self.residual_label = BodyLabel("残差 / Residual: N/A")
        self.residual_label.setStyleSheet("font-size: 14px; color: #888;")
        
        status_layout.addWidget(status_title)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.residual_label)
        
        # 控制按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        self.start_btn = PrimaryPushButton("开始校准\nStart Calibration")
        self.start_btn.setFixedSize(160, 60)
        self.start_btn.clicked.connect(self._handle_start_calibration)
        
        self.save_btn = PushButton("保存校准\nSave Calibration")
        self.save_btn.setFixedSize(160, 60)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._handle_save_calibration)
        
        self.load_btn = PushButton("加载校准\nLoad Calibration")
        self.load_btn.setFixedSize(160, 60)
        self.load_btn.clicked.connect(self._handle_load_calibration)

        self.home_btn = PushButton("返回主页\nHome")
        self.home_btn.setFixedSize(160, 60)
        self.home_btn.clicked.connect(self.return_home_requested.emit)
        self.home_btn.hide()
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.home_btn)
        button_layout.addStretch(1)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 30, 40, 30)
        main_layout.setSpacing(20)
        main_layout.addWidget(title)
        main_layout.addWidget(description)
        main_layout.addWidget(status_card)
        main_layout.addLayout(button_layout)
        main_layout.addStretch(1)
    
    def set_tracker(self, tracker: TrackerPipeline) -> None:
        """设置 TrackerPipeline 实例。
        
        参数:
            tracker: TrackerPipeline 实例（必须已启动）
        """
        self.tracker = tracker
        self._configure_for_backend(tracker.config.normalized_backend)

    def _configure_for_backend(self, backend: str) -> None:
        """Switch calibration strategy for the active tracker backend."""
        backend = backend if backend in {"classic", "deep"} else "classic"
        if backend == self.backend and self.calibrator is not None:
            return

        self.backend = backend
        self.calibration_path = calibration_path_for_backend(backend)
        if backend == "classic":
            self.calibrator = CalibrationModule(num_points=25, max_residual_px=300.0, method="polynomial")
            self.status_label.setText("Classic 5x5 校准 / Classic 5x5 Calibration")
        else:
            self.calibrator = CalibrationModule(num_points=9, max_residual_px=300.0, method="affine")
            self.status_label.setText("Deep 9点校准 / Deep 9-point Calibration")
        self.calibration_success = False
        self.calibration_residual = 0.0
        self.residual_label.setText("残差 / Residual: N/A")
        self.save_btn.setEnabled(False)
        self._show_default_actions()
    
    def _handle_start_calibration(self) -> None:
        """启动校准流程。"""
        if self.tracker is None:
            QMessageBox.warning(
                self,
                "错误",
                "TrackerPipeline 未初始化。请先启动实时追踪。"
            )
            return

        self._show_default_actions()
        
        if not self.tracker.is_running():
            QMessageBox.warning(
                self,
                "错误",
                "TrackerPipeline 未运行。请先启动实时追踪。"
            )
            return
        
        # 创建全屏校准窗口（不传 parent，使其作为独立顶层窗口以正确全屏）
        self.fullscreen_widget = CalibrationFullscreenWidget(
            tracker=self.tracker,
            calibrator=self.calibrator,
        )
        
        # 连接信号
        self.fullscreen_widget.calibration_finished.connect(self._on_calibration_finished)
        self.fullscreen_widget.calibration_cancelled.connect(self._on_calibration_cancelled)
        
        # 启动校准
        self.fullscreen_widget.start_calibration()
    
    def _on_calibration_finished(self, success: bool, residual: float) -> None:
        """校准完成回调。"""
        self.calibration_success = success
        self.calibration_residual = residual
        
        if success:
            self.status_label.setText(
                f"校准成功，请保存并进入验证 / Calibration successful, residual {residual:.2f}px"
            )
            self.status_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: 600;")
        else:
            self.status_label.setText(
                f"校准失败，建议重新校准 / Calibration failed, residual {residual:.2f}px"
            )
            self.status_label.setStyleSheet("font-size: 14px; color: #FF9800; font-weight: 600;")
        
        self.residual_label.setText(f"残差 / Residual: {residual:.2f} px")
        self._show_result_actions(success)
    
    def _on_calibration_cancelled(self) -> None:
        """用户取消校准回调。"""
        print("[CALIBRATION_PAGE] 用户取消校准")
    
    def _handle_save_calibration(self) -> None:
        """保存校准参数。"""
        if not self.calibrator.is_calibrated:
            QMessageBox.warning(self, "错误", "没有可保存的校准数据。")
            return
        
        save_path = self.calibration_path
        
        try:
            save_calibration(self.calibrator, str(save_path))
            self.status_label.setText(f"已保存，正在进入验证 / Saved to {save_path.name}")
            self.status_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: 600;")
            print(f"[CALIBRATION_PAGE] 校准参数已保存：{save_path}")
            self.calibration_ready.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存校准参数失败：{e}")
            print(f"[CALIBRATION_PAGE] 保存失败：{e}")
    
    def _handle_load_calibration(self) -> None:
        """加载校准参数。"""
        load_path = self.calibration_path
        
        if not load_path.exists():
            QMessageBox.warning(
                self,
                "文件不存在",
                f"校准文件不存在：{load_path.absolute()}\n\n请先进行校准并保存。"
            )
            return
        
        try:
            load_calibration(self.calibrator, str(load_path))
            
            # 更新 UI
            self.calibration_success = not self.calibrator.needs_recalibration()
            self.calibration_residual = self.calibrator.residual_mean
            
            self.status_label.setText("校准已加载 / Calibration Loaded")
            self.status_label.setStyleSheet("font-size: 14px; color: #2196F3; font-weight: 600;")
            self.residual_label.setText(f"残差 / Residual: {self.calibration_residual:.2f} px")
            self.save_btn.setEnabled(True)
            
            MessageBox(
                "加载成功",
                f"校准参数已加载。残差：{self.calibration_residual:.2f} 像素",
                self
            ).exec()
            print(f"[CALIBRATION_PAGE] 校准参数已加载：{load_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"加载校准参数失败：{e}")
            print(f"[CALIBRATION_PAGE] 加载失败：{e}")

    def _show_default_actions(self) -> None:
        self.start_btn.setText("开始校准\nStart Calibration")
        self.start_btn.setEnabled(True)
        self.save_btn.setText("保存校准\nSave Calibration")
        self.save_btn.setEnabled(self.calibrator.is_calibrated)
        self.load_btn.show()
        self.home_btn.hide()

    def _show_result_actions(self, success: bool) -> None:
        self.start_btn.setText("重新校准\nRecalibrate")
        self.start_btn.setEnabled(True)
        self.save_btn.setText("保存并进入验证\nSave & Verify")
        self.save_btn.setEnabled(success and self.calibrator.is_calibrated)
        self.load_btn.hide()
        self.home_btn.show()
