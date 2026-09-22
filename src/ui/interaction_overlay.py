from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.ui.i18n import tx, tx_button


if sys.platform == "win32":
    import ctypes


def perform_left_click() -> bool:
    if sys.platform != "win32":
        return False

    user32 = ctypes.windll.user32
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    return True


def launch_browser() -> None:
    webbrowser.open("https://www.bing.com")


def launch_explorer() -> None:
    if sys.platform == "win32":
        os.startfile(str(Path.home()))
        return
    subprocess.Popen(["xdg-open", str(Path.home())])


def launch_notepad() -> None:
    if sys.platform == "win32":
        subprocess.Popen(["notepad.exe"])
        return
    subprocess.Popen(["gedit"])


def launch_osk() -> None:
    if sys.platform == "win32":
        subprocess.Popen(["osk.exe"])


def launch_magnifier() -> None:
    if sys.platform == "win32":
        subprocess.Popen(["magnify.exe"])


class FullscreenStageWindow(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAutoFillBackground(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()


class GazeTrailOverlay(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._points: deque[QPointF] = deque(maxlen=30)
        self._current_point: Optional[QPointF] = None
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

    def clear(self) -> None:
        self._points.clear()
        self._current_point = None
        self.update()

    def update_gaze_point(self, x: float, y: float) -> None:
        point = QPointF(x, y)
        self._current_point = point
        self._points.append(point)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        if len(self._points) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        points = list(self._points)
        for index in range(1, len(points)):
            alpha = int(255 * (index / len(points)))
            pen = QPen(QColor(100, 180, 255, alpha), 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(points[index - 1], points[index])

        if self._current_point is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(QColor(255, 0, 0, 100)))
            painter.drawEllipse(self._current_point, 12, 12)
            painter.setBrush(QBrush(QColor(255, 0, 0, 200)))
            painter.drawEllipse(self._current_point, 8, 8)
            painter.setBrush(QBrush(QColor(255, 255, 255, 255)))
            painter.drawEllipse(self._current_point, 4, 4)


class GazeVerificationWindow(FullscreenStageWindow):
    verified = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._gaze_point: Optional[tuple[float, float]] = None
        self._target_point: Optional[tuple[int, int]] = None
        self._target_points: list[tuple[float, float]] = []

        self._init_ui()
        self._build_reference_points()

    def _init_ui(self) -> None:
        self.setStyleSheet("background-color: black;")

        self.trail_overlay = GazeTrailOverlay(self)
        self.trail_overlay.raise_()

    def _build_reference_points(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            w, h = 1920, 1080
        else:
            geo = screen.geometry()
            w, h = geo.width(), geo.height()

        margin_x = int(w * 0.12)
        margin_y = int(h * 0.14)

        self._target_points = [
            (margin_x, margin_y),
            (w - margin_x, margin_y),
            (w // 2, h // 2),
            (margin_x, h - margin_y),
            (w - margin_x, h - margin_y),
        ]

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.trail_overlay.setGeometry(self.rect())
        self.trail_overlay.raise_()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._gaze_point = None
        self._target_point = None
        self.trail_overlay.clear()

    def update_gaze_point(self, x: float, y: float) -> None:
        self._gaze_point = (x, y)
        self.trail_overlay.update_gaze_point(x, y)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for index, (tx, ty) in enumerate(self._target_points, start=1):
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
            painter.setBrush(QColor(0, 0, 255, 220))
            painter.drawEllipse(int(tx - 15), int(ty - 15), 30, 30)
            painter.setBrush(QColor(255, 255, 255, 255))
            painter.drawEllipse(int(tx - 4), int(ty - 4), 8, 8)

        vector_length = 0.0
        if self._target_point is not None:
            tx, ty = self._target_point
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 255, 255))
            painter.drawEllipse(tx - 15, ty - 15, 30, 30)

        if self._gaze_point is not None and self._target_point is not None:
            gx, gy = self._gaze_point
            tx, ty = self._target_point
            painter.setPen(QPen(QColor(0, 255, 255), 3))
            painter.drawLine(int(gx), int(gy), tx, ty)
            vector_length = ((gx - tx) ** 2 + (gy - ty) ** 2) ** 0.5

        self._draw_vector_info(painter, vector_length)

    def mousePressEvent(self, event) -> None:
        self._target_point = (int(event.position().x()), int(event.position().y()))
        self.trail_overlay.clear()
        self.update()

    def _draw_vector_info(self, painter: QPainter, vector_length: float) -> None:
        font = QFont("Arial", 20)
        painter.setFont(font)
        lines = [
            tx(f"向量长度：{vector_length:.1f} px", f"Vector Length: {vector_length:.1f} px"),
            tx(
                f"精度：{'优秀' if vector_length < 50 else '良好' if vector_length < 100 else '需改进'}",
                f"Accuracy: {'Excellent' if vector_length < 50 else 'Good' if vector_length < 100 else 'Needs Improvement'}",
            ),
            tx("Enter：确认    Esc：取消", "Enter: Confirm    Esc: Cancel"),
        ]
        if self._target_point is None:
            lines.append(tx("点击屏幕设置目标点", "Click screen to set target point"))

        metrics = painter.fontMetrics()
        width = max(metrics.horizontalAdvance(line) for line in lines) + 28
        height = len(lines) * 32 + 22
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 0, 0, 210))
        painter.drawRect(10, 10, width, height)

        painter.setPen(QColor(255, 255, 255))
        for idx, line in enumerate(lines):
            color = QColor(100, 200, 255) if "Enter" in line else QColor(255, 255, 255)
            painter.setPen(color)
            painter.drawText(24, 42 + idx * 32, line)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._confirm()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._cancel()
            return
        super().keyPressEvent(event)

    def _confirm(self) -> None:
        self.verified.emit()
        self.close()

    def _cancel(self) -> None:
        self.cancelled.emit()
        self.close()


@dataclass(frozen=True)
class LauncherAction:
    index: int
    title: str
    icon: str
    subtitle: str
    callback: Optional[Callable[[], None]]


class LauncherRegionCard(QFrame):
    def __init__(self, title: str, icon: str, subtitle: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = title
        self.icon = icon
        self._progress = 0.0
        self._active = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title_label = QLabel(f"{icon}\n{title}")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            """
            QLabel {
                background: transparent;
                border: none;
                color: #333333;
                font-size: 24px;
                font-weight: 600;
                font-family: "Microsoft YaHei", "SimHei", Arial;
            }
            """
        )
        subtitle_label = QLabel(subtitle)
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet(
            """
            QLabel {
                background: transparent;
                border: none;
                color: #666666;
                font-size: 14px;
                font-family: "Microsoft YaHei", "SimHei", Arial;
            }
            """
        )

        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        layout.addStretch(1)

        self._apply_style()

    def _apply_style(self) -> None:
        border = "#E0E0E0"
        background = "#FAF9F6"
        if self._active:
            border = "#4CAF50"
            background = "#F4FFF4"

        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {background};
                border: 2px solid {border};
                border-radius: 10px;
            }}
            """
        )

    def set_progress(self, progress: float, active: bool) -> None:
        self._progress = max(0.0, min(progress, 1.0))
        self._active = active
        self._apply_style()
        self.update()

    def reset_progress(self) -> None:
        self.set_progress(0.0, False)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._progress <= 0.0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        radius = 20
        center_x = self.width() - radius - 10
        center_y = self.height() - radius - 10

        painter.setPen(QPen(QColor(200, 200, 200), 4))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        painter.setPen(QPen(QColor(76, 175, 80), 4))
        painter.drawArc(
            center_x - radius,
            center_y - radius,
            radius * 2,
            radius * 2,
            90 * 16,
            -int(self._progress * 360 * 16),
        )


class InteractionLauncherOverlay(FullscreenStageWindow):
    closed = pyqtSignal()
    request_toggle_dwell = pyqtSignal()
    request_open_gomoku = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._actions = [
            LauncherAction(0, tx("浏览器", "Browser"), "🌐", tx("打开默认浏览器", "Open default browser"), launch_browser),
            LauncherAction(1, tx("文件", "Files"), "📁", tx("打开用户文件夹", "Open user folder"), launch_explorer),
            LauncherAction(2, tx("记事本", "Notepad"), "📝", tx("打开轻量编辑器", "Open lightweight editor"), launch_notepad),
            LauncherAction(3, tx("五子棋", "Gomoku"), "🎮", tx("进入凝视落子游戏", "Play gaze board game"), None),
            LauncherAction(5, tx("放大镜", "Magnifier"), "🔍", tx("打开系统放大镜", "Open system magnifier"), launch_magnifier),
            LauncherAction(6, tx("键盘", "Keyboard"), "⌨", tx("打开屏幕键盘", "Open on-screen keyboard"), launch_osk),
            LauncherAction(7, tx("返回", "Return"), "⬅", tx("关闭交互页", "Close interaction"), None),
            LauncherAction(8, tx("退出", "Exit"), "✕", tx("关闭交互页", "Close interaction"), None),
        ]
        self._cards: dict[int, LauncherRegionCard] = {}
        self._current_region: Optional[int] = None
        self._region_enter_time = 0.0
        self._last_trigger_time = 0.0
        self._dwell_ms = 3000.0
        self.trail_timer: Optional[QTimer] = None

        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet("QWidget { background-color: #FAF9F6; }")

        grid = QGridLayout(self)
        grid.setContentsMargins(40, 40, 40, 40)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(20)

        positions = {
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (1, 0),
            5: (1, 2),
            6: (2, 0),
            7: (2, 1),
            8: (2, 2),
        }

        for action in self._actions:
            if action.index not in positions:
                continue
            row, col = positions[action.index]
            card = LauncherRegionCard(action.title, action.icon, action.subtitle)
            self._cards[action.index] = card
            grid.addWidget(card, row, col)

        center_hint = QLabel(tx_button("注视任意区域\n保持3秒\n即可触发操作\n\n轨迹只显示在当前页面", "Look at any tile\nHold for 3 seconds\nTrigger action\n\nTrail stays on this page"))
        center_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_hint.setStyleSheet(
            """
            QLabel {
                background-color: transparent;
                color: #999999;
                font-size: 18px;
                font-style: italic;
                font-family: "Microsoft YaHei", "SimHei", Arial;
                line-height: 1.4;
            }
            """
        )
        grid.addWidget(center_hint, 1, 1)

        for row in range(3):
            grid.setRowStretch(row, 1)
        for col in range(3):
            grid.setColumnStretch(col, 1)

        self.trail_overlay = GazeTrailOverlay(self)
        self.trail_overlay.setGeometry(self.rect())
        self.trail_overlay.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.trail_overlay.setGeometry(self.rect())
        self.trail_overlay.raise_()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_progress()
        self.trail_overlay.clear()
        self.trail_overlay.setGeometry(self.rect())
        self.trail_overlay.raise_()
        if self.trail_timer is None:
            self.trail_timer = QTimer(self)
            self.trail_timer.timeout.connect(self.trail_overlay.update)
            self.trail_timer.start(10)

    def reset_progress(self) -> None:
        self._current_region = None
        self._region_enter_time = 0.0
        for card in self._cards.values():
            card.reset_progress()

    def update_gaze_point(self, x: float, y: float, *, observed_ms: float) -> None:
        self.trail_overlay.update_gaze_point(x, y)

        now = observed_ms
        if now - self._last_trigger_time < 500.0:
            return

        hovered_region = self._region_at_global_point(int(x), int(y))
        if hovered_region != self._current_region:
            self._current_region = hovered_region
            self._region_enter_time = now
            for index, card in self._cards.items():
                card.set_progress(0.0, index == hovered_region)

        if hovered_region is None:
            return

        progress = min((now - self._region_enter_time) / self._dwell_ms, 1.0)
        for index, card in self._cards.items():
            card.set_progress(progress if index == hovered_region else 0.0, index == hovered_region)

        action = self._action_by_region(hovered_region)
        if progress >= 1.0:
            self._trigger_region(hovered_region)
            self._last_trigger_time = now
            self.reset_progress()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        if self.trail_timer is not None:
            self.trail_timer.stop()
            self.trail_timer = None
        self.trail_overlay.clear()
        self.closed.emit()
        super().closeEvent(event)

    def _region_at_global_point(self, x: int, y: int) -> Optional[int]:
        point = QPoint(x, y)
        for index, card in self._cards.items():
            rect = QRect(card.mapToGlobal(QPoint(0, 0)), card.size())
            if rect.contains(point):
                return index
        return None

    def _action_by_region(self, region_index: int) -> Optional[LauncherAction]:
        for action in self._actions:
            if action.index == region_index:
                return action
        return None

    def _trigger_region(self, region_index: int) -> None:
        action = self._action_by_region(region_index)
        if action is None:
            return

        if region_index == 3:
            self.request_open_gomoku.emit()
            self.close()
            return

        if region_index in {7, 8}:
            self.close()
            return

        if action.callback is not None:
            action.callback()
            self.close()

    @staticmethod
    def _now_ms() -> float:
        from time import perf_counter

        return perf_counter() * 1000.0
