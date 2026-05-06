from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from qfluentwidgets import BodyLabel, PushButton, TitleLabel


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
        self._points: list[QPointF] = []
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

    def clear(self) -> None:
        self._points.clear()
        self.update()

    def update_gaze_point(self, x: float, y: float) -> None:
        self._points.append(QPointF(x, y))
        if len(self._points) > 36:
            self._points.pop(0)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        if len(self._points) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath(self._points[0])
        for idx in range(1, len(self._points)):
            prev = self._points[idx - 1]
            cur = self._points[idx]
            mid = QPointF((prev.x() + cur.x()) / 2.0, (prev.y() + cur.y()) / 2.0)
            path.quadTo(prev, mid)
        path.lineTo(self._points[-1])

        glow = QPen(QColor(59, 130, 246, 70), 18)
        glow.setCapStyle(Qt.PenCapStyle.RoundCap)
        glow.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(glow)
        painter.drawPath(path)

        pen = QPen(QColor(96, 165, 250, 230), 8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawPath(path)


class GazeVerificationWindow(FullscreenStageWindow):
    verified = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._gaze_point: Optional[tuple[float, float]] = None
        self._target_points: list[tuple[float, float]] = []

        self._init_ui()
        self._build_reference_points()

    def _init_ui(self) -> None:
        self.setStyleSheet("background-color: rgb(5, 10, 18);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(16)

        header = QHBoxLayout()
        title_block = QVBoxLayout()

        title = TitleLabel("校准验证")
        title.setStyleSheet("color: white; font-size: 34px; font-weight: 900;")
        subtitle = BodyLabel(
            "移动视线观察蓝色轨迹是否稳定跟随。确认后进入交互阶段。"
        )
        subtitle.setStyleSheet("color: rgba(226, 232, 240, 210); font-size: 16px;")

        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.status_label = QLabel("请注视参考点并观察轨迹")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgba(15, 23, 42, 230);
                border: 1px solid rgba(148, 163, 184, 90);
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 15px;
                font-weight: 600;
            }
            """
        )

        header.addLayout(title_block)
        header.addStretch(1)
        header.addWidget(self.status_label)
        layout.addLayout(header)
        layout.addStretch(1)

        footer = QHBoxLayout()
        footer.addStretch(1)

        confirm = PushButton("确认并继续")
        confirm.clicked.connect(self._confirm)
        cancel = PushButton("取消")
        cancel.clicked.connect(self._cancel)

        footer.addWidget(confirm)
        footer.addSpacing(12)
        footer.addWidget(cancel)
        layout.addLayout(footer)

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
            painter.setPen(QPen(QColor(248, 250, 252, 220), 3))
            painter.setBrush(QColor(239, 68, 68, 230))
            painter.drawEllipse(int(tx - 18), int(ty - 18), 36, 36)
            painter.setBrush(QColor(255, 255, 255, 255))
            painter.drawEllipse(int(tx - 6), int(ty - 6), 12, 12)
            painter.setPen(QPen(QColor(226, 232, 240, 190), 1))
            painter.drawText(int(tx + 20), int(ty - 18), f"P{index}")

        if self._gaze_point is not None:
            gx, gy = self._gaze_point
            painter.setPen(QPen(QColor(191, 219, 254, 230), 4))
            painter.setBrush(QColor(37, 99, 235, 160))
            painter.drawEllipse(int(gx - 24), int(gy - 24), 48, 48)
            painter.setBrush(QColor(255, 255, 255, 255))
            painter.drawEllipse(int(gx - 6), int(gy - 6), 12, 12)

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
    subtitle: str
    callback: Optional[Callable[[], None]]


class LauncherRegionCard(QFrame):
    def __init__(self, title: str, subtitle: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._progress = 0.0
        self._active = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(12)

        title_label = TitleLabel(title)
        title_label.setStyleSheet("color: white; font-size: 30px; font-weight: 900;")
        subtitle_label = BodyLabel(subtitle)
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet("color: rgba(226, 232, 240, 210); font-size: 15px;")

        layout.addStretch(1)
        layout.addWidget(title_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle_label, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        self._apply_style()

    def _apply_style(self) -> None:
        border = "rgba(51, 65, 85, 255)"
        background = "rgb(15, 23, 42)"
        if self._active:
            border = "rgba(96, 165, 250, 255)"
            background = "rgb(30, 64, 120)"

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

        radius = min(self.width(), self.height()) // 7
        center_x = self.width() - radius - 16
        center_y = self.height() - radius - 16

        painter.setPen(QPen(QColor(255, 255, 255, 80), 7))
        painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)

        painter.setPen(QPen(QColor(96, 165, 250, 245), 7))
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
            LauncherAction(0, "浏览器", "打开默认浏览器", launch_browser),
            LauncherAction(1, "文件", "打开用户文件夹", launch_explorer),
            LauncherAction(2, "记事本", "打开轻量编辑器", launch_notepad),
            LauncherAction(3, "五子棋", "进入凝视落子游戏", None),
            LauncherAction(5, "放大镜", "打开系统放大镜", launch_magnifier),
            LauncherAction(6, "键盘", "打开屏幕键盘", launch_osk),
            LauncherAction(7, "返回", "关闭交互页", None),
            LauncherAction(8, "退出", "关闭交互页", None),
        ]
        self._cards: dict[int, LauncherRegionCard] = {}
        self._current_region: Optional[int] = None
        self._region_enter_time = 0.0
        self._last_trigger_time = 0.0
        self._dwell_ms = 1200.0

        self._init_ui()

    def _init_ui(self) -> None:
        self.setStyleSheet("background-color: rgb(5, 10, 18);")

        root = QVBoxLayout(self)
        root.setContentsMargins(38, 28, 38, 28)
        root.setSpacing(18)

        header = QHBoxLayout()
        title_block = QVBoxLayout()

        title = TitleLabel("Look2Act 交互页")
        title.setStyleSheet("color: white; font-size: 36px; font-weight: 900;")
        subtitle = BodyLabel(
            "凝视大目标完成启动。轨迹只显示在当前独立窗口内。"
        )
        subtitle.setStyleSheet("color: rgba(226, 232, 240, 210); font-size: 16px;")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.status_label = QLabel("等待凝视目标")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: white;
                background-color: rgba(15, 23, 42, 230);
                border: 1px solid rgba(148, 163, 184, 90);
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 15px;
                font-weight: 600;
            }
            """
        )

        close_button = PushButton("关闭")
        close_button.clicked.connect(self.close)

        header.addLayout(title_block)
        header.addStretch(1)
        header.addWidget(self.status_label)
        header.addSpacing(12)
        header.addWidget(close_button)
        root.addLayout(header)

        board = QFrame()
        board.setStyleSheet(
            """
            QFrame {
                background-color: rgb(8, 13, 24);
                border: 1px solid rgba(51, 65, 85, 255);
                border-radius: 12px;
            }
            """
        )

        grid = QGridLayout(board)
        grid.setContentsMargins(28, 28, 28, 28)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(24)

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
            card = LauncherRegionCard(action.title, action.subtitle)
            self._cards[action.index] = card
            grid.addWidget(card, row, col)

        center_card = QFrame()
        center_card.setStyleSheet(
            """
            QFrame {
                background-color: rgb(15, 23, 42);
                border: 1px dashed rgba(96, 165, 250, 160);
                border-radius: 10px;
            }
            """
        )
        center_layout = QVBoxLayout(center_card)
        center_layout.setContentsMargins(20, 20, 20, 20)
        center_layout.setSpacing(12)

        center_title = TitleLabel("流程")
        center_title.setStyleSheet("color: white;")
        center_text = BodyLabel(
            "校准 -> 验证 -> 交互\n"
            "保持视线稳定即可触发当前目标。"
        )
        center_text.setWordWrap(True)
        center_text.setStyleSheet("color: rgba(226, 232, 240, 210); font-size: 15px;")

        center_layout.addWidget(center_title, alignment=Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(center_text)
        center_layout.addStretch(1)
        grid.addWidget(center_card, 1, 1)

        for row in range(3):
            grid.setRowStretch(row, 1)
        for col in range(3):
            grid.setColumnStretch(col, 1)

        root.addWidget(board, 1)

        self.trail_overlay = GazeTrailOverlay(self)
        self.trail_overlay.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.trail_overlay.setGeometry(self.rect())
        self.trail_overlay.raise_()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_progress()
        self.trail_overlay.clear()

    def reset_progress(self) -> None:
        self._current_region = None
        self._region_enter_time = 0.0
        for card in self._cards.values():
            card.reset_progress()

    def update_gaze_point(self, x: float, y: float) -> None:
        self.trail_overlay.update_gaze_point(x, y)

        now = self._now_ms()
        if now - self._last_trigger_time < 500.0:
            return

        hovered_region = self._region_at_global_point(int(x), int(y))
        if hovered_region != self._current_region:
            self._current_region = hovered_region
            self._region_enter_time = now
            for index, card in self._cards.items():
                card.set_progress(0.0, index == hovered_region)

        if hovered_region is None:
            self.status_label.setText("请凝视一个目标")
            return

        progress = min((now - self._region_enter_time) / self._dwell_ms, 1.0)
        for index, card in self._cards.items():
            card.set_progress(progress if index == hovered_region else 0.0, index == hovered_region)

        action = self._action_by_region(hovered_region)
        if action is not None:
            self.status_label.setText(f"{action.title}：{int(progress * 100)}%")

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

        self.status_label.setText(f"已触发：{action.title}")

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
