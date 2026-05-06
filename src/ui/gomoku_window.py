from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class GomokuWindow(QWidget):
    """Full-screen 5x5 gaze-playable O/X board.

    The old 15x15 Gomoku board was too dense for gaze. This keeps the same
    TrackingPage integration point but presents a coarse 5x5 tic-tac-toe style
    game with large cells.
    """

    closed = pyqtSignal()
    return_to_launcher = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.board_size = 5
        self.win_len = 3
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_hover: Optional[tuple[int, int]] = None
        self._hover_started_at = 0.0
        self._last_gaze: Optional[QPointF] = None
        self._dwell_ms = 900.0
        self._game_over = False
        self._result_message = ""
        self._result_emoji = ""
        self._return_countdown_ms = 0
        self._return_timer = QTimer(self)
        self._return_timer.timeout.connect(self._on_return_tick)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet("background-color: #FAF9F6;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 24, 36, 24)

        header = QHBoxLayout()
        self.status_label = QLabel("X 落子")
        self.status_label.setStyleSheet(
            """
            QLabel {
                color: #263238;
                font-size: 24px;
                font-weight: 800;
                font-family: "Microsoft YaHei", "SimHei", Arial;
                background: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
                padding: 10px 16px;
            }
            """
        )
        self.reset_button = QPushButton("重新开始")
        self.close_button = QPushButton("关闭")
        for button in (self.reset_button, self.close_button):
            button.setFixedSize(110, 44)
            button.setStyleSheet(
                """
                QPushButton {
                    color: #263238;
                    background: #FFFFFF;
                    border: 1px solid #D0D0D0;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 700;
                    font-family: "Microsoft YaHei", "SimHei", Arial;
                }
                QPushButton:hover { background: #F0F7F0; }
                """
            )
        self.reset_button.clicked.connect(self._reset_board)
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.status_label)
        header.addStretch(1)
        header.addWidget(self.reset_button)
        header.addSpacing(10)
        header.addWidget(self.close_button)
        layout.addLayout(header)
        layout.addStretch(1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def update_gaze_point(self, x: float, y: float) -> None:
        self._last_gaze = QPointF(x, y)
        pos = self._board_pos_from_point(x, y)
        now = self._now_ms()

        if self._game_over:
            self.update()
            return

        if pos != self.current_hover:
            self.current_hover = pos
            self._hover_started_at = now
        elif pos is not None:
            progress = min((now - self._hover_started_at) / self._dwell_ms, 1.0)
            if progress >= 1.0:
                self._place_x(pos)
                self.current_hover = None
                self._hover_started_at = now

        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self._board_rect()
        painter.setPen(QPen(QColor(224, 224, 224), 2))
        painter.setBrush(QColor(255, 255, 255))
        painter.drawRoundedRect(rect, 12, 12)

        cell = rect.width() / self.board_size
        painter.setPen(QPen(QColor(84, 125, 93), 4))
        for i in range(self.board_size + 1):
            x = rect.left() + i * cell
            y = rect.top() + i * cell
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))

        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.board[row][col]
                if piece:
                    self._draw_piece(painter, rect, row, col, piece)

        if self.current_hover is not None and not self._game_over:
            row, col = self.current_hover
            cell_rect = self._cell_rect(rect, row, col)
            progress = min((self._now_ms() - self._hover_started_at) / self._dwell_ms, 1.0)
            painter.setPen(QPen(QColor(76, 175, 80, 220), 5))
            painter.setBrush(QColor(76, 175, 80, 45))
            painter.drawRoundedRect(cell_rect.adjusted(8, 8, -8, -8), 10, 10)
            radius = min(cell_rect.width(), cell_rect.height()) * 0.16
            cx = cell_rect.right() - radius - 14
            cy = cell_rect.bottom() - radius - 14
            painter.setPen(QPen(QColor(200, 200, 200), 4))
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            painter.setPen(QPen(QColor(76, 175, 80), 4))
            painter.drawArc(
                int(cx - radius),
                int(cy - radius),
                int(radius * 2),
                int(radius * 2),
                90 * 16,
                -int(progress * 360 * 16),
            )

        if self._last_gaze is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 0, 0, 110))
            painter.drawEllipse(self._last_gaze, 13, 13)
            painter.setBrush(QColor(255, 255, 255, 245))
            painter.drawEllipse(self._last_gaze, 4, 4)

        if self._game_over:
            self._draw_result_overlay(painter)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_R:
            self._reset_board()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._return_timer.stop()
        self.closed.emit()
        super().closeEvent(event)

    def _board_rect(self) -> QRectF:
        width = self.width()
        height = self.height()
        top = 92.0
        size = min(width - 120.0, height - top - 44.0)
        left = (width - size) / 2.0
        return QRectF(left, top, size, size)

    def _board_pos_from_point(self, x: float, y: float) -> Optional[tuple[int, int]]:
        rect = self._board_rect()
        if not rect.contains(QPointF(x, y)):
            return None

        cell = rect.width() / self.board_size
        col = int((x - rect.left()) // cell)
        row = int((y - rect.top()) // cell)
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return None
        if self.board[row][col] != 0:
            return None
        return (row, col)

    def _cell_rect(self, rect: QRectF, row: int, col: int) -> QRectF:
        cell = rect.width() / self.board_size
        return QRectF(rect.left() + col * cell, rect.top() + row * cell, cell, cell)

    def _draw_piece(self, painter: QPainter, rect: QRectF, row: int, col: int, piece: int) -> None:
        cell_rect = self._cell_rect(rect, row, col).adjusted(24, 24, -24, -24)
        if piece == 1:
            painter.setPen(QPen(QColor(38, 50, 56), 10, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            painter.drawLine(cell_rect.topLeft(), cell_rect.bottomRight())
            painter.drawLine(cell_rect.topRight(), cell_rect.bottomLeft())
        else:
            painter.setPen(QPen(QColor(83, 125, 93), 10))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(cell_rect)

    def _place_x(self, pos: tuple[int, int]) -> None:
        row, col = pos
        if self.board[row][col] != 0:
            return
        self.board[row][col] = 1
        if self._winner_from(row, col, 1):
            self._finish_game("X 获胜", "🎉")
            return
        if self._is_full():
            self._finish_game("平局", "😐")
            return
        self._place_o()

    def _place_o(self) -> None:
        move = self._best_o_move()
        if move is None:
            return
        row, col = move
        self.board[row][col] = 2
        if self._winner_from(row, col, 2):
            self._finish_game("O 获胜", "😅")
        else:
            self.status_label.setText("X 落子")

    def _best_o_move(self) -> Optional[tuple[int, int]]:
        center = (self.board_size // 2, self.board_size // 2)
        empty = [
            (row, col)
            for row in range(self.board_size)
            for col in range(self.board_size)
            if self.board[row][col] == 0
        ]
        if not empty:
            return None
        for player in (2, 1):
            for row, col in empty:
                self.board[row][col] = player
                wins = self._winner_from(row, col, player)
                self.board[row][col] = 0
                if wins:
                    return (row, col)
        return min(empty, key=lambda pos: math.dist(pos, center))

    def _winner_from(self, row: int, col: int, piece: int) -> bool:
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            count += self._count_dir(row, col, dr, dc, piece)
            count += self._count_dir(row, col, -dr, -dc, piece)
            if count >= self.win_len:
                return True
        return False

    def _count_dir(self, row: int, col: int, dr: int, dc: int, piece: int) -> int:
        count = 0
        row += dr
        col += dc
        while 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == piece:
            count += 1
            row += dr
            col += dc
        return count

    def _is_full(self) -> bool:
        return all(self.board[row][col] != 0 for row in range(self.board_size) for col in range(self.board_size))

    def _finish_game(self, message: str, emoji: str) -> None:
        self._game_over = True
        self._result_message = message
        self._result_emoji = emoji
        self._return_countdown_ms = 2000
        self.status_label.setText(f"{message}，2秒后返回")
        self._return_timer.start(250)
        self.update()

    def _on_return_tick(self) -> None:
        self._return_countdown_ms = max(0, self._return_countdown_ms - 250)
        if self._return_countdown_ms <= 0:
            self._return_timer.stop()
            self.return_to_launcher.emit()
            self.close()
            return
        seconds = max(1, math.ceil(self._return_countdown_ms / 1000.0))
        self.status_label.setText(f"{self._result_message}，{seconds}秒后返回")
        self.update()

    def _draw_result_overlay(self, painter: QPainter) -> None:
        overlay_rect = self.rect().adjusted(120, 120, -120, -120)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(250, 249, 246, 232))
        painter.drawRoundedRect(overlay_rect, 24, 24)

        seconds = math.ceil(max(0, self._return_countdown_ms) / 1000.0)
        painter.setPen(QColor(38, 50, 56))

        emoji_font = painter.font()
        emoji_font.setPointSize(82)
        emoji_font.setBold(True)
        painter.setFont(emoji_font)
        painter.drawText(self.rect().adjusted(0, 190, 0, 0), Qt.AlignmentFlag.AlignHCenter, self._result_emoji)

        text_font = painter.font()
        text_font.setPointSize(34)
        text_font.setBold(True)
        painter.setFont(text_font)
        painter.drawText(self.rect().adjusted(0, 330, 0, 0), Qt.AlignmentFlag.AlignHCenter, self._result_message)

        countdown_font = painter.font()
        countdown_font.setPointSize(20)
        countdown_font.setBold(False)
        painter.setFont(countdown_font)
        painter.setPen(QColor(84, 125, 93))
        painter.drawText(
            self.rect().adjusted(0, 410, 0, 0),
            Qt.AlignmentFlag.AlignHCenter,
            f"{seconds} 秒后返回九宫格",
        )

    def _reset_board(self) -> None:
        self._return_timer.stop()
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_hover = None
        self._last_gaze = None
        self._game_over = False
        self._result_message = ""
        self._result_emoji = ""
        self._return_countdown_ms = 0
        self.status_label.setText("X 落子")
        self.update()

    @staticmethod
    def _now_ms() -> float:
        from time import perf_counter

        return perf_counter() * 1000.0
