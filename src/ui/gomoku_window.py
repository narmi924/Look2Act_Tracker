from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class GomokuWindow(QWidget):
    """Full-screen gaze-playable Gomoku board.

    The window does not own a camera or tracker. `TrackingPage` feeds it
    calibrated gaze points via `update_gaze_point`.
    """

    closed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.board_size = 15
        self.win_len = 5
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_hover: Optional[tuple[int, int]] = None
        self._hover_started_at = 0.0
        self._last_gaze: Optional[QPointF] = None
        self._dwell_ms = 950.0
        self._game_over = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setStyleSheet("background-color: rgb(8, 15, 26);")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)
        header = QHBoxLayout()
        self.status_label = QLabel("Black to move")
        self.status_label.setStyleSheet(
            "color: white; font-size: 16px; font-weight: 700; "
            "background: rgba(255,255,255,0.08); border-radius: 8px; padding: 10px 14px;"
        )
        self.reset_button = QPushButton("Restart")
        self.close_button = QPushButton("Close")
        for button in (self.reset_button, self.close_button):
            button.setFixedSize(92, 40)
            button.setStyleSheet(
                "QPushButton { color: white; background: rgba(255,255,255,0.12); "
                "border: 1px solid rgba(255,255,255,0.18); border-radius: 8px; }"
                "QPushButton:hover { background: rgba(255,255,255,0.20); }"
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
                self._place_black(pos)
                self.current_hover = None
                self._hover_started_at = now

        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self._board_rect()
        painter.setPen(QPen(QColor(255, 255, 255, 80), 2))
        painter.setBrush(QColor(236, 202, 142, 245))
        painter.drawRoundedRect(rect, 8, 8)

        cell = rect.width() / (self.board_size - 1)
        painter.setPen(QPen(QColor(70, 44, 25), 2))
        for i in range(self.board_size):
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
            cx, cy = self._cell_center(rect, row, col)
            progress = min((self._now_ms() - self._hover_started_at) / self._dwell_ms, 1.0)
            painter.setPen(QPen(QColor(15, 23, 42, 180), 3))
            painter.setBrush(QColor(15, 23, 42, 65))
            radius = cell * 0.36
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
            painter.setPen(QPen(QColor(245, 158, 11, 230), 5))
            painter.drawArc(
                int(cx - radius - 8),
                int(cy - radius - 8),
                int((radius + 8) * 2),
                int((radius + 8) * 2),
                90 * 16,
                -int(progress * 360 * 16),
            )

        if self._last_gaze is not None:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(96, 165, 250, 170))
            painter.drawEllipse(self._last_gaze, 9, 9)
            painter.setBrush(QColor(255, 255, 255, 240))
            painter.drawEllipse(self._last_gaze, 3, 3)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        if event.key() == Qt.Key.Key_R:
            self._reset_board()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self.closed.emit()
        super().closeEvent(event)

    def _board_rect(self) -> QRectF:
        w = self.width()
        h = self.height()
        top = 82.0
        size = min(w - 96.0, h - top - 52.0)
        left = (w - size) / 2.0
        return QRectF(left, top, size, size)

    def _board_pos_from_point(self, x: float, y: float) -> Optional[tuple[int, int]]:
        rect = self._board_rect()
        margin = 24.0
        if not rect.adjusted(-margin, -margin, margin, margin).contains(QPointF(x, y)):
            return None

        cell = rect.width() / (self.board_size - 1)
        col = int(round((x - rect.left()) / cell))
        row = int(round((y - rect.top()) / cell))
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return None
        if self.board[row][col] != 0:
            return None

        cx, cy = self._cell_center(rect, row, col)
        if math.dist((x, y), (cx, cy)) > cell * 0.48:
            return None
        return (row, col)

    def _cell_center(self, rect: QRectF, row: int, col: int) -> tuple[float, float]:
        cell = rect.width() / (self.board_size - 1)
        return (rect.left() + col * cell, rect.top() + row * cell)

    def _draw_piece(self, painter: QPainter, rect: QRectF, row: int, col: int, piece: int) -> None:
        cx, cy = self._cell_center(rect, row, col)
        radius = rect.width() / (self.board_size - 1) * 0.36
        if piece == 1:
            painter.setBrush(QColor(15, 23, 42))
            painter.setPen(QPen(QColor(255, 255, 255, 90), 1))
        else:
            painter.setBrush(QColor(248, 250, 252))
            painter.setPen(QPen(QColor(15, 23, 42, 120), 1))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

    def _place_black(self, pos: tuple[int, int]) -> None:
        row, col = pos
        if self.board[row][col] != 0:
            return
        self.board[row][col] = 1
        if self._winner_from(row, col, 1):
            self._game_over = True
            self.status_label.setText("Black wins")
            return
        if self._is_full():
            self._game_over = True
            self.status_label.setText("Draw")
            return
        self._place_white()

    def _place_white(self) -> None:
        move = self._best_white_move()
        if move is None:
            return
        row, col = move
        self.board[row][col] = 2
        if self._winner_from(row, col, 2):
            self._game_over = True
            self.status_label.setText("White wins")
        else:
            self.status_label.setText("Black to move")

    def _best_white_move(self) -> Optional[tuple[int, int]]:
        center = (self.board_size // 2, self.board_size // 2)
        empty = [
            (r, c)
            for r in range(self.board_size)
            for c in range(self.board_size)
            if self.board[r][c] == 0
        ]
        if not empty:
            return None
        for player in (2, 1):
            for r, c in empty:
                self.board[r][c] = player
                wins = self._winner_from(r, c, player)
                self.board[r][c] = 0
                if wins:
                    return (r, c)
        return min(empty, key=lambda p: math.dist(p, center))

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
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == piece:
            count += 1
            r += dr
            c += dc
        return count

    def _is_full(self) -> bool:
        return all(self.board[r][c] != 0 for r in range(self.board_size) for c in range(self.board_size))

    def _reset_board(self) -> None:
        self.board = [[0 for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_hover = None
        self._last_gaze = None
        self._game_over = False
        self.status_label.setText("Black to move")
        self.update()

    @staticmethod
    def _now_ms() -> float:
        from time import perf_counter

        return perf_counter() * 1000.0
