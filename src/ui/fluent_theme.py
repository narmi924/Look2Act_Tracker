"""
UI 主题和配色模块。

本模块负责统一整个应用的视觉风格，包括颜色、字体、控件样式等。
使用 PyQt6-Fluent-Widgets 的主题系统作为基础，并应用自定义配色方案。

配色方案：
- 背景色：浅米色（#F3E8DF）
- 卡片色：浅褐色（#E8D1C5，用于输入框）
- 文字色：深灰色（#57595B，用于次要文字）
- 强调色：深褐色（#452829，用于按钮和重要元素）
- 退出色：红色（#B00020，用于退出按钮）
"""
from __future__ import annotations

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication


PALETTE = {
    "bg": "#F3E8DF",  # 背景色：浅米色
    "card": "#E8D1C5",  # 卡片色：浅褐色（用于输入框背景）
    "text": "#57595B",  # 文字色：深灰色（用于次要文字）
    "accent": "#452829",  # 强调色：深褐色（用于按钮）
    "red_exit": "#B00020",  # 退出按钮红色
}


def apply_fluent_theme(app: QApplication) -> None:
    from qfluentwidgets import Theme, setTheme, setThemeColor

    setTheme(Theme.LIGHT)
    setThemeColor(QColor(PALETTE["accent"]))

    app.setStyleSheet(
        f"""
        QWidget {{
            background: {PALETTE['bg']};
            color: #000000;
            font-family: "Microsoft YaHei", "SimHei", Arial;
        }}

        QLabel {{
            background: transparent;
            color: #000000;
        }}

        QLabel[secondary="true"] {{
            color: {PALETTE['text']};
        }}

        CardWidget {{
            background: rgba(255, 255, 255, 0.0);
            border: 1px solid rgba(69, 40, 41, 0.22);
            border-radius: 18px;
        }}

        PrimaryPushButton, PushButton, QPushButton {{
            background: {PALETTE['accent']};
            color: {PALETTE['bg']};
            border: none;
            border-radius: 12px;
            padding: 10px 18px;
        }}
        PrimaryPushButton:hover, PushButton:hover, QPushButton:hover {{
            background: #3A2223;
        }}
        PrimaryPushButton:pressed, PushButton:pressed, QPushButton:pressed {{
            background: #2D191A;
        }}

        LineEdit, QLineEdit, ComboBox, QComboBox {{
            background: {PALETTE['card']};
            color: #000000;
            border: 1px solid rgba(69, 40, 41, 0.55);
            border-radius: 12px;
            padding: 10px 12px;
        }}
        """
    )
