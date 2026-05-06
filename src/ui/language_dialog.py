from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QHBoxLayout, QVBoxLayout
from qfluentwidgets import PrimaryPushButton, PushButton

from src.ui.fluent_theme import PALETTE
from src.ui.i18n import save_language_config


class LanguageSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_language: str | None = None
        self.setWindowTitle("")
        self.setModal(True)
        self.setFixedSize(560, 180)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setStyleSheet(
            f"""
            QDialog {{
                background: {PALETTE['bg']};
                border: 1px solid rgba(69, 40, 41, 0.18);
                border-radius: 10px;
            }}
            PushButton, PrimaryPushButton {{
                font-size: 18px;
                font-weight: 800;
                border-radius: 8px;
            }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 28, 28, 28)
        outer.setSpacing(0)

        row = QHBoxLayout()
        row.setSpacing(18)
        outer.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        zh_btn = PrimaryPushButton("中文")
        en_btn = PushButton("English")
        bilingual_btn = PushButton("双语 / Bilingual")
        for button in (zh_btn, en_btn, bilingual_btn):
            button.setFixedSize(150, 72)
            row.addWidget(button)

        zh_btn.clicked.connect(lambda: self._select("zh"))
        en_btn.clicked.connect(lambda: self._select("en"))
        bilingual_btn.clicked.connect(lambda: self._select("bilingual"))

    def _select(self, language: str) -> None:
        self.selected_language = language
        save_language_config(language)
        self.accept()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - self.width()) // 2,
                geo.y() + (geo.height() - self.height()) // 2,
            )
