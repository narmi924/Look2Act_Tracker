from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QDialog, QFrame, QHBoxLayout, QPushButton, QVBoxLayout
from qfluentwidgets import BodyLabel, CaptionLabel, PrimaryPushButton, PushButton, TitleLabel

from src.ui.fluent_theme import PALETTE
from src.ui.i18n import read_language_config, set_language


MODE_TEMPLATE_CONFIGS = {
    "classic": Path("configs/classic.yaml"),
    "deep": Path("configs/deep.yaml"),
}


def user_config_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Look2Act"
    return Path.home() / "AppData" / "Roaming" / "Look2Act"


def application_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def template_config_path_for_mode(mode: str) -> Path:
    normalized = mode if mode in MODE_TEMPLATE_CONFIGS else "classic"
    template_path = MODE_TEMPLATE_CONFIGS[normalized]
    if template_path.is_absolute():
        return template_path
    return application_base_dir() / template_path


def user_config_path_for_mode(mode: str) -> Path:
    normalized = mode if mode in MODE_TEMPLATE_CONFIGS else "classic"
    return user_config_root() / "configs" / f"{normalized}.yaml"


def ensure_user_config(mode: str) -> Path:
    normalized = mode if mode in MODE_TEMPLATE_CONFIGS else "classic"
    target_path = user_config_path_for_mode(normalized)
    if not target_path.exists():
        template_path = template_config_path_for_mode(normalized)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(template_path, target_path)
    return target_path


class LanguageSelectionDialog(QDialog):
    def __init__(self, parent=None, config_path: Path | str | None = None):
        super().__init__(parent)
        self.selected_mode: str = self._mode_from_config_path(config_path)
        self.selected_config_path: Path = user_config_path_for_mode(self.selected_mode)
        self.config_path = Path(config_path) if config_path is not None else None
        user_config_path = user_config_path_for_mode(self.selected_mode)
        template_config_path = template_config_path_for_mode(self.selected_mode)
        self.selected_language: str | None = (
            read_language_config(user_config_path)
            or read_language_config(template_config_path)
            or (read_language_config(self.config_path) if self.config_path is not None else None)
            or "bilingual"
        )
        self.setWindowTitle("Look2Act Tracker")
        self.setModal(True)
        self.setFixedSize(720, 420)
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
                font-size: 17px;
                font-weight: 800;
                border-radius: 8px;
            }}
            QFrame#Divider {{
                background: rgba(69, 40, 41, 0.20);
                min-height: 1px;
                max-height: 1px;
            }}
            QPushButton#CloseButton {{
                background: transparent;
                color: {PALETTE['accent']};
                border: none;
                font-size: 20px;
                font-weight: 800;
                padding: 0;
            }}
            QPushButton#CloseButton:hover {{
                background: rgba(69, 40, 41, 0.10);
            }}
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(28, 14, 28, 24)
        outer.setSpacing(9)

        top_row = QHBoxLayout()
        top_row.addStretch(1)
        close_btn = QPushButton("X")
        close_btn.setObjectName("CloseButton")
        close_btn.setFixedSize(32, 30)
        close_btn.clicked.connect(self.reject)
        top_row.addWidget(close_btn)
        outer.addLayout(top_row)

        title = TitleLabel("Look2Act Tracker")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        subtitle = BodyLabel("Gaze-driven interaction with a standard webcam")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setProperty("secondary", True)
        outer.addWidget(subtitle)

        credit = CaptionLabel("Author: Imranjan Mamtimin  |  Website: https://imranjan.cn")
        credit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        credit.setProperty("secondary", True)
        outer.addWidget(credit)

        divider = QFrame()
        divider.setObjectName("Divider")
        outer.addWidget(divider)

        row = QHBoxLayout()
        row.setSpacing(14)
        language_label = BodyLabel("Language / 语言")
        language_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(language_label)
        outer.addLayout(row)

        zh_btn = PrimaryPushButton("中文")
        en_btn = PushButton("English")
        bilingual_btn = PushButton("双语 / Bilingual")
        for button in (zh_btn, en_btn, bilingual_btn):
            button.setCheckable(True)
            button.setFixedSize(200, 50)
            row.addWidget(button)

        mode_label = BodyLabel("Mode / 模式")
        mode_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(mode_label)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)
        outer.addLayout(mode_row)

        classic_btn = PushButton("Classic")
        deep_btn = PushButton("Deep")
        for button in (classic_btn, deep_btn):
            button.setCheckable(True)
            button.setFixedSize(300, 50)
            mode_row.addWidget(button)
        self._mode_buttons = {
            "classic": classic_btn,
            "deep": deep_btn,
        }
        self._set_mode(self.selected_mode)

        start_btn = PrimaryPushButton("Start / 启动")
        start_btn.setFixedSize(200, 50)
        start_row = QHBoxLayout()
        start_row.addStretch(1)
        start_row.addWidget(start_btn)
        start_row.addStretch(1)
        outer.addLayout(start_row)

        self._language_buttons = {
            "zh": zh_btn,
            "en": en_btn,
            "bilingual": bilingual_btn,
        }
        self._select_language(self.selected_language)
        zh_btn.clicked.connect(lambda: self._select_language("zh"))
        en_btn.clicked.connect(lambda: self._select_language("en"))
        bilingual_btn.clicked.connect(lambda: self._select_language("bilingual"))
        classic_btn.clicked.connect(lambda: self._set_mode("classic"))
        deep_btn.clicked.connect(lambda: self._set_mode("deep"))
        start_btn.clicked.connect(self._accept_selection)

    def _select_language(self, language: str) -> None:
        self.selected_language = language
        for key, button in self._language_buttons.items():
            button.setChecked(key == language)
            self._apply_button_style(button, selected=key == language)

    def _set_mode(self, mode: str) -> None:
        self.selected_mode = mode if mode in MODE_TEMPLATE_CONFIGS else "classic"
        self.selected_config_path = user_config_path_for_mode(self.selected_mode)
        for key, button in self._mode_buttons.items():
            button.setChecked(key == self.selected_mode)
            self._apply_button_style(button, selected=key == self.selected_mode)

    def _accept_selection(self) -> None:
        language = self.selected_language or "bilingual"
        self.selected_config_path = self._save_startup_config(language, self.selected_mode)
        self.accept()

    @staticmethod
    def _apply_button_style(button: PushButton, selected: bool) -> None:
        if selected:
            button.setStyleSheet(
                f"""
                background: {PALETTE['accent']};
                color: {PALETTE['bg']};
                border: 1px solid {PALETTE['accent']};
                border-radius: 8px;
                font-size: 17px;
                font-weight: 800;
                """
            )
        else:
            button.setStyleSheet(
                f"""
                background: rgba(255, 255, 255, 0.62);
                color: #000000;
                border: 1px solid rgba(69, 40, 41, 0.24);
                border-radius: 8px;
                font-size: 17px;
                font-weight: 800;
                """
            )

    @staticmethod
    def _save_startup_config(language: str, mode: str) -> Path:
        config_path = ensure_user_config(mode)
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        ui = data.setdefault("ui", {})
        ui["language"] = language
        ui["window_mode"] = "fullscreen"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        set_language(language)
        return config_path

    @staticmethod
    def _mode_from_config_path(config_path: Path | str | None) -> str:
        if config_path is None:
            return "classic"
        path = Path(config_path)
        path_text = path.as_posix().lower()
        for mode, mode_path in MODE_TEMPLATE_CONFIGS.items():
            resolved_mode_path = template_config_path_for_mode(mode)
            user_path = user_config_path_for_mode(mode)
            if path_text in {
                mode_path.as_posix().lower(),
                resolved_mode_path.as_posix().lower(),
                user_path.as_posix().lower(),
            }:
                return mode
        return "classic"

    def showEvent(self, event) -> None:
        super().showEvent(event)
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(
                geo.x() + (geo.width() - self.width()) // 2,
                geo.y() + (geo.height() - self.height()) // 2,
            )
