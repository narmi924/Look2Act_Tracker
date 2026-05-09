from pathlib import Path

import yaml

from src.ui.i18n import (
    get_language,
    load_language,
    save_language_config,
    set_language,
    tx,
    tx_button,
)
from src.ui import language_dialog


def test_tx_switches_language_modes():
    original = get_language()
    try:
        set_language("zh")
        assert tx("中文", "English") == "中文"
        assert tx_button("启动", "Start") == "启动"

        set_language("en")
        assert tx("中文", "English") == "English"
        assert tx_button("启动", "Start") == "Start"

        set_language("bilingual")
        assert tx("中文", "English") == "中文 / English"
        assert tx_button("启动", "Start") == "启动\nStart"
    finally:
        set_language(original)


def test_language_config_roundtrip(tmp_path: Path):
    original = get_language()
    config_path = tmp_path / "system_config.yaml"
    config_path.write_text(
        yaml.dump({"ui": {"window_mode": "adaptive"}, "camera": {"index": 0}}, allow_unicode=True),
        encoding="utf-8",
    )

    try:
        save_language_config("en", config_path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert data["ui"]["language"] == "en"
        assert data["ui"]["window_mode"] == "adaptive"
        assert data["camera"]["index"] == 0

        set_language("zh")
        assert load_language(config_path) == "en"
        assert get_language() == "en"
    finally:
        set_language(original)


def test_language_dialog_source_has_only_three_language_choices():
    source = Path("src/ui/language_dialog.py").read_text(encoding="utf-8")

    assert 'PrimaryPushButton("中文")' in source
    assert 'PushButton("English")' in source
    assert 'PushButton("双语 / Bilingual")' in source
    assert "FramelessWindowHint" in source


def test_startup_config_preserves_saved_window_mode(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "classic.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "ui": {"language": "zh", "window_mode": "adaptive"},
                "camera": {"index": 0},
                "model": {},
                "tracker": {"backend": "classic"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(language_dialog, "ensure_user_config", lambda mode: config_path)

    language_dialog.LanguageSelectionDialog._save_startup_config("en", "classic")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["ui"]["language"] == "en"
    assert data["ui"]["window_mode"] == "adaptive"


def test_startup_config_defaults_missing_window_mode_to_fullscreen(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "classic.yaml"
    config_path.write_text(
        yaml.dump(
            {
                "ui": {"language": "zh"},
                "camera": {"index": 0},
                "model": {},
                "tracker": {"backend": "classic"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(language_dialog, "ensure_user_config", lambda mode: config_path)

    language_dialog.LanguageSelectionDialog._save_startup_config("en", "classic")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["ui"]["language"] == "en"
    assert data["ui"]["window_mode"] == "fullscreen"
