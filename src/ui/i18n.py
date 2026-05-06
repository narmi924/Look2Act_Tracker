from __future__ import annotations

from pathlib import Path

import yaml


SUPPORTED_LANGUAGES = {"zh", "en", "bilingual"}
DEFAULT_LANGUAGE = "bilingual"
CONFIG_PATH = Path("configs/system_config.yaml")

_current_language = DEFAULT_LANGUAGE


def normalize_language(language: str | None) -> str:
    if language in SUPPORTED_LANGUAGES:
        return str(language)
    return DEFAULT_LANGUAGE


def set_language(language: str) -> None:
    global _current_language
    _current_language = normalize_language(language)


def get_language() -> str:
    return _current_language


def language_label(language: str) -> str:
    return {
        "zh": "中文",
        "en": "English",
        "bilingual": "双语 / Bilingual",
    }.get(normalize_language(language), "双语 / Bilingual")


def read_language_config(config_path: Path = CONFIG_PATH) -> str | None:
    if not config_path.exists():
        return None
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        language = data.get("ui", {}).get("language")
        return normalize_language(language) if language in SUPPORTED_LANGUAGES else None
    except Exception:
        return None


def load_language(config_path: Path = CONFIG_PATH) -> str | None:
    language = read_language_config(config_path)
    if language is not None:
        set_language(language)
    return language


def save_language_config(language: str, config_path: Path = CONFIG_PATH) -> None:
    language = normalize_language(language)
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}
    ui = data.setdefault("ui", {})
    ui["language"] = language
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    set_language(language)


def tx(zh: str, en: str, bilingual: str | None = None) -> str:
    language = get_language()
    if language == "zh":
        return zh
    if language == "en":
        return en
    return bilingual if bilingual is not None else f"{zh} / {en}"


def tx_button(zh: str, en: str) -> str:
    return tx(zh, en, f"{zh}\n{en}")
