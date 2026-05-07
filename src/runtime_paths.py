from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "Look2Act"


def app_base_dir() -> Path:
    """返回源码根目录或打包后的资源根目录。"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def app_data_dir() -> Path:
    """返回当前用户可写的应用数据目录。"""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_NAME
    return Path.home() / "AppData" / "Roaming" / APP_NAME


def resource_path(path: str | Path) -> Path:
    """解析源码或打包资源路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return app_base_dir() / candidate


def user_data_path(path: str | Path) -> Path:
    """解析 AppData 下的当前用户可写路径。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return app_data_dir() / candidate


def user_config_path(mode: str) -> Path:
    normalized = mode if mode in {"classic", "deep"} else "classic"
    return user_data_path(Path("configs") / f"{normalized}.yaml")


def calibration_path_for_backend(backend: str) -> Path:
    if backend == "deep":
        filename = "calibration_deep.json"
    elif backend == "deep_pog":
        filename = "calibration_deep_pog.json"
    else:
        filename = "calibration_classic.json"
    return user_data_path(Path("calibration") / filename)
