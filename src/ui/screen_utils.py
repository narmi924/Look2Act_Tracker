from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QRect
from PyQt6.QtWidgets import QApplication


DEFAULT_SCREEN_W_MM = 344.0
DEFAULT_SCREEN_H_MM = 194.0


@dataclass(frozen=True)
class RuntimeScreenInfo:
    geometry: QRect
    available_geometry: QRect
    screen_w_px: int
    screen_h_px: int
    available_w_px: int
    available_h_px: int
    screen_w_mm: float
    screen_h_mm: float
    physical_source: str


def _valid_physical_size(width_mm: float, height_mm: float) -> bool:
    return 100.0 <= width_mm <= 2000.0 and 80.0 <= height_mm <= 1400.0


def _qt_physical_size_mm() -> Optional[tuple[float, float]]:
    screen = QApplication.primaryScreen()
    if screen is None:
        return None
    size = screen.physicalSize()
    width_mm = float(size.width())
    height_mm = float(size.height())
    if _valid_physical_size(width_mm, height_mm):
        return width_mm, height_mm
    return None


def _win32_physical_size_mm() -> Optional[tuple[float, float]]:
    if sys.platform != "win32":
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hdc = user32.GetDC(None)
        if not hdc:
            return None
        try:
            HORZSIZE = 4
            VERTSIZE = 6
            width_mm = float(gdi32.GetDeviceCaps(hdc, HORZSIZE))
            height_mm = float(gdi32.GetDeviceCaps(hdc, VERTSIZE))
        finally:
            user32.ReleaseDC(None, hdc)
        if _valid_physical_size(width_mm, height_mm):
            return width_mm, height_mm
    except Exception:
        return None
    return None


def current_screen_info(
    fallback_w_mm: float = DEFAULT_SCREEN_W_MM,
    fallback_h_mm: float = DEFAULT_SCREEN_H_MM,
) -> RuntimeScreenInfo:
    screen = QApplication.primaryScreen()
    if screen is None:
        geometry = QRect(0, 0, 1920, 1080)
        available_geometry = QRect(0, 0, 1920, 1040)
    else:
        geometry = screen.geometry()
        available_geometry = screen.availableGeometry()

    physical = _qt_physical_size_mm()
    source = "qt"
    if physical is None:
        physical = _win32_physical_size_mm()
        source = "win32" if physical is not None else "fallback"

    if physical is None:
        width_mm, height_mm = float(fallback_w_mm), float(fallback_h_mm)
    else:
        width_mm, height_mm = physical

    return RuntimeScreenInfo(
        geometry=geometry,
        available_geometry=available_geometry,
        screen_w_px=int(geometry.width()),
        screen_h_px=int(geometry.height()),
        available_w_px=int(available_geometry.width()),
        available_h_px=int(available_geometry.height()),
        screen_w_mm=round(float(width_mm), 1),
        screen_h_mm=round(float(height_mm), 1),
        physical_source=source,
    )
