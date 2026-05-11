"""Generate an English 25-point calibration reference image.

This renders the same target grid geometry and visual style used by
CalibrationFullscreenWidget, but draws all 25 calibration targets at once.
It does not start the tracker or enter the real calibration flow.

Usage:
    conda run --no-capture-output -n gaze-env python scripts/generate_25pt_calibration_image.py
    conda run --no-capture-output -n gaze-env python scripts/generate_25pt_calibration_image.py --width 1920 --height 1080
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QApplication


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.calibration_points import generate_calibration_points_for_screen  # noqa: E402
from src.ui.i18n import set_language, tx  # noqa: E402


def primary_screen_size(app: QApplication) -> tuple[int, int]:
    screen = app.primaryScreen()
    if screen is None:
        return 1920, 1080
    geometry = screen.geometry()
    return geometry.width(), geometry.height()


def draw_calibration_image(
    width: int,
    height: int,
    *,
    output_path: Path,
    include_text: bool = True,
    annotate_indices: bool = False,
) -> None:
    set_language("en")

    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(QColor(30, 30, 30))

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    for target in generate_calibration_points_for_screen(width, height, num_points=25):
        painter.setPen(QPen(QColor(255, 0, 0), 3))
        painter.setBrush(QColor(255, 0, 0, 200))
        painter.drawEllipse(int(target.x - 15), int(target.y - 15), 30, 30)

        painter.setBrush(QColor(255, 255, 255))
        painter.drawEllipse(int(target.x - 5), int(target.y - 5), 10, 10)

        if annotate_indices:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            painter.drawText(
                QRect(int(target.x - 20), int(target.y + 18), 40, 18),
                Qt.AlignmentFlag.AlignCenter,
                str(target.index + 1),
            )

    if include_text:
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 18))
        progress_text = tx(
            "校准进度：1 / 25  有效点：0",
            "Progress: 1 / 25  Valid: 0",
        )
        painter.drawText(width // 2 - 100, 50, progress_text)

        painter.setPen(QColor(150, 150, 150))
        painter.setFont(QFont("Arial", 14))
        hint_text = tx(
            "请注视红色圆点，保持头部稳定 | ESC 取消",
            "Look at the red dot and keep your head stable | ESC Cancel",
        )
        painter.drawText(width // 2 - 200, height - 50, hint_text)

    painter.end()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(output_path)):
        raise RuntimeError(f"Failed to save image: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an English fullscreen-style 25-point calibration image."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "calibration_25pt_en_fullscreen.png",
        help="Output PNG path.",
    )
    parser.add_argument("--width", type=int, default=None, help="Output image width in pixels.")
    parser.add_argument("--height", type=int, default=None, help="Output image height in pixels.")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Draw only the 25 target dots without the fullscreen instruction text.",
    )
    parser.add_argument(
        "--annotate-indices",
        action="store_true",
        help="Add small point numbers below each target for debugging or documentation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    screen_w, screen_h = primary_screen_size(app)
    width = args.width or screen_w
    height = args.height or screen_h

    draw_calibration_image(
        width,
        height,
        output_path=args.output,
        include_text=not args.no_text,
        annotate_indices=args.annotate_indices,
    )
    print(f"Wrote {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
