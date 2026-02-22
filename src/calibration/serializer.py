"""校准参数序列化模块。

将校准参数（仿射矩阵、校准点、残差等）保存为 JSON 文件，
支持反序列化恢复校准状态。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from src.calibration.calibrator import CalibrationModule


def save_calibration(
    calibrator: CalibrationModule,
    filepath: str,
    screen_w: int = 0,
    screen_h: int = 0,
) -> None:
    """将校准参数序列化保存到 JSON 文件。

    参数:
        calibrator: 已校准的 CalibrationModule 实例
        filepath: 输出 JSON 文件路径
        screen_w: 屏幕宽度（像素），用于记录
        screen_h: 屏幕高度（像素），用于记录
    """
    data = calibrator.get_calibration_data()
    data["version"] = "1.0"
    data["timestamp"] = datetime.now().isoformat()
    data["screen_w"] = screen_w
    data["screen_h"] = screen_h

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_calibration(
    calibrator: CalibrationModule,
    filepath: str,
) -> None:
    """从 JSON 文件反序列化并恢复校准状态。

    参数:
        calibrator: CalibrationModule 实例，将被恢复校准状态
        filepath: JSON 文件路径
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    calibrator.set_calibration_data(data)
