"""Print live Look2Act tracker diagnostics without opening the full GUI.

Usage:
    conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend classic
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import load_calibration
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finite live tracker diagnostics.")
    parser.add_argument("--config", default="configs/system_config.yaml")
    parser.add_argument("--backend", choices=["classic", "deep"], default=None)
    parser.add_argument("--deep-space", choices=["head", "camera"], default=None)
    parser.add_argument("--deep-pose-input", choices=["live", "zero"], default=None)
    parser.add_argument("--smoother", choices=["kalman", "ema", "none"], default=None)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--csv", dest="csv_path", default=None, help="Optional CSV output path.")
    parser.add_argument("--no-calibration", action="store_true")
    return parser.parse_args()


def _fmt_point(point: Optional[tuple[float, float]]) -> str:
    if point is None:
        return "None"
    return f"({point[0]:.3f},{point[1]:.3f})"


def _load_calibrator(config: SystemConfig) -> Optional[CalibrationModule]:
    path = ROOT / config.calibration_path
    if not path.exists():
        print(f"[diagnose] calibration not found: {path.name}")
        return None

    if config.normalized_backend == "classic":
        calibrator = CalibrationModule(num_points=25, max_residual_px=300.0, method="polynomial")
    else:
        calibrator = CalibrationModule(num_points=9, max_residual_px=300.0, method="affine")
    load_calibration(calibrator, str(path))
    print(
        "[diagnose] calibration loaded: "
        f"{path.name} method={calibrator.method.value} residual={calibrator.residual_mean:.2f}"
    )
    return calibrator


def _calibrated_point(
    result: TrackerResult,
    calibrator: Optional[CalibrationModule],
) -> Optional[tuple[float, float]]:
    if result.gaze_point is None:
        return None
    if calibrator is None or not calibrator.is_calibrated:
        return result.gaze_point
    return calibrator.apply(result.gaze_point)


def main() -> int:
    args = parse_args()
    config_path = ROOT / args.config
    config = SystemConfig.from_yaml(str(config_path)) if config_path.exists() else SystemConfig()
    if args.backend is not None:
        config.tracker_backend = args.backend
    if args.deep_space is not None:
        config.deep_gaze_space = args.deep_space
    if args.deep_pose_input is not None:
        config.deep_pose_input = args.deep_pose_input
    if args.smoother is not None:
        config.smoother_type = args.smoother

    calibrator = None if args.no_calibration else _load_calibrator(config)

    errors: list[str] = []

    def on_error(message: str) -> None:
        errors.append(message)
        print(f"[diagnose:error] {message}")

    pipeline = TrackerPipeline(
        model_path=config.checkpoint_path,
        config=config,
        error_callback=on_error,
    )

    print(
        "[diagnose] starting "
        f"backend={config.normalized_backend} frames={args.frames} "
        f"deep_space={config.deep_gaze_space} pose_input={config.normalized_deep_pose_input} "
        f"smoother={config.normalized_smoother_type} "
        f"calibration={config.calibration_path}"
    )
    if not pipeline.start():
        print("[diagnose] failed to start tracker")
        return 2

    printed = 0
    last_print = 0.0
    csv_file = None
    csv_writer = None
    if args.csv_path:
        csv_path = Path(args.csv_path)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "index",
                "time_s",
                "backend",
                "valid",
                "face_detected",
                "fps",
                "raw_x",
                "raw_y",
                "calibrated_x",
                "calibrated_y",
                "error",
                "timings",
            ],
        )
        csv_writer.writeheader()
        print(f"[diagnose] csv output: {csv_path}")
    try:
        while printed < args.frames and pipeline.is_running():
            now = time.perf_counter()
            if now - last_print < args.interval:
                time.sleep(0.01)
                continue
            last_print = now

            result = pipeline.get_latest_result()
            if result is None:
                print("[diagnose] waiting for first frame...")
                printed += 1
                continue

            raw = result.raw_point or result.gaze_point
            calibrated = _calibrated_point(result, calibrator)
            timing = " ".join(f"{k}={v:.1f}ms" for k, v in result.timings.items())
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "index": printed,
                        "time_s": f"{time.time():.3f}",
                        "backend": result.backend,
                        "valid": result.valid,
                        "face_detected": result.face_detected,
                        "fps": f"{result.fps:.3f}",
                        "raw_x": "" if raw is None else f"{raw[0]:.6f}",
                        "raw_y": "" if raw is None else f"{raw[1]:.6f}",
                        "calibrated_x": "" if calibrated is None else f"{calibrated[0]:.6f}",
                        "calibrated_y": "" if calibrated is None else f"{calibrated[1]:.6f}",
                        "error": result.error_message or "",
                        "timings": timing,
                    }
                )
            print(
                f"[diagnose] #{printed:04d} "
                f"backend={result.backend} valid={result.valid} face={result.face_detected} "
                f"fps={result.fps:.1f} raw={_fmt_point(raw)} "
                f"calibrated={_fmt_point(calibrated)} "
                f"err={result.error_message or '-'} {timing}"
            )
            printed += 1
    except KeyboardInterrupt:
        print("[diagnose] interrupted")
    finally:
        pipeline.stop()
        if csv_file is not None:
            csv_file.close()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
