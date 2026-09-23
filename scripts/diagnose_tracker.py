"""Print live Look2Act tracker diagnostics without opening the full GUI.

Usage:
    conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend classic
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import load_calibration
from src.tracker.pipeline import SystemConfig, TrackerPipeline
from src.experiment.consumer import Consumer, PollSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finite live tracker diagnostics.")
    parser.add_argument("--config", default="configs/system_config.yaml")
    parser.add_argument("--backend", choices=["classic", "deep_pog", "deep"], default=None)
    parser.add_argument("--deep-space", choices=["head", "camera"], default=None)
    parser.add_argument("--deep-pose-input", choices=["live", "zero"], default=None)
    parser.add_argument("--deep-ray-origin", choices=["face_translation", "zero_origin"], default=None)
    parser.add_argument("--smoother", choices=["kalman", "ema", "none"], default=None)
    parser.add_argument("--frames", type=int, default=300, help="Number of 33 ms consumption polls (including empty/duplicate), not printed rows")
    parser.add_argument("--interval", type=float, default=0.2, help="Print interval only; CSV records every 33 ms consumption poll")
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


def diagnostic_row(event, index):
    identity = event['observation_id'] or [None, None]
    row = dict(index=index, session=identity[0], sequence=identity[1],
               backend=event['backend'], source_valid=event['source_valid'],
               face_detected=event.get('source_face_detected'), fps=event.get('source_fps'),
               source_time=event['source_time'], source_time_source=event['source_time_source'],
               continuity=event['source_continuity'], published_at=event['published_at'],
               consume_time=event['read_at'], processed_at=event['processed_at'], age_s=event['age_s'],
               gate_state=event['gate_state'], gate_reason=event['gate_reason'], reset=event['reset'],
               raw_units=event['raw_units'], screen_rejection=event['screen_rejection'],
               calibrated_in_bounds=event['calibrated_in_bounds'], smoothed_in_bounds=event['smoothed_in_bounds'],
               dispatch_rejection=event['dispatch_rejection'], dispatch_allowed=event['dispatch_allowed'])
    for stage in ('raw', 'calibrated', 'smoothed', 'display'):
        point = event[stage + '_point']
        row[stage + '_x'], row[stage + '_y'] = point if point is not None else (None, None)
    for name in ('source_timings', 'processing_timings', 'processing_status', 'dispatch_state'):
        row[name] = json.dumps(event[name])
    return row


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
    if args.deep_ray_origin is not None:
        config.deep_ray_origin = args.deep_ray_origin
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
        f"ray_origin={config.normalized_deep_ray_origin} "
        f"smoother={config.normalized_smoother_type} "
        f"calibration={config.calibration_path}"
    )
    if not pipeline.start():
        print("[diagnose] failed to start tracker")
        return 2

    csv_file = None
    csv_writer = None
    geometry = pipeline.screen_geometry
    consumer = Consumer(config, calibrator, (geometry.screen_w_px, geometry.screen_h_px))
    consumer.reset(pipeline.session_id, 'diagnostic_start')
    schedule = PollSchedule(print_interval=args.interval)
    count = 0
    try:
        if args.csv_path:
            csv_file = Path(args.csv_path).open('w', newline='', encoding='utf-8')
        while count < args.frames and pipeline.is_running():
            due, printing = schedule.due(time.perf_counter())
            if not due:
                time.sleep(.005)
                continue
            event = consumer.consume(pipeline.get_latest_result(), pipeline.get_dispatch_snapshot)
            row = diagnostic_row(event, count)
            if csv_file:
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
            if printing:
                print(f"[diagnose] #{count} gate={event['gate_state']} reason={event['gate_reason']} "
                      f"raw[{event['raw_units']}]={event['raw_point']} calibrated={event['calibrated_point']} "
                      f"smoothed={event['smoothed_point']} display={event['display_point']} "
                      f"screen_rejection={event['screen_rejection']} age_s={event['age_s']}")
            count += 1
    except KeyboardInterrupt:
        print("[diagnose] interrupted")
    finally:
        pipeline.stop()
        if csv_file is not None:
            csv_file.close()

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
