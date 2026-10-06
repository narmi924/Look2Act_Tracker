"""R8: record fixations, pursuit paths, pursuit-choice trials and short head motions with eye crops.

collect  opens the fullscreen window; the camera starts only after you click 开始.
check    integrity plus F2-only feasibility numbers for one recorded session (offline, no camera).
selftest synthetic session in a temporary directory (no camera, no real data).
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect')
    collect.add_argument('--config', default='configs/classic.yaml')
    collect.add_argument('--protocol-config', help='JSON overrides of R8 protocol parameters')
    check = commands.add_parser('check')
    check.add_argument('session')
    commands.add_parser('selftest')
    args = parser.parse_args()
    if args.command == 'collect':
        # Windows: load MediaPipe's native module before Qt, without opening a camera (see scripts/experiment.py).
        import src.vision.face_detector  # noqa: F401

        from PyQt6.QtWidgets import QApplication
        from src.tracker.pipeline import SystemConfig
        from src.ui.r8_record_window import R8RecordWindow
        config = SystemConfig.from_yaml(args.config)
        parameters = json.loads(Path(args.protocol_config).read_text(encoding='utf-8')) if args.protocol_config else None
        app = QApplication(sys.argv)
        window = R8RecordWindow(config, parameters)
        window.showFullScreen()
        return app.exec()
    from src.experiment.r8_pursuit import brief, check as run_check, write_synthetic_r8_session
    if args.command == 'check':
        report = run_check(Path(args.session))
        print(brief(report))
        print('written r8_check.json next to the session; stimuli are instructions, not gaze truth')
        integrity = report['integrity']
        # Exit 0 only when the record is complete AND its images/landmarks are all present and consistent.
        return 0 if integrity['complete'] and not report['issues'] and integrity['artifacts_ok'] else 1
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / 'r8-synthetic'
        write_synthetic_r8_session(source)
        report = run_check(source)
        print(brief(report))
        accuracy = report['choice']['lag_150ms']['accuracy']
        assert accuracy is not None and accuracy >= .9, accuracy
        assert report['integrity']['landmark_index_consistent'] and report['integrity']['missing_png'] == 0
        assert report['integrity']['artifacts_ok']
        print('R8 synthetic selftest: passed')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
