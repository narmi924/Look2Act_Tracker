"""Local numeric experiment: selftest / collect / replay. Camera only in collect after Start."""
import argparse
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    synthetic = commands.add_parser('selftest')
    synthetic.add_argument('--output', default=None)
    playback = commands.add_parser('replay')
    playback.add_argument('session')
    playback.add_argument('--speed', type=float, default=0., help='0 = immediately; positive speed changes waits only')
    collect = commands.add_parser('collect')
    collect.add_argument('--config', default='configs/classic.yaml')
    collect.add_argument('--protocol', choices=['A', 'B', 'AB'], default='AB')
    collect.add_argument('--protocol-config', help='JSON overrides of small protocol parameters')
    collect.add_argument('--calibration', help='Explicit existing calibration; never auto-loaded')
    collect.add_argument('--calibration-backend', choices=['classic', 'deep', 'deep_pog'],
                         help='Required with --calibration: confirm its source backend')
    args = parser.parse_args()
    if args.command == 'collect':
        from PyQt6.QtWidgets import QApplication
        from src.tracker.pipeline import SystemConfig
        from src.ui.experiment_window import ExperimentWindow
        config = SystemConfig.from_yaml(args.config)
        if args.calibration and args.calibration_backend != config.normalized_backend:
            parser.error('--calibration requires matching --calibration-backend (legacy files do not identify it)')
        parameters = json.loads(Path(args.protocol_config).read_text(encoding='utf-8')) if args.protocol_config else None
        app = QApplication(sys.argv)
        window = ExperimentWindow(config, args.protocol, parameters, args.calibration)
        window.showFullScreen()
        return app.exec()
    if args.command == 'selftest':
        from src.experiment.synthetic import record_synthetic
        path = Path(args.output) if args.output else ROOT / 'experiment_sessions' / ('synthetic-' + uuid.uuid4().hex)
        summary = record_synthetic(path)
    else:
        from src.experiment.session import replay
        path = Path(args.session)
        summary, _ = replay(path, args.speed, time.sleep)
    print(f"Session: {path.name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('Numeric replay only; instructed_target is not independent gaze ground truth.')
    return 0 if summary['replay']['strictly_reproducible'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
