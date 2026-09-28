"""R6 F2 shadow: collect, explicit loaded-map validation, synthetic check and replay."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collect = commands.add_parser('collect', help='New nine-point personal calibration; camera opens after Start')
    collect.add_argument('--config', default='configs/classic.yaml')
    loaded = commands.add_parser('validate-loaded', help='Explicit mapping load; independent targets then free look')
    loaded.add_argument('--config', default='configs/classic.yaml')
    loaded.add_argument('--mapping', required=True)
    synthetic = commands.add_parser('selftest', help='No camera, Qt or system action')
    synthetic.add_argument('--output')
    replay = commands.add_parser('replay', help='Recompute the F2 candidate chain, not just R3 fields')
    replay.add_argument('session')
    historical = commands.add_parser('historical', help='Read-only causal first-round fit on an R5 AB session')
    historical.add_argument('--session', help='One explicit R3/R5 source directory')
    historical.add_argument('--eligible-r5', action='store_true', help='Both fixed R5-eligible sessions')
    args = parser.parse_args()

    if args.command in ('collect', 'validate-loaded'):
        # Preserve R3's Windows native-module-before-Qt startup order.
        import src.vision.face_detector

        from PyQt6.QtWidgets import QApplication
        from src.tracker.pipeline import SystemConfig
        from src.ui.r6_shadow_window import ShadowWindow

        config = SystemConfig.from_yaml(args.config)
        if config.normalized_backend != 'classic':
            parser.error('R6 F2 requires the Classic perception backend')
        app = QApplication(sys.argv)
        window = ShadowWindow(config, load_mapping_path=getattr(args, 'mapping', None))
        window.showFullScreen()
        return app.exec()

    from src.experiment.r6_shadow import historical_consistency, replay_shadow, synthetic_selftest
    if args.command == 'selftest':
        path = Path(args.output) if args.output else ROOT / 'experiment_sessions' / 'r6_runs' / (
            'synthetic-' + uuid.uuid4().hex)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = synthetic_selftest(path)
        okay = result['replay']['strictly_reproducible']
    elif args.command == 'replay':
        result = replay_shadow(args.session)
        okay = result['strictly_reproducible']
    else:
        if bool(args.session) == bool(args.eligible_r5):
            parser.error('choose exactly one of --session and --eligible-r5')
        if args.session:
            result = [historical_consistency(args.session)]
        else:
            from src.experiment.r5_replication import audit_candidates

            root = ROOT / 'experiment_sessions'
            records = audit_candidates(root, root / 'r4_runs')
            chosen = [record for record in records if record['eligible']]
            if len(chosen) != 2:
                parser.error(f'expected exactly two R5-eligible sources; found {len(chosen)}')
            result = [historical_consistency(record['source_local']) for record in chosen]
        okay = all(entry['strictly_consistent'] for entry in result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if okay else 1


if __name__ == '__main__':
    raise SystemExit(main())
