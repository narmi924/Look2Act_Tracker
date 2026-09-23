"""R4 offline only: legacy read-only audit and fixed numerical feature comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment.r4_analysis import guarded_output, run_session, select_complete_ab  # noqa: E402
from src.experiment.r4_audit import audit_legacy, audit_r3_sessions  # noqa: E402
from src.experiment.r4_features import local_eye_point  # noqa: E402
from src.experiment.recording import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='read-only audit of old data; decode at most 32 pairs per group')
    audit.add_argument('--raw', type=Path, default=ROOT / 'dataset_raw')
    audit.add_argument('--processed', type=Path, default=ROOT / 'dataset_processed')
    audit.add_argument('--image-limit', type=int, default=32)
    audit.add_argument('--output', type=Path)
    run = sub.add_parser('run', help='fixed A0 fit, A1/B evaluation from one real Classic AB')
    run.add_argument('--session', type=Path, required=True)
    run.add_argument('--output', type=Path)
    sub.add_parser('select', help='print the sole eligible real complete AB source directory')
    sub.add_parser('selftest', help='pure synthetic local-coordinate check; no camera or files')
    args = parser.parse_args()
    if args.command == 'selftest':
        value, reason = local_eye_point((0, 0), (10, 0), (7, 2))
        assert reason is None and value == [0.2, 0.2]
        print('R4 synthetic numerical selftest: passed')
        return
    if args.command == 'select':
        print(select_complete_ab(ROOT / 'experiment_sessions'))
        return
    output = args.output or ROOT / 'experiment_sessions' / 'r4_runs' / (args.command + '-' + uuid4().hex[:12])
    if not output.resolve().is_relative_to((ROOT / 'experiment_sessions' / 'r4_runs').resolve()):
        parser.error('output must be under ignored experiment_sessions/r4_runs/')
    protected = [ROOT / 'dataset_raw', ROOT / 'dataset_processed']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.command == 'audit':
        output = guarded_output(output, args.raw, [args.processed, *protected])
        output.mkdir(parents=True, exist_ok=False)
        result = audit_legacy(args.raw, args.processed, image_limit=args.image_limit)
        result['r3_sessions'] = audit_r3_sessions(ROOT / 'experiment_sessions')
        write_json(output / 'manifest.json', dict(analysis_run=True, kind='legacy_read_only_audit',
                                                  code_commit=commit, image_limit=args.image_limit))
        write_json(output / 'audit.json', result)
        print(json.dumps({'output': str(output), 'raw_rows': result['raw']['rows'],
                          'processed_rows': result['processed']['rows'],
                          'matched': result['processed']['matched_raw']}, ensure_ascii=False))
    else:
        result = run_session(args.session, output, protected, commit)
        print(json.dumps({'output': str(output), 'producers': result['quality']['producers'],
                          'splits': result['quality']['splits'],
                          'elapsed_s': result['elapsed_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
