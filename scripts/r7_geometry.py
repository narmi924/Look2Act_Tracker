"""R7 read-only geometry check on saved Classic AB sessions: F2 vs head-normalized vs geometric."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment.r5_replication import audit_candidates, write_inventory  # noqa: E402
from src.experiment.r7_geometry import (MAIN_PROTOCOL, MODELS, ROTATION_SOURCES, run_session,  # noqa: E402
                                        summarize_sessions, write_synthetic_session)
from src.experiment.recording import write_json  # noqa: E402

SESSIONS = ROOT / 'experiment_sessions'
R7_RUNS = SESSIONS / 'r7_runs'
PROTECTED = [ROOT / 'dataset_raw', ROOT / 'dataset_processed', SESSIONS / 'r4_runs', SESSIONS / 'r5_runs',
             SESSIONS / 'r6_runs']


def _output(path, command):
    output = (path or R7_RUNS / (command + '-' + uuid4().hex[:12])).resolve()
    if not output.is_relative_to(R7_RUNS.resolve()) or output == R7_RUNS.resolve():
        raise ValueError('output must be a new directory under ignored experiment_sessions/r7_runs/')
    if output.exists():
        raise FileExistsError(output)
    return output


def _table(aggregate, rotation):
    main = aggregate['rotation'][rotation]['protocols'][MAIN_PROTOCOL]
    lines = [f'rotation={rotation:5s} ' + ''.join(f'{name:>12}' for name in MODELS)]
    for split, entry in main['splits'].items():
        cells = [entry['models'][name]['metrics']['error_px']['mean'] if entry['models'][name]['metrics'].get('error_px')
                 else float('nan') for name in MODELS]
        lines.append(f'  {split:12s} ' + ''.join(f'{c:12.1f}' for c in cells))
    return '\n'.join(lines)


def _selftest():
    """Noise-free and noisy synthetic subjects; no camera, no real data, temporary directory only."""
    with tempfile.TemporaryDirectory() as tmp:
        for noise in (0.0, 0.3):
            source = Path(tmp) / f'fixture-{noise}'
            write_synthetic_session(source, noise_px=noise)
            aggregate = run_session(source, Path(tmp) / f'out-{noise}', PROTECTED, 'selftest')
            main = aggregate['rotation']['pnp6']['protocols'][MAIN_PROTOCOL]
            geo = main['splits']['B_yaw']['models']['GEO']['metrics']['error_px']['mean']
            f2 = main['splits']['B_yaw']['models']['F2']['metrics']['error_px']['mean']
            print(f'synthetic pupil noise {noise} px, kappa {main["models"]["GEO"]["kappa"]}')
            for rotation in ROTATION_SOURCES:
                print(_table(aggregate, rotation))
            if noise == 0.0:
                assert geo < 1.0, geo  # exact forward model with exact PnP: geometry recovers the target
                assert f2 > 50.0, f2  # F2 trained with a still head cannot follow head rotation
        print('R7 synthetic selftest: passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('audit', 'run', 'run-all'):
        item = sub.add_parser(command)
        item.add_argument('--output', type=Path, help='new directory under experiment_sessions/r7_runs/')
        if command == 'run':
            item.add_argument('--candidate', required=True, help='C01, C02, ... from audit')
    sub.add_parser('selftest')
    args = parser.parse_args()
    if args.command == 'selftest':
        _selftest()
        return 0
    try:
        output = _output(args.output, args.command)
        output.parent.mkdir(parents=True, exist_ok=True)
        records = audit_candidates(SESSIONS, SESSIONS / 'r4_runs')
        usable = [r for r in records if r.get('analysis_eligible')]
        write_inventory(output, records)
        if args.command == 'audit':
            print(json.dumps(dict(candidate_count=len(records), analysis_eligible=len(usable),
                candidates=[dict(id=r['candidate_id'], protocol=r.get('protocol'), analysis_eligible=r.get('analysis_eligible'),
                                 r4_used=r.get('r4_used'), reasons=r['reasons']) for r in records],
                output=str(output)), ensure_ascii=False))
            return 0
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        chosen = usable if args.command == 'run-all' else [r for r in usable if r['candidate_id'] == args.candidate]
        if not chosen:
            raise ValueError('no analysis-eligible session selected')
        results = []
        for index, record in enumerate(chosen, 1):
            alias = 'S' + str(index).zfill(2)
            aggregate = run_session(Path(record['source_local']), output / 'sessions' / alias, PROTECTED, commit)
            results.append((alias, aggregate))
            print(alias, 'r4_used' if record.get('r4_used') else 'independent')
            for rotation in ROTATION_SOURCES:
                print(_table(aggregate, rotation))
        write_json(output / 'summary.json', summarize_sessions(results))
        print(json.dumps(dict(output=str(output), analyzed=len(results)), ensure_ascii=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
