"""R5 offline-only independent-session F2 replication and actual-motion QC."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.experiment.r5_replication import (aggregate_sessions, audit_candidates,  # noqa: E402
    collection_instructions, motion_quality, run_one, write_inventory)
from src.experiment.recording import write_json  # noqa: E402


SESSIONS = ROOT / 'experiment_sessions'
R5_RUNS = SESSIONS / 'r5_runs'


def _output(path, command):
    output = (path or R5_RUNS / (command + '-' + uuid4().hex[:12])).resolve()
    if not output.is_relative_to(R5_RUNS.resolve()) or output == R5_RUNS.resolve():
        raise ValueError('output must be a new directory under ignored experiment_sessions/r5_runs/')
    if output.exists():
        raise FileExistsError(output)
    return output


def _selftest():
    rows, rotations = [], {}
    for split, angle in [('B_natural', .2), ('B_yaw', 8.), ('B_pitch', 8.)]:
        for index in range(10):
            key = ('tracker', len(rows))
            rows.append(dict(id=['synthetic', *key], split=split, label_status='measurement',
                             segment=len(rows) // 10, epoch=0, continuous_part=0))
            axis = [1., 0., 0.] if split == 'B_pitch' else [0., 1., 0.]
            rotations[key] = cv2.Rodrigues(np.asarray(axis) * np.deg2rad(angle * index / 9))[0]
    assert motion_quality(rows, rotations)['status'] == 'pass'
    assert aggregate_sessions([])['status'] == 'insufficient_sessions'
    print('R5 synthetic motion/decision selftest: passed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('audit', 'run', 'run-all'):
        item = sub.add_parser(command)
        item.add_argument('--output', type=Path, help='new directory under experiment_sessions/r5_runs/')
        if command == 'run':
            choice = item.add_mutually_exclusive_group(required=True)
            choice.add_argument('--session', type=Path, help='explicit path from local audit inventory')
            choice.add_argument('--candidate', help='C01, C02, ... from current audit; re-audited on run')
    sub.add_parser('selftest')
    sub.add_parser('instructions')
    args = parser.parse_args()
    if args.command == 'selftest':
        _selftest()
        return
    records = audit_candidates(SESSIONS, SESSIONS / 'r4_runs')
    main_records = [r for r in records if r['eligible']]
    shortfall = max(0, 2 - len(main_records))
    if args.command == 'instructions':
        print(collection_instructions(shortfall) if shortfall else '已有至少 2 份合格独立会话，无需补采。')
        return
    try:
        output = _output(args.output, args.command)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_inventory(output, records)
        if args.command == 'audit':
            print(json.dumps(dict(candidate_count=len(records), main_eligible_count=len(main_records),
                r4_used_count=sum(r.get('r4_used', False) for r in records), shortfall=shortfall,
                candidates=[dict(id=r['candidate_id'], protocol=r.get('protocol'),
                                 eligible=r['eligible'], reasons=r['reasons']) for r in records],
                output=str(output)), ensure_ascii=False))
            return
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        protected = [ROOT / 'dataset_raw', ROOT / 'dataset_processed', SESSIONS / 'r4_runs']
        if args.command == 'run':
            selected = next((r for r in records if
                (r['candidate_id'] == args.candidate if args.candidate else
                 Path(r['source_local']) == args.session.resolve())), None)
            if selected is None:
                raise ValueError('specified session was not found in experiment_sessions')
            chosen = [selected]
        else:
            chosen = [r for r in records if r['analysis_eligible'] and not r['r4_used']]
        results = []
        for index, record in enumerate(chosen, 1):
            alias = 'S' + str(index).zfill(2)
            aggregate = run_one(record, output / 'sessions' / alias, protected, commit)
            results.append((alias, record, aggregate))
        summary = aggregate_sessions(results)
        write_json(output / 'replication_summary.json', summary)
        if shortfall:
            (output / 'collection_instructions.txt').write_text(collection_instructions(shortfall), encoding='utf-8')
        print(json.dumps(dict(output=str(output), analyzed=len(results), main_eligible=summary['main_sessions'],
                              shortfall=summary['shortfall'], status=summary['status'],
                              online_candidate=summary['online_candidate']), ensure_ascii=False))
        if args.command == 'run-all' and shortfall:
            print(collection_instructions(shortfall), file=sys.stderr)
            return 2
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
