"""Synthetic-only R5 selection, SO(3) QC, fixed R4 fits and session aggregation."""
import json
import shutil

import cv2
import numpy as np
import pytest

from scripts.r5_replication import _output
import scripts.r5_replication as r5_cli
from src.experiment.protocol import make_plan
from src.experiment.r4_analysis import hash_file
from src.experiment.r5_replication import (MIN_MEASUREMENTS_PER_SPLIT, _source_reasons,
    aggregate_sessions, audit_candidates, audit_session, collection_instructions, motion_quality,
    r4_source_hashes, rotation_from_snapshot, run_one, write_inventory)
from src.experiment.recording import write_json
from src.vision.head_pose import _MODEL_POINTS_3D, _REQUIRED_KEYS


def _snapshot():
    return dict(units='camera_px', frame_size=[100, 80],
        eyes={'left': dict(roi_origin=[0, 0], iris_center=[8, 2], iris_status='available'),
              'right': dict(roi_origin=[20, 0], iris_center=[28, 2], iris_status='available')},
        eye_landmarks={'33': [0, 0], '133': [10, 0], '362': [20, 0], '263': [30, 0]},
        classic_dark_centroid=dict(units='camera_px', left_roi=[7, 2], right_roi=[7, 2],
                                   left_frame=[7, 2], right_frame=[27, 2]))


def _session(path, *, full=True, offset=0, synthetic=False, backend='classic', selection='AB'):
    path.mkdir(parents=True)
    plan = make_plan((100, 80), selection)
    meta = dict(schema_version=1, experiment_id='fixture-' + str(offset), complete=True,
                synthetic=synthetic, backend=backend, screen_size=[100, 80], plan=plan, pnp={})
    events = []
    for segment in plan['segments']:
        at = 100. + offset + segment['planned_offset_s']
        events.append(dict(kind='target_painted', at=at, event_id=len(events) + 1, epoch=0, **segment))
        if not full:
            continue
        n = (9 if segment['protocol'] == 'A' else
             14 if segment['requested_motion'] == 'natural' else 27)
        for j in range(n):
            source_time = at + .61 + .8 * j / n
            obs = dict(session='tracker-' + str(offset), sequence=len(events), timestamp=source_time,
                       continuity=0)
            result = dict(observation=obs, valid=True, point_kind='observed',
                          raw_point=[.17, .025], numeric_snapshot=_snapshot())
            events.append(dict(kind='producer', at=source_time + .001,
                               event_id=len(events) + 1, result=result))
    meta['written'] = len(events)
    write_json(path / 'session.json', meta)
    (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    (path / 'summary.json').write_text('{"historical":true}', encoding='utf-8')
    (path / 'calibration.json').write_text('{"private":"synthetic"}', encoding='utf-8')
    return meta, events


def _rotation(axis, degrees):
    return cv2.Rodrigues(np.asarray(axis, dtype=float) * np.deg2rad(degrees))[0]


def _motion_rows():
    rows, rotations = [], {}
    for segment, (split, angle, axis) in enumerate((
            ('B_natural', 1., (0, 1, 0)), ('B_yaw', 12., (0, 1, 0)),
            ('B_pitch', 10., (1, 0, 0)))):
        for j in range(20):
            key = ('tracker', segment * 20 + j)
            rows.append(dict(id=['fixture', *key], split=split, segment=segment, epoch=0,
                             continuous_part=0, label_status='measurement'))
            # Yaw crosses an Euler branch but relative rotation stays near 12 degrees.
            origin = 179. if split == 'B_yaw' else 0.
            rotations[key] = _rotation(axis, origin + angle * j / 19)
    return rows, rotations


def test_fixed_motion_quality_uses_relative_matrices_across_branch():
    rows, rotations = _motion_rows()
    quality = motion_quality(rows, rotations)
    assert quality['status'] == 'pass'
    assert quality['splits']['B_natural']['piece_p95_rotation_deg'] < 1
    assert 8 < quality['splits']['B_yaw']['yaw_like_p95_deg'] < 13
    assert 7 < quality['splits']['B_pitch']['pitch_like_p95_deg'] < 11
    assert quality['splits']['B_yaw']['pieces'] == 1
    assert quality['splits']['B_yaw']['piece_details'][0]['n'] == 20
    assert MIN_MEASUREMENTS_PER_SPLIT == 80


def test_motion_unavailable_is_unknown_not_zero_and_weak_motion_fails():
    rows, rotations = _motion_rows()
    missing = motion_quality(rows, {})
    assert missing['status'] == 'motion_quality_unknown'
    assert missing['splits']['B_yaw']['piece_p95_rotation_deg'] is None
    assert missing['splits']['B_yaw']['measurements'] == 20
    assert motion_quality(rows, {key: np.eye(3) for key in rotations})['status'] == 'motion_quality_insufficient'


def test_saved_pnp_rotation_qc_uses_r4_contract_and_rejects_missing():
    camera = np.asarray([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    projected, _ = cv2.projectPoints(_MODEL_POINTS_3D, np.array([0., np.deg2rad(179.), 0.]),
                                     np.array([0., 0., 1200.]), camera, np.zeros(4))
    snapshot = dict(frame_size=[100, 80], pnp_points={key: value.tolist()
        for key, value in zip(_REQUIRED_KEYS, projected.reshape(-1, 2))})
    pnp = dict(keys=list(_REQUIRED_KEYS), model_points_mm=_MODEL_POINTS_3D.tolist(),
               camera_matrix=camera.tolist(), dist_coeffs=[0., 0., 0., 0.])
    rotation = rotation_from_snapshot(snapshot, pnp)
    assert rotation is not None and np.linalg.det(rotation) == pytest.approx(1.)
    assert rotation_from_snapshot(snapshot, {**pnp, 'keys': []}) is None
    assert rotation_from_snapshot({}, pnp) is None


@pytest.mark.parametrize('change,reason', [
    ({'synthetic': True}, 'synthetic_or_unknown'),
    ({'backend': 'deep'}, 'not_classic'),
    ({'complete': False}, 'incomplete_session'),
    ({'plan': {'selection': 'A', 'segments': []}}, 'not_complete_AB_plan'),
])
def test_source_exclusions_are_input_only(change, reason):
    plan = make_plan((100, 80), 'AB')
    meta = dict(synthetic=False, backend='classic', complete=True, plan=plan)
    meta.update(change)
    issues = ['incomplete_session'] if not meta['complete'] else []
    reasons = _source_reasons(meta, [], issues)
    assert reason in reasons
    assert 'not_all_30_painted' in reasons


def test_audit_r4_exclusion_duplicate_detection_and_fixed_measurement_floor(tmp_path, monkeypatch):
    sessions = tmp_path / 'experiment_sessions'
    sessions.mkdir()
    first = sessions / 'first'
    _session(first)
    marker = sessions / 'r4_runs' / 'old'
    marker.mkdir(parents=True)
    first_hash = hash_file(first / 'session.json')
    write_json(marker / 'manifest.json', dict(analysis_run=True, source_session_sha256=first_hash))
    assert r4_source_hashes(sessions / 'r4_runs') == {first_hash}
    # Synthetic matrices isolate selection logic; QC itself is tested above.
    monkeypatch.setattr('src.experiment.r5_replication.motion_quality', lambda rows, rotations:
        dict(status='pass', splits={}))
    used = audit_session(first, {first_hash})
    assert used['measurement_counts'] == dict(A_train=81, A_holdout=81,
        B_natural=84, B_yaw=81, B_pitch=81)
    assert used['analysis_eligible'] and not used['eligible'] and 'r4_source_used' in used['reasons']
    second = sessions / 'second'
    _session(second, offset=1000)
    assert audit_session(second, {first_hash})['eligible']
    copied = sessions / 'copied'
    shutil.copytree(second, copied)
    records = audit_candidates(sessions, sessions / 'r4_runs')
    assert len(records) == 3
    assert sum(r['eligible'] for r in records) == 0
    assert sum('copied_session' in r['reasons'] for r in records) == 2
    inventory = sessions / 'r5_runs' / 'audit'
    write_inventory(inventory, records)
    assert len(json.loads((inventory / 'all_candidates.json').read_text())) == 3
    assert len(json.loads((inventory / 'eligible_sessions.json').read_text())) == 0
    before = {p: p.read_bytes() for p in first.iterdir()}
    assert {p: p.read_bytes() for p in first.iterdir()} == before
    with pytest.raises(FileExistsError):
        write_inventory(inventory, records)


def test_insufficient_measurements_excluded_even_if_motion_passes(tmp_path, monkeypatch):
    source = tmp_path / 'short'
    _session(source, full=False)
    monkeypatch.setattr('src.experiment.r5_replication.motion_quality', lambda rows, rotations:
        dict(status='pass', splits={}))
    record = audit_session(source, {'not-the-r4-hash'})
    assert not record['eligible'] and not record['analysis_eligible']
    assert 'no_producer' in record['reasons']


def test_r4_events_remain_excluded_if_copied_metadata_changes(tmp_path, monkeypatch):
    sessions = tmp_path / 'experiment_sessions'
    sessions.mkdir()
    original = sessions / 'original'
    _session(original)
    marker = sessions / 'r4_runs' / 'old'
    marker.mkdir(parents=True)
    write_json(marker / 'manifest.json', dict(analysis_run=True,
        source_session_sha256=hash_file(original / 'session.json'),
        source_events_sha256=hash_file(original / 'events.jsonl')))
    copied = sessions / 'copied_with_changed_metadata'
    shutil.copytree(original, copied)
    metadata = json.loads((copied / 'session.json').read_text())
    metadata['experiment_id'] = 'changed-metadata'
    write_json(copied / 'session.json', metadata)
    monkeypatch.setattr('src.experiment.r5_replication.motion_quality', lambda rows, rotations:
        dict(status='pass', splits={}))
    records = audit_candidates(sessions, sessions / 'r4_runs')
    assert all(r['r4_used'] for r in records)
    assert all(not r['eligible'] for r in records)
    assert all('r4_source_used' in r['reasons'] for r in records)


def test_two_empty_incomplete_attempts_are_not_called_copies(tmp_path):
    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    for name in ('one', 'two'):
        source = sessions / name
        source.mkdir()
        write_json(source / 'session.json', dict(schema_version=1, complete=False,
            experiment_id=name, synthetic=False, backend='classic', plan=make_plan((100, 80), 'AB')))
        (source / 'events.jsonl').write_text('', encoding='utf-8')
    records = audit_candidates(sessions, sessions / 'r4_runs')
    assert len(records) == 2
    assert all('incomplete_session' in r['reasons'] for r in records)
    assert all('copied_session' not in r['reasons'] for r in records)


def test_single_session_r4_fit_and_read_only_manifest_with_extra_pair(tmp_path):
    source = tmp_path / 'source'
    _session(source)
    original = {p: hash_file(p) for p in source.iterdir()}
    record = audit_session(source, {'other-source-hash'})
    assert record['analysis_eligible'] and record['motion_quality']['status'] == 'motion_quality_unknown'
    output = tmp_path / 'r5_runs' / 'single'
    result = run_one(record, output, [tmp_path / 'r4_runs'], 'test-commit')
    assert result['quality']['splits']['A_train'] == 81
    assert result['paired']['F0_vs_F2']['common_train_n'] == 81
    assert result['paired']['F0_vs_F2']['splits']['A_holdout']['common_test_n'] == 81
    assert len(json.loads((output / 'sample_manifest.json').read_text())) == 408
    support = json.loads((output / 'paired_support.json').read_text())
    assert len(support['F0_vs_F2']['common_train_ids']) == 81
    assert len(support['F0_vs_F2']['common_test_ids']['B_natural']) == 84
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['ridge_lambda'] == .01 and manifest['weight_sum'] == 1
    assert json.loads((output / 'r5_manifest.json').read_text())['source_events_sha256'] == original[source / 'events.jsonl']
    assert {p: hash_file(p) for p in source.iterdir()} == original
    with pytest.raises(FileExistsError):
        run_one(record, output, [], 'test-commit')
    metadata = json.loads((source / 'session.json').read_text())
    metadata['changed_after_audit'] = True
    write_json(source / 'session.json', metadata)
    with pytest.raises(ValueError, match='source changed after qualification'):
        run_one(record, tmp_path / 'r5_runs' / 'stale', [], 'test-commit')
    assert not (tmp_path / 'r5_runs' / 'stale').exists()


def _result(mean, deltas, n):
    own = {name: dict(splits={split: dict(error_px=dict(mean=mean + (0 if name == 'F2' else 10)))
                                    for split in ('A_holdout', 'B_natural', 'B_yaw', 'B_pitch')})
           for name in ('F0', 'F1', 'F2', 'F3', 'C0', 'H0', 'F2+H', 'F3+H')}
    paired = {pair: dict(splits={split: dict(mean_delta_right_minus_left_px=deltas.get(split))
                                 for split in ('A_holdout', 'B_natural', 'B_yaw', 'B_pitch')})
              for pair in ('F1_vs_F2', 'F0_vs_F2', 'F2_vs_F3')}
    return dict(own=own, paired=paired, quality=dict(splits={'A_holdout': n}))


def _candidate(name, n):
    return dict(eligible=True, session_sha256=name * 64, measurement_counts={'A_holdout': n},
                motion_quality=dict(status='pass'))


def test_session_level_aggregation_uses_sessions_and_handles_missing_split():
    first = _result(100, dict(A_holdout=-10, B_natural=-20, B_yaw=-30, B_pitch=-40), 10000)
    second = _result(200, dict(A_holdout=-5, B_natural=-10, B_yaw=-15), 80)
    summary = aggregate_sessions([('S01', _candidate('a', 10000), first),
                                  ('S02', _candidate('b', 80), second)])
    assert summary['status'] == 'complete' and summary['main_sessions'] == 2
    assert summary['session_level_methods']['F2']['A_holdout']['mean'] == 150
    assert summary['session_level_methods']['F2']['A_holdout']['n'] == 2
    assert summary['comparisons']['F2_vs_F1']['session_wins'] == 2
    assert summary['comparisons']['F2_vs_F1']['split_delta_px']['B_pitch']['n'] == 1
    assert summary['online_candidate'] == 'support'
    assert aggregate_sessions([('S01', _candidate('a', 10000), first)])['online_candidate'].startswith('unavailable')


def test_no_result_based_qualification_and_insufficient_collection_notice(tmp_path):
    good = _candidate('a', 80)
    terrible = _result(100000, dict(A_holdout=1000, B_natural=1000, B_yaw=1000, B_pitch=1000), 80)
    summary = aggregate_sessions([('S01', good, terrible)])
    assert summary['main_sessions'] == 1 and summary['status'] == 'insufficient_sessions'
    assert summary['comparisons']['F2_vs_F1']['status'].startswith('unavailable')
    assert '还缺 2 份' in collection_instructions(2)
    assert 'collect --config configs/classic.yaml --protocol AB' in collection_instructions(2)
    with pytest.raises(ValueError):
        _output(tmp_path / 'r4_runs' / 'bad', 'audit')


def test_cli_audit_and_explicit_candidate_run_without_main_claim(tmp_path, monkeypatch, capsys):
    sessions = tmp_path / 'experiment_sessions'
    sessions.mkdir()
    source = sessions / 'source'
    _session(source)
    marker = sessions / 'r4_runs' / 'old'
    marker.mkdir(parents=True)
    write_json(marker / 'manifest.json', dict(analysis_run=True,
        source_session_sha256=hash_file(source / 'session.json')))
    monkeypatch.setattr(r5_cli, 'SESSIONS', sessions)
    monkeypatch.setattr(r5_cli, 'R5_RUNS', sessions / 'r5_runs')
    monkeypatch.setattr('sys.argv', ['r5_replication.py', 'audit',
                                    '--output', str(sessions / 'r5_runs' / 'audit')])
    assert r5_cli.main() is None
    audit = json.loads(capsys.readouterr().out)
    assert audit['candidate_count'] == 1 and audit['candidates'][0]['id'] == 'C01'
    assert audit['candidates'][0]['reasons'] == ['r4_source_used', 'motion_quality_unknown']
    monkeypatch.setattr('sys.argv', ['r5_replication.py', 'run', '--candidate', 'C01',
                                    '--output', str(sessions / 'r5_runs' / 'trial')])
    assert r5_cli.main() == 0
    trial = json.loads(capsys.readouterr().out)
    assert trial['analyzed'] == 1 and trial['main_eligible'] == 0
    assert trial['online_candidate'].startswith('unavailable')
