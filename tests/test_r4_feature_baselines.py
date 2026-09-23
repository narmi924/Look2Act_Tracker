"""Synthetic-only R4 checks: no webcam, real data, model weights or OS action."""
import csv
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest
import cv2

from src.experiment.protocol import make_plan
from src.experiment.r4_features import extract_features, local_eye_point
from src.experiment.r4_analysis import (build_rows, compare, fit_ridge, guarded_output,
                                        metrics, motion_diagnostics, pose_from_snapshot, predict, run_session,
                                        select_complete_ab)
from src.experiment.r4_audit import audit_legacy, old_source_key
from src.experiment.recording import write_json
from src.vision.head_pose import _MODEL_POINTS_3D, _REQUIRED_KEYS


def snapshot(left=(7., 2.), right=(27., 2.), iris_left=(8., 2.), iris_right=(28., 2.)):
    eyes = {'left': dict(roi_origin=[0., 0.], iris_center=iris_left, iris_status='available'),
            'right': dict(roi_origin=[20., 0.], iris_center=iris_right, iris_status='available')}
    return dict(units='camera_px', frame_size=[100, 80], eyes=eyes,
                eye_landmarks={'33': [0., 0.], '133': [10., 0.], '362': [20., 0.], '263': [30., 0.]},
                classic_dark_centroid=dict(units='camera_px', left_roi=left,
                    right_roi=None if right is None else [right[0] - 20, right[1]],
                    left_frame=left, right_frame=right))


def test_eye_local_numerics_and_similarity_invariance():
    base, reason = local_eye_point((0, 0), (10, 0), (7, 2))
    assert reason is None and base == pytest.approx([.2, .2])
    theta = .73
    rotation = np.array([[math.cos(theta), -math.sin(theta)],
                         [math.sin(theta), math.cos(theta)]])
    transform = lambda v: 3.7 * rotation @ np.asarray(v) + [13., -5.]
    transformed, reason = local_eye_point(transform((0, 0)), transform((10, 0)), transform((7, 2)))
    assert reason is None and transformed == pytest.approx(base)


def test_feature_source_separation_and_missing_degenerate_rejection():
    features, reasons = extract_features(snapshot())
    assert features['F0'] == pytest.approx([.17, .025])
    assert features['F1'] == pytest.approx([.07, .025, .27, .025])
    assert features['F2'] == pytest.approx([.2, .2, .2, .2])
    assert features['F3'] == pytest.approx([.3, .2, .3, .2])
    assert not any(reasons.values())
    data = snapshot(right=None)
    features, reasons = extract_features(data)
    assert features['F0'] is not None and all(features[n] is None for n in ('F1', 'F2'))
    assert reasons['F1'] == 'one_or_both_dark_points_unavailable'
    data = snapshot()
    data['eye_landmarks']['133'] = data['eye_landmarks']['33']
    assert extract_features(data)[1]['F2'] == 'eye_width_below_2px_left'
    data = snapshot()
    data['eyes']['left']['iris_center'] = None
    assert extract_features(data)[0]['F3'] is None
    data = snapshot()
    data['classic_dark_centroid']['left_frame'] = [math.nan, 2]
    assert extract_features(data)[0]['F1'] is None
    data = snapshot()
    data['classic_dark_centroid']['left_frame'] = [999, 2]  # ROI and frame contradict.
    assert extract_features(data)[0]['F0'] is not None  # right eye remains.
    assert extract_features(data)[0]['F2'] is None
    data = snapshot()
    data['units'] = 'eye_crop_px'
    assert all(v is None for v in extract_features(data)[0].values())


def test_offline_pose_uses_saved_contract_not_zero_model_input():
    camera = np.asarray([[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]])
    distortion = np.zeros((4, 1))
    projected, _ = cv2.projectPoints(_MODEL_POINTS_3D, np.array([.1, -.1, .05]),
                                     np.array([0., 0., 1200.]), camera, distortion)
    sample = snapshot()
    sample['pnp_points'] = {key: xy.tolist() for key, xy in zip(_REQUIRED_KEYS, projected.reshape(-1, 2))}
    sample['model_pose_input_deg'] = [0, 0, 0]
    saved = dict(keys=list(_REQUIRED_KEYS), model_points_mm=_MODEL_POINTS_3D.tolist(),
                 camera_matrix=camera.tolist(), dist_coeffs=distortion.tolist())
    pose, info = pose_from_snapshot(sample, saved)
    assert pose is not None and info['origin'] == 'derived_offline'
    assert info['reprojection_rmse_px'] < 1e-5
    pose, info = pose_from_snapshot(sample, {**saved, 'camera_matrix': None})
    assert pose is None and info['origin'] == 'unavailable'


def _row(i, segment, split, x, y, *, f2=True):
    return dict(id=['fixture', 'tracker', i], split=split, segment=segment, epoch=0, target_id=segment,
                target_norm=list(y), target_px=(np.asarray(y) * [100, 80]).tolist(),
                continuous_part=0, source_time=float(i),
                features={'C0': [], 'F0': list(x), 'F1': list(x) + list(x),
                          'F2': list(x) + list(x) if f2 else None,
                          'F3': list(x) + list(x), 'H0': [0, 0, 0],
                          'F2+H': list(x) + list(x) + [0, 0, 0] if f2 else None,
                          'F3+H': list(x) + list(x) + [0, 0, 0]})


def test_ridge_training_only_intercept_weight_rank_reproducibility():
    rows = []
    for segment, n, x in [(0, 40, [0., 0.]), (1, 2, [1., 0.]), (2, 2, [0., 1.]), (3, 2, [1., 1.])]:
        rows.extend(_row(len(rows) + j, segment, 'A_train', x, [.1 + .2*x[0], .3 + .1*x[1]]) for j in range(n))
    model = fit_ridge(rows, 'F0')
    assert model == fit_ridge(rows, 'F0')
    assert model['presentations'] == 4 and model['rank'] == 2
    assert predict(model, [_row(500, 5, 'A_holdout', [0., 0.], [.1, .3])])[0] == pytest.approx([.1, .3], abs=.015)
    original = json.dumps(model, sort_keys=True)
    test = _row(501, 5, 'A_holdout', [99., -99.], [1., 1.])
    test['target_norm'] = [-100, -100]
    assert json.dumps(fit_ridge(rows, 'F0'), sort_keys=True) == original
    zero = fit_ridge([_row(i, 0, 'A_train', [1., 1.], [.3, .7]) for i in range(3)], 'F0')
    assert zero['zero_variance_columns'] == [0, 1] and zero['rank'] == 0
    assert predict(zero, [_row(4, 1, 'A_holdout', [1., 1.], [.2, .2])])[0] == pytest.approx([.3, .7])
    assert fit_ridge(rows, 'C0')['intercept'] == pytest.approx([.2, .35], abs=.01)
    imbalanced = ([_row(i, 0, 'A_train', [0, 0], [0, 0]) for i in range(100)] +
                  [_row(101, 1, 'A_train', [1, 1], [1, 1])])
    assert fit_ridge(imbalanced, 'C0')['intercept'] == pytest.approx([.5, .5])


def test_metrics_count_outside_and_pair_support():
    rows = [_row(1, 0, 'B_yaw', [.1, .2], [.1, .2]),
            _row(2, 0, 'B_yaw', [.2, .2], [.1, .2]),
            _row(3, 1, 'B_yaw', [.2, .3], [.1, .2])]
    output = np.asarray([[1.5, .2], [1.5, .2], [.1, .2]])
    score = metrics(rows, output, (100, 80))
    assert score['error_px']['mean'] == pytest.approx(280 / 3)  # outside values included
    assert score['outside'] == 2 and score['segments'] == 2
    assert [s['n'] for s in score['segment_errors_px']] == [2, 1]
    motion = motion_diagnostics(rows, output, (100, 80))
    assert motion['B_yaw']['pieces'] == 1 and motion['B_yaw']['n'] == 2
    assert motion['B_yaw']['dispersion_px']['mean'] == 0  # stability alone says nothing about accuracy


def test_split_and_pair_use_identical_train_and_test_ids():
    plan = make_plan((100, 80), 'AB')
    rows = [_row(i, i, '', [i/20, i/30], [.2 + i/100, .4], f2=(i != 0)) for i in range(9)]
    rows += [_row(i, i, '', [i/20, i/30], [.2 + i/100, .4], f2=(i != 10)) for i in range(9, 18)]
    for row in rows:
        row['label_status'] = 'measurement'
        row['producer_valid'] = True
        row['target_id'] = plan['segments'][row['segment']]['target_id']
    _, own, paired, _ = compare(rows, plan, (100, 80))
    assert own['F1']['train_n'] == 9 and own['F2']['train_n'] == 8
    assert own['F1']['splits']['A_holdout']['n'] == 9
    assert paired['F1_vs_F2']['common_train_n'] == 8
    assert paired['F1_vs_F2']['splits']['A_holdout']['common_test_n'] == 8
    assert own['F1']['splits']['A_holdout_B_targets']['n'] == sum(
        plan['segments'][i]['target_id'] in (3, 4, 5) for i in range(9, 18))
    assert all(row['split'] == 'A_train' for row in rows[:9])
    assert all(row['split'] == 'A_holdout' for row in rows[9:])
    original = copy.deepcopy(rows)
    models1, _, _, predicted1 = compare(original, plan, (100, 80))
    altered = copy.deepcopy(rows)
    altered[9]['features']['F1'] = [100, 100, 100, 100]
    altered[9]['target_norm'] = [-10, -10]
    models2, _, _, predicted2 = compare(altered, plan, (100, 80))
    assert models1 == models2  # held-out feature and label cannot refit training statistics
    assert next(p['predicted_norm'] for p in predicted1 if p['feature'] == 'F0' and p['id'] == rows[9]['id']) == \
           next(p['predicted_norm'] for p in predicted2 if p['feature'] == 'F0' and p['id'] == rows[9]['id'])
    assert next(p['predicted_norm'] for p in predicted1 if p['feature'] == 'F1' and p['id'] == rows[9]['id']) != \
           next(p['predicted_norm'] for p in predicted2 if p['feature'] == 'F1' and p['id'] == rows[9]['id'])


def _event(kind, at, event_id, **payload):
    return dict(kind=kind, at=at, event_id=event_id, **payload)


def test_adapter_uses_source_time_dedup_and_tracks_unaccepted_producers():
    plan = make_plan((100, 80), 'AB')
    meta = dict(schema_version=1, synthetic=False, experiment_id='fixture',
                screen_size=[100, 80], plan=plan, pnp={})
    painted = _event('target_painted', 10., 1, **plan['segments'][0], epoch=0)
    obs = dict(session='tracker', sequence=1, timestamp=10.6, continuity=0)
    producer = _event('producer', 10.7, 2, result=dict(observation=obs, valid=True,
                  point_kind='observed', raw_point=[.17, .025], numeric_snapshot=snapshot()))
    consumed = _event('consume', 10.8, 3, observation_id=['tracker', 1], gate_state='invalid', gate_reason='stale')
    rows, counts = build_rows(meta, [painted, producer, consumed, producer])
    assert len(rows) == 1 and counts['duplicate_producer_identity'] == 1
    assert rows[0]['label_status'] == 'measurement' and rows[0]['ui_rejections'] == ['stale']
    assert rows[0]['features']['F0'] is not None and rows[0]['f0_raw_difference'] == 0
    invalid = dict(producer, event_id=4, result={**producer['result'], 'observation': {**obs, 'sequence': 2},
                                                  'valid': False, 'point_kind': 'held'})
    rows, counts = build_rows(meta, [painted, producer, invalid])
    assert counts['producer_invalid'] == 1 and rows[-1]['features']['F0'] is None
    assert rows[-1]['continuous_part'] > rows[0]['continuous_part']
    rows, counts = build_rows(meta, [painted, {**producer, 'result': {**producer['result'],
                   'observation': {**obs, 'timestamp': None}}}])
    assert rows == [] and counts['missing_observation_identity_or_time'] == 1


def test_legacy_mapping_and_no_synthetic_roi_time():
    row = dict(session_id='S_X', eye_img_path='images/S_X_000123_L.jpg',
               right_eye_img_path='images/S_X_000123_R.jpg')
    assert old_source_key(row) == ('S_X', 123)
    assert old_source_key({**row, 'eye_img_path': 'images/other_000123_L.jpg'}) is None
    assert 'roi_origin' not in row and 'timestamp_ms' not in row and 'gaze_x' not in ('F0', 'F1', 'F2', 'F3')


def test_read_only_legacy_audit_and_output_guard(tmp_path):
    raw, processed, out = (tmp_path / name for name in ('dataset_raw', 'dataset_processed', 'r4_runs'))
    session = raw / 'one'; session.mkdir(parents=True)
    split = processed / 'train'; split.mkdir(parents=True)
    with (session / 'labels.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['session_id', 'frame_idx', 'user_id', 'device_id', 'valid',
            'img_path', 'timestamp_ms', 'screen_w', 'screen_h', 'frame_w', 'frame_h', 'target_x', 'target_y'])
        writer.writeheader(); writer.writerow(dict(session_id='one', frame_idx=1, user_id='u', device_id='d', valid=0,
            img_path='missing.jpg', timestamp_ms=10, screen_w=100, screen_h=80, frame_w=200, frame_h=100,
            target_x=20, target_y=40))
    with (split / 'labels.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['session_id', 'user_id', 'eye_img_path', 'right_eye_img_path',
                                                   'norm_target_x', 'norm_target_y'])
        writer.writeheader(); writer.writerow(dict(session_id='one', user_id='u', eye_img_path='images/one_000001_L.jpg',
                right_eye_img_path='images/one_000001_R.jpg', norm_target_x=.2, norm_target_y=.5))
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.csv')}
    result = audit_legacy(raw, processed, image_limit=0)
    assert result['processed']['matched_raw'] == 1 and result['processed']['norm_target_mismatch'] == 0
    assert result['raw']['missing_images'] == 1
    assert {p: p.read_bytes() for p in before} == before
    other = processed / 'val'; other.mkdir()
    (other / 'labels.csv').write_bytes((split / 'labels.csv').read_bytes())
    overlapping = audit_legacy(raw, processed, image_limit=0)
    assert overlapping['processed']['split_source_overlap']['train_val'] == 1
    assert overlapping['processed']['split_user_overlap']['train_val'] == 1
    with pytest.raises(ValueError):
        guarded_output(raw / 'bad', session, [raw, processed])
    assert guarded_output(out, session, [raw, processed]) == out.resolve()


def test_complete_tiny_run_does_not_modify_source(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'analysis'
    source.mkdir()
    plan = make_plan((100, 80), 'AB')
    meta = dict(schema_version=1, experiment_id='fixture', complete=True, synthetic=False,
                backend='classic', screen_size=[100, 80], plan=plan, pnp={})
    events = []
    for segment in plan['segments']:
        at = 100. + segment['planned_offset_s']
        events.append(_event('target_painted', at, len(events) + 1, **segment, epoch=0))
        obs = dict(session='tracker', sequence=segment['segment'] + 1, timestamp=at + .7, continuity=0)
        events.append(_event('producer', at + .8, len(events) + 1,
            result=dict(observation=obs, valid=True, point_kind='observed', raw_point=[.17, .025],
                        numeric_snapshot=snapshot())))
    meta['written'] = len(events)
    write_json(source / 'session.json', meta)
    (source / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf8')
    (source / 'summary.json').write_text('{"historical":true}', encoding='utf8')
    (source / 'calibration.json').write_text('{"private":"fixture"}', encoding='utf8')
    assert select_complete_ab(tmp_path) == source
    before = {p: p.read_bytes() for p in source.iterdir()}
    result = run_session(source, output, [], 'test-commit')
    assert result['quality']['splits']['A_train'] == 9
    assert result['quality']['splits']['A_holdout'] == 9
    assert result['paired']['F1_vs_F2']['splits']['A_holdout']['common_test_n'] == 9
    assert {p: p.read_bytes() for p in source.iterdir()} == before
    assert json.loads((output / 'manifest.json').read_text())['analysis_run'] is True
    with pytest.raises(FileExistsError):
        run_session(source, output, [], 'test-commit')
    shutil.copytree(source, tmp_path / 'another_eligible_source')
    with pytest.raises(ValueError, match='found 2'):
        select_complete_ab(tmp_path)
