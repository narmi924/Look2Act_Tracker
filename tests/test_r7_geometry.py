"""Synthetic-only R7 checks: forward model, fits, leakage guards and read-only outputs."""
import copy
import json
import math

import cv2
import numpy as np
import pytest

from scripts.r7_geometry import _output
from src.experiment.r4_analysis import build_rows, hash_file, split_name
from src.experiment.r7_geometry import (EYEBALL_RADIUS_MM, HEAD_FLIP, MODEL_PARAMETERS, MODELS, ROTATION_SOURCES,
                                        camera_matrix, decomposition, evaluate, fit_geo, frame_geometry, geo_predict,
                                        mm_per_logical_px, pnp_yaw_consistency, project, ray_sphere_direction,
                                        run_session, screen_plane, summarize_sessions, synthetic_snapshot, tangent,
                                        write_synthetic_session)
from src.experiment.recording import read_session, write_json
from src.vision.head_pose import _MODEL_POINTS_3D, _REQUIRED_KEYS

CAMERA = np.array([[1280., 0., 640.], [0., 1280., 360.], [0., 0., 1.]])
PNP_META = dict(keys=list(_REQUIRED_KEYS), model_points_mm=_MODEL_POINTS_3D.tolist(),
                camera_matrix=CAMERA.tolist(), dist_coeffs=[[0.], [0.], [0.], [0.]])
SCREEN = (1280, 800)
MM_PER_PX = 0.2359
KAPPA = (0.02, -0.035)


def _plane():
    return screen_plane(MM_PER_PX, SCREEN)


def _target_mm(px):
    plane = _plane()
    return plane['origin'] + px[0] * MM_PER_PX * plane['ex'] + px[1] * MM_PER_PX * plane['ey']


def _pose(yaw_deg=0., pitch_deg=0., shift=(0., 0., 0.)):
    R = cv2.Rodrigues(np.array([math.radians(pitch_deg), math.radians(yaw_deg), 0.]))[0] @ HEAD_FLIP
    return R, np.array([0., 150., 600.]) + np.asarray(shift, dtype=float)


def test_ray_sphere_direction_hit_and_miss_fallback():
    center = np.array([0., 0., 600.])
    pixel = project(CAMERA, center)[0]
    axis, hit = ray_sphere_direction(CAMERA, pixel, center, EYEBALL_RADIUS_MM)
    assert hit and np.allclose(axis, [0., 0., -1.])
    axis, hit = ray_sphere_direction(CAMERA, pixel + np.array([400., 0.]), center, EYEBALL_RADIUS_MM)
    assert not hit and math.isclose(np.linalg.norm(axis), 1.0)


def test_tangent_uses_model_frame_forward():
    assert tangent(np.array([0., 0., -1.])) is None
    assert np.allclose(tangent(np.array([0.1, -0.2, 1.0])), [0.1, -0.2])


def test_screen_right_is_camera_negative_x():
    plane = _plane()
    right_edge = plane['origin'] + SCREEN[0] * MM_PER_PX * plane['ex']
    assert right_edge[0] < plane['origin'][0] and math.isclose(right_edge[0], -plane['origin'][0])


@pytest.mark.parametrize('pose', [dict(), dict(yaw_deg=12.), dict(pitch_deg=-8.), dict(shift=(45., -20., -70.)),
                                  dict(yaw_deg=-15., pitch_deg=6., shift=(30., 10., 40.))])
def test_forward_model_is_recovered_exactly_with_pnp_rotation(pose):
    target_px = np.array([1023., 160.])
    R, t = _pose(**pose)
    snapshot = synthetic_snapshot(CAMERA, R, t, _target_mm(target_px), KAPPA)
    geo, reason = frame_geometry(snapshot, PNP_META, CAMERA, 'dark', 'pnp6')
    assert reason is None and geo['sphere_misses'] == 0
    predicted = geo_predict(geo, KAPPA, _plane(), SCREEN) * np.asarray(SCREEN)
    assert np.allclose(predicted, target_px, atol=1e-3)
    assert np.allclose(geo['R'], R, atol=1e-6)


def test_rotation_none_ignores_head_rotation_but_keeps_parallax():
    target_px = np.array([640., 400.])
    still = synthetic_snapshot(CAMERA, *_pose(), _target_mm(target_px), KAPPA)
    moved = synthetic_snapshot(CAMERA, *_pose(shift=(60., 0., 0.)), _target_mm(target_px), KAPPA)
    turned = synthetic_snapshot(CAMERA, *_pose(yaw_deg=15.), _target_mm(target_px), KAPPA)
    for snapshot in (still, moved):
        geo, _ = frame_geometry(snapshot, PNP_META, CAMERA, 'dark', 'none')
        assert np.allclose(geo_predict(geo, KAPPA, _plane(), SCREEN) * np.asarray(SCREEN), target_px, atol=1e-3)
    geo, _ = frame_geometry(turned, PNP_META, CAMERA, 'dark', 'none')
    assert np.linalg.norm(geo_predict(geo, KAPPA, _plane(), SCREEN) * np.asarray(SCREEN) - target_px) > 100
    with pytest.raises(ValueError):
        frame_geometry(turned, PNP_META, CAMERA, 'dark', 'pnp4')


def test_frame_geometry_rejects_missing_or_foreign_inputs():
    snapshot = synthetic_snapshot(CAMERA, *_pose(), _target_mm(np.array([640., 400.])), KAPPA)
    assert frame_geometry({**snapshot, 'units': 'normalized'}, PNP_META, CAMERA)[1] == 'wrong_or_missing_coordinate_source'
    assert frame_geometry({k: v for k, v in snapshot.items() if k != 'pnp_points'}, PNP_META, CAMERA)[1] == 'pnp_points_missing'
    assert frame_geometry({k: v for k, v in snapshot.items() if k != 'classic_dark_centroid'}, PNP_META, CAMERA)[1] == 'pupil_missing_dark'
    other = dict(PNP_META, model_points_mm=(_MODEL_POINTS_3D * 1.1).tolist())
    assert frame_geometry(snapshot, other, CAMERA)[1] == 'pnp_model_differs'
    broken = copy.deepcopy(snapshot)
    del broken['eye_landmarks']['33']
    assert frame_geometry(broken, PNP_META, CAMERA)[1] == 'eye_corners_missing'


def _rows(tmp_path, name='fixture', **kwargs):
    source = tmp_path / name
    meta, events = write_synthetic_session(source, kappa=KAPPA, **kwargs)
    meta, events, issues = read_session(source)
    assert not issues
    rows, _ = build_rows(meta, events)
    for row in rows:
        row['split'] = split_name(row, meta['plan'])
    from src.experiment.r7_geometry import attach_geometry
    attach_geometry(rows, events, meta['pnp'], camera_matrix(meta['pnp']), 'dark', 'pnp6')
    return source, meta, events, rows


def test_fit_recovers_kappa_and_identity_gain(tmp_path):
    _, meta, _, rows = _rows(tmp_path)
    train = [r for r in rows if r['split'] == 'A_train' and r['geo'] is not None]
    assert len(train) > 100
    pure = fit_geo(train, _plane(), SCREEN)
    assert np.allclose(pure['kappa'], KAPPA, atol=1e-6)
    gained = fit_geo(train, _plane(), SCREEN, with_gain=True)
    assert np.allclose(gained['kappa'], KAPPA, atol=1e-4) and np.allclose(gained['gain'], np.eye(2), atol=1e-4)


def test_parameter_counts_match_fitted_models(tmp_path):
    _, meta, _, rows = _rows(tmp_path)
    result = evaluate(rows, _plane(), SCREEN, 'P1_static_train')
    fitted = result['fitted']
    assert MODEL_PARAMETERS['F2'] == np.asarray(fitted['F2']['coef']).size + len(fitted['F2']['intercept']) == 10
    assert MODEL_PARAMETERS['G'] == np.asarray(fitted['G']['coef']).size + len(fitted['G']['intercept']) == 10
    assert MODEL_PARAMETERS['GEO'] == len(fitted['GEO']['params']) == 2
    assert MODEL_PARAMETERS['GEO+gain'] == len(fitted['GEO+gain']['params']) == 6
    assert MODEL_PARAMETERS['GEO+affine'] == len(fitted['GEO+affine']['params']) + np.asarray(fitted['GEO+affine']['affine']).size == 8
    assert tuple(result['models']) == MODELS


def test_holdout_labels_cannot_change_any_fit(tmp_path):
    _, meta, _, rows = _rows(tmp_path)
    before = evaluate(rows, _plane(), SCREEN, 'P1_static_train')
    for row in rows:
        if row['split'] in ('A_holdout', 'B_natural', 'B_yaw', 'B_pitch'):
            row['target_norm'] = [0.9, 0.1]
    after = evaluate(rows, _plane(), SCREEN, 'P1_static_train')
    assert json.dumps(before['fitted'], sort_keys=True) == json.dumps(after['fitted'], sort_keys=True)
    assert before['train_ids'] == after['train_ids']
    for split in before['splits']:
        assert before['splits'][split]['ids'] == after['splits'][split]['ids']
        assert before['splits'][split]['models']['F2']['metrics']['error_px'] != after['splits'][split]['models']['F2']['metrics']['error_px']


def test_every_model_scores_identical_test_ids(tmp_path):
    _, meta, _, rows = _rows(tmp_path)
    rows[5]['features']['F2'] = None  # a frame without F2 must leave every model's support
    result = evaluate(rows, _plane(), SCREEN, 'P1_static_train')
    for split, entry in result['splits'].items():
        counts = {name: entry['models'][name]['metrics']['n'] for name in MODELS}
        assert len(set(counts.values())) == 1 and counts['F2'] == entry['n']
    assert rows[5]['id'] not in result['train_ids']


def test_decomposition_identity_and_piece_separation():
    rows, predicted = [], []
    for segment, (bias, spread) in enumerate([((30., -40.), 10.), ((0., 0.), 25.)]):
        for i in range(20):
            rows.append(dict(segment=segment, epoch=0, continuous_part=segment, target_norm=[.5, .5]))
            predicted.append([.5 + (bias[0] + spread * (1 if i % 2 else -1)) / 1280,
                              .5 + (bias[1] + spread * (1 if (i // 2) % 2 else -1)) / 800])
    out = decomposition(rows, np.asarray(predicted), (1280, 800))
    assert out['pieces'] == 2
    mse = out['pooled_rms_bias_px'] ** 2 + out['pooled_rms_within_px'] ** 2
    expected = np.mean([50. ** 2 + 2 * 10. ** 2, 0. + 2 * 25. ** 2])
    assert math.isclose(mse, expected, rel_tol=1e-9)
    assert math.isclose(out['bias_px']['max'], 50.) and math.isclose(out['within_rms_px']['max'], 25. * math.sqrt(2))


def test_pnp_yaw_consistency_is_flat_for_a_still_synthetic_head(tmp_path):
    _, meta, _, rows = _rows(tmp_path)
    report = pnp_yaw_consistency(rows)
    assert len(report['per_target']) == 9
    assert report['pnp_yaw_range_deg'] < 1e-3 and report['width_range_px'] < 1e-3


def test_run_session_is_read_only_and_keeps_private_data_local(tmp_path):
    source, meta, events, _ = _rows(tmp_path)
    hashes = {name: hash_file(source / name) for name in ('session.json', 'events.jsonl')}
    with pytest.raises(ValueError):
        run_session(source, source / 'inside', [], 'test')
    with pytest.raises(ValueError):
        run_session(source, tmp_path / 'out', [tmp_path], 'test')
    aggregate = run_session(source, tmp_path / 'out', [tmp_path / 'protected'], 'test')
    assert hashes == {name: hash_file(source / name) for name in ('session.json', 'events.jsonl')}
    with pytest.raises(FileExistsError):
        run_session(source, tmp_path / 'out', [], 'test')
    text = (tmp_path / 'out' / 'aggregate.json').read_text(encoding='utf-8')
    assert '"ids"' not in text and '"predictions"' not in text and '"train_ids"' not in text
    assert aggregate['analysis_run'] and set(aggregate['rotation']) == set(ROTATION_SOURCES)
    manifest = json.loads((tmp_path / 'out' / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['analysis_run'] and manifest['fixed']['primary_rotation'] == 'pnp6'
    local = json.loads((tmp_path / 'out' / 'local_predictions.json').read_text(encoding='utf-8'))
    assert local['pnp6']['P1_static_train']['predictions']
    main = aggregate['rotation']['pnp6']['protocols']['P1_static_train']
    assert main['splits']['B_yaw']['models']['GEO']['metrics']['error_px']['mean'] < 1.0
    assert main['splits']['B_yaw']['models']['F2']['metrics']['error_px']['mean'] > 50.0
    summary = summarize_sessions([('S01', aggregate)])
    assert summary['rotation']['pnp6']['per_session']['S01']['B_yaw']['GEO'] < 1.0


def test_run_session_refuses_non_real_sources(tmp_path):
    source, meta, events, _ = _rows(tmp_path)
    meta['synthetic'] = True
    write_json(source / 'session.json', meta)
    with pytest.raises(ValueError):
        run_session(source, tmp_path / 'out', [], 'test')


def test_screen_scale_prefers_reported_dpi_then_config():
    assert math.isclose(mm_per_logical_px(dict(screen_scale=dict(physical_dpi_reported=107.66)))[0], 25.4 / 107.66)
    scale, source = mm_per_logical_px(dict(config=dict(screen_w_mm=320.), screen_size=[1280, 800]))
    assert math.isclose(scale, .25) and source.startswith('config')
    with pytest.raises(ValueError):
        mm_per_logical_px({})


def test_cli_output_must_be_new_directory_under_r7_runs(tmp_path):
    with pytest.raises(ValueError):
        _output(tmp_path / 'elsewhere', 'run')
