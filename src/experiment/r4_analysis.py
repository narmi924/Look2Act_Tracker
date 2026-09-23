"""Read-only R3 adapter and fixed, un-smoothed R4 ridge comparison."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from src.experiment.protocol import target_at
from src.experiment.r4_features import FEATURES, extract_features, point
from src.experiment.recording import read_session, write_json
from src.vision.head_pose import HeadPoseEstimator, _MODEL_POINTS_3D, _REQUIRED_KEYS


RIDGE_LAMBDA = 0.01
MAX_CONTINUOUS_GAP_S = 0.25  # Segmentation only; not a feature or tuned parameter.
MAX_REPROJECTION_RMSE_PX = 20.0  # Fixed validity guard for generic-model PnP.
PAIRS = (('F0', 'F1'), ('F1', 'F2'), ('F2', 'F3'), ('F2', 'F2+H'), ('F3', 'F3+H'))


def _stat(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else dict(mean=float(np.mean(values)), median=float(np.median(values)),
                                             p95=float(np.percentile(values, 95)), min=float(np.min(values)),
                                             max=float(np.max(values)))


def hash_file(path):
    digest = sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def guarded_output(output, source, protected):
    output, source = Path(output).resolve(), Path(source).resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('analysis output overlaps source session')
    for folder in protected:
        folder = Path(folder).resolve()
        if output == folder or output.is_relative_to(folder) or folder.is_relative_to(output):
            raise ValueError('analysis output overlaps protected data')
    if output.exists():
        raise FileExistsError(output)
    return output


def select_complete_ab(directory):
    """Deterministic source-only selection; refuse ambiguity instead of cherry-picking error."""
    candidates = []
    for child in sorted(Path(directory).iterdir()):
        if not (child / 'session.json').is_file():
            continue
        try:
            meta, events, issues = read_session(child)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        producers = [e for e in events if e['kind'] == 'producer']
        if (issues or not meta.get('complete') or meta.get('synthetic') or meta.get('backend') != 'classic'
                or meta.get('plan', {}).get('selection') != 'AB' or len(meta['plan']['segments']) != 30
                or {e['segment'] for e in events if e['kind'] == 'target_painted'} != set(range(30))
                or not producers):
            continue
        snapshot_count = sum(bool((p['result'].get('numeric_snapshot') or {}).get('classic_dark_centroid')
                                  and (p['result'].get('numeric_snapshot') or {}).get('eye_landmarks'))
                             for p in producers)
        if snapshot_count / len(producers) < .9:  # source-field completeness, not prediction error
            continue
        candidates.append(child)
    if len(candidates) != 1:
        raise ValueError(f'expected exactly one eligible real Classic AB session; found {len(candidates)}')
    return candidates[0]


def pose_from_snapshot(snapshot, pnp_meta):
    """Use the recorded online pose, or the existing estimator with recorded PnP contract."""
    online = snapshot.get('head_pose') or {}
    if online.get('valid'):
        vector = [online.get(k) for k in ('yaw', 'pitch', 'roll')]
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
            return vector, dict(origin='estimated_online', reprojection_rmse_px=None)
    try:
        if list(pnp_meta['keys']) != list(_REQUIRED_KEYS):
            raise ValueError('model keys differ')
        model = np.asarray(pnp_meta['model_points_mm'], dtype=float)
        if model.shape != (6, 3) or not np.array_equal(model, _MODEL_POINTS_3D):
            raise ValueError('model differs from existing estimator')
        camera = np.asarray(pnp_meta['camera_matrix'], dtype=float)
        distortion = np.asarray(pnp_meta['dist_coeffs'], dtype=float)
        if (camera.shape != (3, 3) or distortion.size not in (4, 5, 8, 12, 14)
                or not np.isfinite(camera).all() or not np.isfinite(distortion).all()
                or camera[0, 0] <= 0 or camera[1, 1] <= 0):
            raise ValueError('invalid camera parameters')
        saved = snapshot['pnp_points']
        image = np.asarray([saved[k] for k in _REQUIRED_KEYS], dtype=float)
        if image.shape != (6, 2) or not np.isfinite(image).all():
            raise ValueError('invalid PnP points')
        size = snapshot['frame_size']
        estimator = HeadPoseEstimator(tuple(size))
        estimator.camera_matrix = camera
        estimator.dist_coeffs = distortion.reshape(-1, 1)
        result = estimator.estimate(saved)
        rotation = result.rotation_matrix
        if not result.valid or rotation is None or not np.isfinite(rotation).all():
            raise ValueError('solvePnP failed')
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            raise ValueError('invalid rotation')
        rvec, _ = cv2.Rodrigues(rotation)
        projection, _ = cv2.projectPoints(model, rvec, result.translation_vec,
                                          camera, estimator.dist_coeffs)
        rmse = float(np.sqrt(np.mean(np.sum((projection.reshape(-1, 2) - image) ** 2, axis=1))))
        angles = [result.yaw, result.pitch, result.roll]
        if not all(math.isfinite(v) for v in (*angles, rmse)) or rmse > MAX_REPROJECTION_RMSE_PX:
            raise ValueError('reprojection rejected')
        return angles, dict(origin='derived_offline', reprojection_rmse_px=rmse)
    except (KeyError, TypeError, ValueError, cv2.error) as exc:
        return None, dict(origin='unavailable', reason=type(exc).__name__ + ': ' + str(exc)[:80])


def build_rows(meta, events):
    """Producer records only. Labels use the original source timestamp and paint history."""
    if meta.get('schema_version') != 1 or meta.get('synthetic') or meta.get('analysis_run'):
        raise ValueError('requires a real R3 schema-1 source session')
    plan = meta['plan']
    params = plan['parameters']
    width, height = map(float, meta['screen_size'])
    consumed = defaultdict(list)
    for event in events:
        if event['kind'] == 'consume' and event.get('observation_id'):
            consumed[tuple(event['observation_id'])].append(event)
    rows, seen = [], set()
    counts = Counter()
    part = 0
    previous = None
    for event in events:
        if event['kind'] in ('gap', 'context', 'pause', 'resume', 'skip'):
            part += 1
            counts['boundary_' + event['kind']] += 1
        if event['kind'] != 'producer':
            continue
        result = event['result']
        obs = result.get('observation') or {}
        source_time = obs.get('timestamp')
        key = (meta['experiment_id'], obs.get('session'), obs.get('sequence'))
        if not obs.get('session') or type(obs.get('sequence')) is not int or not isinstance(source_time, (int, float)) or not math.isfinite(source_time):
            counts['missing_observation_identity_or_time'] += 1
            continue
        if key in seen:
            counts['duplicate_producer_identity'] += 1
            part += 1
            continue
        seen.add(key)
        if previous and (obs['session'] != previous.get('session') or obs.get('continuity') != previous.get('continuity')
                         or source_time <= previous['timestamp'] or source_time - previous['timestamp'] > MAX_CONTINUOUS_GAP_S):
            part += 1
            counts['boundary_source_discontinuity'] += 1
        previous = obs
        label = target_at(source_time, events, params['settling_s'], params['transition_guard_s'])
        counts['label_' + label['status']] += 1
        valid = result.get('valid') is True and result.get('point_kind') == 'observed'
        if not valid:
            counts['producer_invalid'] += 1
            part += 1
        snapshot = result.get('numeric_snapshot') or {}
        feats, reasons = extract_features(snapshot) if valid else ({name: None for name in ('F0', 'F1', 'F2', 'F3')},
                                                               {name: 'producer_invalid' for name in ('F0', 'F1', 'F2', 'F3')})
        pose, pose_info = pose_from_snapshot(snapshot, meta.get('pnp') or {}) if valid else (None, {'origin': 'unavailable', 'reason': 'producer_invalid'})
        feat = {'C0': [], **feats, 'H0': pose,
                'F2+H': None if feats.get('F2') is None or pose is None else feats['F2'] + pose,
                'F3+H': None if feats.get('F3') is None or pose is None else feats['F3'] + pose}
        consume = consumed.get((obs['session'], obs['sequence']), [])
        ui_rejections = sorted({str(e.get('gate_reason') or e.get('screen_rejection') or e.get('dispatch_rejection'))
                                for e in consume if e.get('gate_state') == 'invalid' or e.get('screen_rejection') or e.get('dispatch_rejection')})
        raw = point(result.get('raw_point'))
        f0 = point(feat.get('F0'))
        row = dict(id=list(key), event_id=event['event_id'], source_time=source_time,
                   tracker_continuity=obs.get('continuity'), continuous_part=part,
                   producer_valid=valid, producer_error=result.get('error_message'),
                   label_status=label['status'], segment=label.get('segment'), epoch=label.get('epoch'),
                   target_id=None if label.get('segment') is None else plan['segments'][label['segment']]['target_id'],
                   protocol=label.get('protocol'), motion=label.get('requested_motion'),
                   target_px=label.get('instructed_target'),
                   target_norm=None if label.get('instructed_target') is None else
                   [label['instructed_target'][0] / width, label['instructed_target'][1] / height],
                   features=feat, feature_reasons=reasons, pose_info=pose_info,
                   ui_consumed=bool(consume), ui_rejections=ui_rejections,
                   f0_raw_difference=None if f0 is None or raw is None else float(np.max(np.abs(f0 - raw))))
        rows.append(row)
    return rows, dict(counts)


def split_name(row, plan):
    if row['label_status'] != 'measurement' or not row['producer_valid'] or row['segment'] is None:
        return None
    segment = plan['segments'][row['segment']]
    if segment['protocol'] == 'A':
        return 'A_train' if segment['round'] == 0 else 'A_holdout'
    return 'B_' + segment['requested_motion']


def fit_ridge(rows, name, lam=RIDGE_LAMBDA):
    if not rows:
        raise ValueError('no training support')
    X = np.asarray([r['features'][name] for r in rows], dtype=float)
    if name == 'C0':
        X = np.empty((len(rows), 0))
    Y = np.asarray([r['target_norm'] for r in rows], dtype=float)
    groups = Counter((r['segment'], r['epoch']) for r in rows)
    weights = np.asarray([1 / groups[r['segment'], r['epoch']] for r in rows])
    # Each presentation has total weight 1; do not renormalize across presentations,
    # which would silently change the fixed penalty strength.
    weights = weights.astype(float)
    if X.shape[1]:
        mean = np.average(X, axis=0, weights=weights)
        scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=weights))
        zero = np.flatnonzero(scale < 1e-12).tolist()
        scale[zero] = 1
        Z = (X - mean) / scale
    else:
        mean, scale, zero, Z = np.array([]), np.array([]), [], X
    ymean = np.average(Y, axis=0, weights=weights)
    zmean = np.average(Z, axis=0, weights=weights) if Z.shape[1] else np.array([])
    centered = Z - zmean
    if Z.shape[1]:
        design = np.vstack([np.sqrt(weights)[:, None] * centered, np.sqrt(lam) * np.eye(Z.shape[1])])
        response = np.vstack([np.sqrt(weights)[:, None] * (Y - ymean), np.zeros((Z.shape[1], 2))])
        coef = np.linalg.lstsq(design, response, rcond=None)[0]
        singular = np.linalg.svd(np.sqrt(weights)[:, None] * centered, compute_uv=False)
        rank = int(np.linalg.matrix_rank(centered))
        condition = None if not len(singular) or singular[-1] < 1e-12 else float(singular[0] / singular[-1])
    else:
        coef, rank, condition = np.empty((0, 2)), 0, None
    intercept = ymean - zmean @ coef
    return dict(feature=name, mean=mean.tolist(), scale=scale.tolist(), zero_variance_columns=zero,
                coef=coef.tolist(), intercept=intercept.tolist(), rank=rank, condition=condition,
                n=len(rows), presentations=len(groups), lambda_=lam)


def predict(model, rows):
    if not rows:
        return np.empty((0, 2))
    X = np.asarray([r['features'][model['feature']] for r in rows], dtype=float)
    if model['feature'] == 'C0':
        X = np.empty((len(rows), 0))
    return (X - model['mean']) / model['scale'] @ np.asarray(model['coef']).reshape(X.shape[1], 2) + model['intercept']


def metrics(rows, predicted, size):
    if not rows:
        return dict(n=0, error_px=None, error_norm=None, finite_rate=None, outside_rate=None, segment_macro_px=None)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(predicted).all(axis=1)
    labels = np.asarray([r['target_norm'] for r in rows])
    errors_norm = np.linalg.norm(predicted[finite] - labels[finite], axis=1)
    errors_px = np.linalg.norm((predicted[finite] - labels[finite]) * np.asarray(size), axis=1)
    outside = np.any((predicted[finite] < 0) | (predicted[finite] > (np.asarray(size) - 1) / size), axis=1)
    by_segment = defaultdict(list)
    for row, error in zip((r for r, ok in zip(rows, finite) if ok), errors_px):
        by_segment[(row['segment'], row['epoch'])].append(float(error))
    return dict(n=len(rows), finite=int(finite.sum()), finite_rate=float(np.mean(finite)),
                outside=int(outside.sum()), outside_rate=float(np.mean(outside)) if len(outside) else None,
                error_px=_stat(errors_px), error_norm=_stat(errors_norm),
                segment_macro_px=_stat([np.mean(v) for v in by_segment.values()]),
                segments=len(by_segment),
                segment_errors_px=[dict(segment=segment, epoch=epoch, n=len(values), error_px=_stat(values))
                                   for (segment, epoch), values in sorted(by_segment.items())])


def _support(rows, name, split):
    b_targets = {3, 4, 5}
    return [r for r in rows if (r['split'] == split or
            (split == 'A_holdout_B_targets' and r['split'] == 'A_holdout' and r['target_id'] in b_targets))
            and r['features'][name] is not None]


def motion_diagnostics(rows, predicted, size):
    """Fixed-target contiguous pieces only; no differencing across gaps or targets."""
    groups = defaultdict(list)
    for row, output in zip(rows, predicted):
        if row['split'] and row['split'].startswith('B_') and np.isfinite(output).all():
            groups[(row['split'], row['segment'], row['epoch'], row['continuous_part'])].append((row, output))
    by_motion = defaultdict(list)
    for (split, _, _, _), group in groups.items():
        if len(group) < 2:
            continue
        xy = np.asarray([output for _, output in group]) * np.asarray(size)
        target = np.asarray(group[0][0]['target_px'])
        record = dict(n=len(group), bias_px=float(np.linalg.norm(np.mean(xy, axis=0) - target)),
                      dispersion_px=float(np.sqrt(np.sum(np.var(xy, axis=0)))),
                      first_last_shift_px=float(np.linalg.norm(xy[-1] - xy[0])))
        poses = [r['features']['H0'] for r, _ in group]
        if all(p is not None for p in poses) and len(group) >= 3:
            pose = np.asarray(poses)
            record['head_prediction_abs_correlation'] = {
                axis: max(abs(float(np.corrcoef(pose[:, i], xy[:, j])[0, 1]))
                          for j in (0, 1) if np.std(xy[:, j]) > 1e-9)
                for i, axis in enumerate(('yaw', 'pitch', 'roll'))
                if np.std(pose[:, i]) > 1e-9 and any(np.std(xy[:, j]) > 1e-9 for j in (0, 1))}
        by_motion[split].append(record)
    return {split: dict(pieces=len(records), n=sum(r['n'] for r in records),
                        bias_px=_stat([r['bias_px'] for r in records]),
                        dispersion_px=_stat([r['dispersion_px'] for r in records]),
                        first_last_shift_px=_stat([r['first_last_shift_px'] for r in records]),
                        head_prediction_abs_correlation={axis: _stat([r['head_prediction_abs_correlation'][axis]
                            for r in records if axis in r.get('head_prediction_abs_correlation', {})])
                            for axis in ('yaw', 'pitch', 'roll')})
            for split, records in by_motion.items()}


def compare(rows, plan, size):
    for row in rows:
        row['split'] = split_name(row, plan)
    splits = ('A_holdout', 'A_holdout_B_targets', 'B_natural', 'B_yaw', 'B_pitch')
    models, own, paired, predictions = {}, {}, {}, []
    for name in FEATURES:
        train = _support(rows, name, 'A_train')
        if not train:
            own[name] = {'status': 'no_training_support'}
            continue
        model = fit_ridge(train, name)
        models[name] = model
        own[name] = {'train_n': len(train), 'training_presentations': model['presentations'],
                     'feature_dim': len(model['mean']), 'rank': model['rank'], 'condition': model['condition'],
                     'zero_variance_columns': model['zero_variance_columns'], 'splits': {}, 'motion': {}}
        for split in splits:
            support = _support(rows, name, split)
            outputs = predict(model, support)
            own[name]['splits'][split] = metrics(support, outputs, size)
            denominator = sum(r['split'] == split or (split == 'A_holdout_B_targets' and
                              r['split'] == 'A_holdout' and r['target_id'] in (3, 4, 5)) for r in rows)
            own[name]['splits'][split]['eligible_measurements'] = denominator
            own[name]['splits'][split]['missing_feature'] = denominator - len(support)
            for row, output in zip(support, outputs):
                predictions.append(dict(id=row['id'], feature=name, split=split, predicted_norm=output.tolist(),
                                        target_norm=row['target_norm'], segment=row['segment'], epoch=row['epoch'],
                                        continuous_part=row['continuous_part']))
        own[name]['motion'] = motion_diagnostics(
            [r for r in rows if r['split'] and r['split'].startswith('B_') and r['features'][name] is not None],
            predict(model, [r for r in rows if r['split'] and r['split'].startswith('B_') and r['features'][name] is not None]), size)
    for left, right in PAIRS:
        train = [r for r in rows if r['split'] == 'A_train' and r['features'][left] is not None and r['features'][right] is not None]
        key = left + '_vs_' + right
        if not train:
            paired[key] = {'status': 'no_common_training_support'}
            continue
        model_l, model_r = fit_ridge(train, left), fit_ridge(train, right)
        paired[key] = {'common_train_n': len(train), 'common_train_presentations': model_l['presentations'], 'splits': {}}
        for split in splits:
            support = [r for r in rows if r['split'] == split and r['features'][left] is not None and r['features'][right] is not None]
            ml, mr = metrics(support, predict(model_l, support), size), metrics(support, predict(model_r, support), size)
            paired[key]['splits'][split] = dict(common_test_n=len(support), left=ml, right=mr,
                mean_delta_right_minus_left_px=None if not support or ml['error_px'] is None or mr['error_px'] is None
                else mr['error_px']['mean'] - ml['error_px']['mean'])
    return models, own, paired, predictions


def quality(rows, counts, plan):
    available = {name: sum(r['features'][name] is not None for r in rows) for name in FEATURES}
    reasons = {name: dict(Counter(r['feature_reasons'].get(name) for r in rows if r['features'][name] is None))
               for name in ('F0', 'F1', 'F2', 'F3')}
    poses = [r for r in rows if r['features']['H0'] is not None]
    reproj = [r['pose_info']['reprojection_rmse_px'] for r in poses if r['pose_info']['reprojection_rmse_px'] is not None]
    by_split = Counter(r['split'] for r in rows)
    pose_ranges = {}
    for split in ('A_train', 'A_holdout', 'B_natural', 'B_yaw', 'B_pitch'):
        group = [r['features']['H0'] for r in poses if r['split'] == split]
        pose_ranges[split] = None if not group else {axis: _stat(np.asarray(group)[:, i]) for i, axis in enumerate(('yaw', 'pitch', 'roll'))}
    f0_differences = [r['f0_raw_difference'] for r in rows if r['f0_raw_difference'] is not None]
    jumps = []
    for first, second in zip(rows, rows[1:]):
        if (first['features']['H0'] is not None and second['features']['H0'] is not None
                and first['continuous_part'] == second['continuous_part']):
            jumps.append(float(np.max(np.abs(np.asarray(second['features']['H0']) - first['features']['H0']))))
    return dict(producers=len(rows), source_counts=counts, splits=dict(by_split), available=available,
                missing_reasons=reasons, ui_rejected=sum(bool(r['ui_rejections']) for r in rows),
                ui_not_consumed=sum(not r['ui_consumed'] for r in rows),
                f0_raw_max_abs_difference=max(f0_differences, default=None),
                pose_origin=dict(Counter(r['pose_info']['origin'] for r in rows)),
                pose_reprojection_rmse_px=_stat(reproj), pose_max_axis_jump_deg=_stat(jumps),
                pose_jumps_over_90_deg=sum(j > 90 for j in jumps), pose_ranges=pose_ranges)


def run_session(source, output, protected, code_commit):
    """Freeze selection and source hashes before any fit; write only to new output."""
    start = time.perf_counter()
    output = guarded_output(output, source, protected)
    meta, events, issues = read_session(source)
    if (issues or not meta.get('complete') or meta.get('synthetic') or meta.get('backend') != 'classic'
            or meta.get('plan', {}).get('selection') != 'AB' or len(meta['plan']['segments']) != 30):
        raise ValueError('requires complete, real Classic AB with two A rounds and B')
    if {e['segment'] for e in events if e['kind'] == 'target_painted'} != set(range(30)):
        raise ValueError('not all target presentations were painted')
    inventory = []
    for candidate in sorted(Path(source).parent.iterdir()):
        metadata_path = candidate / 'session.json'
        if not metadata_path.is_file():
            continue
        try:
            candidate_meta, candidate_events, candidate_issues = read_session(candidate)
            producer_events = [e for e in candidate_events if e['kind'] == 'producer']
            has_dark = sum(bool((e['result'].get('numeric_snapshot') or {}).get('classic_dark_centroid'))
                           for e in producer_events)
            has_eyes = sum(bool((e['result'].get('numeric_snapshot') or {}).get('eye_landmarks'))
                           for e in producer_events)
            has_pnp = sum(bool((e['result'].get('numeric_snapshot') or {}).get('pnp_points'))
                          for e in producer_events)
            inventory.append(dict(session_sha256=hash_file(metadata_path),
                synthetic=bool(candidate_meta.get('synthetic')),
                complete=bool(candidate_meta.get('complete')),
                backend=candidate_meta.get('backend'),
                protocol=candidate_meta.get('plan', {}).get('selection'),
                planned_segments=len(candidate_meta.get('plan', {}).get('segments', [])),
                painted=sum(e['kind'] == 'target_painted' for e in candidate_events),
                producers=len(producer_events), snapshot_dark=has_dark, snapshot_eye_landmarks=has_eyes,
                snapshot_pnp=has_pnp, issues=candidate_issues))
        except (OSError, ValueError, KeyError, TypeError):
            inventory.append(dict(session_sha256=hash_file(metadata_path), status='unreadable'))
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(analysis_run=True, source_session_sha256=hash_file(Path(source) / 'session.json'),
                  source_events_sha256=hash_file(Path(source) / 'events.jsonl'),
                  source_session_local=str(Path(source).resolve()), code_commit=code_commit,
                  analysis_code_sha256={name: hash_file(Path(__file__).with_name(name)) for name in
                                        ('r4_analysis.py', 'r4_features.py', 'r4_audit.py')},
                  selection='complete_real_classic_AB_30_painted', training='A_round_0',
                  holdouts=['A_round_1', 'B_natural', 'B_yaw', 'B_pitch'],
                  ridge_lambda=RIDGE_LAMBDA, min_eye_width_px=2.0,
                  max_reprojection_rmse_px=MAX_REPROJECTION_RMSE_PX,
                  continuous_gap_s=MAX_CONTINUOUS_GAP_S,
                  screen_size=meta['screen_size'], schema_version=1)
    write_json(output / 'manifest.json', frozen)  # recorded before fitting
    write_json(output / 'selection_inventory.json', inventory)
    rows, counts = build_rows(meta, events)
    if not rows:
        raise ValueError('no producer observations')
    models, own, paired, predictions = compare(rows, meta['plan'], meta['screen_size'])
    aggregate = dict(analysis_run=True, quality=quality(rows, counts, meta['plan']), own=own, paired=paired,
                     elapsed_s=time.perf_counter() - start)
    # Local-only files include sensitive numerical observations/identity mapping.
    write_json(output / 'rows.json', rows)
    write_json(output / 'models.json', models)
    write_json(output / 'predictions.json', predictions)
    write_json(output / 'aggregate.json', aggregate)
    return aggregate
