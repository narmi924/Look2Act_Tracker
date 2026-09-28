"""R5 read-only session selection, SO(3) motion QC, and session-level replication."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np

from src.experiment.r4_analysis import (MAX_REPROJECTION_RMSE_PX, PAIRS, build_rows,
                                        hash_file, run_session, split_name)
from src.experiment.r4_features import FEATURES
from src.experiment.recording import read_session, write_json
from src.vision.head_pose import HeadPoseEstimator, _MODEL_POINTS_3D, _REQUIRED_KEYS


MIN_MEASUREMENTS_PER_SPLIT = 80  # Fixed engineering floor, before looking at R5 errors.
MOTION_MARGIN_DEG = 3.0  # Fixed difference over natural; approximate PnP QC, not angular truth.
BASELINE_FRAMES = 5  # First steady measurement frames of each continuous B piece.
SPLITS = ('A_train', 'A_holdout', 'B_natural', 'B_yaw', 'B_pitch')
TEST_SPLITS = SPLITS[1:]
R5_PAIRS = (*PAIRS, ('F0', 'F2'))  # R4 fits unchanged; one extra common-support comparison.
COMPARISONS = {'F2_vs_F1': 'F1_vs_F2', 'F2_vs_F0': 'F0_vs_F2', 'F3_vs_F2': 'F2_vs_F3'}


def r4_source_hashes(r4_runs):
    """Only frozen R4 run manifests identify used sources; never infer from error."""
    hashes = set()
    for path in sorted(Path(r4_runs).glob('*/manifest.json')):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            value = manifest.get('source_session_sha256')
            if manifest.get('analysis_run') and isinstance(value, str) and len(value) == 64:
                hashes.add(value)
        except (OSError, ValueError, TypeError):
            continue
    return hashes


def r4_event_hashes(r4_runs):
    hashes = set()
    for path in sorted(Path(r4_runs).glob('*/manifest.json')):
        try:
            manifest = json.loads(path.read_text(encoding='utf-8'))
            value = manifest.get('source_events_sha256')
            if manifest.get('analysis_run') and isinstance(value, str) and len(value) == 64:
                hashes.add(value)
        except (OSError, ValueError, TypeError):
            continue
    return hashes


def rotation_from_snapshot(snapshot, pnp_meta):
    """Derive a matrix from saved PnP points under R4's same 20 px validity contract.

    This is QC only. It is never inserted into F0/F1/F2/F3 or substituted for H0.
    """
    try:
        if list(pnp_meta['keys']) != list(_REQUIRED_KEYS):
            return None
        model = np.asarray(pnp_meta['model_points_mm'], dtype=float)
        camera = np.asarray(pnp_meta['camera_matrix'], dtype=float)
        distortion = np.asarray(pnp_meta['dist_coeffs'], dtype=float)
        if (model.shape != (6, 3) or not np.array_equal(model, _MODEL_POINTS_3D)
                or camera.shape != (3, 3) or distortion.size not in (4, 5, 8, 12, 14)
                or not np.isfinite(camera).all() or not np.isfinite(distortion).all()
                or camera[0, 0] <= 0 or camera[1, 1] <= 0):
            return None
        saved = snapshot['pnp_points']
        image = np.asarray([saved[k] for k in _REQUIRED_KEYS], dtype=float)
        if image.shape != (6, 2) or not np.isfinite(image).all():
            return None
        estimator = HeadPoseEstimator(tuple(snapshot['frame_size']))
        estimator.camera_matrix = camera
        estimator.dist_coeffs = distortion.reshape(-1, 1)
        pose = estimator.estimate(saved)
        rotation = pose.rotation_matrix
        if not pose.valid or rotation is None or not np.isfinite(rotation).all():
            return None
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            return None
        rvec, _ = cv2.Rodrigues(rotation)
        projected, _ = cv2.projectPoints(model, rvec, pose.translation_vec,
                                          camera, estimator.dist_coeffs)
        rmse = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - image) ** 2, axis=1))))
        return rotation if math.isfinite(rmse) and rmse <= MAX_REPROJECTION_RMSE_PX else None
    except (KeyError, TypeError, ValueError, cv2.error):
        return None


def rotations_for_rows(meta, events, rows):
    wanted = {tuple(row['id'][1:]) for row in rows if (row.get('split') or '').startswith('B_')
              and row['producer_valid'] and row['label_status'] == 'measurement'}
    found = {}
    for event in events:
        if event['kind'] != 'producer':
            continue
        result = event.get('result') or {}
        obs = result.get('observation') or {}
        key = (obs.get('session'), obs.get('sequence'))
        if key in wanted and key not in found:
            found[key] = rotation_from_snapshot(result.get('numeric_snapshot') or {}, meta.get('pnp') or {})
    return found


def _median(values):
    return None if not values else float(np.median(values))


def motion_quality(rows, rotations):
    """Relative camera-axis rotation vectors avoid raw Euler ±180 degree branches.

    R_i R_base^T maps baseline camera orientation to this frame. Rodrigues vector
    norm is total angle; abs x/y/z are pitch/yaw/roll-like *components*, not
    independently calibrated anatomical angles. Each piece needs five poses.
    """
    groups = defaultdict(list)
    expected_groups = Counter()
    expected = Counter()
    valid = Counter()
    for row in rows:
        split = row.get('split')
        if split not in ('B_natural', 'B_yaw', 'B_pitch') or row['label_status'] != 'measurement':
            continue
        expected[split] += 1
        group_key = (split, row['segment'], row['epoch'], row['continuous_part'])
        expected_groups[group_key] += 1
        rotation = rotations.get(tuple(row['id'][1:]))
        if rotation is None:
            continue
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
            continue
        valid[split] += 1
        groups[group_key].append(rotation)
    pieces = defaultdict(list)
    for (split, segment, epoch, part), measurement_count in sorted(expected_groups.items()):
        matrices = groups[(split, segment, epoch, part)]
        if len(matrices) < BASELINE_FRAMES:
            pieces[split].append(dict(segment=segment, epoch=epoch, continuous_part=part,
                                      measurements=measurement_count, valid_pose_frames=len(matrices),
                                      status='unavailable_fewer_than_5_pose_frames'))
            continue
        u, _, vt = np.linalg.svd(np.sum(matrices[:BASELINE_FRAMES], axis=0))
        base = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
        vectors = np.asarray([cv2.Rodrigues(matrix @ base.T)[0].reshape(3) for matrix in matrices])
        degrees = np.rad2deg(vectors)
        total = np.linalg.norm(degrees, axis=1)
        pieces[split].append(dict(segment=segment, epoch=epoch, continuous_part=part, n=len(matrices),
            measurements=measurement_count, valid_pose_frames=len(matrices), status='available',
            total_rotation_deg=dict(median=float(np.median(total)), p95=float(np.percentile(total, 95))),
            pitch_like_change_deg=dict(median_abs=float(np.median(np.abs(degrees[:, 0]))),
                                       p95_abs=float(np.percentile(np.abs(degrees[:, 0]), 95))),
            yaw_like_change_deg=dict(median_abs=float(np.median(np.abs(degrees[:, 1]))),
                                     p95_abs=float(np.percentile(np.abs(degrees[:, 1]), 95))),
            roll_like_change_deg=dict(median_abs=float(np.median(np.abs(degrees[:, 2]))),
                                      p95_abs=float(np.percentile(np.abs(degrees[:, 2]), 95)))))
    splits = {}
    for split in ('B_natural', 'B_yaw', 'B_pitch'):
        entries = [p for p in pieces[split] if p['status'] == 'available']
        splits[split] = dict(measurements=expected[split], valid_pose_frames=valid[split],
            pieces=len(entries), unavailable_pieces=len(pieces[split]) - len(entries),
            piece_details=pieces[split],
            piece_median_rotation_deg=_median([p['total_rotation_deg']['median'] for p in entries]),
            piece_p95_rotation_deg=_median([p['total_rotation_deg']['p95'] for p in entries]),
            yaw_like_p95_deg=_median([p['yaw_like_change_deg']['p95_abs'] for p in entries]),
            pitch_like_p95_deg=_median([p['pitch_like_change_deg']['p95_abs'] for p in entries]),
            roll_like_p95_deg=_median([p['roll_like_change_deg']['p95_abs'] for p in entries]))
    if any(not splits[name]['pieces'] or splits[name]['unavailable_pieces'] for name in splits):
        status = 'motion_quality_unknown'
    else:
        natural, yaw, pitch = (splits[name] for name in ('B_natural', 'B_yaw', 'B_pitch'))
        passed = (yaw['piece_p95_rotation_deg'] >= natural['piece_p95_rotation_deg'] + MOTION_MARGIN_DEG
                  and pitch['piece_p95_rotation_deg'] >= natural['piece_p95_rotation_deg'] + MOTION_MARGIN_DEG
                  and yaw['yaw_like_p95_deg'] >= natural['yaw_like_p95_deg'] + MOTION_MARGIN_DEG
                  and pitch['pitch_like_p95_deg'] >= natural['pitch_like_p95_deg'] + MOTION_MARGIN_DEG)
        status = 'pass' if passed else 'motion_quality_insufficient'
    return dict(status=status, method='SO3_R_i_R_baseline_T_Rodrigues_camera_axes',
                baseline_frames=BASELINE_FRAMES, margin_over_natural_deg=MOTION_MARGIN_DEG,
                splits=splits)


def _source_reasons(meta, events, issues):
    reasons = list(issues)
    plan = meta.get('plan') or {}
    segments = plan.get('segments') or []
    if meta.get('synthetic') is not False:
        reasons.append('synthetic_or_unknown')
    if meta.get('backend') != 'classic':
        reasons.append('not_classic')
    if plan.get('selection') != 'AB' or len(segments) != 30:
        reasons.append('not_complete_AB_plan')
    else:
        a = Counter(s.get('round') for s in segments if s.get('protocol') == 'A')
        b = Counter(s.get('requested_motion') for s in segments if s.get('protocol') == 'B')
        if a != {0: 9, 1: 9} or b != {'natural': 6, 'yaw': 3, 'pitch': 3}:
            reasons.append('unexpected_AB_plan')
    painted = [e.get('segment') for e in events if e['kind'] == 'target_painted']
    if set(painted) != set(range(30)):
        reasons.append('not_all_30_painted')
    if meta.get('write_unconfirmed') or meta.get('writer_error') or meta.get('writer_still_alive'):
        reasons.append('recording_integrity')
    return reasons


def audit_session(source, r4_sources):
    source = Path(source)
    record = dict(source_local=str(source.resolve()), session_sha256=None, events_sha256=None,
                  reasons=[], eligible=False, analysis_eligible=False)
    try:
        record['session_sha256'] = hash_file(source / 'session.json')
        record['events_sha256'] = (hash_file(source / 'events.jsonl')
                                   if (source / 'events.jsonl').is_file() else None)
        meta, events, issues = read_session(source)
        record.update(complete=bool(meta.get('complete')), synthetic=meta.get('synthetic'),
                      backend=meta.get('backend'), protocol=(meta.get('plan') or {}).get('selection'),
                      event_count=len(events),
                      painted=len({e.get('segment') for e in events if e['kind'] == 'target_painted'}),
                      r4_used=record['session_sha256'] in r4_sources,
                      source_experiment_id=meta.get('experiment_id'))
        reasons = _source_reasons(meta, events, issues)
        if not r4_sources:
            reasons.append('r4_source_reference_missing')
        if reasons and any(reason not in ('r4_source_reference_missing',) for reason in reasons):
            record['reasons'] = reasons
            return record
        producers = [e for e in events if e['kind'] == 'producer']
        if not producers:
            reasons.append('no_producer')
            record['reasons'] = reasons
            return record
        snapshots = sum(bool(((e.get('result') or {}).get('numeric_snapshot') or {}).get('classic_dark_centroid')
                             and ((e.get('result') or {}).get('numeric_snapshot') or {}).get('eye_landmarks'))
                        for e in producers)
        record['numeric_snapshot_coverage'] = snapshots / len(producers)
        if record['numeric_snapshot_coverage'] < .9:  # R4 source-field completeness rule.
            reasons.append('numeric_snapshot_coverage_below_90_percent')
        rows, counts = build_rows(meta, events)
        for row in rows:
            row['split'] = split_name(row, meta['plan'])
        split_counts = Counter(row['split'] for row in rows if row['split'] in SPLITS)
        record.update(producers=len(producers), rows=len(rows), source_counts=counts,
                      measurement_counts={name: split_counts[name] for name in SPLITS},
                      feature_support={name: {feature: sum(row['split'] == name and row['features'][feature] is not None
                                                          for row in rows) for feature in FEATURES} for name in SPLITS})
        for name in SPLITS:
            if split_counts[name] < MIN_MEASUREMENTS_PER_SPLIT:
                reasons.append('measurement_below_80_' + name)
        b_segments = {segment['segment'] for segment in meta['plan']['segments'] if segment['protocol'] == 'B'}
        measured_b_segments = {row['segment'] for row in rows if row['split'] in
                               ('B_natural', 'B_yaw', 'B_pitch')}
        if b_segments - measured_b_segments:
            reasons.append('B_presentation_without_measurement')
        motion = motion_quality(rows, rotations_for_rows(meta, events, rows))
        record['motion_quality'] = motion
        if record['r4_used']:
            reasons.append('r4_source_used')
        if motion['status'] != 'pass':
            reasons.append(motion['status'])
        record['reasons'] = reasons
        record['analysis_eligible'] = not any(r for r in reasons if r not in
            ('r4_source_used', 'motion_quality_unknown', 'motion_quality_insufficient'))
        record['eligible'] = not reasons
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, cv2.error) as exc:
        record['reasons'] = ['unreadable_or_invalid_' + type(exc).__name__]
    return record


def audit_candidates(sessions, r4_runs):
    sources = r4_source_hashes(r4_runs)
    r4_events = r4_event_hashes(r4_runs)
    records = []
    for child in sorted(Path(sessions).iterdir()):
        if not (child / 'session.json').is_file():
            continue
        record = audit_session(child, sources)
        record['candidate_id'] = 'C' + str(len(records) + 1).zfill(2)
        if record.get('event_count') and record['events_sha256'] in r4_events:
            record['r4_used'] = True
            if 'r4_source_used' not in record['reasons']:
                record['reasons'].append('r4_source_used')
            record['eligible'] = False
        records.append(record)
    by_events, by_experiment = defaultdict(list), defaultdict(list)
    for record in records:
        if record.get('event_count') and record['events_sha256'] is not None:
            by_events[record['events_sha256']].append(record)
        if record.get('source_experiment_id') is not None:
            by_experiment[record['source_experiment_id']].append(record)
    for group in (*by_events.values(), *by_experiment.values()):
        if len(group) < 2:
            continue
        r4_linked = any(r.get('r4_used') for r in group)
        for record in group:
            if 'copied_session' not in record['reasons']:
                record['reasons'].append('copied_session')
            if r4_linked and 'r4_source_used' not in record['reasons']:
                record['reasons'].append('r4_source_used')
                record['r4_used'] = True
            record['eligible'] = record['analysis_eligible'] = False
    return records


def write_inventory(output, records):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'all_candidates.json', records)
    write_json(output / 'eligible_sessions.json', [r for r in records if r['eligible']])
    write_json(output / 'excluded_sessions.json', [r for r in records if not r['eligible']])


def run_one(record, output, protected, code_commit):
    """Explicit source can be R4-used for a smoke run; never enters main summary."""
    if not record.get('analysis_eligible'):
        raise ValueError('session lacks fixed source/measurement qualification')
    source = Path(record['source_local'])
    for name, expected in (('session.json', record['session_sha256']),
                           ('events.jsonl', record['events_sha256'])):
        if hash_file(source / name) != expected:
            raise ValueError('source changed after qualification: ' + name)
    aggregate = run_session(source, output, protected, code_commit, pairs=R5_PAIRS)
    for name, expected in (('session.json', record['session_sha256']),
                           ('events.jsonl', record['events_sha256'])):
        if hash_file(source / name) != expected:
            raise ValueError('source changed during analysis: ' + name)
    manifest = dict(source_session_sha256=record['session_sha256'],
                    source_events_sha256=record['events_sha256'], code_commit=code_commit,
                    r5_code_sha256=hash_file(__file__), r4_code_sha256=hash_file(Path(__file__).with_name('r4_analysis.py')),
                    eligible=record['eligible'], r4_used=record['r4_used'],
                    fixed_rules=dict(min_measurements_per_split=MIN_MEASUREMENTS_PER_SPLIT,
                                     motion_margin_deg=MOTION_MARGIN_DEG, baseline_frames=BASELINE_FRAMES,
                                     pair_support=[list(pair) for pair in R5_PAIRS]))
    write_json(Path(output) / 'r5_manifest.json', manifest)
    write_json(Path(output) / 'motion_quality.json', record['motion_quality'])
    write_json(Path(output) / 'r5_quality.json', dict(measurement_counts=record['measurement_counts'],
        feature_support=record['feature_support'], numeric_snapshot_coverage=record['numeric_snapshot_coverage'],
        reasons=record['reasons']))
    return aggregate


def _stats(values):
    return None if not values else dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                                        min=float(np.min(values)), max=float(np.max(values)))


def aggregate_sessions(results):
    """Each result contributes one observation per split/method, regardless of frame count."""
    per_session = []
    for alias, candidate, result in results:
        methods = {name: {split: ((entry.get('splits') or {}).get(split) or {}).get('error_px', {}).get('mean')
                          if ((entry.get('splits') or {}).get(split) or {}).get('error_px') else None
                          for split in TEST_SPLITS} for name, entry in result['own'].items()}
        deltas = {name: {split: ((result['paired'].get(pair) or {}).get('splits') or {}).get(split, {}).get(
            'mean_delta_right_minus_left_px') for split in TEST_SPLITS}
            for name, pair in COMPARISONS.items()}
        per_session.append(dict(alias=alias,
            measurement_counts=candidate['measurement_counts'], motion_quality=candidate['motion_quality'],
            methods=methods, paired_deltas_px=deltas))
    main = [row for row, (_, candidate, _) in zip(per_session, results) if candidate['eligible']]
    methods = {name: {split: _stats([row['methods'].get(name, {}).get(split) for row in main
                                    if row['methods'].get(name, {}).get(split) is not None])
                      for split in TEST_SPLITS} for name in FEATURES}
    comparisons = {}
    for name in COMPARISONS:
        split_stats = {split: _stats([row['paired_deltas_px'][name][split] for row in main
                                      if row['paired_deltas_px'][name][split] is not None]) for split in TEST_SPLITS}
        wins = []
        for row in main:
            values = row['paired_deltas_px'][name]
            wins.append(values['A_holdout'] is not None and values['A_holdout'] < 0
                        and sum(values[split] is not None and values[split] < 0 for split in TEST_SPLITS[1:]) >= 2)
        aggregate_direction = (split_stats['A_holdout'] is not None and split_stats['A_holdout']['mean'] < 0
            and sum(split_stats[split] is not None and split_stats[split]['mean'] < 0
                    for split in TEST_SPLITS[1:]) >= 2)
        if len(main) < 2:
            status = 'unavailable_insufficient_independent_sessions'
        elif sum(wins) > len(main) / 2 and aggregate_direction:
            status = 'support'
        elif not any(wins):
            status = 'not_supported'
        else:
            status = 'mixed'
        comparisons[name] = dict(status=status, session_wins=sum(wins), session_denominator=len(main),
                                 split_delta_px=split_stats, evidence_aliases=[r['alias'] for r in main])
    enough = len(main) >= 2
    return dict(status='complete' if enough else 'insufficient_sessions', required_main_sessions=2,
                main_sessions=len(main), shortfall=max(0, 2 - len(main)), per_session=per_session,
                session_level_methods=methods, comparisons=comparisons,
                online_candidate='support' if enough and comparisons['F2_vs_F1']['status'] == 'support'
                    and comparisons['F2_vs_F0']['status'] == 'support' else
                    ('not_supported' if enough else 'unavailable_insufficient_independent_sessions'),
                decision_rule='strict majority of main sessions: A_holdout and >=2 B split paired improvements; '
                              'same aggregate mean directions; online requires F2 over F1 and F0')


def collection_instructions(shortfall):
    return (f'还缺 {shortfall} 份独立、完整的真实 Classic AB 会话。每份在 Git Bash 从仓库根目录运行：\n'
            'R1_PY="E:/TMP/look2act-r1-env/Scripts/python.exe"\n'
            '"$R1_PY" scripts/experiment.py collect --config configs/classic.yaml --protocol AB\n'
            '在安全桌面完成 30 个目标；A 两轮正常注视，B 自然保持、左右缓慢转头、抬头/低头都实际执行。'
            '每份重新启动录制；若需要校准，请先重新校准并按现有显式校准参数加载，不要覆盖旧校准文件。'
            '程序自动生成不同 experiment_sessions/<会话目录>，无需重命名或复制。'
            '完成后保持原文件不变，运行 scripts/r5_replication.py audit 与 run-all；'
            '工具自动发现目录。不要提交会话目录或在含敏感按钮的页面试操作。')
