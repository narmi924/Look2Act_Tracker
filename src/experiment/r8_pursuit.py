"""R8 offline check: integrity of a recorded session plus two feasibility numbers from F2 only.

1. Pursuit-choice accuracy: which orbiting marker did the eyes follow, by motion correlation.
2. Dense pursuit calibration vs the 9-point fixation calibration, both tested on the second
   fixation round. Everything is numerical; images are only counted here, never analysed.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from src.experiment.frames import load_landmarks, read_frames_index
from src.experiment.protocol import target_at
from src.experiment.r4_analysis import fit_ridge, predict
from src.experiment.r4_features import extract_features
from src.experiment.recording import read_session

CHOICE_SKIP_S = .75  # eyes need time to find the highlighted marker; fixed before any real session
PATH_SETTLE_S = .5
LAGS_S = (0., .15)  # gaze follows the target with a delay; both values are reported, none selected


def load(directory):
    meta, events, issues = read_session(directory)
    if (meta.get('plan') or {}).get('selection') != 'R8':
        raise ValueError('not an R8 session')
    return meta, events, issues


def producer_rows(meta, events):
    params = meta['plan']['parameters']
    segments = meta['plan']['segments']
    rows = []
    for event in events:
        if event['kind'] != 'producer':
            continue
        result = event['result']
        obs = result.get('observation') or {}
        if not obs.get('session') or type(obs.get('sequence')) is not int:
            continue
        valid = result.get('valid') is True and result.get('point_kind') == 'observed'
        features = extract_features(result.get('numeric_snapshot') or {})[0] if valid else {}
        label = target_at(obs['timestamp'], events, params['settling_s'], params['transition_guard_s'])
        segment = segments[label['segment']] if label.get('segment') is not None else None
        rows.append(dict(id=[obs['session'], obs['sequence']], event_id=event['event_id'], source_time=obs['timestamp'],
                         valid=valid, F2=features.get('F2'), status=label['status'], segment=label.get('segment'),
                         epoch=label.get('epoch'), stimulus=None if segment is None else segment['stimulus'],
                         protocol=None if segment is None else segment['protocol'],
                         round=None if segment is None else segment.get('round'),
                         instructed_target=None if segment is None else segment.get('instructed_target')))
    return rows


def moving_tracks(events):
    """Painted positions per (segment, epoch): path dots and choice markers, sorted by paint time."""
    paths, markers = defaultdict(list), defaultdict(list)
    for event in events:
        key = (event.get('segment'), event.get('epoch'))
        if event['kind'] == 'target_moved':
            paths[key].append((event['at'], event['x'], event['y']))
        elif event['kind'] == 'markers_moved':
            markers[key].append((event['at'], event['positions'], event['instructed']))
    return ({k: np.asarray(sorted(v)) for k, v in paths.items()},
            {k: (np.asarray([t for t, _, _ in sorted(v, key=lambda r: r[0])]),
                 np.asarray([p for _, p, _ in sorted(v, key=lambda r: r[0])]), v[0][2]) for k, v in markers.items()})


def interpolate(times, values, t, tolerance_s=.1):
    """Linear interpolation of painted positions; None outside the painted span."""
    if len(times) < 2 or t < times[0] - tolerance_s or t > times[-1] + tolerance_s:
        return None
    t = min(max(t, times[0]), times[-1])
    return np.asarray([np.interp(t, times, values[:, i]) for i in range(values.shape[-1])])


def gaze_signal(row, sign):
    f = row['F2']
    return np.asarray([sign[0] * (f[0] + f[2]) / 2, sign[1] * (f[1] + f[3]) / 2])


def feature_sign(rows):
    """Direction of F2 against the fixation targets of round 0; +1 when unknown."""
    sign = [1., 1.]
    use = [r for r in rows if r['valid'] and r['F2'] and r['protocol'] == 'A' and r['round'] == 0 and r['status'] == 'measurement']
    if len(use) < 20:
        return sign, len(use)
    f = np.asarray([[(r['F2'][0] + r['F2'][2]) / 2, (r['F2'][1] + r['F2'][3]) / 2] for r in use])
    t = np.asarray([r['instructed_target'] for r in use], dtype=float)
    for axis in (0, 1):
        if np.std(f[:, axis]) > 1e-9 and np.std(t[:, axis]) > 1e-9:
            sign[axis] = 1. if np.corrcoef(f[:, axis], t[:, axis])[0, 1] >= 0 else -1.
    return sign, len(use)


def _pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.
    return float(np.corrcoef(a, b)[0, 1])


def choice_accuracy(rows, markers, segments, sign, lag_s):
    trials, per_k = [], defaultdict(lambda: [0, 0])
    for (segment, epoch), (times, positions, instructed) in markers.items():
        frames = [r for r in rows if r['segment'] == segment and r['epoch'] == epoch and r['valid'] and r['F2']
                  and r['source_time'] >= times[0] + CHOICE_SKIP_S]
        if len(frames) < 10:
            trials.append(dict(segment=segment, epoch=epoch, status='insufficient_frames', frames=len(frames)))
            continue
        gaze = np.asarray([gaze_signal(r, sign) for r in frames])
        k = positions.shape[1]
        scores = []
        for i in range(k):
            track = np.asarray([interpolate(times, positions[:, i, :], r['source_time'] - lag_s) for r in frames], dtype=object)
            keep = [j for j, p in enumerate(track) if p is not None]
            if len(keep) < 10:
                scores.append(-1.)
                continue
            mx = np.asarray([track[j][0] for j in keep]); my = np.asarray([track[j][1] for j in keep])
            scores.append(.65 * _pearson(gaze[keep, 0], mx) + .35 * _pearson(gaze[keep, 1], my))
        order = np.argsort(scores)[::-1]
        predicted = int(order[0])
        margin = float(scores[order[0]] - scores[order[1]]) if k > 1 else None
        correct = predicted == instructed
        per_k[k][0] += correct
        per_k[k][1] += 1
        trials.append(dict(segment=segment, epoch=epoch, k=k, instructed=instructed, predicted=predicted, correct=correct,
                           best_score=float(scores[order[0]]), margin=margin, frames=len(frames)))
    scored = [t for t in trials if 'correct' in t]
    return dict(lag_s=lag_s, trials=len(trials), scored=len(scored),
                accuracy=None if not scored else sum(t['correct'] for t in scored) / len(scored),
                per_k={str(k): dict(correct=v[0], trials=v[1]) for k, v in per_k.items()},
                margin_median=None if not scored else float(np.median([t['margin'] for t in scored if t['margin'] is not None])),
                per_trial=trials)


def _ridge_rows(frames, targets, size):
    return [dict(features={'F2': r['F2']}, target_norm=[t[0] / size[0], t[1] / size[1]],
                 segment=r['segment'], epoch=r['epoch']) for r, t in zip(frames, targets)]


def calibration_comparison(rows, paths, size, lag_s):
    """Fit F2 ridge on (a) round-0 fixations, (b) pursuit paths, (c) both; test on round-1 fixations."""
    fix0 = [r for r in rows if r['valid'] and r['F2'] and r['protocol'] == 'A' and r['round'] == 0 and r['status'] == 'measurement']
    fix1 = [r for r in rows if r['valid'] and r['F2'] and r['protocol'] == 'A' and r['round'] == 1 and r['status'] == 'measurement']
    path_frames, path_targets = [], []
    for r in rows:
        if not (r['valid'] and r['F2'] and r['stimulus'] == 'path' and r['status'] == 'measurement'):
            continue
        track = paths.get((r['segment'], r['epoch']))
        if track is None:
            continue
        target = interpolate(track[:, 0], track[:, 1:], r['source_time'] - lag_s)
        if target is not None:
            path_frames.append(r)
            path_targets.append(target)
    sets = {'fixation_round0': _ridge_rows(fix0, [r['instructed_target'] for r in fix0], size),
            'pursuit_paths': _ridge_rows(path_frames, path_targets, size)}
    sets['both'] = sets['fixation_round0'] + sets['pursuit_paths']
    test = _ridge_rows(fix1, [r['instructed_target'] for r in fix1], size)
    out = dict(lag_s=lag_s, test_n=len(test), train={})
    for name, train in sets.items():
        if len(train) < 20 or len(test) < 10 or len({(r['segment'], r['epoch']) for r in train}) < 2:
            out['train'][name] = dict(n=len(train), status='insufficient')
            continue
        model = fit_ridge(train, 'F2')
        errors = np.linalg.norm((predict(model, test) - np.asarray([r['target_norm'] for r in test])) * np.asarray(size), axis=1)
        out['train'][name] = dict(n=len(train), presentations=model['presentations'],
                                  holdout_mean_px=float(errors.mean()), holdout_median_px=float(np.median(errors)),
                                  holdout_p95_px=float(np.percentile(errors, 95)))
    return out


def integrity(meta, events, rows, directory):
    index = read_frames_index(directory)
    landmarks = load_landmarks(directory)
    missing_png = sum(not (Path(directory) / f).is_file() for record in index for f in record['files'].values())
    by_kind = Counter()
    for event in events:
        if event['kind'] in ('target_moved', 'markers_moved', 'target_painted', 'producer'):
            by_kind[event['kind']] += 1
    valid_ids = {tuple(r['id']) for r in rows if r['valid']}
    frames = meta.get('frames') or {}
    landmark_index_consistent = landmarks.shape[0] == len(index) and all(
        r.get('landmark_record') == i for i, r in enumerate(index))
    artifacts_ok = bool(missing_png == 0 and landmark_index_consistent and frames.get('writer_error') is None
                        and not frames.get('writer_still_alive') and not frames.get('unwritten_at_close'))
    return dict(session_type=meta.get('session_type'), complete=bool(meta.get('complete')), write_lost=meta.get('write_lost'),
                frames=meta.get('frames'), events=dict(by_kind), producers=len(rows), valid_producers=len(valid_ids),
                with_f2=sum(1 for r in rows if r['F2']), image_records=len(index), missing_png=missing_png,
                valid_producers_with_images=len(valid_ids & {tuple(r['observation']) for r in index}),
                landmark_records=int(landmarks.shape[0]),
                landmark_index_consistent=landmark_index_consistent, artifacts_ok=artifacts_ok,
                segments=dict(planned=len(meta['plan']['segments']),
                              painted=len({e['segment'] for e in events if e['kind'] == 'target_painted'}),
                              by_stimulus=dict(Counter(s['stimulus'] for s in meta['plan']['segments']))))


def check(directory):
    directory = Path(directory)
    meta, events, issues = load(directory)
    rows = producer_rows(meta, events)
    paths, markers = moving_tracks(events)
    size = meta['screen_size']
    sign, sign_n = feature_sign(rows)
    report = dict(session=directory.name, issues=issues, integrity=integrity(meta, events, rows, directory),
                  feature_sign=dict(sign=sign, fixation_frames=sign_n),
                  choice={f'lag_{int(l * 1000)}ms': choice_accuracy(rows, markers, meta['plan']['segments'], sign, l) for l in LAGS_S},
                  calibration={f'lag_{int(l * 1000)}ms': calibration_comparison(rows, paths, size, l) for l in LAGS_S},
                  interpretation='instructed stimuli are not independent gaze truth; F2 only; images counted, not analysed')
    (directory / 'r8_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
    return report


def brief(report):
    i = report['integrity']
    lines = [f"session {report['session']}  complete={i['complete']}  issues={report['issues']}  artifacts_ok={i['artifacts_ok']}"]
    lines.append(f"producers {i['producers']} (valid {i['valid_producers']}, F2 {i['with_f2']}), images {i['image_records']} "
                 f"(missing png {i['missing_png']}), landmarks {i['landmark_records']}, painted {i['segments']['painted']}/{i['segments']['planned']}")
    if i.get('frames'):
        lines.append(f"frame writer: {i['frames']}")
    for key, c in report['choice'].items():
        lines.append(f"choice {key}: accuracy {c['accuracy']} over {c['scored']}/{c['trials']} trials, margin median {c['margin_median']}, per_k {c['per_k']}")
    for key, c in report['calibration'].items():
        cells = ', '.join(f"{name}: n={v['n']}" + (f" holdout mean {v['holdout_mean_px']:.1f} px" if 'holdout_mean_px' in v else ' insufficient')
                          for name, v in c['train'].items())
        lines.append(f"calibration {key} (test n={c['test_n']}): {cells}")
    return '\n'.join(lines)


# --- synthetic session for tests and the selftest --------------------------------------------------
def write_synthetic_r8_session(path, *, noise_px=.3, lag_s=.12, fps=15, size=(1280, 800), seed=3, with_images=True):
    """An R8-shaped session whose dark centroids follow the stimulus linearly; marked synthetic."""
    from src.experiment.frames import FrameWriter
    from src.experiment.protocol import make_plan
    from src.experiment.r8_protocol import stimulus_at
    from src.experiment.recording import write_json
    from types import SimpleNamespace
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    plan = make_plan(size, 'R8')
    corners = {'33': np.array([560., 470.]), '133': np.array([606., 470.]), '362': np.array([664., 470.]), '263': np.array([712., 470.])}
    meta = dict(schema_version=1, experiment_id='r8-synthetic', complete=True, synthetic=True, session_type='r8_pursuit_v1',
                backend='classic', screen_size=list(size), plan=plan, pnp={}, calibration=None,
                config=dict(max_observation_age_ms=250.), monotonic_origin_s=100.)
    events = []
    emit = lambda kind, at, **payload: events.append(dict(kind=kind, at=at, event_id=len(events) + 1, **payload))
    emit('start', 100.)
    writer = FrameWriter(path) if with_images else None
    sequence = 0
    gaze_history = []  # (time, target position) so the eye can lag the stimulus
    for segment in plan['segments']:
        start = 100. + segment['planned_offset_s']
        emit('target_request', start, segment=segment['segment'], epoch=0, reason='schedule')
        emit('target_painted', start + .02, epoch=0, **segment)
        ticks = int(segment['duration_s'] * fps)
        for j in range(ticks):
            at = start + .02 + j / fps
            phase = at - start
            stimulus = stimulus_at(segment, phase, size)
            if segment['stimulus'] == 'path':
                emit('target_moved', at, segment=segment['segment'], epoch=0, phase_s=phase, x=stimulus['dot'][0], y=stimulus['dot'][1])
                looked = stimulus['dot']
            elif segment['stimulus'] == 'choice':
                emit('markers_moved', at, segment=segment['segment'], epoch=0, phase_s=phase, instructed=stimulus['instructed'],
                     positions=stimulus['markers'])
                looked = stimulus['markers'][stimulus['instructed']]
            else:
                looked = stimulus['dot']
            gaze_history.append((at, np.asarray(looked, dtype=float)))
            source_time = at + .004
            lagged = [p for t, p in gaze_history if t <= source_time - lag_s]
            target = lagged[-1] if lagged else gaze_history[0][1]
            gx, gy = target[0] / size[0] - .5, target[1] / size[1] - .5
            sequence += 1
            snapshot = dict(units='camera_px', frame_size=list((1280, 720)), eye_landmarks={k: v.tolist() for k, v in corners.items()},
                            eyes={}, classic_dark_centroid=dict(units='camera_px'))
            for side, (a, b) in (('left', ('33', '133')), ('right', ('362', '263'))):
                mid = (corners[a] + corners[b]) / 2
                pupil = mid + np.array([10. * gx, 4. * gy]) + rng.normal(0, noise_px, 2)
                origin = np.floor(corners[a] - [0, 10])
                snapshot['eyes'][side] = dict(roi_origin=origin.tolist(), iris_center=pupil.tolist(), iris_status='available')
                snapshot['classic_dark_centroid'][side + '_frame'] = pupil.tolist()
                snapshot['classic_dark_centroid'][side + '_roi'] = (pupil - origin).tolist()
            observation = dict(session='synthetic-tracker', sequence=sequence, timestamp=source_time, continuity=0)
            emit('producer', source_time + .003, result=dict(observation=observation, valid=True, point_kind='observed',
                                                             raw_point=[.5, .5], published_at=source_time + .003,
                                                             numeric_snapshot=snapshot))
            if writer is not None:
                while writer.queue.qsize() > 32:  # the real camera paces submissions; the generator must not
                    time.sleep(.002)
                landmarks = np.zeros((478, 3), dtype=np.float32) + np.array([640., 400., 0.], dtype=np.float32)
                for key, value in corners.items():
                    landmarks[int(key), :2] = value
                frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
                writer.submit(SimpleNamespace(observation=SimpleNamespace(**observation), published_at=source_time + .003, valid=True),
                              frame, landmarks)
        emit('target_closed', start + segment['duration_s'], segment=segment['segment'], epoch=0, outcome='completed_after_paint')
    emit('end', 100. + plan['duration_s'], reason='protocol_complete')
    if writer is not None:
        meta['frames'] = writer.close()
    meta['written'] = len(events)
    write_json(path / 'session.json', meta)
    (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    return meta, events
