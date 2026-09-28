"""Synthetic R6 checks only: no camera, model weights, GUI actions or private sessions."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiment.r4_analysis import fit_ridge, predict
from src.experiment.r4_features import extract_features
from src.experiment.r6_shadow import (CalibrationCollector, ShadowConsumer, feature_from_result,
                                      fit_personal, load_mapping, mapping_context, mapping_from_dict,
                                      replay_shadow, save_mapping, synthetic_selftest)
from src.experiment.snapshots import result_snapshot
from src.tracker.observation import Observation
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult


class Clock:
    def __init__(self, now=0.):
        self.now = now

    def __call__(self):
        return self.now


def snapshot(left=(7., 2.), right=(27., 2.)):
    return dict(units='camera_px', frame_size=[100, 80],
                eye_landmarks={'33': [0., 0.], '133': [10., 0.],
                               '362': [20., 0.], '263': [30., 0.]},
                eyes={'left': dict(roi_origin=[0., 0.]), 'right': dict(roi_origin=[20., 0.])},
                classic_dark_centroid=dict(units='camera_px', left_frame=left, left_roi=left,
                                           right_frame=right,
                                           right_roi=None if right is None else [right[0] - 20, right[1]]))


def result(seq, source, *, session='s', valid=True, right=(27., 2.), continuity=0,
           candidate_continuity=0, published=None):
    data = snapshot(right=right)
    feature, reasons = extract_features(data)
    return TrackerResult((.5, .5) if valid else None, valid, 30., backend='classic',
                         face_detected=valid, raw_point=(.5, .5) if valid else None,
                         point_kind='observed' if valid else 'held', numeric_snapshot=data,
                         candidate_feature=None if feature['F2'] is None else tuple(feature['F2']),
                         candidate_rejection=reasons['F2'], candidate_continuity=candidate_continuity,
                         observation=Observation(session, seq, source, continuity, 'host_read_completed'),
                         published_at=source + .01 if published is None else published)


def samples():
    rows = []
    for segment in range(9):
        for i in range(10 + segment):
            x, y = (segment % 3) / 2, (segment // 3) / 2
            rows.append(dict(id=['s', len(rows) + 1], source_time=1 + len(rows) * .01,
                             segment=segment, epoch=0, target_norm=[.2 + .6*x, .2 + .6*y],
                             feature=[x + i*.003, y + i*.002, x*.7 + i*.004, y*.8 + i*.001]))
    return rows


def context():
    return mapping_context(SystemConfig(tracker_backend='classic', camera_width=100, camera_height=80),
                           (100, 80), (1000, 800), 1.)


def test_pipeline_one_detection_same_frame_and_switch(monkeypatch):
    face = SimpleNamespace(detected=True, frame_size=(100, 80),
                           left_eye_roi=np.full((20, 20, 3), 200, np.uint8),
                           right_eye_roi=np.full((20, 20, 3), 200, np.uint8),
                           left_eye_origin=(0, 0), right_eye_origin=(20, 0),
                           left_iris_center=None, right_iris_center=None,
                           pnp_points_2d={}, landmarks_68=np.zeros((68, 2)))
    face.landmarks_68[36] = [0, 0]
    face.landmarks_68[39] = [10, 0]
    face.landmarks_68[42] = [20, 0]
    face.landmarks_68[45] = [30, 0]
    face.left_eye_roi[7:11, 5:9] = 0
    face.right_eye_roi[7:11, 5:9] = 0
    calls = []
    detector = SimpleNamespace(detect=lambda frame: (calls.append(frame[0, 0, 0]), face)[1])
    pipe = TrackerPipeline('', SystemConfig(tracker_backend='classic'))
    pipe.face_detector = detector
    pipe.head_pose_estimator = SimpleNamespace(estimate=lambda _: pytest.fail('extra PnP'))
    frame = np.zeros((80, 100, 3), np.uint8)
    ordinary = pipe.process_frame(frame, captured_at=1.)
    assert ordinary.numeric_snapshot is None and ordinary.candidate_feature is None
    pipe.numeric_observation_enabled = True
    shadow = pipe.process_frame(frame, captured_at=2.)
    assert len(calls) == 2  # exactly one detector call per frame
    assert shadow.valid and shadow.candidate_feature == pytest.approx(extract_features(shadow.numeric_snapshot)[0]['F2'])
    assert ordinary.raw_point == shadow.raw_point
    face.right_eye_roi = None
    missing = pipe.process_frame(frame, captured_at=3.)
    assert missing.valid and missing.candidate_feature is None  # Classic single-eye remains valid
    assert len(calls) == 3
    face.right_eye_roi = np.full((20, 20, 3), 200, np.uint8)
    monkeypatch.setattr('src.experiment.r4_features.extract_features',
                        lambda *_: (_ for _ in ()).throw(RuntimeError('synthetic')))
    fallback = pipe.process_frame(frame, captured_at=4.)
    assert fallback.valid and fallback.candidate_feature is None
    assert fallback.candidate_rejection == 'candidate_extract_RuntimeError'


def test_producer_candidate_continuity_catches_overwritten_f2_loss():
    pipe = TrackerPipeline('', SystemConfig(tracker_backend='classic'))
    pipe.numeric_observation_enabled = True
    pipe._running = True
    published = []
    pipe.observation_sink = published.append
    a = result(1, 1., session=pipe.session_id)
    b = result(2, 1.05, session=pipe.session_id, right=None)
    c = result(3, 1.1, session=pipe.session_id)
    for item in (a, b, c):
        pipe._publish_result(item)
    assert [p.candidate_continuity for p in published] == [0, 1, 1]
    state = pipe.get_dispatch_snapshot(clock=lambda: 1.11)
    assert state['latest_candidate_valid'] and state['candidate_continuity'] == 1


def test_f2_personal_math_matches_r4_and_training_is_frozen():
    training = samples()
    model = fit_personal(training, context())
    rows = [dict(features={'F2': s['feature']}, target_norm=s['target_norm'],
                 segment=s['segment'], epoch=s['epoch']) for s in training]
    reference = fit_ridge(rows, 'F2')
    assert reference['weight_sum'] == pytest.approx(1.)
    assert np.asarray(model.coefficients) == pytest.approx(np.asarray(reference['coef']))
    assert model.intercept == pytest.approx(reference['intercept'])
    candidate = [0.3, 0.4, 0.2, 0.35]
    expected = predict(reference, [dict(features={'F2': candidate})])[0]
    assert model.predict_norm(candidate) == pytest.approx(expected, abs=1e-12)
    original = model.to_dict()
    test_only = copy.deepcopy(training)
    test_only[0]['target_norm'] = [100., -100.]
    assert model.to_dict() == original  # no validation sample or display state enters fit
    with pytest.raises(ValueError, match='degenerate'):
        fit_personal([{**s, 'feature': [1., 1., 1., 1.]} for s in training], context())
    with pytest.raises(ValueError, match='fewer_than_10'):
        fit_personal(training[:-18], context())


@pytest.mark.parametrize('change', [
    {'schema_version': 99}, {'feature_schema': 'old_2d'}, {'mean': [0, 0]},
    {'scale': [1, 1, 0, 1]}, {'coefficients': [[0, 0]]},
    {'intercept': [math.inf, 0]}, {'lambda_': .001},
])
def test_mapping_rejects_incompatible_or_malformed(tmp_path, change):
    model = fit_personal(samples(), context())
    path = tmp_path / 'f2.json'
    save_mapping(path, model)
    loaded = load_mapping(path, context())
    assert loaded.to_dict() == model.to_dict()
    with pytest.raises(FileExistsError):
        save_mapping(path, model)
    bad = {**model.to_dict(), **change}
    with pytest.raises(ValueError):
        mapping_from_dict(bad, context())
    with pytest.raises(ValueError, match='context'):
        load_mapping(path, {**context(), 'display_size': [200, 100]})
    assert load_mapping(path, context()).model_id == model.model_id


def test_collector_uses_source_and_actual_paint_history_not_current_target():
    collector = CalibrationCollector('s')
    collector.bind_display((100, 80))
    events = [dict(kind='target_request', at=10., segment=0, epoch=0),
              dict(kind='target_painted', at=10., segment=0, epoch=0,
                   instructed_target=[20, 16], requested_motion='stable', protocol='A'),
              dict(kind='target_closed', at=12., segment=0, epoch=0),
              dict(kind='target_request', at=12., segment=1, epoch=0),
              dict(kind='target_painted', at=12.02, segment=1, epoch=0,
                   instructed_target=[50, 16], requested_motion='stable', protocol='A')]
    first = collector.consider(result(1, 11.9, published=12.1), events, 14.)
    assert first['segment'] == 0 and first['target_norm'] == [.2, .2]
    assert collector.consider(result(2, 11.99), events, 14.) is None  # ±50ms guard
    second = collector.consider(result(3, 12.6), events, 14.)
    assert second['segment'] == 1
    assert collector.consider(result(3, 12.6), events, 14.) is None
    assert collector.consider(result(4, 14.1), events, 14.) is None  # late source cut off
    events[-1]['planned_at'] = 12.
    events[-1]['duration_s'] = 2.
    assert collector.consider(result(4, 14.01), events, 20.) is None  # delayed UI cannot extend segment
    assert collector.rejections['segment_deadline_passed'] == 1
    events += [dict(kind='pause', at=13.), dict(kind='resume', at=20.),
               dict(kind='target_painted', at=20., segment=1, epoch=1,
                    planned_at=20., duration_s=2.,
                    instructed_target=[50, 16], requested_motion='stable', protocol='A')]
    assert collector.consider(result(5, 13.5), events, 21.) is None
    assert collector.consider(result(6, 20.7), events, 21.)['epoch'] == 1
    assert len(collector.samples) == 3
    missing = result(8, 20.8)
    missing.observation = None
    assert collector.consider(missing, events, 21.) is None
    missing = result(9, 20.8)
    missing.published_at = None
    assert collector.consider(missing, events, 21.) is None
    assert collector.rejections['invalid_identity'] >= 1
    assert collector.rejections['expired_before_publication'] >= 1


def _state(clock, *, continuity=0, candidate_continuity=0, valid=True, session='s'):
    return dict(checked_at=clock(), running=True, worker_alive=True, calibrating=False,
                session=session, continuity=continuity, candidate_continuity=candidate_continuity,
                latest_valid=valid, latest_candidate_valid=valid, latest_sequence=99)


def test_shadow_gate_duplicate_timeout_and_continuity_precheck():
    clock = Clock(1.)
    consumer = ShadowConsumer(250., clock)
    consumer.activate(fit_personal(samples(), context()))
    consumer.reset('s', 'start', 'free')
    clock.now = 1.1
    a = result(1, 1.08)
    good = consumer.consume(a, lambda: _state(clock))
    assert good['gate_state'] == 'new' and good['valid']
    assert consumer.consume(a, lambda: _state(clock))['gate_state'] == 'duplicate'
    clock.now = 1.15
    assert consumer.consume(a, lambda: _state(clock, candidate_continuity=1))['rejection_reason'] == 'candidate_continuity_changed'
    assert consumer.last is None
    b = result(2, 1.16, candidate_continuity=1)
    clock.now = 1.17
    assert consumer.consume(b, lambda: _state(clock, candidate_continuity=1))['valid']
    clock.now = 1.42
    assert consumer.consume(b, lambda: _state(clock, candidate_continuity=1))['rejection_reason'] == 'expired'
    assert consumer.last is None
    clock.now = 1.43
    assert not consumer.consume(result(3, 1.42, candidate_continuity=1),
                                lambda: _state(clock, candidate_continuity=1))['valid']  # long sampling gap
    clock.now = 1.44
    assert consumer.consume(result(4, 1.43, candidate_continuity=1),
                            lambda: _state(clock, candidate_continuity=1))['valid']
    bad = result(5, 1.45, right=None, candidate_continuity=2)
    clock.now = 1.46
    assert not consumer.consume(bad, lambda: _state(clock, candidate_continuity=2, valid=False))['valid']
    assert consumer.last is None
    clock.now = 1.47
    assert not consumer.consume(result(2, 1.2), lambda: _state(clock))['valid']  # out of order


def test_shadow_rechecks_after_mapping_and_keeps_unclipped_outside_point():
    clock = Clock(5.)
    consumer = ShadowConsumer(250., clock)
    model = fit_personal(samples(), context())
    consumer.activate(model)
    consumer.reset('s', 'start', 'free')
    clock.now = 5.05
    latest = result(1, 5.04)
    state = _state(clock)
    original = model.predict_norm
    class Delayed:
        model_id = model.model_id
        context = model.context
        def predict_norm(self, feature):
            state['candidate_continuity'] = 1  # failure published during calculation
            return original(feature)
    consumer.model = Delayed()
    record = consumer.consume(latest, lambda: dict(state))
    assert record['rejection_reason'] == 'candidate_continuity_changed'
    assert not record['visible'] and consumer.last is None

    consumer.reset('s', 'recover', 'free')
    class Outside:
        model_id = model.model_id
        context = model.context
        def predict_norm(self, feature):
            return (2., -.5)
    consumer.model = Outside()
    clock.now = 5.1
    outside = consumer.consume(result(2, 5.09, candidate_continuity=1),
                               lambda: _state(clock, candidate_continuity=1))
    assert outside['screen_point'] == [2000., -400.]
    assert outside['display_point'] is None and outside['rejection_reason'] == 'out_of_bounds'


@pytest.mark.parametrize('source,session,point', [
    (1.2, 's', (.5, .5)),  # future source time
    (0.9, 'old', (.5, .5)),
    (0.9, 's', (math.nan, .5)),
    (0.9, 's', (math.inf, .5)),
])
def test_shadow_rejects_future_old_session_and_nonfinite(source, session, point):
    clock = Clock(1.)
    consumer = ShadowConsumer(250., clock)
    consumer.activate(fit_personal(samples(), context()))
    consumer.reset('s', 'start', 'free')
    clock.now = 1.01
    item = result(1, source, session=session)
    item.gaze_point = point
    assert not consumer.consume(item, lambda: _state(clock))['valid']


def test_r3_replay_refuses_to_claim_r6_candidate_coverage(tmp_path):
    from src.experiment.session import replay

    session = tmp_path / 'synthetic'
    synthetic_selftest(session)
    with pytest.raises(ValueError, match='r6_shadow.py'):
        replay(session)


def test_r5_audit_explicitly_excludes_r6_session_type():
    from src.experiment.protocol import make_plan
    from src.experiment.r5_replication import _source_reasons

    meta = dict(session_type='f2_shadow_r6', synthetic=False, backend='classic',
                plan=make_plan((100, 80), 'AB'))
    painted = [dict(kind='target_painted', segment=i) for i in range(30)]
    assert 'not_standard_r3_session' in _source_reasons(meta, painted, [])


def test_shadow_replay_recomputes_and_detects_tampering(tmp_path):
    session = tmp_path / 'synthetic'
    report = synthetic_selftest(session)
    assert report['replay']['strictly_reproducible'] and report['replay']['compared'] == 5
    path = session / 'events.jsonl'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    changed = next(e for e in events if e['kind'] == 'shadow_consume' and e['screen_point'] is not None)
    changed['screen_point'][0] += 1.
    path.write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    replay = replay_shadow(session)
    assert not replay['strictly_reproducible'] and replay['mismatches']
