"""F2-only personal mapping and action-free shadow observation consumer.

This module has no OS action imports. Source times are VideoCapture.read return
times in the host monotonic clock domain, not exposure timestamps.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np

from src.experiment.protocol import target_at
from src.experiment.r4_analysis import RIDGE_LAMBDA, fit_ridge, predict
from src.experiment.r4_features import EYE_CORNERS, MIN_EYE_WIDTH_PX, extract_features
from src.experiment.recording import read_session, write_json
from src.experiment.session import VirtualClock, equivalent
from src.experiment.snapshots import observation_id, restore_result
from src.tracker.observation import Observation, ObservationGate, ObservationState, dispatch_rejection


FEATURE_SCHEMA = 'r4_f2_dark_local_v1'
FEATURE_DEFINITION = dict(order=['left_rx', 'left_ry', 'right_rx', 'right_ry'],
                          eye_corners={side: list(pair) for side, pair in EYE_CORNERS.items()},
                          units='camera_px_local_eye_width', min_eye_width_px=MIN_EYE_WIDTH_PX,
                          source='same_frame_dark_centroid_and_ROI')
MAPPING_SCHEMA = 1
REPLAY_ATOL_PX = 1e-6
REPLAY_RTOL = 1e-10
MIN_SAMPLES_PER_TARGET = 10
LATE_WAIT_S = .25


def feature_from_result(result):
    """Use the exact R4 extractor on the numerical snapshot from this frame."""
    if result is None or not result.valid:
        return None, 'producer_invalid'
    features, reasons = extract_features(result.numeric_snapshot)
    return features['F2'], reasons['F2']


def mapping_context(config, camera_size, display_size, dpr, *, screen_origin=(0, 0),
                    window_origin=(0, 0), logical_dpi=None):
    if (camera_size is None or display_size is None or
            len(camera_size) != 2 or len(display_size) != 2 or
            any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                for v in (*camera_size, *display_size, dpr)) or
            len(screen_origin) != 2 or len(window_origin) != 2 or
            any(not math.isfinite(v) for v in (*screen_origin, *window_origin)) or
            logical_dpi is not None and (not math.isfinite(logical_dpi) or logical_dpi <= 0)):
        raise ValueError('invalid_camera_or_display_geometry')
    return dict(camera_actual=list(camera_size), display_size=list(display_size),
                display_bounds=[0, 0, display_size[0] - 1, display_size[1] - 1],
                screen_origin=list(screen_origin), window_origin=list(window_origin),
                logical_dpi=logical_dpi,
                device_pixel_ratio=float(dpr), camera_index=config.camera_index,
                camera_backend=config.camera_backend, eye_crop_size=config.eye_crop_size,
                min_detection_confidence=config.min_detection_confidence,
                min_tracking_confidence=config.min_tracking_confidence,
                face_refine_landmarks=True,
                roi_convention='original_unscaled_ROI_origin_top_left_frame_pixels',
                coordinate_system='single_Qt_window_logical_px_normalized_by_width_height')


@dataclass(frozen=True)
class F2Mapping:
    model_id: str
    mean: tuple
    scale: tuple
    coefficients: tuple
    intercept: tuple
    context_json: str
    presentations: int
    sample_count: int
    sample_sha256: str
    rank: int
    condition: float | None
    training_rmse_norm: float
    code_commit: str

    @property
    def context(self):
        return json.loads(self.context_json)

    def to_dict(self):
        return dict(schema_version=MAPPING_SCHEMA, model_id=self.model_id,
                    feature_schema=FEATURE_SCHEMA,
                    feature_definition=copy.deepcopy(FEATURE_DEFINITION),
                    mean=list(self.mean), scale=list(self.scale),
                    coefficients=[list(row) for row in self.coefficients],
                    intercept=list(self.intercept), lambda_=RIDGE_LAMBDA,
                    weight_definition='1/(G*n_presentation); total weight=1; intercept unpenalized',
                    objective='sum_i w_i ||z_i B + b - y_i||^2 + 0.01 ||B||^2',
                    training_presentations=self.presentations, training_samples=self.sample_count,
                    sample_sha256=self.sample_sha256, rank=self.rank, condition=self.condition,
                    training_rmse_norm=self.training_rmse_norm, context=self.context,
                    code_commit=self.code_commit)

    def predict_norm(self, feature):
        x = np.asarray(feature, dtype=float)
        if x.shape != (4,) or not np.isfinite(x).all():
            raise ValueError('invalid_f2_vector')
        output = (x - self.mean) / self.scale @ np.asarray(self.coefficients) + self.intercept
        if not np.isfinite(output).all():
            raise ValueError('nonfinite_prediction')
        return tuple(float(v) for v in output)


def _array(value, shape, name):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid_' + name) from exc
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError('invalid_' + name)
    return array


def mapping_from_dict(data, expected_context):
    if not isinstance(data, dict) or data.get('schema_version') != MAPPING_SCHEMA:
        raise ValueError('unsupported_f2_mapping_schema')
    if data.get('feature_schema') != FEATURE_SCHEMA or data.get('feature_definition') != FEATURE_DEFINITION:
        raise ValueError('incompatible_f2_feature_definition')
    if data.get('context') != expected_context:
        raise ValueError('incompatible_camera_or_display_context')
    if not isinstance(data.get('model_id'), str) or not data['model_id']:
        raise ValueError('invalid_model_id')
    if (data.get('weight_definition') != '1/(G*n_presentation); total weight=1; intercept unpenalized'
            or data.get('objective') != 'sum_i w_i ||z_i B + b - y_i||^2 + 0.01 ||B||^2'):
        raise ValueError('incompatible_ridge_definition')
    mean = _array(data.get('mean'), (4,), 'mean')
    scale = _array(data.get('scale'), (4,), 'scale')
    coef = _array(data.get('coefficients'), (4, 2), 'coefficients')
    intercept = _array(data.get('intercept'), (2,), 'intercept')
    if np.any(scale <= 0) or data.get('lambda_') != RIDGE_LAMBDA:
        raise ValueError('invalid_scale_or_lambda')
    if (type(data.get('training_presentations')) is not int or data['training_presentations'] < 9
            or type(data.get('training_samples')) is not int or data['training_samples'] < 90
            or type(data.get('rank')) is not int or not 1 <= data['rank'] <= 4
            or not isinstance(data.get('sample_sha256'), str) or len(data['sample_sha256']) != 64
            or any(ch not in '0123456789abcdef' for ch in data['sample_sha256'])):
        raise ValueError('invalid_training_provenance')
    condition = data.get('condition')
    if condition is not None and (not isinstance(condition, (int, float)) or not math.isfinite(condition)):
        raise ValueError('invalid_condition')
    rmse = data.get('training_rmse_norm')
    if not isinstance(rmse, (int, float)) or not math.isfinite(rmse) or rmse < 0:
        raise ValueError('invalid_training_rmse')
    return F2Mapping(data['model_id'], tuple(mean), tuple(scale),
                     tuple(tuple(row) for row in coef), tuple(intercept),
                     json.dumps(expected_context, sort_keys=True), data['training_presentations'],
                     data['training_samples'], data['sample_sha256'], data['rank'], condition,
                     float(rmse), str(data.get('code_commit', 'unknown')))


def load_mapping(path, expected_context):
    return mapping_from_dict(json.loads(Path(path).read_text(encoding='utf-8')), expected_context)


def save_mapping(path, mapping):
    path = Path(path)
    if path.exists():
        raise FileExistsError('mapping already exists')
    write_json(path, mapping.to_dict())


def fit_personal(samples, context, code_commit='unknown'):
    """A first-round-only immutable fit; rows follow R4's weighted ridge contract."""
    counts = Counter(row['segment'] for row in samples)
    if set(counts) != set(range(9)) or any(counts[i] < MIN_SAMPLES_PER_TARGET for i in range(9)):
        raise ValueError('fewer_than_10_unique_measurements_per_target')
    rows = [dict(features={'F2': list(s['feature'])}, target_norm=list(s['target_norm']),
                 segment=s['segment'], epoch=s['epoch']) for s in samples]
    X = _array([row['features']['F2'] for row in rows], (len(rows), 4), 'training_features')
    _array([row['target_norm'] for row in rows], (len(rows), 2), 'training_labels')
    if not np.any(np.std(X, axis=0) > 1e-12):
        raise ValueError('degenerate_f2_training_input')
    fit = fit_ridge(rows, 'F2')
    if fit['rank'] < 1 or not math.isclose(fit['weight_sum'], 1., abs_tol=1e-12):
        raise ValueError('invalid_ridge_fit')
    output = predict(fit, rows)
    if not np.isfinite(output).all():
        raise ValueError('nonfinite_ridge_fit')
    rmse = float(np.sqrt(np.mean(np.sum((output - [r['target_norm'] for r in rows]) ** 2, axis=1))))
    canonical = [dict(id=s['id'], source_time=s['source_time'], segment=s['segment'],
                      epoch=s['epoch'], feature=s['feature'], target_norm=s['target_norm']) for s in samples]
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    return F2Mapping(uuid.uuid4().hex, tuple(fit['mean']), tuple(fit['scale']),
                     tuple(tuple(row) for row in fit['coef']), tuple(fit['intercept']),
                     json.dumps(context, sort_keys=True), fit['presentations'], len(samples),
                     digest, fit['rank'], fit['condition'], rmse, code_commit)


class CalibrationCollector:
    def __init__(self, session, max_age_ms=250., sink=None):
        self.session = session
        self.max_age = max_age_ms / 1000.
        self.sink = sink
        self.seen = set()
        self.samples = []
        self.rejections = Counter()
        self.last_sequence = 0

    def consider(self, result, events, cutoff):
        obs = getattr(result, 'observation', None)
        reason = None
        if (not isinstance(obs, Observation) or obs.session != self.session or
                type(obs.sequence) is not int or obs.sequence < 1 or
                type(obs.continuity) is not int or obs.continuity < 0 or
                not isinstance(obs.timestamp, (int, float)) or isinstance(obs.timestamp, bool) or
                not math.isfinite(obs.timestamp) or
                not isinstance(obs.time_source, str) or not obs.time_source):
            reason = 'invalid_identity'
        elif obs.sequence in self.seen:
            reason = 'duplicate'
        elif obs.sequence <= self.last_sequence:
            reason = 'out_of_order'
        elif obs.timestamp > cutoff:
            reason = 'outside_calibration_cutoff'
        elif (not isinstance(result.published_at, (int, float)) or isinstance(result.published_at, bool)
              or not math.isfinite(result.published_at)
              or obs.timestamp > result.published_at or result.published_at - obs.timestamp > self.max_age):
            reason = 'expired_before_publication'
        if reason is None:
            self.seen.add(obs.sequence)
            self.last_sequence = obs.sequence
            if (not result.valid or result.candidate_feature is None or
                    np.asarray(result.candidate_feature).shape != (4,) or
                    not np.isfinite(result.candidate_feature).all()):
                reason = result.candidate_rejection or 'candidate_unavailable'
            else:
                label = target_at(obs.timestamp, events)
                if label['status'] != 'measurement' or label['segment'] not in range(9):
                    reason = 'label_' + label['status']
                else:
                    paint = max((e for e in events if e['kind'] == 'target_painted' and
                                 e['segment'] == label['segment'] and e['epoch'] == label['epoch']),
                                key=lambda e: e['at'])
                    if (paint.get('planned_at') is not None and paint.get('duration_s') is not None
                            and obs.timestamp >= paint['planned_at'] + paint['duration_s']):
                        reason = 'segment_deadline_passed'
                    elif any(e['kind'] in ('target_closed', 'target_request') and
                             paint['at'] < e['at'] <= obs.timestamp for e in events):
                        # A frame after a request/close cannot inherit the previous paint.
                        reason = 'target_no_longer_active_at_source_time'
                    else:
                        target = label['instructed_target']
                        size = self.display_size
                        sample = dict(id=[obs.session, obs.sequence], source_time=obs.timestamp,
                                      source_continuity=obs.continuity,
                                      candidate_continuity=result.candidate_continuity,
                                      segment=label['segment'], epoch=label['epoch'],
                                      feature=list(result.candidate_feature),
                                      target_norm=[target[0] / size[0], target[1] / size[1]])
                        self.samples.append(sample)
                        if self.sink:
                            self.sink('calibration_sample', result.published_at, **sample)
                        return sample
        self.rejections[reason] += 1
        if self.sink:
            at = result.published_at if isinstance(result.published_at, (int, float)) and math.isfinite(
                result.published_at) else (cutoff if math.isfinite(cutoff) else time.perf_counter())
            self.sink('calibration_reject', at,
                      observation_id=observation_id(result), reason=reason)
        return None

    def bind_display(self, size):
        self.display_size = tuple(size)


class ShadowConsumer:
    """A display-only consumer; there is deliberately no action callback."""
    def __init__(self, max_age_ms=250., clock=time.perf_counter, sink=None):
        self.clock, self.sink = clock, sink
        self.gate = ObservationGate(max_age_ms, clock)
        self.model = None
        self.last = None
        self.phase = 'prepare'

    def reset(self, session, reason, phase):
        self.gate.reset(session)
        self.last = None
        self.phase = phase
        if self.sink:
            self.sink('shadow_context', self.clock(), session=session, reason=reason, phase=phase)

    def activate(self, model, source_attempt_id=None):
        self.model = model
        if self.sink:
            self.sink('model_active', self.clock(), mapping=model.to_dict(),
                      source_attempt_id=source_attempt_id)

    def _check(self, obs, candidate_continuity, state):
        reason = dispatch_rejection(obs, self.gate.max_age, state, state['checked_at'])
        if reason is None and (candidate_continuity is None or
                               candidate_continuity != state.get('candidate_continuity')):
            reason = 'candidate_continuity_changed'
        if reason is None and not state.get('latest_candidate_valid'):
            reason = 'latest_candidate_unavailable'
        return reason

    def recheck_display(self, state_provider):
        if self.last is None or not self.last['visible']:
            return False, 'no_visible_candidate', None
        state = state_provider()
        reason = self._check(self.last['observation'], self.last['candidate_continuity'], state)
        if reason:
            self.gate.reject(reason)
            self.last = None
        return reason is None, reason, state

    def consume(self, result, state_provider):
        at = self.clock()
        obs = None if result is None else result.observation
        gate_state = self.gate.consume(result, now=at)
        record = dict(observation_id=observation_id(result), consumed_at=at,
                      source_observation=None if obs is None else asdict(obs),
                      published_at=None if result is None else result.published_at,
                      source_valid=None if result is None else result.valid,
                      feature_schema=FEATURE_SCHEMA, feature_vector=None,
                      mapping_id=None if self.model is None else self.model.model_id,
                      candidate_continuity=None if result is None else result.candidate_continuity,
                      gate_state=gate_state.value, gate_reason=self.gate.reason,
                      predicted_norm=None, screen_point=None, display_point=None,
                      in_bounds=None, valid=False, rejection_reason=None,
                      computed_at=None, feature_ms=None, mapping_ms=None,
                      producer_state=None, visible=False, phase=self.phase,
                      age_s=None if obs is None else at - obs.timestamp)
        if gate_state is ObservationState.DUPLICATE and self.last is not None:
            state = state_provider()
            record['producer_state'] = state
            reason = self._check(obs, result.candidate_continuity, state)
            if reason:
                self.gate.reject(reason)
                self.last = None
                record['rejection_reason'] = reason
            else:
                record['visible'] = self.last['visible']
                record['display_point'] = self.last['display_point']
                record['rejection_reason'] = 'duplicate'
        elif gate_state is not ObservationState.NEW:
            self.last = None
            record['rejection_reason'] = self.gate.reason
        elif self.model is None:
            self.last = None
            self.gate.reject('no_mapping')
            record['rejection_reason'] = 'no_mapping'
        else:
            t0 = self.clock()
            feature, reason = feature_from_result(result)
            record['feature_ms'] = (self.clock() - t0) * 1000
            if reason is None:
                try:
                    source_feature = np.asarray(result.candidate_feature, dtype=float)
                    if (source_feature.shape != (4,) or not np.isfinite(source_feature).all() or
                            not np.allclose(feature, source_feature, rtol=0, atol=1e-12)):
                        reason = 'producer_feature_mismatch'
                except (TypeError, ValueError):
                    reason = result.candidate_rejection or 'candidate_unavailable'
            if reason is None:
                record['feature_vector'] = list(feature)
                t1 = self.clock()
                try:
                    predicted = self.model.predict_norm(feature)
                    size = self.model.context['display_size']
                    screen = tuple(float(predicted[i] * size[i]) for i in (0, 1))
                    record['predicted_norm'] = list(predicted)
                    record['screen_point'] = list(screen)
                    record['in_bounds'] = all(0 <= screen[i] <= size[i] - 1 for i in (0, 1))
                    if record['in_bounds']:
                        record['display_point'] = list(screen)
                    else:
                        reason = 'out_of_bounds'
                except ValueError as exc:
                    reason = str(exc)
                record['mapping_ms'] = (self.clock() - t1) * 1000
            state = state_provider()  # locked producer snapshot immediately before display eligibility
            record['producer_state'] = state
            reason = reason or self._check(obs, result.candidate_continuity, state)
            record['computed_at'] = self.clock()
            if reason:
                self.gate.reject(reason)
                self.last = None
                record['rejection_reason'] = reason
            else:
                record['valid'] = True
                record['visible'] = self.phase == 'free'
                self.last = dict(observation=obs, candidate_continuity=result.candidate_continuity,
                                 visible=record['visible'], display_point=record['display_point'])
        if self.sink:
            self.sink('shadow_consume', at, **record)
        return record


def _verify_model_activation(meta, events, activation, producers):
    """Re-fit the sealed calibration support; never trust recorded coefficients alone."""
    issues = []
    saved = activation['mapping']
    context = meta.get('mapping_context')
    if context is not None and saved.get('context') != context:
        issues.append('mapping_context_mismatch')
    mode = meta.get('mode')
    attempt = activation.get('source_attempt_id')
    if mode == 'loaded_mapping_validation':
        return issues + (['loaded_mapping_has_calibration_source'] if attempt is not None else [])
    if mode != 'new_personal_calibration':
        return issues  # synthetic trace has no live calibration phase
    seals = [e for e in events if e['kind'] == 'calibration_sealed' and
             e.get('attempt_id') == attempt and e['event_id'] < activation['event_id']]
    if not attempt or not seals:
        return issues + ['missing_sealed_calibration_attempt']
    seal = seals[-1]
    sample_events = [e for e in events if e['kind'] == 'calibration_sample' and
                     e.get('attempt_id') == attempt and e['event_id'] < seal['event_id']]
    if len(sample_events) != seal.get('sample_count') or seal.get('queue_drops'):
        issues.append('calibration_support_or_queue_mismatch')
    history = [e for e in events if e.get('attempt_id') == attempt and
               e['kind'] in ('target_painted', 'target_request', 'target_closed',
                             'pause', 'resume', 'skip', 'end', 'gap')]
    samples = []
    for event in sample_events:
        sample = {k: event[k] for k in ('id', 'source_time', 'segment', 'epoch',
                                       'feature', 'target_norm')}
        samples.append(sample)
        source = producers.get(tuple(sample['id']))
        if source is None:
            issues.append('calibration_sample_missing_producer')
            continue
        source_result = restore_result(source)
        actual_feature, reason = feature_from_result(source_result)
        obs = source_result.observation
        if (reason is not None or obs is None or obs.timestamp != sample['source_time'] or
                not np.allclose(actual_feature, sample['feature'], rtol=0, atol=1e-12)):
            issues.append('calibration_sample_source_mismatch')
        # Recreate only the paint history known when this sample was accepted.
        known = [e for e in history if e['event_id'] < event['event_id']]
        label = target_at(sample['source_time'], known)
        target = label.get('instructed_target')
        size = saved['context']['display_size']
        paints = [e for e in known if e['kind'] == 'target_painted' and
                  e.get('segment') == label.get('segment') and e.get('epoch') == label.get('epoch')]
        paint = max(paints, key=lambda e: e['at']) if paints else None
        deadline_passed = (paint is not None and paint.get('planned_at') is not None and
                           paint.get('duration_s') is not None and
                           sample['source_time'] >= paint['planned_at'] + paint['duration_s'])
        closed = paint is not None and any(e['kind'] in ('target_closed', 'target_request') and
                                           paint['at'] < e['at'] <= sample['source_time'] for e in known)
        if (label['status'] != 'measurement' or label.get('segment') != sample['segment'] or
                label.get('epoch') != sample['epoch'] or target is None or
                not np.allclose([target[0] / size[0], target[1] / size[1]],
                                sample['target_norm'], rtol=0, atol=1e-12) or
                sample['source_time'] > seal['cutoff'] or deadline_passed or closed):
            issues.append('calibration_sample_label_mismatch')
    try:
        fitted = fit_personal(samples, saved['context'], meta.get('code_commit', 'unknown')).to_dict()
        fields = ('mean', 'scale', 'coefficients', 'intercept', 'lambda_',
                  'training_presentations', 'training_samples', 'sample_sha256',
                  'rank', 'condition', 'training_rmse_norm')
        if any(not equivalent(fitted[name], saved[name]) for name in fields):
            issues.append('frozen_fit_mismatch')
    except (KeyError, TypeError, ValueError) as exc:
        issues.append('frozen_fit_recompute_' + type(exc).__name__)
    return issues


def replay_shadow(directory):
    """Recompute F2, the gate, mapping and display from a virtual monotonic clock."""
    meta, events, issues = read_session(directory)
    if meta.get('session_type') != 'f2_shadow_r6':
        raise ValueError('not_an_r6_shadow_session')
    clock = VirtualClock(meta['monotonic_origin_s'])
    consumer = ShadowConsumer(meta['config']['max_observation_age_ms'], clock)
    producers = {}
    for event in events:
        if event['kind'] == 'producer':
            data = event['result']
            stamp = data.get('observation') or {}
            key = (stamp.get('session'), stamp.get('sequence'))
            if key in producers:
                issues.append('duplicate_producer_identity')
            producers[key] = data
    compared, mismatches = 0, []
    fields = ('gate_state', 'gate_reason', 'feature_vector', 'mapping_id',
              'predicted_norm', 'screen_point', 'display_point', 'in_bounds',
              'valid', 'rejection_reason', 'visible', 'candidate_continuity')
    for event in events:
        clock.now = event['at']
        if event['kind'] == 'gap':
            consumer.reset(consumer.gate.session, 'record_gap', consumer.phase)
        elif event['kind'] == 'shadow_context':
            consumer.reset(event['session'], event['reason'], event['phase'])
        elif event['kind'] == 'model_active':
            try:
                issues.extend(_verify_model_activation(meta, events, event, producers))
                consumer.activate(mapping_from_dict(event['mapping'], event['mapping']['context']))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                issues.append('invalid_model_activation_' + type(exc).__name__)
                consumer.model = None
        elif event['kind'] == 'shadow_consume':
            compared += 1
            key = event['observation_id']
            data = None if key is None else producers.get(tuple(key))
            if key is not None and data is None:
                issues.append('missing_producer')
                continue
            try:
                result = None if data is None else restore_result(data)
            except (KeyError, TypeError, ValueError):
                issues.append('invalid_producer_result')
                consumer.reset(consumer.gate.session, 'invalid_producer_result', consumer.phase)
                continue
            if result is not None:
                computed, reason = feature_from_result(result)
                if result.candidate_feature is not None:
                    try:
                        saved = np.asarray(result.candidate_feature, dtype=float)
                        if (reason is not None or saved.shape != (4,) or not np.isfinite(saved).all() or
                                not np.allclose(computed, saved, rtol=0, atol=1e-12)):
                            issues.append('producer_feature_mismatch')
                    except (TypeError, ValueError):
                        issues.append('producer_feature_mismatch')
            def state():
                snapshot = event['producer_state']
                if snapshot is None:
                    raise ValueError('missing_producer_state')
                clock.now = snapshot['checked_at']
                return snapshot
            try:
                actual = consumer.consume(result, state)
                wrong = [name for name in fields if not equivalent(actual[name], event[name])]
                if actual['screen_point'] is not None and event['screen_point'] is not None:
                    if not np.allclose(actual['screen_point'], event['screen_point'],
                                       rtol=REPLAY_RTOL, atol=REPLAY_ATOL_PX):
                        wrong.append('screen_point_tolerance')
                if wrong:
                    mismatches.append(dict(event_id=event['event_id'], fields=wrong))
            except (KeyError, ValueError, TypeError) as exc:
                mismatches.append(dict(event_id=event['event_id'], fields=['recompute_' + type(exc).__name__]))
        elif event['kind'] == 'shadow_paint':
            compared += 1
            def paint_state():
                snapshot = event.get('checked_state')
                if snapshot is None:
                    raise ValueError('missing_paint_state')
                clock.now = snapshot['checked_at']
                return snapshot
            try:
                visible, reason, _ = consumer.recheck_display(paint_state)
                stamp = None if consumer.last is None else consumer.last['observation']
                key = None if stamp is None else [stamp.session, stamp.sequence]
                actual_delay = None if stamp is None or not visible else event['at'] - stamp.timestamp
                wrong = [name for name, value in (
                    ('visible', visible), ('rejection_reason', reason),
                    ('observation_id', key), ('source_to_paint_s', actual_delay))
                    if not equivalent(value, event.get(name))]
                if wrong:
                    mismatches.append(dict(event_id=event['event_id'], fields=wrong))
            except (KeyError, ValueError, TypeError) as exc:
                mismatches.append(dict(event_id=event['event_id'], fields=['recompute_' + type(exc).__name__]))
    return dict(compared=compared, mismatches=mismatches, issues=sorted(set(issues)),
                strictly_reproducible=not mismatches and not issues,
                tolerance=dict(atol_px=REPLAY_ATOL_PX, rtol=REPLAY_RTOL))


def shadow_summary(directory):
    meta, events, issues = read_session(directory)
    producers = [e for e in events if e['kind'] == 'producer']
    consumed = [e for e in events if e['kind'] == 'shadow_consume']
    shown = [e for e in events if e['kind'] == 'shadow_paint' and e.get('visible')]
    samples = [e for e in events if e['kind'] == 'calibration_sample']
    new = [e for e in consumed if e['gate_state'] == 'new']
    valid_new = sum(e['valid'] for e in new)
    missing_new = sum(bool(e['rejection_reason']) and str(e['rejection_reason']).startswith(
        ('missing_', 'eye_width_', 'wrong_or_missing_')) for e in new)
    outside_new = sum(e['rejection_reason'] == 'out_of_bounds' for e in new)
    visible_new = sum(e['visible'] for e in new)
    sealed = [e for e in events if e['kind'] == 'calibration_sealed']
    activated = [e for e in events if e['kind'] == 'model_active']
    started = [e for e in events if e['kind'] == 'start' and e.get('phase') == 'calibrating']
    def stats(values):
        values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
        return None if not values else dict(n=len(values), median=float(np.median(values)),
                                            p95=float(np.percentile(values, 95)))
    def rate(items):
        return None if len(items) < 2 or items[-1]['at'] <= items[0]['at'] else (
            len(items) - 1) / (items[-1]['at'] - items[0]['at'])
    return dict(session_type=meta.get('session_type'), complete=meta.get('complete'),
                integrity_issues=issues, producer_count=len(producers), consume_polls=len(consumed),
                consumed_new=len(new), display_paints=len(shown),
                calibration_samples=len(samples), per_target_samples=dict(Counter(e['segment'] for e in samples)),
                calibration_duration_s=None if not started or not sealed else sealed[0]['at'] - started[0]['at'],
                fitting_elapsed_s=None if not sealed or not activated else activated[0]['at'] - sealed[0]['at'],
                candidate_new_denominator=len(new), valid_new=valid_new,
                valid_new_rate=None if not new else valid_new / len(new),
                missing_feature_new=missing_new,
                missing_feature_new_rate=None if not new else missing_new / len(new),
                outside_new=outside_new,
                outside_new_rate=None if not new else outside_new / len(new),
                visible_new=visible_new,
                visible_new_rate=None if not new else visible_new / len(new),
                producer_hz=rate(producers), consumer_new_hz=rate([e for e in consumed if e['gate_state'] == 'new']),
                display_hz=rate(shown), age_s=stats(e.get('age_s') for e in consumed),
                feature_ms=stats(e.get('feature_ms') for e in consumed),
                producer_feature_ms=stats((e.get('result') or {}).get('candidate_feature_ms') for e in producers),
                mapping_ms=stats(e.get('mapping_ms') for e in consumed),
                read_return_to_paint_submit_s=stats(e.get('source_to_paint_s') for e in shown),
                rejections=dict(Counter(e.get('rejection_reason') for e in consumed if e.get('rejection_reason'))),
                calibration_queue_drops=meta.get('calibration_queue_drops'),
                write_lost=meta.get('write_lost'), write_unconfirmed=meta.get('write_unconfirmed'),
                timing_limit=meta.get('timing_limits'))


def historical_consistency(directory):
    """Read-only, causal first-A-round fit against an existing R3/R5 AB session."""
    from src.experiment.r4_analysis import build_rows, fit_ridge, hash_file, predict, split_name

    directory = Path(directory)
    before = [hash_file(directory / name) for name in ('session.json', 'events.jsonl')]
    meta, events, issues = read_session(directory)
    if issues or meta.get('synthetic') or meta.get('backend') != 'classic' or (
            meta.get('plan') or {}).get('selection') != 'AB':
        raise ValueError('requires a complete real Classic AB source')
    rows, _ = build_rows(meta, events)
    for row in rows:
        row['split'] = split_name(row, meta['plan'])
    cutoff_events = [e['at'] for e in events if e['kind'] == 'target_request' and e.get('segment') == 9]
    if not cutoff_events:
        raise ValueError('missing first-round cutoff')
    cutoff = min(cutoff_events)
    train = [r for r in rows if r['split'] == 'A_train' and r['features']['F2'] is not None
             and r['source_time'] < cutoff]
    test = [r for r in rows if r['split'] in ('A_holdout', 'B_natural', 'B_yaw', 'B_pitch')
            and r['features']['F2'] is not None and r['source_time'] >= cutoff]
    samples = [dict(id=r['id'][1:], source_time=r['source_time'], segment=r['segment'],
                    epoch=r['epoch'], feature=r['features']['F2'], target_norm=r['target_norm']) for r in train]
    context = dict(display_size=meta['screen_size'], camera_actual=meta['camera_actual'],
                   source='historical_read_only_first_A_round')
    model = fit_personal(samples, context, meta.get('code_commit', 'unknown'))
    reference = fit_ridge(train, 'F2')
    expected = predict(reference, test) * np.asarray(meta['screen_size'])
    actual = np.asarray([model.predict_norm(r['features']['F2']) for r in test]) * np.asarray(meta['screen_size'])
    difference = np.abs(actual - expected)
    after = [hash_file(directory / name) for name in ('session.json', 'events.jsonl')]
    return dict(training_samples=len(train), training_presentations=model.presentations,
                validation_samples=len(test), first_validation_after_cutoff=all(r['source_time'] >= cutoff for r in test),
                max_abs_difference_px=float(difference.max()) if difference.size else None,
                same_support_ids=len(test), source_hashes_unchanged=before == after,
                strictly_consistent=bool(before == after and len(test) and difference.size and
                                         np.allclose(actual, expected, rtol=REPLAY_RTOL, atol=REPLAY_ATOL_PX)),
                support_note='all historical producer measurement frames; not live latest-result UI support')


def synthetic_selftest(directory):
    """Small complete numerical trace; no Qt, camera, detector, weights or OS actions."""
    from src.experiment.recording import Recorder
    from src.experiment.snapshots import result_snapshot
    from src.tracker.observation import Observation
    from src.tracker.pipeline import TrackerResult

    clock = VirtualClock(1.)
    context = dict(display_size=[100, 80], camera_actual=[100, 80], source='synthetic')
    samples = []
    for segment in range(9):
        for i in range(10):
            x, y = (segment % 3) / 2, (segment // 3) / 2
            samples.append(dict(id=['synthetic', segment * 10 + i + 1], source_time=segment * 2 + .6 + i * .05,
                                segment=segment, epoch=0,
                                feature=[x + i*.001, y + i*.002, x*.7 + i*.001, y*.8 + i*.002],
                                target_norm=[.2 + .6*x, .2 + .6*y]))
    model = fit_personal(samples, context, 'synthetic')
    recorder = Recorder(directory, dict(session_type='f2_shadow_r6', synthetic=True,
                                        monotonic_origin_s=1., config=dict(max_observation_age_ms=250.),
                                        timing_limits='read return and paint submit are host times'))
    consumer = ShadowConsumer(250., clock, recorder.emit)
    consumer.activate(model)
    consumer.reset('synthetic', 'start', 'free')
    latest = None
    candidate_continuity = 0
    for seq, left, right, valid in ((1, (7., 2.), (27., 2.), True),
                                    (2, (8., 2.), (26., 3.), True),
                                    (3, (8., 2.), None, False),
                                    (4, (7., 3.), (27., 2.), True)):
        clock.now = 1. + seq * .05
        snapshot = dict(units='camera_px', frame_size=[100, 80],
                        eye_landmarks={'33': [0., 0.], '133': [10., 0.],
                                       '362': [20., 0.], '263': [30., 0.]},
                        eyes={'left': dict(roi_origin=[0., 0.]), 'right': dict(roi_origin=[20., 0.])},
                        classic_dark_centroid=dict(units='camera_px', left_frame=left, left_roi=left,
                                                   right_frame=right,
                                                   right_roi=None if right is None else [right[0] - 20, right[1]]))
        feature, reason = extract_features(snapshot)
        f2 = feature['F2']
        if not valid:
            candidate_continuity += 1
        result = TrackerResult((.5, .5) if valid else None, valid, 20.,
                               face_detected=valid, raw_point=(.5, .5) if valid else None,
                               backend='classic', point_kind='observed' if valid else 'held',
                               observation=Observation('synthetic', seq, clock.now - .01,
                                                       candidate_continuity, 'host_read_completed'),
                               numeric_snapshot=snapshot, candidate_feature=None if f2 is None else tuple(f2),
                               candidate_rejection=reason['F2'], candidate_continuity=candidate_continuity,
                               published_at=clock.now - .005)
        latest = result
        recorder.emit('producer', result.published_at, result=result_snapshot(result))
        def state():
            return dict(checked_at=clock.now, running=True, worker_alive=True, calibrating=False,
                        session='synthetic', continuity=candidate_continuity,
                        candidate_continuity=candidate_continuity, latest_valid=result.valid,
                        latest_candidate_valid=result.valid and result.candidate_feature is not None,
                        latest_sequence=seq)
        consumer.consume(result, state)
        if seq == 1:
            consumer.consume(result, state)  # duplicate poll cannot add a new prediction
    recorder.close(True)
    result = replay_shadow(directory)
    return dict(session=Path(directory).name, replay=result,
                producer_count=4, last_observation_id=observation_id(latest))
