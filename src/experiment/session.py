"""Frozen session inputs and replay; no camera/model initialization on replay."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np
from src.calibration.calibrator import CalibrationModule
from src.experiment.recording import encode, read_session, write_json
from src.experiment.consumer import Consumer
from src.experiment.snapshots import restore_result
from src.tracker.pipeline import SystemConfig
from src.vision.head_pose import _MODEL_POINTS_3D, _REQUIRED_KEYS
from src.runtime_paths import resource_path

FLOAT_ATOL = 1e-8
FLOAT_RTOL = 1e-10


def checksum(value):
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def calibration_from_snapshot(snapshot):
    if snapshot is None:
        return None
    if checksum(snapshot['data']) != snapshot['sha256']:
        raise ValueError('calibration checksum mismatch')
    cal = CalibrationModule()
    cal.set_calibration_data(snapshot['data'])
    return cal


def config_snapshot(config):
    settings = asdict(config)
    for key, value in settings.items():
        if isinstance(value, str) and ('path' in key or 'checkpoint' in key):
            settings[key] = Path(value).name
    return settings


def metadata(config, calibrator, screen_size, plan, origin, *, synthetic=False):
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[2],
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return 'unknown'
    settings = config_snapshot(config)
    # Config paths identify artifacts, not a personal directory. Never replay-load them.
    versions = {'python': '.'.join(map(str, sys.version_info[:3]))}
    for package in ('numpy', 'opencv-python', 'opencv-contrib-python', 'mediapipe', 'onnxruntime', 'PyQt6', 'scipy'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    calibration = None
    if calibrator is not None and calibrator.is_calibrated:
        data = calibrator.get_calibration_data()
        calibration = dict(data=data, sha256=checksum(data))
    model = dict(name=Path(config.onnx_path if config.use_onnx else config.checkpoint_path).name,
                 sha256=None, used=config.normalized_backend != 'classic' and not synthetic)
    model_path = resource_path(config.onnx_path if config.use_onnx else config.checkpoint_path)
    model['artifact_status'] = 'not_used' if not model['used'] else ('present' if model_path.is_file() else 'missing_or_runtime_random_initialization')
    if model['used'] and model_path.is_file():
        digest = hashlib.sha256()
        with model_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        model['sha256'] = digest.hexdigest()
    return dict(experiment_id=uuid.uuid4().hex, created_utc=datetime.now(timezone.utc).isoformat(),
                code_commit=git('rev-parse', 'HEAD'), dirty=git('status', '--porcelain') != '',
                synthetic=synthetic, monotonic_origin_s=origin, time_unit='seconds',
                time_domain='host_perf_counter',
                timing_limits='read completion != exposure; paint submission != lit pixels; no hardware sync',
                config=settings, backend=config.normalized_backend, camera_requested=[config.camera_width, config.camera_height],
                camera_actual=None, screen_size=list(screen_size), screen_units='single_screen_px_existing_Qt_coordinates',
                screen_scale=None, plan=plan, calibration=calibration,
                model=model, software=versions, poll_period_s=.033,
                face_model=dict(name='MediaPipe FaceMesh', max_num_faces=1,
                                refine_landmarks=config.normalized_backend == 'classic',
                                eye_landmarks_source='existing_float32_68_point_subset_in_camera_pixels',
                                deep_crop='eye_contour_bbox_center; side=max(2*max(width,height),40); black_border_padding; resize_to_eye_crop_size',
                                pnp_pixels='existing_clipped_0_to_width_minus_1_and_height_minus_1'),
                pnp=dict(keys=list(_REQUIRED_KEYS), model_points_mm=_MODEL_POINTS_3D.tolist(),
                         model_source='existing_generic_face_model_not_subject_measurement',
                         intrinsics_source='focal_length_approx_image_width_zero_distortion',
                         coordinates='camera_pixels; cv2.solvePnP_ITERATIVE; RQDecomp3x3_degrees'),
                geometry_source='configuration_not_measured_physical_calibration',
                replay_tolerance=dict(atol=FLOAT_ATOL, rtol=FLOAT_RTOL))


class VirtualClock:
    def __init__(self, now=0.):
        self.now = now

    def __call__(self):
        return self.now


COMPARE_FIELDS = ('observation_id', 'source_valid', 'gate_state', 'gate_reason', 'reset',
                  'raw_point', 'calibrated_point', 'smoothed_point', 'display_point',
                  'calibrated_in_bounds', 'smoothed_in_bounds', 'screen_rejection',
                  'dispatch_rejection', 'dispatch_allowed', 'processing_status')


def equivalent(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, int) and isinstance(right, int):
        return left == right  # discrete identity is exact, not floating tolerance
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equivalent(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equivalent(a, b) for a, b in zip(left, right))
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return bool(np.isclose(left, right, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, equal_nan=True))
    return left == right


def replay(directory, speed=0., wait=None):
    """speed=0 computes immediately; optional wait affects presentation only."""
    if speed < 0:
        raise ValueError('speed must be nonnegative')
    meta, events, issues = read_session(directory)
    clock = VirtualClock(meta['monotonic_origin_s'])
    consumer = Consumer(SystemConfig(**meta['config']), calibration_from_snapshot(meta['calibration']),
                        meta['screen_size'], clock)
    producers = {}
    for event in events:
        if event['kind'] == 'producer':
            data = event['result']
            stamp = data['observation']
            if stamp:
                key = (stamp['session'], stamp['sequence'])
                if key in producers:
                    issues.append('duplicate_producer_identity')
                producers[key] = data
    comparisons, recomputed = [], []
    previous_at = None
    for event in events:
        at = event['at']
        if wait and speed and previous_at is not None:
            wait(max(0., at - previous_at) / speed)
        previous_at = at
        clock.now = at
        kind = event['kind']
        if kind == 'gap':
            consumer.reset(consumer.gate.session, 'record_gap')
        elif kind == 'context':
            consumer.reset(event['session'], event['reason'], event['active'])
        elif kind == 'consume':
            key = event['observation_id']
            data = None if key is None else producers.get(tuple(key))
            if key is not None and data is None:
                issues.append('missing_producer')
                consumer.reset(consumer.gate.session, 'missing_producer')
                comparisons.append(dict(event_id=event['event_id'], mismatch=['missing_producer']))
                continue
            result = None if data is None else restore_result(data)
            def snapshot():
                state = event['dispatch_state']
                if state is None:
                    # Missing check cannot be replaced by an always-allow state.
                    raise ValueError('required dispatch snapshot missing')
                clock.now = state['checked_at']
                return state
            try:
                actual = consumer.consume(result, snapshot)
                wrong = [field for field in COMPARE_FIELDS if not equivalent(actual[field], event[field])]
                recomputed.append(dict(kind='consume', at=at, event_id=event['event_id'], **actual))
            except (ValueError, KeyError, TypeError) as exc:
                wrong = ['recompute_error:' + type(exc).__name__]
                consumer.reset(consumer.gate.session, 'recompute_error')
            comparisons.append(dict(event_id=event['event_id'], mismatch=wrong))
    report = dict(compared=len(comparisons), mismatches=[c for c in comparisons if c['mismatch']],
                  issues=sorted(set(issues)), strictly_reproducible=not issues and all(not c['mismatch'] for c in comparisons),
                  tolerance=dict(atol=FLOAT_ATOL, rtol=FLOAT_RTOL))
    from src.experiment.summary import summarize
    summary = summarize(meta, events)
    summary['recorded_complete'] = summary['complete']
    summary['complete'] = bool(summary['complete'] and not issues)
    summary['replay'] = report
    write_json(Path(directory) / 'summary.json', summary)
    return summary, recomputed
