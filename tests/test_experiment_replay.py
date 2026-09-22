"""R3 numerical tests: no cameras, model weights, network or real OS actions."""
import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.calibration.calibrator import CalibrationModule
from src.experiment.consumer import Consumer, PollSchedule
from src.experiment.protocol import Protocol, make_plan, target_at
from src.experiment.recording import Recorder, encode, read_session, write_json
from src.experiment.session import VirtualClock, metadata, replay, equivalent
from src.experiment.snapshots import face_snapshot, result_snapshot
from src.experiment.synthetic import record_synthetic
from src.tracker.observation import Observation
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult
from src.tracker.screen_mapping import ScreenMapper
from tests.test_observation_safety import deep_pipeline, qapp


def calibration(matrix=None, method='affine'):
    cal = CalibrationModule()
    cal.set_calibration_data(dict(method=method, transform_matrix=matrix or [[1., 0., 0.], [0., 1., 0.]]))
    return cal


class Harness:
    def __init__(self, directory, monkeypatch, config=None, cal=None):
        self.clock = VirtualClock(10.)
        monkeypatch.setattr('src.tracker.pipeline.time.perf_counter', self.clock)
        self.config = config or SystemConfig(tracker_backend='deep')
        self.cal = cal or calibration()
        self.tracker = TrackerPipeline('', self.config)
        self.tracker._running = True
        self.tracker.session_id = 'test-session'
        meta = metadata(self.config, self.cal, (1000, 800), make_plan((1000, 800)), self.clock(), synthetic=True)
        self.recorder = Recorder(directory, meta)
        self.tracker.record_sink = self.recorder.emit
        self.consumer = Consumer(self.config, self.cal, (1000, 800), self.clock, self.recorder.emit)
        self.consumer.reset(self.tracker.session_id)

    def publish(self, at, point=(100., 100.), valid=True):
        self.clock.now = at
        result = TrackerResult(point, valid, 30., raw_point=point, backend=self.config.normalized_backend,
                               error_message=None if valid else 'synthetic_failure')
        result = self.tracker._finish_observation(result, self.tracker._stamp(at, 'host_read_completed'))
        self.tracker._publish_result(result)
        return result

    def consume(self):
        return self.consumer.consume(self.tracker.get_latest_result(),
                                     lambda: self.tracker.get_dispatch_snapshot(clock=self.clock))

    def finish(self):
        self.recorder.close()
        summary, actual = replay(self.recorder.directory)
        assert summary['replay']['strictly_reproducible'], summary['replay']
        return summary, actual


def test_same_frame_snapshot_detached_no_images(monkeypatch):
    pipeline, face, pose = deep_pipeline()
    face.landmarks_68 = np.arange(136, dtype=float).reshape(68, 2)
    face.left_eye_roi = np.zeros((5, 8, 3), dtype=np.uint8)
    face.left_eye_origin = (3, 4)
    face.pnp_points_2d = {'nose_tip': (1., 2.)}
    face.left_iris_center = None
    face.frame_size = (640, 480)
    pipeline.record_sink = lambda *a, **k: None
    result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), captured_at=10.)
    snapshot = result_snapshot(result)
    expected = copy.deepcopy(snapshot)
    face.landmarks_68[:] = 999
    face.pnp_points_2d['nose_tip'] = (999., 999.)
    pose.rotation_matrix[:] = 99
    pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8), captured_at=10.1)
    assert snapshot == expected
    assert snapshot['numeric_snapshot']['eye_landmarks']['33'] == [72., 73.]
    assert snapshot['numeric_snapshot']['eyes']['left']['roi_size'] == [8, 5]
    assert snapshot['numeric_snapshot']['eyes']['left']['iris_status'].startswith('unavailable')
    assert snapshot['observation']['timestamp'] == 10.
    json.dumps(encode(snapshot), allow_nan=False)
    with pytest.raises(TypeError):
        encode(np.zeros((3, 3, 3)))


def test_classic_keeps_pnp_and_centroids_without_extra_pose(monkeypatch):
    from tests.test_classic_pipeline import _classic_face_result
    pipeline = TrackerPipeline('', SystemConfig())
    face = _classic_face_result(detected=True)
    face.pnp_points_2d = {'nose_tip': (12., 34.)}
    pipeline.face_detector = SimpleNamespace(detect=lambda frame: face)
    pipeline.head_pose_estimator = SimpleNamespace(estimate=lambda *a: pytest.fail('must not add live Classic pose'))
    pipeline.record_sink = lambda *a, **k: None
    result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.valid
    assert result.numeric_snapshot['pnp_points']['nose_tip'] == (12., 34.)
    assert result.numeric_snapshot['classic_dark_centroid']['left_frame'] is not None
    assert result.numeric_snapshot['head_pose'] is None
    pipeline.record_sink = None
    assert pipeline.process_frame(None).numeric_snapshot is None


def test_zero_model_pose_is_not_measured_head_pose():
    pipeline, face, pose = deep_pipeline()
    pipeline.config.deep_pose_input = 'zero'
    pose.yaw = 12.
    pipeline.record_sink = lambda *a, **k: None
    result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.numeric_snapshot['head_pose']['yaw'] == 12.
    assert result.numeric_snapshot['head_pose']['origin'] == 'estimated_online'
    assert result.numeric_snapshot['model_pose_input_deg'] == [0., 0., 0.]


def test_target_assignment_uses_source_time_not_completed_time():
    plan = make_plan((1000, 800), 'A')
    events = [dict(kind='target_painted', at=10., epoch=0, **plan['segments'][0]),
              dict(kind='target_painted', at=12.2, epoch=0, **plan['segments'][1])]
    # Source at 11.9, inference ends 12.3: still belongs to old displayed target.
    assert target_at(11.9, events)['segment'] == 0
    assert target_at(12.19, events)['status'] == 'time_uncertain'
    assert target_at(12.3, events)['status'] == 'settling'
    assert target_at(9.9, events)['status'] == 'unavailable'


def test_protocol_skipped_targets_pause_resume_and_fixed_orders():
    plan = make_plan((1000, 800), 'AB')
    assert len(plan['segments']) == 30
    assert [s['target_id'] for s in plan['segments'][:9]] != [s['target_id'] for s in plan['segments'][9:18]]
    assert plan == make_plan((1000, 800), 'AB')
    events = []
    clock = VirtualClock(10.)
    protocol = Protocol(plan, lambda kind, at, **p: events.append(dict(kind=kind, at=at, **p)), clock)
    protocol.start()
    clock.now = 16.1  # first three targets never actually painted
    protocol.tick()
    protocol.painted()
    assert [e['segment'] for e in events if e['kind'] == 'skipped'] == [0, 1, 2]
    assert target_at(15., events)['status'] == 'unavailable'
    clock.now = 16.8
    protocol.pause()
    assert target_at(17., events)['status'] == 'unavailable'
    clock.now = 18.
    protocol.resume()
    protocol.tick()
    protocol.painted()
    assert target_at(18.2, events)['status'] == 'settling'
    clock.now = 18.6
    protocol.skip()
    protocol.painted()
    assert any(e['kind'] == 'skip' for e in events)


def test_replay_synthetic_speed_and_expected_values_not_copied(tmp_path):
    directory = tmp_path / 'synthetic'
    summary = record_synthetic(directory)
    assert summary['producer_count'] == 19
    assert summary['source_failure'] == dict(count=1, denominator=19, ratio=1/19)
    assert summary['duplicate_ticks'] == 1
    assert summary['replay']['compared'] == 20
    fast, actual1 = replay(directory, speed=20., wait=lambda dt: None)
    slow, actual2 = replay(directory, speed=.1, wait=lambda dt: None)
    assert fast['replay'] == slow['replay']
    assert [e['smoothed_point'] for e in actual1] == [e['smoothed_point'] for e in actual2]
    path = directory / 'events.jsonl'
    rows = path.read_text(encoding='utf-8').splitlines()
    for i, line in enumerate(rows):
        row = json.loads(line)
        if row['kind'] == 'consume' and row['smoothed_point']:
            row['smoothed_point'][0] += 50.
            rows[i] = json.dumps(row)
            break
    path.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    changed, _ = replay(directory)
    assert changed['replay']['mismatches'][0]['mismatch'] == ['smoothed_point']


@pytest.mark.parametrize('event', ['invalid', 'invalid_then_valid', 'new_valid'])
def test_dispatch_interleaving_record_and_replay(tmp_path, monkeypatch, event):
    h = Harness(tmp_path / event, monkeypatch)
    h.publish(10.125)
    original = h.cal.apply
    def processing(point):
        if event != 'new_valid':
            h.publish(10.15, valid=False)
        if event != 'invalid':
            h.publish(10.1875)
        return original(point)
    h.cal.apply = processing
    result = h.consume()
    assert result['dispatch_allowed'] == (event == 'new_valid')
    if event != 'new_valid':
        assert result['dispatch_rejection'] == 'producer_continuity_changed'
        assert h.consumer.mapper.ema._prev is None
    h.cal.apply = original
    h.publish(10.25, point=(300., 100.))
    recovered = h.consume()
    assert recovered['dispatch_allowed']
    if event != 'new_valid':
        assert recovered['smoothed_point'] == (300., 100.)
    h.finish()


def test_consumer_missing_duplicate_expired_pause_context(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'states', monkeypatch)
    assert h.consume()['gate_state'] == 'invalid'
    h.publish(10.125)
    assert h.consume()['gate_state'] == 'new'
    assert h.consume()['gate_state'] == 'duplicate'
    h.clock.now = 10.5
    expired = h.consume()
    assert expired['source_valid'] and expired['gate_reason'] == 'expired'
    h.consumer.reset(h.tracker.session_id, 'pause', active=False)
    h.publish(10.625)
    assert h.consume()['gate_reason'] == 'paused'
    h.consumer.reset(h.tracker.session_id, 'resume')
    h.publish(10.75)
    assert h.consume()['smoothed_point'] == (100., 100.)
    summary, _ = h.finish()
    assert summary['independent_consumed'] == 3
    assert summary['rejection_reasons_per_tick']['expired'] == 1


@pytest.mark.parametrize('point', [(-200., 100.), (900., 100.), (float('nan'), 100.), (float('inf'), 100.)])
def test_raw_mapping_rejections_recovery_replay(tmp_path, monkeypatch, point):
    h = Harness(tmp_path / 'points', monkeypatch, cal=calibration([[1., 0., 300.], [0., 1., 0.]]))
    h.publish(10.125, point)
    output = h.consume()
    if point[0] == -200.:
        assert output['calibrated_point'] == (100., 100.)
    elif point[0] == 900.:
        assert output['calibrated_point'] == (1200., 100.)
        assert output['screen_rejection'] == 'calibrated_out_of_bounds'
    else:
        assert output['gate_state'] == 'invalid'
    h.publish(10.25, (-100., 100.))
    assert h.consume()['dispatch_allowed']
    h.finish()


def test_nonlinear_real_calibration_replay(tmp_path, monkeypatch):
    # Existing polynomial basis [x, y, xy, x², y², 1]. No fit on evaluation samples.
    cal = calibration([[0., 0., 0., 1., 0., 0.], [0., 1., 0., 0., 0., 0.]], 'polynomial')
    h = Harness(tmp_path / 'polynomial', monkeypatch, cal=cal)
    values = []
    for i, x in enumerate((2., 4., 6.), 1):
        h.publish(10. + i * .125, (x, 100.))
        values.append(h.consume()['smoothed_point'][0])
    assert values == pytest.approx([4., 7.6, 16.12])
    h.finish()


def test_screen_mapper_measures_only_executed_stages_and_preserves_source():
    times = iter([10., 10.0002, 11., 11.0003])
    mapper = ScreenMapper(clock=lambda: next(times))
    result = TrackerResult((100., 100.), True, 30., raw_point=(100., 100.), backend='deep', timings={'model': 4.})
    out = mapper.process(result, SystemConfig(), calibration(), (1000, 800))
    assert out.processing_timings == pytest.approx(dict(calibration=.2, smoothing=.3))
    assert out.timings == result.timings == {'model': 4.}
    assert result.processing_timings == {}
    none = ScreenMapper().process(result, SystemConfig(smoother_type='none'), None, (1000, 800))
    assert none.processing_timings == {'calibration': None, 'smoothing': None}
    assert none.processing_status['smoothing'] == 'disabled'


def test_print_interval_does_not_select_filter_samples():
    def sample(interval):
        schedule = PollSchedule(period=.033, print_interval=interval)
        consumed, printed = [], []
        for t in np.arange(0., 1., .005):
            due, printing = schedule.due(t)
            if due: consumed.append(t)
            if printing: printed.append(t)
        return consumed, printed
    first, a = sample(.01)
    second, b = sample(.4)
    assert first == second
    assert len(a) != len(b)


def test_bounded_queue_overflow_and_nonblocking_producer(tmp_path):
    entered, release = threading.Event(), threading.Event()
    class Writer:
        def __enter__(self):
            entered.set()
            assert release.wait(3.)
            self.file = (tmp_path / 'bounded' / 'events.jsonl').open('w', encoding='utf-8')
            return self.file
        def __exit__(self, *args): self.file.close()
    rec = Recorder(tmp_path / 'bounded', {'synthetic': True}, capacity=1, opener=lambda p: Writer())
    assert entered.wait(3.)
    try:
        assert rec.emit('one', 1.)
        assert not rec.emit('two', 2.)
        assert rec.lost == 1 and rec.queue.qsize() == 1
    finally:
        release.set()
        rec.close()
    meta, events, issues = read_session(rec.directory)
    assert not meta['complete'] and meta['lost_event_range'] == [2, 2]
    assert issues == ['incomplete_session', 'write_lost']


def test_disk_failure_and_incomplete_marker(tmp_path):
    failed = threading.Event()
    def opener(path):
        failed.set()
        raise OSError('private-path-must-not-be-recorded')
    rec = Recorder(tmp_path / 'failed', {}, opener=opener)
    assert failed.wait(3.)
    rec.close()
    data = (rec.directory / 'session.json').read_text()
    assert 'private-path' not in data
    assert not json.loads(data)['complete']
    assert json.loads(data)['writer_error'] == 'OSError'


def test_prefix_tail_corruption_gap_and_absent_completion(tmp_path):
    directory = tmp_path / 'prefix'
    record_synthetic(directory)
    path = directory / 'events.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines()
    del lines[6]
    path.write_text('\n'.join(lines) + '\n{"broken":', encoding='utf-8')
    meta = json.loads((directory / 'session.json').read_text(encoding='utf-8'))
    meta['complete'] = False
    write_json(directory / 'session.json', meta)
    summary, _ = replay(directory)
    assert {'event_gap', 'corrupt_tail_or_record', 'incomplete_session'} <= set(summary['replay']['issues'])
    assert not summary['complete']
    assert not summary['replay']['strictly_reproducible']


def test_snapshot_checksum_and_privacy(tmp_path):
    directory = tmp_path / 'privacy'
    record_synthetic(directory)
    data = (directory / 'session.json').read_text(encoding='utf-8')
    assert 'Users' not in data and 'dataset_raw' not in data
    meta = json.loads(data)
    meta['calibration']['data']['transform_matrix'][0][0] += 1.
    write_json(directory / 'session.json', meta)
    with pytest.raises(ValueError, match='checksum'):
        replay(directory)
    import subprocess
    assert subprocess.run(['git', 'check-ignore', 'experiment_sessions/test/session.json'], capture_output=True).returncode == 0


def test_analyzer_units_unknown_and_no_bridging(tmp_path):
    from scripts.analyze_tracker_diagnostics import analyze_file
    legacy = tmp_path / 'legacy.csv'
    pd.DataFrame({'raw_x': [.1, .2], 'raw_y': [.2, .3]}).to_csv(legacy, index=False)
    out = analyze_file(legacy)
    assert out['raw_units'] == 'unknown'
    assert out['gate_state_counts'] == {'unknown': 2}
    current = tmp_path / 'current.csv'
    pd.DataFrame(dict(raw_x=[.1, .5, .9], raw_y=[.1, .5, .9], raw_units=['camera_normalized_feature'] * 3,
                      gate_state=['new', 'invalid', 'new'], session=['s']*3, sequence=[1, 2, 3],
                      continuity=[0, 0, 0], reset=[False, True, False])).to_csv(current, index=False)
    out = analyze_file(current)
    assert out['raw_step']['count'] == 0
    assert out['raw_step']['units'] == 'camera_normalized_feature'


def test_experiment_window_start_pause_resume_finish_offscreen(tmp_path, monkeypatch, qapp):
    from src.ui import experiment_window as module
    clock = VirtualClock(10.)
    monkeypatch.setattr(module, 'OUTPUT_ROOT', tmp_path / 'sessions')
    monkeypatch.setattr('src.tracker.pipeline.time.perf_counter', clock)
    started = []
    class FakePipeline(TrackerPipeline):
        def start(self):
            started.append(True)
            self._running = True
            self.actual_camera_size = (640, 480)
            self.head_pose_estimator = None
            return True
    window = module.ExperimentWindow(SystemConfig(), 'A', clock=clock, pipeline_factory=FakePipeline)
    window.resize(1000, 800)
    window.show()
    qapp.processEvents()
    assert not started and not (tmp_path / 'sessions').exists()
    window.start_recording()
    qapp.processEvents()
    assert started == [True]
    target_x, target_y = window.target['instructed_target']
    assert window.grab().toImage().pixelColor(target_x, target_y).name() == '#f2d35e'
    clock.now = 10.125
    result = TrackerResult((.2, .3), True, 30., raw_point=(.2, .3), backend='classic',
                           observation=Observation(window.pipeline.session_id, 1, clock(), 0))
    window.pipeline._publish_result(result)
    window.tick()
    clock.now = 10.2
    window.pause()
    assert window.protocol.paused_at == 10.2 and not window.consumer.active
    clock.now = 10.5
    window.resume()
    qapp.processEvents()
    assert window.protocol.epoch == 1 and window.consumer.active
    clock.now = 10.75
    window.finish()
    directory = window.recorder.directory
    window.close()
    summary, _ = replay(directory)
    assert summary['replay']['strictly_reproducible']
    assert summary['head_pose']['source'] == 'unavailable'
    assert summary['stages']['calibrated_point']['finite'] == 0
    assert not window.recorder.thread.is_alive()
    assert not window.pipeline.is_running()


def test_timing_labels_distinguish_missing_from_zero(qapp):
    from src.ui.tracking_page import TrackingPage
    page = TrackingPage()
    result = TrackerResult((0., 0.), True, 30., timings={'face_detection': 0.})
    result.processing_timings = dict(calibration=.2, smoothing=None)
    page._update_timing_labels(result)
    assert page.timing_labels['face_detection'].text() == '0.00 ms'
    assert page.timing_labels['calibration'].text() == '0.20 ms'
    assert page.timing_labels['smoothing'].text() == '—'
    assert page.timing_labels['head_pose'].text() == '—'
    page.close()


def test_csv_projection_contains_rejected_valid_source(tmp_path, monkeypatch):
    from scripts.diagnose_tracker import diagnostic_row
    h = Harness(tmp_path / 'csv', monkeypatch)
    h.publish(10.125)
    h.consume()
    h.clock.now = 10.5
    row = diagnostic_row(h.consume(), 2)
    assert row['source_valid'] is True and row['gate_state'] == 'invalid'
    assert row['gate_reason'] == 'expired'
    assert row['source_time'] == 10.125 and row['consume_time'] == 10.5
    assert row['source_time_source'] == 'host_read_completed'
    assert json.loads(row['processing_timings'])['smoothing'] is None
    h.finish()


def test_replay_missing_producer_and_no_calibration(tmp_path):
    directory = tmp_path / 'missing'
    record_synthetic(directory)
    path = directory / 'events.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    missing = next(row for row in rows if row['kind'] == 'producer')
    rows.remove(missing)
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n', encoding='utf-8')
    summary, _ = replay(directory)
    assert 'missing_producer' in summary['replay']['issues']
    assert not summary['replay']['strictly_reproducible']


def test_post_smoothing_outside_is_recorded_and_cleared(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'smooth', monkeypatch)
    h.publish(10.125)
    h.consume()
    monkeypatch.setattr(h.consumer.mapper.ema, 'update', lambda p: (-1., 100.))
    h.publish(10.25)
    event = h.consume()
    assert event['smoothed_point'] == (-1., 100.)
    assert event['display_point'] == (0., 100.)
    assert event['smoothed_in_bounds'] is False
    assert event['screen_rejection'] == 'smoothed_out_of_bounds'
    assert h.consumer.mapper.ema._prev is None
    h.recorder.close()
    # Injected filter is not in the frozen config: replay must detect, not copy it.
    summary, _ = replay(h.recorder.directory)
    assert summary['replay']['mismatches']


def test_delayed_result_records_old_target_not_current(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'delayed', monkeypatch)
    segments = h.recorder.metadata['plan']['segments']
    h.recorder.emit('target_painted', 10., **segments[0], epoch=0, planned_at=10.)
    h.publish(10.875)
    h.clock.now = 11.
    h.recorder.emit('target_painted', 11., **segments[1], epoch=0, planned_at=11.)
    h.clock.now = 11.0625
    assert h.consume()['dispatch_allowed']
    summary, _ = h.finish()
    assert summary['target_segments'][0]['segment'] == 0


def test_invalid_utf8_tail_preserves_complete_prefix(tmp_path):
    directory = tmp_path / 'utf8'
    record_synthetic(directory)
    original = (directory / 'events.jsonl').read_bytes().splitlines(keepends=True)
    with (directory / 'events.jsonl').open('ab') as stream:
        stream.write(b'\xff\xff\n')
    summary, _ = replay(directory)
    assert summary['replay']['compared'] == 20
    assert 'corrupt_tail_or_record' in summary['replay']['issues']
    assert summary['recorded_complete'] and not summary['complete']
    (directory / 'events.jsonl').write_bytes(b''.join(original[:-1]))
    truncated, _ = replay(directory)
    assert 'record_count_mismatch' in truncated['replay']['issues']
    assert not truncated['complete']


def test_failed_write_reports_unconfirmed_records(tmp_path):
    class BrokenWriter:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def write(self, line): raise OSError('synthetic disk full')
    recorder = Recorder(tmp_path / 'diskfull', {}, opener=lambda path: BrokenWriter())
    recorder.emit('start', 1.)
    recorder.close()
    assert recorder.metadata['write_unconfirmed'] == 1
    assert recorder.metadata['writer_error'] == 'OSError'
    assert not recorder.metadata['complete']


def test_existing_classic_filter_overshoot_replays_without_mock(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'classic_overshoot', monkeypatch, config=SystemConfig(tracker_backend='classic'),
                cal=calibration([[1000., 0., 0.], [0., 1000., 0.]]))
    rejections = []
    for i, x in enumerate([0.] * 10 + [.999] * 60, 1):
        h.publish(10. + i * .125, (x, .1))
        event = h.consume()
        if event['screen_rejection']:
            rejections.append(event)
    assert len(rejections) == 1
    assert rejections[0]['screen_rejection'] == 'smoothed_out_of_bounds'
    assert rejections[0]['calibrated_in_bounds'] is True
    assert rejections[0]['smoothed_point'][0] > 999.
    h.publish(h.clock() + .125, (.4, .1))
    assert h.consume()['smoothed_point'] == (400., 100.)
    h.finish()


def test_private_source_error_redacted_consistently_in_replay(tmp_path, monkeypatch):
    h = Harness(tmp_path / 'redacted', monkeypatch)
    h.clock.now = 10.125
    result = TrackerResult(None, False, 0., error_message='failed reading C:/Users/Private/model.onnx',
                           observation=Observation(h.tracker.session_id, 1, h.clock(), 0))
    h.tracker._publish_result(result)
    assert h.consume()['gate_reason'] == 'source_error_details_redacted'
    h.finish()
    assert 'Private' not in (h.recorder.directory / 'events.jsonl').read_text(encoding='utf-8')


@pytest.mark.parametrize('parameters', [{'dwell_s': -1.}, {'settling_s': float('nan')},
                                        {'b_durations': [1., 2.]}, {'a_rounds': 0},
                                        {'transition_guard_s': -1.}, {'unknown': 2}])
def test_protocol_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        make_plan((1000, 800), parameters=parameters)


def test_context_and_gate_use_exact_recorded_clock_instants(tmp_path):
    # Time advances between every call, exposing reset/consume double-clock reads.
    ticks = iter(10. + i * .001 for i in range(100))
    clock = lambda: next(ticks)
    config, cal = SystemConfig(tracker_backend='deep'), calibration()
    rec = Recorder(tmp_path / 'clock', metadata(config, cal, (1000, 800), make_plan((1000, 800)), 10., synthetic=True))
    consumer = Consumer(config, cal, (1000, 800), clock, rec.emit)
    consumer.reset('s')
    result = TrackerResult((100., 100.), True, 30., raw_point=(100., 100.), backend='deep',
                           observation=Observation('s', 1, 10.0015, 0), published_at=10.002)
    rec.emit('producer', 10.002, result=result_snapshot(result))
    def state():
        return dict(checked_at=clock(), running=True, worker_alive=True, calibrating=False,
                    session='s', continuity=0, latest_valid=True, latest_sequence=1)
    event = consumer.consume(result, state)
    assert event['gate_state'] == 'new'
    rec.close()
    summary, _ = replay(rec.directory)
    assert summary['replay']['strictly_reproducible'], summary['replay']


def test_actual_diagnostic_entry_print_interval_independent(tmp_path, monkeypatch, capsys):
    from scripts import diagnose_tracker as script
    clock = VirtualClock(10.)
    monkeypatch.setattr(script.time, 'perf_counter', clock)
    monkeypatch.setattr(script.time, 'sleep', lambda dt: setattr(clock, 'now', clock.now + dt))
    monkeypatch.setattr(script, 'Consumer', lambda config, cal, size: Consumer(config, cal, size, clock))
    class FakePipeline:
        session_id = 's'
        screen_geometry = SimpleNamespace(screen_w_px=1000, screen_h_px=800)
        def __init__(self, **kwargs): self.seq = 0
        def start(self): return True
        def stop(self): pass
        def is_running(self): return True
        def get_latest_result(self):
            self.seq += 1
            point = (100. + self.seq, 100.)
            return TrackerResult(point, True, 30., raw_point=point, backend='deep',
                                 observation=Observation('s', self.seq, clock(), 0), published_at=clock())
        def get_dispatch_snapshot(self):
            return dict(checked_at=clock(), running=True, worker_alive=True, calibrating=False,
                        session='s', continuity=0, latest_valid=True, latest_sequence=self.seq)
    monkeypatch.setattr(script, 'TrackerPipeline', FakePipeline)
    outputs, printing = [], []
    for i, interval in enumerate((.01, .5)):
        clock.now = 10.
        csv_path = tmp_path / f'diagnostic-{i}.csv'
        args = SimpleNamespace(config=str(tmp_path / 'not-present.yaml'), backend='deep', deep_space=None,
                               deep_pose_input=None, deep_ray_origin=None, smoother=None, no_calibration=True,
                               frames=10, interval=interval, csv_path=str(csv_path))
        monkeypatch.setattr(script, 'parse_args', lambda: args)
        assert script.main() == 0
        outputs.append(pd.read_csv(csv_path))
        printing.append(capsys.readouterr().out.count('[diagnose] #'))
    # CPU duration is telemetry, not part of deterministic numerical output.
    pd.testing.assert_frame_equal(outputs[0].drop(columns='processing_timings'),
                                  outputs[1].drop(columns='processing_timings'))
    assert printing[0] > printing[1]
