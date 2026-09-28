"""Offscreen complete shadow flow with a fake producer and failing OS action hooks."""
import json
import queue
import threading
from types import SimpleNamespace

import pytest
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

from src.experiment.r4_features import extract_features
from src.experiment.r6_shadow import replay_shadow
from src.experiment.snapshots import result_snapshot
from src.tracker.observation import Observation
from src.tracker.pipeline import SystemConfig, TrackerResult
from src.ui.r6_shadow_window import ShadowWindow


_APP = None


@pytest.fixture(scope='module')
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class Clock:
    def __init__(self):
        self.now = 100.

    def __call__(self):
        return self.now


def snapshot(x, y, *, right=True):
    left = [5. + 3*x, 2. + 2*y]
    other = [25. + 3*x, 2. + 2*y] if right else None
    return dict(units='camera_px', frame_size=[100, 80],
                eye_landmarks={'33': [0., 0.], '133': [10., 0.],
                               '362': [20., 0.], '263': [30., 0.]},
                eyes={'left': dict(roi_origin=[0., 0.]), 'right': dict(roi_origin=[20., 0.])},
                classic_dark_centroid=dict(units='camera_px', left_frame=left, left_roi=left,
                                           right_frame=other,
                                           right_roi=None if other is None else [other[0] - 20., other[1]]))


class FakePipeline:
    starts = 0

    def __init__(self, _, config, clock):
        self.config = config
        self.clock = clock
        self.session_id = 'fake-session'
        self.actual_camera_size = (100, 80)
        self.numeric_observation_enabled = False
        self.record_sink = self.observation_sink = None
        self._thread = None
        self._running = False
        self.latest = None
        self.seq = self.candidate_continuity = 0

    def start(self):
        type(self).starts += 1
        self._running = True
        return True

    def stop(self):
        self._running = False

    def get_latest_result(self):
        return self.latest

    def get_dispatch_snapshot(self, *, clock):
        return dict(checked_at=clock(), running=self._running, worker_alive=self._running,
                    calibrating=False, session=self.session_id,
                    continuity=self.candidate_continuity,
                    candidate_continuity=self.candidate_continuity,
                    latest_valid=self.latest is not None and self.latest.valid,
                    latest_candidate_valid=(self.latest is not None and self.latest.valid and
                                            self.latest.candidate_feature is not None),
                    latest_sequence=self.seq)

    def publish(self, x=.5, y=.5, *, right=True):
        self.seq += 1
        data = snapshot(x, y, right=right)
        feature, reasons = extract_features(data)
        if feature['F2'] is None:
            self.candidate_continuity += 1
        stamp = Observation(self.session_id, self.seq, self.clock.now - .01,
                            self.candidate_continuity, 'host_read_completed')
        item = TrackerResult((.5, .5), True, 30., backend='classic', raw_point=(.5, .5),
                             observation=stamp, published_at=self.clock.now,
                             numeric_snapshot=data, candidate_feature=None if feature['F2'] is None
                             else tuple(feature['F2']), candidate_rejection=reasons['F2'],
                             candidate_continuity=self.candidate_continuity)
        self.latest = item
        if self.record_sink:
            self.record_sink('producer', self.clock.now, result=result_snapshot(item))
        if self.observation_sink:
            self.observation_sink(item)
        return item


def test_full_offscreen_shadow_flow_never_calls_system_actions(qapp, tmp_path, monkeypatch):
    import src.ui.interaction_overlay as overlay

    def forbidden(*args, **kwargs):
        pytest.fail('shadow attempted a system action')
    monkeypatch.setattr(QCursor, 'setPos', forbidden)
    for name in ('perform_left_click', 'launch_browser', 'launch_explorer',
                 'launch_notepad', 'launch_osk', 'launch_magnifier'):
        monkeypatch.setattr(overlay, name, forbidden)

    clock = Clock()
    created = []
    def factory(path, config):
        instance = FakePipeline(path, config, clock)
        created.append(instance)
        return instance
    FakePipeline.starts = 0
    window = ShadowWindow(SystemConfig(tracker_backend='classic', camera_width=100, camera_height=80),
                          clock=clock, pipeline_factory=factory, output_root=tmp_path)
    window.resize(1000, 800)
    window.show()
    qapp.processEvents()
    assert FakePipeline.starts == 0 and window.phase == 'prepare'
    window.start_session()
    window.timer.stop()  # deterministic virtual clock; all ticks below are explicit
    assert FakePipeline.starts == 1 and window.phase == 'calibrating'
    pipe = created[0]
    window.repaint()
    for segment in range(9):
        item = window.protocol.plan['segments'][segment]
        base = window.protocol.started + item['planned_offset_s']
        x = item['instructed_target'][0] / window.width()
        y = item['instructed_target'][1] / window.height()
        for i in range(12):
            clock.now = base + .62 + i * .09
            pipe.publish(x + i*.001, y + i*.001)
            window.tick()
        clock.now = base + item['duration_s'] + (.1 if segment == 8 else .001)
        if segment == 8:
            pipe.publish(x, y)  # source is after the predeclared final cutoff
        window.tick()
        window.repaint()
    assert window.phase == 'fitting'
    clock.now += .26
    window.tick()
    assert window.collector is not None, (window.phase, window.status.text())
    assert window.collector.rejections['outside_calibration_cutoff'] >= 1
    sealed = window.collector
    sealed_count = len(sealed.samples)
    clock.now += .01
    pipe.publish(.5, .5)
    window._drain_samples()
    assert len(sealed.samples) == sealed_count  # post-seal producer frame cannot enter the fit
    token, answer = window.fit_results.get(timeout=2.)
    window.fit_results.put((token, answer))
    window.tick()
    assert window.phase == 'validation'
    assert (window.recorder.directory / 'f2_mapping.json').exists()
    # The independent A second round and all B segments run on a frozen map.
    model_id = window.consumer.model.model_id
    for segment in range(21):
        item = window.protocol.plan['segments'][segment]
        base = window.protocol.started + item['planned_offset_s']
        window.repaint()
        clock.now = base + item['duration_s'] + .001
        window.tick()
    assert window.phase == 'free' and window.consumer.model.model_id == model_id
    clock.now += .1
    pipe.publish(.5, .5)
    window.tick()
    window.repaint()
    assert window.candidate is not None
    assert len(sealed.samples) == sealed_count  # validation/free frames never train
    clock.now += .3
    window.tick()
    assert window.candidate is None  # no new producer result, still expires
    window.pause()
    assert window.phase == 'paused' and window.candidate is None
    window.resume()
    clock.now += .1
    pipe.publish(.5, .5, right=False)  # Classic valid, F2 missing
    window.tick()
    assert window.candidate is None
    clock.now += .1
    pipe.publish(.5, .5)
    window.tick()
    assert window.candidate is not None
    window.finish()
    assert window.phase == 'ended' and not pipe._running
    summary = json.loads((window.recorder.directory / 'shadow_summary.json').read_text(encoding='utf-8'))
    assert summary['calibration_samples'] >= 90
    assert summary['shadow_replay']['strictly_reproducible'], summary['shadow_replay']
    assert summary['producer_count'] >= 110 and summary['write_lost'] == 0
    loaded = ShadowWindow(SystemConfig(tracker_backend='classic', camera_width=100, camera_height=80),
                          load_mapping_path=window.recorder.directory / 'f2_mapping.json',
                          clock=clock, pipeline_factory=factory, output_root=tmp_path)
    loaded.resize(window.size())
    loaded.show()
    qapp.processEvents()
    assert FakePipeline.starts == 1
    loaded.start_session()
    loaded.timer.stop()
    assert loaded.phase == 'validation' and loaded.consumer.model.model_id == model_id
    assert loaded.recorder.metadata['mode'] == 'loaded_mapping_validation'
    loaded.finish()
    loaded.close()
    trace = window.recorder.directory / 'events.jsonl'
    recorded = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
    active = next(e for e in recorded if e['kind'] == 'model_active')
    original_coefficient = active['mapping']['coefficients'][0][0]
    active['mapping']['coefficients'][0][0] += 1.
    trace.write_text(''.join(json.dumps(e) + '\n' for e in recorded), encoding='utf-8')
    assert 'frozen_fit_mismatch' in replay_shadow(window.recorder.directory)['issues']
    active['mapping']['coefficients'][0][0] = original_coefficient
    sample = next(e for e in recorded if e['kind'] == 'calibration_sample')
    sample['feature'][0] += 1.
    trace.write_text(''.join(json.dumps(e) + '\n' for e in recorded), encoding='utf-8')
    assert 'calibration_sample_source_mismatch' in replay_shadow(window.recorder.directory)['issues']
    window.close()


def test_bounded_calibration_queue_reports_overflow(qapp, tmp_path):
    clock = Clock()
    window = ShadowWindow(SystemConfig(tracker_backend='classic'), clock=clock,
                          pipeline_factory=lambda *_: None, output_root=tmp_path)
    for _ in range(513):
        window._enqueue(object())
    assert window.producer_queue.qsize() == 512 and window.queue_drops == 1
    window.close()


def test_late_fit_cannot_activate_after_recalibration(qapp, tmp_path, monkeypatch):
    clock = Clock()
    pipe = None
    def factory(path, config):
        nonlocal pipe
        pipe = FakePipeline(path, config, clock)
        return pipe
    window = ShadowWindow(SystemConfig(tracker_backend='classic', camera_width=100, camera_height=80),
                          clock=clock, pipeline_factory=factory, output_root=tmp_path)
    window.resize(1000, 800)
    window.show()
    qapp.processEvents()
    window.start_session()
    window.timer.stop()
    entered, release = threading.Event(), threading.Event()
    def slow_fit(*_):
        entered.set()
        assert release.wait(2.)
        return object()
    monkeypatch.setattr('src.ui.r6_shadow_window.fit_personal', slow_fit)
    window.phase = 'fitting'
    window.protocol_events = [dict(kind='target_painted', segment=i, at=clock.now) for i in range(9)]
    window._start_fit()
    assert entered.wait(2.)
    old_token = window.fit_token
    window.recalibrate()
    assert window.phase == 'calibrating' and window.consumer.model is None
    release.set()
    token, _ = window.fit_results.get(timeout=2.)
    assert token == old_token and token != window.fit_token
    window.finish()
    window.close()


def test_window_restart_waits_for_late_worker_cleanup(qapp, tmp_path):
    clock = Clock()
    order = []
    class LatePipeline(FakePipeline):
        def __init__(self, path, config):
            super().__init__(path, config, clock)
            self.alive = True
            self._thread = SimpleNamespace(is_alive=lambda: self.alive)

        def stop(self):
            self._running = False
            if not self.alive:
                order.append('old_cleanup')

    made = []
    def factory(path, config):
        if not made:
            pipe = LatePipeline(path, config)
        else:
            order.append('new_construct')
            pipe = FakePipeline(path, config, clock)
        made.append(pipe)
        return pipe
    window = ShadowWindow(SystemConfig(tracker_backend='classic', camera_width=100, camera_height=80),
                          clock=clock, pipeline_factory=factory, output_root=tmp_path)
    window.show()
    qapp.processEvents()
    window.start_session()
    window.timer.stop()
    window.finish()
    assert window.retiring_pipeline is made[0]
    window.start_session()
    window.timer.stop()
    assert len(made) == 1 and '拒绝' in window.status.text()
    made[0].alive = False
    window.start_session()
    window.timer.stop()
    assert order == ['old_cleanup', 'new_construct']
    assert window.phase == 'calibrating'
    window.finish()
    window.close()
