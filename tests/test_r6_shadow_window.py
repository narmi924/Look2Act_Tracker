"""Offscreen complete shadow flow with a fake producer and failing OS action hooks."""
import json
import queue
import threading
from types import SimpleNamespace

import pytest
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

from src.experiment.r4_features import extract_features
from src.experiment.r6_shadow import replay_shadow, shadow_summary
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
    sealed = window.collector
    clock.now += .26
    window.tick()
    assert sealed.rejections['outside_calibration_cutoff'] >= 1
    sealed_count = len(sealed.samples)
    clock.now += .01
    pipe.publish(.5, .5)
    window._drain_samples()
    assert len(sealed.samples) == sealed_count  # post-seal producer frame cannot enter the fit
    if window.phase == 'fitting':
        token, answer = window.fit_results.get(timeout=5.)
        window.fit_results.put((token, answer))
        window.tick()
    assert window.phase == 'validation'
    assert len(list(window.recorder.directory.glob('f2_mapping*.json'))) == 1
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
                          load_mapping_path=next(window.recorder.directory.glob('f2_mapping*.json')),
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


@pytest.fixture
def cycle_factory(qapp, tmp_path, monkeypatch):
    """Drive real window controls and protocol with synthetic producer frames only."""
    import src.ui.interaction_overlay as overlay

    action_calls = []
    def forbidden(*args, **kwargs):
        action_calls.append((args, kwargs))
        pytest.fail('R6 shadow attempted a system action')
    monkeypatch.setattr(QCursor, 'setPos', forbidden)
    for name in ('perform_left_click', 'launch_browser', 'launch_explorer',
                 'launch_notepad', 'launch_osk', 'launch_magnifier'):
        monkeypatch.setattr(overlay, name, forbidden)

    clock = Clock()
    windows, pipes = [], []
    def pipeline_factory(path, config):
        pipe = FakePipeline(path, config, clock)
        pipe.session_id = f'synthetic-session-{len(pipes) + 1}'
        pipes.append(pipe)
        return pipe
    def make_window(*, load_mapping_path=None):
        window = ShadowWindow(SystemConfig(tracker_backend='classic', camera_width=100,
                                           camera_height=80),
                              load_mapping_path=load_mapping_path, clock=clock,
                              pipeline_factory=pipeline_factory, output_root=tmp_path)
        window.resize(1000, 800)
        window.show()
        qapp.processEvents()
        windows.append(window)
        return window
    yield make_window, clock, pipes, action_calls
    for window in reversed(windows):
        if window.pipeline is not None:
            window.finish()
        window.close()


def advance_calibration_to_fit(window, pipe, clock, *, variation=0., allow_failure=False):
    assert window.phase == 'calibrating'
    window.repaint()
    for segment in range(9):
        item = window.protocol.plan['segments'][segment]
        base = window.protocol.started + item['planned_offset_s']
        x = item['instructed_target'][0] / window.width()
        y = item['instructed_target'][1] / window.height()
        for i in range(12):
            clock.now = base + .62 + i * .09
            pipe.publish(x + variation + i*.001, y + variation + i*.001)
            window.tick()
        clock.now = base + item['duration_s'] + .001
        window.tick()
        window.repaint()
    assert window.phase == 'fitting', window.status.text()
    clock.now += .26
    window.tick()  # seals the actual samples and starts the real fit worker
    expected = ('fitting', 'validation', 'failed') if allow_failure else ('fitting', 'validation')
    assert window.phase in expected, (window.status.text(), window.queue_drops,
                                      window.recorder.lost, window.recorder.error)
    assert window.samples_sealed


def complete_fit(window):
    if window.phase == 'fitting':
        token, answer = window.fit_results.get(timeout=5.)
        window.fit_results.put((token, answer))
        window.tick()
    assert window.phase == 'validation', window.status.text()
    return window.consumer.model.model_id


def advance_validation_to_free(window, clock):
    assert window.phase == 'validation'
    for segment in range(21):
        item = window.protocol.plan['segments'][segment]
        window.repaint()
        clock.now = window.protocol.started + item['planned_offset_s'] + item['duration_s'] + .001
        window.tick()
    assert window.phase == 'free'


def recorded_events(directory):
    return [json.loads(line) for line in (directory / 'events.jsonl').read_text(
        encoding='utf-8').splitlines()]


def test_two_complete_calibrations_preserve_both_mappings_and_replay(cycle_factory):
    make_window, clock, pipes, action_calls = cycle_factory
    window = make_window()
    window.start_session()
    window.timer.stop()
    advance_calibration_to_fit(window, pipes[-1], clock)
    first_model = complete_fit(window)
    first_files = list(window.recorder.directory.glob('f2_mapping*.json'))
    assert len(first_files) == 1
    first_bytes = first_files[0].read_bytes()
    advance_validation_to_free(window, clock)

    window.recalibrate_button.click()
    advance_calibration_to_fit(window, pipes[-1], clock, variation=.02)
    second_model = complete_fit(window)
    assert second_model != first_model
    assert first_files[0].read_bytes() == first_bytes
    assert len(list(window.recorder.directory.glob('f2_mapping*.json'))) == 2
    advance_validation_to_free(window, clock)
    directory = window.recorder.directory
    window.finish()
    events = recorded_events(directory)
    active = [e for e in events if e['kind'] == 'model_active']
    assert len(active) == 2
    assert len({e['source_attempt_id'] for e in active}) == 2
    assert all(e['source_kind'] == 'personal_calibration' and e['mapping_file']
               for e in active)
    summary = json.loads((directory / 'shadow_summary.json').read_text(encoding='utf-8'))
    assert len(summary['calibration_attempts']) == 2
    assert all(a['sample_count'] >= 90 for a in summary['calibration_attempts'])
    assert summary['calibration_duration_s'] is None
    assert replay_shadow(directory)['strictly_reproducible']
    assert not action_calls


def test_finish_then_restart_collects_new_session_samples(cycle_factory):
    make_window, clock, pipes, action_calls = cycle_factory
    window = make_window()
    window.start_session()
    window.timer.stop()
    advance_calibration_to_fit(window, pipes[-1], clock)
    complete_fit(window)
    advance_validation_to_free(window, clock)
    first_directory = window.recorder.directory
    first_mapping = next(first_directory.glob('f2_mapping*.json'))
    first_bytes = first_mapping.read_bytes()
    first_session = pipes[-1].session_id
    window.finish()
    assert replay_shadow(first_directory)['strictly_reproducible']

    clock.now = 500.
    window.start_session()
    window.timer.stop()
    assert window.phase == 'calibrating' and pipes[-1].session_id != first_session
    second_directory = window.recorder.directory
    window.repaint()
    clock.now = 500.6
    pipes[-1].publish(.5, .5)
    clock.now = 500.66  # wait past the 50 ms transition guard without real sleep
    window.tick()
    assert len(window.collector.samples) == 1
    assert window.collector.rejections['outside_calibration_cutoff'] == 0
    advance_calibration_to_fit(window, pipes[-1], clock, variation=.01)
    complete_fit(window)
    advance_validation_to_free(window, clock)
    window.finish()
    second_samples = [e for e in recorded_events(second_directory)
                      if e['kind'] == 'calibration_sample']
    assert second_samples and all(s['id'][0] == pipes[-1].session_id for s in second_samples)
    assert first_mapping.read_bytes() == first_bytes
    assert replay_shadow(second_directory)['strictly_reproducible']
    assert not action_calls


def test_loaded_mapping_then_recalibration_replays_both_sources(cycle_factory):
    make_window, clock, pipes, action_calls = cycle_factory
    first = make_window()
    first.start_session()
    first.timer.stop()
    advance_calibration_to_fit(first, pipes[-1], clock)
    original_model = complete_fit(first)
    advance_validation_to_free(first, clock)
    mapping_path = next(first.recorder.directory.glob('f2_mapping*.json'))
    mapping_bytes = mapping_path.read_bytes()
    first.finish()

    loaded = make_window(load_mapping_path=mapping_path)
    loaded.start_session()
    loaded.timer.stop()
    assert loaded.phase == 'validation' and loaded.consumer.model.model_id == original_model
    loaded.recalibrate_button.click()
    advance_calibration_to_fit(loaded, pipes[-1], clock, variation=.03)
    new_model = complete_fit(loaded)
    assert new_model != original_model and mapping_path.read_bytes() == mapping_bytes
    advance_validation_to_free(loaded, clock)
    directory = loaded.recorder.directory
    loaded.finish()
    active = [e for e in recorded_events(directory) if e['kind'] == 'model_active']
    assert [e['source_kind'] for e in active] == ['explicit_load', 'personal_calibration']
    assert active[0]['source_attempt_id'] is None and active[1]['source_attempt_id']
    assert replay_shadow(directory)['strictly_reproducible']
    # Schema-1 loaded activations lacked provenance fields; the later personal fit
    # still has to be tied to its own sealed attempt.
    events_path = directory / 'events.jsonl'
    events = recorded_events(directory)
    legacy_load = next(e for e in events if e['kind'] == 'model_active')
    for field in ('source_kind', 'mapping_id', 'attempt_number', 'mapping_file',
                  'source_file_name'):
        legacy_load.pop(field, None)
    events_path.write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    assert replay_shadow(directory)['strictly_reproducible']
    legacy_summary = shadow_summary(directory)
    assert legacy_summary['model_activations'][0]['source_kind'] == 'explicit_load'
    assert legacy_summary['model_activations'][0]['source_inferred_from_legacy_mode']
    assert not action_calls


@pytest.mark.parametrize('interrupt', ['stop', 'recalibrate'])
def test_late_second_fit_cannot_save_or_activate(cycle_factory, monkeypatch, interrupt):
    import src.ui.r6_shadow_window as shadow_ui

    make_window, clock, pipes, action_calls = cycle_factory
    window = make_window()
    window.start_session()
    window.timer.stop()
    advance_calibration_to_fit(window, pipes[-1], clock)
    complete_fit(window)
    advance_validation_to_free(window, clock)
    first_files = list(window.recorder.directory.glob('f2_mapping*.json'))
    assert len(first_files) == 1
    directory = window.recorder.directory
    entered, release = threading.Event(), threading.Event()
    actual_fit = shadow_ui.fit_personal
    def delayed_fit(*args):
        entered.set()
        assert release.wait(2.)
        return actual_fit(*args)
    monkeypatch.setattr(shadow_ui, 'fit_personal', delayed_fit)
    window.recalibrate_button.click()
    advance_calibration_to_fit(window, pipes[-1], clock, variation=.02)
    assert entered.wait(2.)
    old_token = window.fit_token
    if interrupt == 'stop':
        window.finish()
    else:
        window.recalibrate_button.click()
        assert window.phase == 'calibrating' and window.consumer.model is None
    release.set()
    token, _ = window.fit_results.get(timeout=5.)
    assert token == old_token and token != window.fit_token
    if window.pipeline is not None:
        window.tick()
        window.finish()
    assert len(list(directory.glob('f2_mapping*.json'))) == 1
    assert first_files[0].exists()
    assert not action_calls


def test_failed_second_mapping_save_never_reactivates_old_model(cycle_factory, monkeypatch):
    import src.ui.r6_shadow_window as shadow_ui

    make_window, clock, pipes, action_calls = cycle_factory
    window = make_window()
    window.start_session()
    window.timer.stop()
    advance_calibration_to_fit(window, pipes[-1], clock)
    complete_fit(window)
    advance_validation_to_free(window, clock)
    first_mapping = next(window.recorder.directory.glob('f2_mapping*.json'))
    first_bytes = first_mapping.read_bytes()
    directory = window.recorder.directory
    def failed_save(*_):
        raise OSError('synthetic_save_failure')
    monkeypatch.setattr(shadow_ui, 'save_mapping', failed_save)
    window.recalibrate_button.click()
    assert window.consumer.model is None
    advance_calibration_to_fit(window, pipes[-1], clock, variation=.02, allow_failure=True)
    complete_fit_or_fail = window.fit_results.get(timeout=5.) if window.phase == 'fitting' else None
    if complete_fit_or_fail is not None:
        window.fit_results.put(complete_fit_or_fail)
        window.tick()
    assert window.phase == 'failed' and '映射保存失败' in window.status.text()
    assert first_mapping.read_bytes() == first_bytes
    assert len(list(directory.glob('f2_mapping*.json'))) == 1
    assert len([e for e in recorded_events(directory) if e['kind'] == 'model_active']) == 1
    assert not action_calls


def test_second_calibration_tampering_still_fails_replay(cycle_factory):
    make_window, clock, pipes, action_calls = cycle_factory
    window = make_window()
    window.start_session()
    window.timer.stop()
    advance_calibration_to_fit(window, pipes[-1], clock)
    complete_fit(window)
    advance_validation_to_free(window, clock)
    window.recalibrate_button.click()
    advance_calibration_to_fit(window, pipes[-1], clock, variation=.02)
    complete_fit(window)
    advance_validation_to_free(window, clock)
    directory = window.recorder.directory
    window.finish()
    assert replay_shadow(directory)['strictly_reproducible']
    path = directory / 'events.jsonl'
    original = recorded_events(directory)
    attempts = [e['attempt_id'] for e in original if e['kind'] == 'calibration_attempt']
    second = attempts[1]

    def corrupted(change):
        events = json.loads(json.dumps(original))
        change(events)
        path.write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
        assert not replay_shadow(directory)['strictly_reproducible']

    def change_coefficient(events):
        active = next(e for e in events if e['kind'] == 'model_active' and
                      e.get('source_attempt_id') == second)
        active['mapping']['coefficients'][0][0] += 1.
    def change_sample(events):
        sample = next(e for e in events if e['kind'] == 'calibration_sample' and
                      e.get('attempt_id') == second)
        sample['feature'][0] += 1.
    def change_source(events):
        active = next(e for e in events if e['kind'] == 'model_active' and
                      e.get('source_attempt_id') == second)
        active['source_kind'] = 'explicit_load'
    for change in (change_coefficient, change_sample, change_source):
        corrupted(change)
    path.write_text(''.join(json.dumps(e) + '\n' for e in original), encoding='utf-8')
    assert replay_shadow(directory)['strictly_reproducible']
    second_active = next(e for e in original if e['kind'] == 'model_active' and
                         e.get('source_attempt_id') == second)
    saved_path = directory / second_active['mapping_file']
    # The event snapshot and saved version must continue to agree on replay.
    saved = saved_path.read_bytes()
    saved_data = json.loads(saved)
    saved_data['coefficients'][0][0] += 1.
    saved_path.write_text(json.dumps(saved_data), encoding='utf-8')
    assert not replay_shadow(directory)['strictly_reproducible']
    saved_path.write_bytes(saved)
    assert replay_shadow(directory)['strictly_reproducible']
    assert not action_calls
