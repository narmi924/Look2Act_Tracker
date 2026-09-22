"""Synthetic observation safety tests: no camera, model weights or OS actions."""
from types import SimpleNamespace

import numpy as np

from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult


def test_no_face_fallback_is_not_actionable():
    pipeline = TrackerPipeline("", SystemConfig())
    pipeline.face_detector = SimpleNamespace(
        detect=lambda frame: SimpleNamespace(detected=False)
    )
    pipeline._last_valid_result = TrackerResult((100., 100.), True, 30.)
    result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
    assert result.gaze_point == (100., 100.)  # display may retain the point
    assert not result.valid

from dataclasses import FrozenInstanceError, replace
import pytest

from src.tracker.observation import Observation, ObservationGate, ObservationState


class Clock:
    now = 10.0
    def __call__(self):
        return self.now


def observation(seq=1, t=10.1, **kwargs):
    stamp = Observation(kwargs.pop('session', 'session'), seq, t, kwargs.pop('continuity', 0))
    return TrackerResult((100., 100.), True, 30., observation=stamp, **kwargs)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), True, '250', None])
def test_invalid_age_configuration(value):
    with pytest.raises(ValueError):
        SystemConfig(max_observation_age_ms=value)
    with pytest.raises(ValueError):
        ObservationGate(value)


def test_gate_boundaries_duplicates_and_gap():
    clock = Clock()
    gate = ObservationGate(clock=clock)
    gate.reset('session')
    clock.now = 10.125
    first = observation(t=10.125)
    assert gate.consume(first) is ObservationState.NEW
    clock.now = 10.375
    assert gate.consume(first) is ObservationState.DUPLICATE  # equality is allowed
    assert gate.consume(observation(2, 10.375)) is ObservationState.NEW
    clock.now = 10.625001
    assert gate.consume(observation(3, clock.now)) is ObservationState.INVALID
    clock.now += .125
    assert gate.consume(observation(4, clock.now)) is ObservationState.NEW
    assert gate.reset_required
    clock.now += .250001
    assert gate.consume(observation(4, 10.750001)) is ObservationState.INVALID


@pytest.mark.parametrize('case', ['missing', 'old_session', 'future', 'nan_time', 'nan_point',
                                  'inf_point', 'held', 'invalid', 'sequence', 'time_order'])
def test_gate_rejects_bad_evidence(case):
    clock = Clock(); gate = ObservationGate(clock=clock); gate.reset('session')
    clock.now = 10.125
    assert gate.consume(observation(2, clock.now)) is ObservationState.NEW
    clock.now = 10.25
    result = observation(3, clock.now)
    if case == 'missing': result.observation = None
    elif case == 'old_session': result.observation = replace(result.observation, session='old')
    elif case == 'future': result.observation = replace(result.observation, timestamp=11.)
    elif case == 'nan_time': result.observation = replace(result.observation, timestamp=float('nan'))
    elif case == 'nan_point': result.gaze_point = (float('nan'), 0.)
    elif case == 'inf_point': result.gaze_point = (0., float('inf'))
    elif case == 'held': result.point_kind = 'held'
    elif case == 'invalid': result.valid = False
    elif case == 'sequence': result.observation = replace(result.observation, sequence=1)
    elif case == 'time_order': result.observation = replace(result.observation, timestamp=10.125)
    assert gate.consume(result) is ObservationState.INVALID


def test_metadata_is_immutable_and_published_results_are_detached():
    pipeline = TrackerPipeline('', SystemConfig())
    pipeline._latest_result = observation()
    fetched = pipeline.get_latest_result()
    with pytest.raises(FrozenInstanceError):
        fetched.observation.sequence = 2
    fetched.observation = None
    fetched.debug['ui'] = True
    assert pipeline.get_latest_result().observation is not None
    assert pipeline.get_latest_result().debug == {}


@pytest.fixture(scope='session')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


class ControlledTracker(TrackerPipeline):
    """Real publication lock/state with synthetic results and no worker/camera."""
    def __init__(self):
        super().__init__('', SystemConfig())
        self.session_id = 'session'
        self.running = True

    @property
    def running(self):
        return self.is_running()

    @running.setter
    def running(self, value):
        with self._lock:
            self._running = value

    @property
    def result(self):
        return self._latest_result

    @result.setter
    def result(self, value):
        with self._lock:
            self._latest_result = value
            stamp = getattr(value, 'observation', None)
            if stamp is not None and stamp.session == self.session_id:
                self._sequence = max(self._sequence, stamp.sequence)
                self._continuity = max(self._continuity, stamp.continuity)


@pytest.fixture(params=['desktop', 'launcher', 'board'])
def interaction(request, qapp, monkeypatch):
    from src.ui import tracking_page as ui
    from src.ui import interaction_overlay as launcher_module
    from src.ui.interaction_overlay import InteractionLauncherOverlay
    from src.ui.gomoku_window import GomokuWindow
    page = ui.TrackingPage()
    clock = Clock()
    page.observation_gate = ObservationGate(clock=clock)
    page.tracker_config = SystemConfig()
    tracker = ControlledTracker()
    page.tracker = tracker
    calls, moves, samples = [], [], []
    monkeypatch.setattr(ui, 'perform_left_click', lambda: calls.append('click') or True)
    monkeypatch.setattr(ui.QCursor, 'setPos', lambda *args: moves.append(args))
    monkeypatch.setattr(page.screen_stabilizer, 'update', lambda p: samples.append(p) or p)
    if request.param == 'desktop':
        page.dwell_enabled = True
        duration = page.dwell_ms
    elif request.param == 'launcher':
        monkeypatch.setattr(launcher_module, "launch_browser", lambda: calls.append("browser"))
        window = InteractionLauncherOverlay()
        page.interaction_overlay = window
        monkeypatch.setattr(window, '_region_at_global_point', lambda *args: 0)
        monkeypatch.setattr(window, 'close', lambda: None)  # retain widget for assertions
        duration = window._dwell_ms
    else:
        window = GomokuWindow()
        page.gomoku_window = window
        monkeypatch.setattr(window, '_board_pos_from_point', lambda *args: (2, 2))
        original_place = window._place_x
        def place(pos):
            calls.append(pos)
            original_place(pos)
        monkeypatch.setattr(window, '_place_x', place)
        duration = window._dwell_ms
    page._update_tracking_data()  # bind context before observations are captured
    sequence = [0]
    epoch = [0]
    def feed(t, *, valid=True, continuity=None, point=(100., 100.)):
        clock.now = t
        sequence[0] = max(sequence[0], tracker._sequence) + 1
        epoch[0] = max(epoch[0], tracker._continuity)
        if continuity is not None: epoch[0] = continuity
        tracker.result = observation(sequence[0], t, continuity=epoch[0], backend='classic')
        tracker.result.valid = valid
        tracker.result.gaze_point = point
        page._update_tracking_data()
    def hold(start, duration_ms=duration):
        # Binary-exact intervals avoid accidental threshold floating point crossings.
        count = int(duration_ms / 125.) + 2
        for i in range(count):
            feed(start + i * .125)
    def cleared():
        assert page._dwell_anchor is None
        if page.interaction_overlay:
            assert page.interaction_overlay._current_region is None
            assert all(c._progress == 0 for c in page.interaction_overlay._cards.values())
        if page.gomoku_window:
            assert page.gomoku_window.current_hover is None
            assert page.gomoku_window._gaze_progress == 0
    yield SimpleNamespace(page=page, clock=clock, tracker=tracker, calls=calls, moves=moves,
                          samples=samples, feed=feed, hold=hold, cleared=cleared,
                          duration=duration, kind=request.param)
    page.close()


def test_real_routes_normal_and_duplicate_reading(interaction):
    h = interaction
    h.feed(10.125)
    n = len(h.samples)
    for _ in range(30):
        h.clock.now = 10.2
        h.page._update_tracking_data()
    assert len(h.samples) == n == 1
    assert h.calls == []
    h.hold(10.25)
    assert len(h.calls) == 1
    for _ in range(30): h.page._update_tracking_data()
    assert len(h.calls) == 1


@pytest.mark.parametrize('failure', ['invalid', 'stalled', 'none', 'hidden_failure', 'nan', 'gap'])
def test_real_routes_interrupt_and_recover(interaction, failure):
    h = interaction
    start = 10.125
    count = int(h.duration / 125.)
    for i in range(count): h.feed(start + i * .125)
    assert h.calls == []
    last = h.clock.now
    if failure == 'invalid':
        for i in range(3): h.feed(last + (i + 1) * .125, valid=False)
    elif failure == 'stalled':
        h.clock.now = last + .251
        h.page._update_tracking_data()
    elif failure == 'none':
        h.tracker.result = None
        h.page._update_tracking_data()
    elif failure == 'nan':
        h.feed(last + .125, point=(float('nan'), 0.))
    elif failure == 'gap':
        h.feed(last + .5)
    else:
        h.feed(last + .125, continuity=1)  # failure overwritten before UI polled
    assert h.calls == []
    if failure != 'hidden_failure': h.cleared()
    if failure in ('stalled', 'none'):
        h.clock.now += 10.
        h.page._update_tracking_data()
        h.cleared()
        assert h.calls == []
    recovered = h.clock.now + .125
    h.feed(recovered, continuity=1)
    assert h.calls == []
    # A first frame after a long source gap may be rejected; start from the next one.
    h.hold(recovered + .125)
    assert len(h.calls) == 1


def test_real_routes_stop_restart_and_stage_boundary(interaction):
    h = interaction
    h.feed(10.125); h.feed(10.25)
    h.tracker.running = False
    h.page._update_tracking_data()
    h.cleared()
    h.tracker.running = True
    h.tracker.session_id = 'new-session'
    h.clock.now = 10.375
    h.page._update_tracking_data()  # old session rejected
    assert h.calls == []
    h.cleared()
    h.tracker.result = observation(1, 10.5, session='new-session')
    h.clock.now = 10.5
    h.page._update_tracking_data()
    assert h.calls == []
    h.page._reset_observation_context()  # entering/re-entering a stage
    h.clock.now = 10.625
    h.page._update_tracking_data()
    h.cleared()
    assert h.calls == []


def test_delayed_launcher_return_cancelled_on_stop(qapp, monkeypatch):
    from src.ui.tracking_page import TrackingPage
    page = TrackingPage()
    calls = []
    monkeypatch.setattr(page, '_open_launcher_overlay', lambda: calls.append(True))
    page._launcher_return_timer.start(0)
    page._stop_tracker_runtime()
    qapp.processEvents()
    assert not calls
    page.close()


def deep_pipeline(backend='deep'):
    pipeline = TrackerPipeline('', SystemConfig(tracker_backend=backend, use_onnx=True,
                                               deep_gaze_space='camera', deep_ray_origin='zero_origin'))
    pipeline.model_version = 'v2'
    face = SimpleNamespace(detected=True, pnp_points_2d={},
                           left_eye_crop=np.zeros((8, 8, 3), dtype=np.uint8),
                           right_eye_crop=np.zeros((8, 8, 3), dtype=np.uint8))
    pose = SimpleNamespace(valid=True, yaw=0., pitch=0., roll=0.,
                           rotation_matrix=np.eye(3), translation_vec=np.zeros(3))
    pipeline.face_detector = SimpleNamespace(detect=lambda frame: face)
    pipeline.head_pose_estimator = SimpleNamespace(estimate=lambda points: pose)
    pipeline.onnx_session = SimpleNamespace(run=lambda *args: [np.array([[0., 0., 1.]])])
    pipeline.screen_geometry = SimpleNamespace(
        ray_plane_intersect=lambda *args: np.array([0., 0., 720.]),
        world_to_screen_px=lambda *args, **kw: (100., 100.), screen_w_px=1920, screen_h_px=1080)
    pipeline.get_diagnostics = lambda: {}
    return pipeline, face, pose


@pytest.mark.parametrize('failure', ['face', 'eyes', 'pose', 'pose_nan', 'model', 'model_nan',
                                     'intersection', 'intersection_nan', 'mapping', 'exception'])
def test_producer_failure_branches_preserve_identity_and_break_continuity(failure):
    pipeline, face, pose = deep_pipeline()
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    good = pipeline.process_frame(frame, captured_at=10.)
    assert good.valid
    if failure == 'face': face.detected = False
    elif failure == 'eyes': face.left_eye_crop = None
    elif failure == 'pose': pose.valid = False
    elif failure == 'pose_nan': pose.yaw = float('nan')
    elif failure == 'model': pipeline.onnx_session.run = lambda *a: 1 / 0
    elif failure == 'model_nan': pipeline.onnx_session.run = lambda *a: [np.array([[np.nan, 0., 1.]])]
    elif failure == 'intersection': pipeline.screen_geometry.ray_plane_intersect = lambda *a: None
    elif failure == 'intersection_nan': pipeline.screen_geometry.ray_plane_intersect = lambda *a: np.full(3, np.nan)
    elif failure == 'mapping': pipeline.screen_geometry.world_to_screen_px = lambda *a, **k: (float('inf'), 1.)
    elif failure == 'exception': pipeline.face_detector.detect = lambda *a: 1 / 0
    for i in range(2):
        bad = pipeline.process_frame(frame, captured_at=10.1 + i * .1)
        assert not bad.valid
        assert bad.point_kind == 'held'
        assert bad.error_message
        assert bad.observation.sequence == good.observation.sequence + i + 1
        assert bad.observation.continuity > good.observation.continuity


def test_classic_does_not_depend_on_pose_and_pog_rejects_nan():
    from tests.test_classic_pipeline import _classic_face_result
    pipeline = TrackerPipeline('', SystemConfig())
    face = _classic_face_result(detected=True)
    pipeline.face_detector = SimpleNamespace(detect=lambda frame: face)
    pipeline.head_pose_estimator = SimpleNamespace(estimate=lambda points: 1 / 0)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert pipeline.process_frame(frame).valid
    face.left_eye_roi = face.right_eye_roi = None
    assert not pipeline.process_frame(frame).valid
    pipeline, _, _ = deep_pipeline('deep_pog')
    pipeline.onnx_session.run = lambda *a: [np.array([[.25, .5]])]
    assert pipeline.process_frame(frame).valid
    pipeline.onnx_session.run = lambda *a: [np.array([[np.inf, .5]])]
    assert not pipeline.process_frame(frame).valid


def test_camera_retry_failure_survives_latest_result_overwrite(monkeypatch):
    import src.tracker.pipeline as module
    clock = Clock()
    pipeline = TrackerPipeline('', SystemConfig())
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    reads = [True, False, True]
    def read():
        clock.now += .125
        if not reads:
            pipeline._running = False
            return False, None
        success = reads.pop(0)
        return success, frame if success else None
    pipeline.cap = SimpleNamespace(read=read)
    pipeline._process_frame = lambda f: TrackerResult((1., 2.), True, 30.)
    monkeypatch.setattr(module.time, 'perf_counter', clock)
    monkeypatch.setattr(module.time, 'sleep', lambda t: None)
    pipeline._running = True
    pipeline.run()
    latest = pipeline.get_latest_result()
    assert latest.valid
    assert latest.observation.sequence == 3
    assert latest.observation.continuity == 1
    assert latest.observation.timestamp == 10.375
    assert latest.observation.time_source == 'host_read_completed'


def test_source_time_precedes_inference_and_sessions_change_on_restart(monkeypatch):
    import src.tracker.pipeline as module
    clock = Clock()
    pipeline = TrackerPipeline('', SystemConfig())
    def process(frame):
        clock.now += 1.
        return TrackerResult((1., 2.), True, 30.)
    pipeline._process_frame = process
    monkeypatch.setattr(module.time, 'perf_counter', clock)
    result = pipeline.process_frame(None)
    assert result.observation.timestamp == 10.
    assert result.observation.time_source == 'processing_entry'
    monkeypatch.setattr(pipeline, 'initialize', lambda: True)
    monkeypatch.setattr(module.threading, 'Thread', lambda **kw: SimpleNamespace(
        start=lambda: None, join=lambda **kw: None, is_alive=lambda: False))
    assert pipeline.start()
    first = pipeline.process_frame(None)
    pipeline.stop()
    assert pipeline.start()
    second = pipeline.process_frame(None)
    assert first.observation.sequence == second.observation.sequence == 1
    assert first.observation.session != second.observation.session
    pipeline.stop()


def test_still_stopping_thread_cannot_restart(monkeypatch):
    pipeline = TrackerPipeline('', SystemConfig())
    pipeline._thread = SimpleNamespace(join=lambda **kw: None, is_alive=lambda: True)
    monkeypatch.setattr(pipeline, 'initialize', lambda: pytest.fail('must not reopen camera'))
    pipeline.stop()
    assert not pipeline.start()


def test_calibrated_nan_cannot_be_clamped_into_action(interaction, monkeypatch):
    h = interaction
    h.page.calibrator = SimpleNamespace(is_calibrated=True, apply=lambda p: (float('nan'), 0.))
    h.page._update_tracking_data()
    h.feed(10.125)
    assert h.calls == h.moves == []
    h.cleared()


@pytest.mark.parametrize('fault', ['missing', 'future', 'out_of_order', 'old_session', 'inf'])
def test_real_routes_reject_bad_metadata(interaction, fault):
    h = interaction
    h.feed(10.125); h.feed(10.25)
    h.clock.now = 10.375
    bad = observation(3, h.clock.now)
    if fault == 'missing': bad.observation = None
    elif fault == 'future': bad.observation = replace(bad.observation, timestamp=11.)
    elif fault == 'out_of_order': bad.observation = replace(bad.observation, sequence=1)
    elif fault == 'old_session': bad.observation = replace(bad.observation, session='retired')
    elif fault == 'inf': bad.gaze_point = (0., float('inf'))
    h.tracker.result = bad
    moves = len(h.moves)
    h.page._update_tracking_data()
    h.cleared()
    assert h.calls == []
    assert len(h.moves) == moves


def test_board_paint_does_not_advance_progress(qapp, monkeypatch):
    from src.ui.gomoku_window import GomokuWindow
    window = GomokuWindow()
    window.resize(900, 700)
    monkeypatch.setattr(window, '_board_pos_from_point', lambda *a: (2, 2))
    window.update_gaze_point(100., 100., observed_ms=1000.)
    window.update_gaze_point(100., 100., observed_ms=1125.)
    progress = window._gaze_progress
    monkeypatch.setattr(window, '_now_ms', lambda: pytest.fail('paint must not read wall time'))
    window.grab()
    assert window._gaze_progress == progress
    assert window.board[2][2] == 0
    window.reset_gaze_progress()
    window.grab()
    assert window.current_hover is None
    window.close()


def test_actual_stage_switch_discards_previous_observation(interaction, monkeypatch):
    from src.ui.gomoku_window import GomokuWindow
    h = interaction
    h.feed(10.125); h.feed(10.25)
    monkeypatch.setattr(GomokuWindow, 'show', lambda self: None)
    h.page._open_gomoku_window()
    h.clock.now = 10.375
    h.page._update_tracking_data()
    assert h.page.gomoku_window.current_hover is None
    assert h.calls == []


def test_duplicate_invalid_copy_cannot_preserve_selection():
    clock = Clock(); gate = ObservationGate(clock=clock); gate.reset('session')
    clock.now = 10.125
    result = observation(t=clock.now)
    assert gate.consume(result) is ObservationState.NEW
    assert gate.consume(replace(result, valid=False)) is ObservationState.INVALID
    assert gate.consume(result) is ObservationState.INVALID


def test_yaml_observation_age_validation(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('tracker:\n  max_observation_age_ms: 100\n', encoding='utf-8')
    assert SystemConfig.from_yaml(str(path)).max_observation_age_ms == 100
    path.write_text('tracker:\n  max_observation_age_ms: .nan\n', encoding='utf-8')
    with pytest.raises(ValueError): SystemConfig.from_yaml(str(path))


def test_processing_delay_cannot_trigger_stale_actions(interaction, monkeypatch):
    h = interaction
    def slow_smoothing(point):
        h.clock.now += .3
        return point
    monkeypatch.setattr(h.page.screen_stabilizer, 'update', slow_smoothing)
    h.feed(10.125)
    assert h.calls == h.moves == []
    h.cleared()


def test_calibration_mode_suspends_all_gaze_actions(interaction):
    h = interaction
    h.feed(10.125)
    h.tracker._calibration_mode = True
    h.feed(10.25)
    h.cleared()
    assert h.calls == []


def test_main_window_stops_interactions_before_recalibration_or_config_change():
    from src.ui.main_window import MainWindow
    calls = []
    fake = SimpleNamespace(page_tracking=SimpleNamespace(_handle_stop_tracking=lambda: calls.append('stop')),
                           tracker=None, _ensure_tracker_initialized=lambda: False)
    assert MainWindow.go_calibration(fake) is False
    MainWindow._on_config_changed(fake, SystemConfig(tracker_backend='deep'))
    assert calls == ['stop', 'stop']


def test_settings_save_preserves_observation_timeout(tmp_path, monkeypatch):
    from src.ui import settings_page
    monkeypatch.setattr(settings_page, 'set_language', lambda value: None)
    control = SimpleNamespace(setText=lambda v: None, setStyleSheet=lambda v: None)
    fake = SimpleNamespace(config=SystemConfig(max_observation_age_ms=125.,
                                               calibration_save_path='synthetic.json'),
                           config_path=tmp_path / 'settings.yaml',
                           _update_config_from_ui=lambda: None, status_label=control,
                           config_changed=SimpleNamespace(emit=lambda c: None))
    settings_page.SettingsPage._handle_save(fake)
    assert SystemConfig.from_yaml(str(fake.config_path)).max_observation_age_ms == 125.


def test_calibration_sampling_consumes_each_observation_once(qapp):
    from src.ui.calibration_page import CalibrationFullscreenWidget
    from src.calibration.calibrator import CalibrationModule
    clock = Clock()
    tracker = SimpleNamespace(config=SystemConfig(), session_id='session', result=None,
                              set_calibration_mode=lambda enabled: None)
    tracker.get_latest_result = lambda: tracker.result
    widget = CalibrationFullscreenWidget(tracker, CalibrationModule())
    widget.observation_gate = ObservationGate(clock=clock)
    widget._start_sampling()
    widget.sampling_timer.stop()
    clock.now = 10.125
    tracker.result = observation(t=clock.now)
    widget._on_sampling_tick()
    for _ in range(10): widget._on_sampling_tick()
    assert widget.current_samples == [(100., 100.)]
    clock.now = 10.25
    tracker.result = observation(2, clock.now)
    tracker.result.valid = False
    widget._on_sampling_tick()
    assert len(widget.current_samples) == 1
    widget._start_sampling()  # old point cannot become evidence for a new target
    widget.sampling_timer.stop()
    widget._on_sampling_tick()
    assert not widget.current_samples
    widget.close()


@pytest.mark.parametrize('phase', ['calibration', 'smoothing'])
@pytest.mark.parametrize('event', ['invalid', 'invalid_then_valid', 'valid', 'stop', 'session', 'calibration'])
def test_dispatch_rechecks_producer_after_processing(interaction, monkeypatch, phase, event):
    """Publish during processing, 62.5 ms before dispatch, while the consumed frame is fresh."""
    h = interaction
    start = 10.125
    steps = int(np.ceil(h.duration / 125.))
    for i in range(steps):
        h.feed(start + i * .125)
    assert not h.calls
    moves_before = len(h.moves)
    target_time = start + steps * .125

    def publish(valid):
        # Exercise real producer stamping/failure accounting, then its publication lock.
        monkeypatch.setattr(h.tracker, '_process_frame', lambda frame: TrackerResult(
            (100., 100.), valid, 30., backend='classic',
            error_message=None if valid else 'synthetic_failure'))
        result = h.tracker.process_frame(None, captured_at=h.clock.now)
        with h.tracker._lock:
            h.tracker._latest_result = result

    def processing_event():
        h.clock.now += .0625
        if event in ('invalid', 'invalid_then_valid'):
            publish(False)
        if event in ('valid', 'invalid_then_valid'):
            publish(True)
        if event == 'stop':
            h.tracker.stop()
        elif event == 'session':
            with h.tracker._lock:
                h.tracker.session_id = 'replacement-session'
        elif event == 'calibration':
            h.tracker.set_calibration_mode(True)

    if phase == 'calibration':
        original = h.page._apply_calibration_and_clamp_with_debug
        def process(*args):
            point = original(*args)
            processing_event()
            return point
        monkeypatch.setattr(h.page, '_apply_calibration_and_clamp_with_debug', process)
    else:
        original = h.page.screen_stabilizer.update
        def process(point):
            result = original(point)
            processing_event()
            return result
        monkeypatch.setattr(h.page.screen_stabilizer, 'update', process)

    h.feed(target_time)
    assert h.clock.now - target_time == .0625
    if event == 'valid':
        # A newer sequence alone must not prevent a normal completed dwell.
        assert len(h.calls) == 1
        return
    assert h.calls == []
    assert len(h.moves) == moves_before
    h.cleared()

    if phase == 'calibration':
        monkeypatch.setattr(h.page, '_apply_calibration_and_clamp_with_debug', original)
    else:
        monkeypatch.setattr(h.page.screen_stabilizer, 'update', original)
    if event in ('invalid', 'invalid_then_valid'):
        # No inherited progress, but the complete new dwell remains usable.
        recovered_at = h.clock.now + .125
        h.feed(recovered_at)
        assert not h.calls
        h.hold(recovered_at + .125)
        assert len(h.calls) == 1


def test_late_worker_exit_cleans_resources_before_restart(monkeypatch):
    import src.tracker.pipeline as module
    pipeline = TrackerPipeline('', SystemConfig())
    order = []
    alive = [True]
    cap = SimpleNamespace(release=lambda: order.append('release'))
    detector = SimpleNamespace(close=lambda: order.append('close'))
    pipeline.cap = cap
    pipeline.face_detector = detector
    pipeline._running = True
    pipeline._latest_result = observation()
    pipeline._thread = SimpleNamespace(
        join=lambda **kw: order.append('join'), is_alive=lambda: alive[0])
    old_session = pipeline.session_id

    def initialize():
        order.append('initialize')
        assert order.count('release') == order.count('close') == 1
        assert pipeline.cap is None and pipeline.face_detector is None
        assert pipeline.get_latest_result() is None
        return True
    monkeypatch.setattr(pipeline, 'initialize', initialize)
    monkeypatch.setattr(module.threading, 'Thread', lambda **kw: SimpleNamespace(
        start=lambda: order.append('new_thread'), join=lambda **kw: None, is_alive=lambda: False))

    pipeline.stop()  # join times out; resources are still in use
    assert pipeline.cap is cap and pipeline.face_detector is detector
    assert order == ['join']
    assert pipeline.get_latest_result() is None
    assert not pipeline.start()
    assert order == ['join']
    alive[0] = False  # old worker exits later, without any sleep
    assert pipeline.start()
    assert order == ['join', 'release', 'close', 'initialize', 'new_thread']
    assert pipeline.session_id != old_session
    assert pipeline._sequence == 0
    pipeline.stop()
    pipeline.stop()
    assert order.count('release') == order.count('close') == 1
