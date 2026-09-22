"""R2 uses synthetic backend outputs; never opens a camera."""
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult
from src.tracker.observation import Observation
from src.tracker.screen_mapping import ScreenMapper
from src.tracker.smoother import GazeSmoother
from src.tracker.classic import ClassicScreenSmoother
from src.calibration.calibrator import CalibrationModule
from src.calibration.serializer import save_calibration, load_calibration
from tests.test_observation_safety import deep_pipeline, interaction, qapp


@pytest.mark.parametrize('backend', ['deep', 'deep_pog'])
def test_backend_raw_is_identical_in_calibration_and_tracking(backend):
    pipeline, _, _ = deep_pipeline(backend)
    if backend == 'deep':
        # Controlled geometric output exposes the requested clamp flag.
        pipeline.screen_geometry.world_to_screen_px = lambda hit, clamp: (-200., 100.) if not clamp else (0., 100.)
    else:
        pipeline.onnx_session.run = lambda *a: [np.array([[-200. / 1920., 100. / 1080.]])]
    points = []
    for mode in (True, False):
        pipeline.set_calibration_mode(mode)
        result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result.valid
        points.append(result.raw_point)
    assert points[0] == pytest.approx((-200., 100.))
    assert points[1] == pytest.approx(points[0])

def raw_result(point, backend='deep'):
    return TrackerResult(point, True, 30., raw_point=point, backend=backend,
                         observation=Observation('s', 1, 10., 0))


def calibrator(fn):
    return SimpleNamespace(is_calibrated=True, apply=fn)


def test_classic_raw_definition_unchanged_between_modes():
    from tests.test_classic_pipeline import _classic_face_result
    pipeline = TrackerPipeline('', SystemConfig())
    face = _classic_face_result(detected=True)
    pipeline.face_detector = SimpleNamespace(detect=lambda frame: face)
    outputs = []
    for mode in (True, False):
        pipeline.set_calibration_mode(mode)
        result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result.valid
        assert result.raw_units == 'camera_normalized_feature'
        outputs.append(result.raw_point)
    assert outputs[0] == outputs[1]
    assert outputs[0] == pytest.approx((.45, .3), abs=.04)


@pytest.mark.parametrize('backend', ['deep', 'deep_pog'])
def test_backend_never_updates_output_filter(backend):
    pipeline, _, _ = deep_pipeline(backend)
    pipeline.smoother = SimpleNamespace(update=lambda p: pytest.fail('backend must not smooth'))
    for x in (-200., 2100.):
        if backend == 'deep':
            pipeline.screen_geometry.world_to_screen_px = lambda *a, **k: (x, 100.)
        else:
            pipeline.onnx_session.run = lambda *a: [np.array([[x / 1920., 100. / 1080.]])]
        result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        assert result.valid
        assert result.raw_point == pytest.approx((x, 100.))
        assert result.gaze_point == result.raw_point
        assert result.smoothed_point is None


def test_raw_outside_calibrates_inside_without_alias_fallback():
    calls = []
    def apply(p):
        calls.append(p)
        return p[0] + 300., p[1]
    mapper = ScreenMapper()
    source = raw_result((-200., 100.))
    source.gaze_point = (999., 999.)  # explicitly not the calibration input
    out = mapper.process(source, SystemConfig(smoother_type='none'), calibrator(apply), (1000, 800))
    assert calls == [(-200., 100.)]
    assert out.raw_point == source.raw_point
    assert out.calibrated_point == out.smoothed_point == out.display_point == (100., 100.)
    assert out.observation is source.observation
    assert source.calibrated_point is None
    assert out.screen_rejection is None
    missing = mapper.process(replace(source, raw_point=None), SystemConfig(), calibrator(apply), (1000, 800))
    assert missing.screen_rejection == 'missing_or_invalid_raw_point'
    assert len(calls) == 1


def test_nonlinear_calibration_precedes_ema_with_independent_reference():
    mapper = ScreenMapper()
    config = SystemConfig(smoother_alpha=.3)
    calls = []
    def nonlinear(p):
        calls.append(p)
        return (p[0] ** 2, p[1])
    cal = calibrator(nonlinear)
    outputs = [mapper.process(raw_result((x, 100.)), config, cal, (1000, 800)) for x in (2., 4., 6.)]
    assert [o.calibrated_point[0] for o in outputs] == [4., 16., 36.]
    assert [o.smoothed_point[0] for o in outputs] == pytest.approx([4., 7.6, 16.12])
    assert outputs[-1].smoothed_point[0] != pytest.approx(13.1044)  # C(S(raw))
    assert calls == [(2., 100.), (4., 100.), (6., 100.)]


@pytest.mark.parametrize('backend', ['classic', 'deep', 'deep_pog'])
@pytest.mark.parametrize('mode', ['none', 'kalman', 'ema'])
def test_existing_filter_algorithms_and_parameters(backend, mode):
    config = SystemConfig(tracker_backend=backend, smoother_type=mode, smoother_alpha=.18)
    mapper = ScreenMapper()
    reference = ClassicScreenSmoother(history_len=60) if backend == 'classic' else GazeSmoother(.18)
    cal = calibrator(lambda p: (p[0] * 1000., p[1] * 800.)) if backend == 'classic' else None
    for point in ((.2, .3), (.4, .5), (.3, .4)) if backend == 'classic' else ((200., 240.), (400., 400.), (300., 320.)):
        output = mapper.process(raw_result(point, backend), config, cal, (1000, 800))
        calibrated = cal.apply(point) if cal else point
        expected = calibrated if mode == 'none' else reference.update(calibrated)
        assert output.smoothed_point == pytest.approx(expected)
        assert not output.screen_rejection
    if mode == 'none':
        assert mapper.ema._prev is None
        assert not mapper.classic.history


@pytest.mark.parametrize('point,allowed', [((0., 0.), True), ((999., 799.), True),
                                          ((1000., 100.), False), ((100., 800.), False),
                                          ((-.001, 100.), False)])
def test_pixel_boundaries_and_display_do_not_replace_estimates(point, allowed):
    mapper = ScreenMapper()
    source = raw_result(point)
    out = mapper.process(source, SystemConfig(), None, (1000, 800))
    assert out.raw_point == out.calibrated_point == point
    assert (out.screen_rejection is None) == allowed
    if not allowed:
        assert out.smoothed_point is None  # filter was not fed an invalid measurement
        assert out.display_point != point
        assert mapper.ema._prev is None
    assert source.raw_point == point


@pytest.mark.parametrize('stage', ['calibration', 'smoothing'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., 1000.])
def test_invalid_stage_never_becomes_actionable_via_clipping(stage, bad):
    class Filter:
        value = (bad, 100.)
        def reset(self): pass
        def update(self, p): return self.value
    smoother = Filter()
    mapper = ScreenMapper(smoother)
    cal = calibrator(lambda p: (bad, 100.) if stage == 'calibration' else (100., 100.))
    out = mapper.process(raw_result((.2, .3), 'classic'), SystemConfig(), cal, (1000, 800))
    assert out.screen_rejection
    value = out.calibrated_point if stage == 'calibration' else out.smoothed_point
    assert value[0] == bad or np.isnan(value[0])
    if not np.isfinite(bad): assert out.display_point is None


@pytest.mark.parametrize('method', ['affine', 'polynomial'])
@pytest.mark.parametrize('backend', ['classic', 'deep', 'deep_pog'])
def test_synthetic_legacy_calibration_file_retains_raw_space(tmp_path, method, backend):
    cal = CalibrationModule(method=method)
    for x, y in ((0., 0.), (1., 0.), (0., 1.), (1., 1.), (.5, .2), (.2, .5), (.7, .8)):
        raw = (x, y) if backend == 'classic' else (x * 1000. - 300., y * 800.)
        target = (100. + 400. * x + (40. * x * x if method == 'polynomial' else 0.), 100. + 400. * y)
        cal.add_calibration_point(raw, target)
    cal.calibrate()
    path = tmp_path / 'synthetic_legacy.json'
    save_calibration(cal, str(path), 1000, 800)
    loaded = CalibrationModule()
    load_calibration(loaded, str(path))
    raw = (.3, .4) if backend == 'classic' else (0., 320.)
    output = ScreenMapper().process(raw_result(raw, backend), SystemConfig(smoother_type='none'), loaded, (1000, 800))
    assert output.calibrated_point == pytest.approx((220. + (3.6 if method == 'polynomial' else 0.), 260.))
    assert output.screen_rejection is None


@pytest.mark.parametrize('stage', ['calibration', 'smoothing'])
def test_real_routes_out_of_bounds_interrupt_and_recover(interaction, monkeypatch, stage):
    h = interaction
    start = 10.125
    for i in range(int(np.ceil(h.duration / 125.))): h.feed(start + i * .125)
    calls_before = len(h.calls)
    moves_before = len(h.moves)
    target = h.page.calibrator if stage == 'calibration' else h.page.screen_stabilizer
    name = 'apply' if stage == 'calibration' else 'update'
    original = getattr(target, name)
    monkeypatch.setattr(target, name, lambda p: (-10., 100.))
    h.feed(h.clock.now + .125)
    assert len(h.calls) == calls_before == 0
    assert len(h.moves) == moves_before
    h.cleared()
    monkeypatch.setattr(target, name, original)
    h.feed(h.clock.now + .125)
    assert not h.calls
    h.hold(h.clock.now + .125)
    assert len(h.calls) == 1


def test_classic_without_calibration_is_not_treated_as_pixels():
    out = ScreenMapper().process(raw_result((.3, .4), 'classic'), SystemConfig(), None, (1000, 800))
    assert out.screen_rejection == 'classic_requires_calibration'
    assert out.calibrated_point is None


@pytest.mark.parametrize('stage', ['calibration', 'smoothing'])
def test_processing_exception_rejects_and_clears_filter(stage):
    mapper = ScreenMapper()
    config = SystemConfig()
    cal = calibrator(lambda p: p)
    mapper.process(raw_result((100., 100.)), config, cal, (1000, 800))
    def fail(point):
        raise ValueError('synthetic failure')
    if stage == 'calibration':
        cal.apply = fail
    else:
        mapper.ema.update = fail
    out = mapper.process(raw_result((200., 100.)), config, cal, (1000, 800))
    assert out.screen_rejection == 'screen_processing_failed: synthetic failure'
    assert mapper.ema._prev is None
    assert out.display_point is None


def test_calibration_change_and_reset_clear_ema_history():
    mapper = ScreenMapper()
    config = SystemConfig()
    mapper.process(raw_result((100., 100.)), config, None, (1000, 800))
    mapper.reset()
    assert mapper.process(raw_result((500., 100.)), config, None, (1000, 800)).smoothed_point == (500., 100.)
    new_cal = calibrator(lambda p: (200., 100.))
    assert mapper.process(raw_result((500., 100.)), config, new_cal, (1000, 800)).smoothed_point == (200., 100.)


@pytest.mark.parametrize('backend', ['deep', 'deep_pog'])
def test_real_routes_ema_deduplicates_and_resets_on_interrupt(interaction, backend):
    h = interaction
    h.page.tracker_config = SystemConfig(tracker_backend=backend)
    h.page._update_tracking_data()  # bind the changed backend before acquisition
    h.feed(10.125, backend=backend, point=(100., 100.))
    h.feed(10.25, backend=backend, point=(200., 100.))
    ema = h.page.screen_mapper.ema
    expected = (100. + h.page.tracker_config.smoother_alpha * 100., 100.)
    assert ema._prev == pytest.approx(expected)
    for _ in range(20):
        h.page._update_tracking_data()
    assert ema._prev == pytest.approx(expected)
    h.feed(10.375, backend=backend, valid=False)
    assert ema._prev is None
    h.cleared()
    h.feed(10.5, backend=backend, point=(300., 100.))
    assert ema._prev == (300., 100.)
    assert not h.calls


def test_verification_receives_same_calibrated_then_smoothed_output(qapp, monkeypatch):
    from tests.test_observation_safety import Clock, ControlledTracker, observation
    from src.tracker.observation import ObservationGate
    from src.ui import tracking_page as ui
    page = ui.TrackingPage()
    clock = Clock()
    page.observation_gate = ObservationGate(clock=clock)
    page.tracker = ControlledTracker()
    page.tracker_config = SystemConfig(tracker_backend='deep', smoother_alpha=.3)
    page.calibrator = calibrator(lambda p: (p[0] ** 2, p[1]))
    received = []
    page.verification_window = SimpleNamespace(update_gaze_point=lambda *p: received.append(p), close=lambda: None)
    monkeypatch.setattr(ui.QCursor, 'setPos', lambda *a: pytest.fail('verification must not move mouse'))
    page._update_tracking_data()
    try:
        for seq, x in enumerate((2., 4., 6.), 1):
            clock.now = 10. + seq * .125
            page.tracker.result = observation(seq, clock.now, backend='deep', raw_point=(x, 100.))
            page._update_tracking_data()
        assert [p[0] for p in received] == pytest.approx([4., 7.6, 16.12])
    finally:
        page.close()
