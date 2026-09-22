import time
import pytest
from src.tracker.observation import Observation, ObservationGate
from src.tracker.pipeline import SystemConfig, TrackerResult
from src.ui.calibration_page import (
    CalibrationPage,
    CalibrationFullscreenWidget,
    calibration_module_for_config,
    calibration_path_for_backend,
    min_samples_per_calibration_point,
    min_valid_points_for_calibration,
)


class FakeControl:
    def __init__(self):
        self._text = ""
        self._enabled = True
        self._hidden = False
        self._style = ""

    def setText(self, value):
        self._text = value

    def text(self):
        return self._text

    def setEnabled(self, value):
        self._enabled = value

    def isEnabled(self):
        return self._enabled

    def hide(self):
        self._hidden = True

    def show(self):
        self._hidden = False

    def isHidden(self):
        return self._hidden

    def setStyleSheet(self, value):
        self._style = value


class FakeCalibrator:
    @property
    def is_calibrated(self):
        return True


def test_calibration_path_is_backend_specific():
    assert calibration_path_for_backend("classic").name == "calibration_classic.json"
    assert calibration_path_for_backend("deep_pog").name == "calibration_deep_pog.json"
    assert calibration_path_for_backend("deep").name == "calibration_deep.json"
    assert calibration_path_for_backend("unknown").name == "calibration_classic.json"


def test_min_valid_points_for_calibration():
    assert min_valid_points_for_calibration(25, "polynomial") == 4
    assert min_valid_points_for_calibration(9, "polynomial") == 6
    assert min_valid_points_for_calibration(9, "affine") == 3


def test_min_samples_per_calibration_point():
    assert min_samples_per_calibration_point(45) == 15
    assert min_samples_per_calibration_point(12) == 6


def test_deep_experiment_config_uses_25_point_polynomial_calibration():
    config = SystemConfig(
        tracker_backend="deep",
        calibration_num_points=25,
        calibration_save_path="calibration_deep.json",
    )

    calibrator = calibration_module_for_config(config)

    assert calibrator.num_points == 25
    assert calibrator.method.value == "polynomial"


def test_calibration_finished_stays_on_result_actions():
    page = CalibrationPage.__new__(CalibrationPage)
    page.status_label = FakeControl()
    page.residual_label = FakeControl()
    page.start_btn = FakeControl()
    page.save_btn = FakeControl()
    page.load_btn = FakeControl()
    page.home_btn = FakeControl()
    page.calibrator = FakeCalibrator()

    page._on_calibration_finished(True, 32.5)

    assert page.save_btn.text().startswith("保存并进入验证")
    assert page.start_btn.text().startswith("重新校准")
    assert page.load_btn.isHidden()
    assert not page.home_btn.isHidden()


def test_calibration_sampling_discards_initial_transition_frames():
    class FakeTracker:
        def get_latest_result(self):
            return TrackerResult(
                gaze_point=(0.25, 0.5),
                raw_point=(0.25, 0.5),
                observation=Observation("test", time.perf_counter_ns(), time.perf_counter(), 0),
                valid=True,
                fps=30.0,
                face_detected=True,
                backend="classic",
            )

        def set_calibration_mode(self, enabled):
            pass

    class FakeTimer:
        def stop(self):
            pass

    widget = CalibrationFullscreenWidget.__new__(CalibrationFullscreenWidget)
    widget.tracker = FakeTracker()
    widget.observation_gate = ObservationGate()
    widget.observation_gate.reset("test")
    widget.current_samples = []
    widget.sampling_ticks = 0
    widget.discard_initial_frames = 10
    widget.sampling_frames = 45
    widget.max_sampling_ticks = 165
    widget.sampling_timer = FakeTimer()
    widget.update = lambda: None

    for _ in range(widget.discard_initial_frames):
        widget._on_sampling_tick()

    assert widget.current_samples == []

    widget._on_sampling_tick()

    assert widget.current_samples == [(0.25, 0.5)]


@pytest.mark.parametrize('raw', [(100.0, 200.0), None])
def test_calibration_sampling_prefers_raw_point_over_smoothed_point(raw):
    class FakeTracker:
        def get_latest_result(self):
            return TrackerResult(
                gaze_point=(900.0, 900.0),
                raw_point=raw,
                observation=Observation("test", time.perf_counter_ns(), time.perf_counter(), 0),
                valid=True,
                fps=30.0,
                face_detected=True,
                backend="deep",
            )

    class FakeTimer:
        def stop(self):
            pass

    widget = CalibrationFullscreenWidget.__new__(CalibrationFullscreenWidget)
    widget.tracker = FakeTracker()
    widget.observation_gate = ObservationGate()
    widget.observation_gate.reset("test")
    widget.current_samples = []
    widget.sampling_ticks = 10
    widget.discard_initial_frames = 10
    widget.sampling_frames = 45
    widget.max_sampling_ticks = 165
    widget.sampling_timer = FakeTimer()
    widget.update = lambda: None

    widget._on_sampling_tick()

    assert widget.current_samples == ([raw] if raw is not None else [])
