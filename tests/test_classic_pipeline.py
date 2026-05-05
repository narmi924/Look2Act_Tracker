from types import SimpleNamespace

import numpy as np

from src.tracker.classic import ClassicKalmanSmoother
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult


def _classic_face_result(**overrides):
    data = {
        "left_iris_center": (110.0, 100.0),
        "right_iris_center": (210.0, 100.0),
        "left_eye_center": (100.0, 100.0),
        "right_eye_center": (200.0, 100.0),
        "left_eye_width": 100.0,
        "right_eye_width": 100.0,
        "left_eye_crop": None,
        "right_eye_crop": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_process_classic_result_uses_iris_offsets_without_smoothing():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="classic"),
    )
    pipeline._calibration_mode = True
    pipeline.classic_smoother = ClassicKalmanSmoother()

    result = pipeline._process_classic_result(_classic_face_result(), {})

    assert result.valid is True
    assert result.backend == "classic"
    assert np.allclose(result.raw_point, (0.6, 0.5))
    assert np.allclose(result.gaze_point, (0.6, 0.5))
    assert result.debug["feature_method"] == "iris_offset"


def test_process_frame_no_face_result_keeps_backend():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="classic"),
    )
    pipeline.face_detector = SimpleNamespace(
        detect=lambda frame: SimpleNamespace(detected=False)
    )

    result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.valid is False
    assert result.face_detected is False
    assert result.backend == "classic"


def test_process_frame_no_face_fallback_keeps_backend_and_last_point():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="classic"),
    )
    pipeline.face_detector = SimpleNamespace(
        detect=lambda frame: SimpleNamespace(detected=False)
    )
    pipeline._last_valid_result = TrackerResult(
        gaze_point=(0.4, 0.6),
        valid=True,
        fps=30.0,
        backend="classic",
    )

    result = pipeline.process_frame(np.zeros((8, 8, 3), dtype=np.uint8))

    assert result.valid is True
    assert result.face_detected is False
    assert result.gaze_point == (0.4, 0.6)
    assert result.backend == "classic"
