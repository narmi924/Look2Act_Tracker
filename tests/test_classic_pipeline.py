from types import SimpleNamespace

import numpy as np

from src.tracker.classic import ClassicKalmanSmoother
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult


def _eye_roi(width=80, height=40, pupil=(40, 20)):
    eye = np.full((height, width, 3), 230, dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    mask = (xx - pupil[0]) ** 2 + (yy - pupil[1]) ** 2 <= 8 ** 2
    eye[mask] = 5
    return eye


def _classic_face_result(**overrides):
    data = {
        "left_eye_roi": _eye_roi(pupil=(20, 10)),
        "right_eye_roi": _eye_roi(pupil=(40, 10)),
        "left_eye_origin": (100, 50),
        "right_eye_origin": (200, 50),
        "frame_size": (400, 200),
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_process_classic_result_uses_absolute_pupil_feature():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="classic"),
    )
    pipeline._calibration_mode = True
    pipeline.classic_smoother = ClassicKalmanSmoother()

    result = pipeline._process_classic_result(_classic_face_result(), {})

    assert result.valid is True
    assert result.backend == "classic"
    assert np.allclose(result.raw_point, (0.45, 0.3), atol=0.04)
    assert np.allclose(result.gaze_point, result.raw_point)
    assert result.debug["feature_method"] == "classic_pupil"


def test_process_classic_result_can_disable_smoothing():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="classic", smoother_type="none"),
    )
    pipeline.classic_smoother = ClassicKalmanSmoother()

    result = pipeline._process_classic_result(_classic_face_result(), {})

    assert np.allclose(result.gaze_point, result.raw_point)
    assert result.debug["smoother_type"] == "none"


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

    assert result.valid is False
    assert result.face_detected is False
    assert result.gaze_point == (0.4, 0.6)
    assert result.backend == "classic"
