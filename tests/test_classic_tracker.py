import numpy as np

from src.calibration.calibrator import CalibrationModule
from src.tracker.classic import (
    ClassicKalmanSmoother,
    detect_pupil_centroid,
    fuse_eye_features,
    normalize_iris_offset,
    normalize_crop_point,
)


def test_normalize_crop_point_clamps_to_unit_range():
    assert normalize_crop_point(-10, 200, 100, 100) == (0.0, 1.0)
    assert normalize_crop_point(49.5, 49.5, 100, 100) == (0.5, 0.5)


def test_fuse_eye_features_averages_available_points():
    feature = fuse_eye_features((0.25, 0.5), (0.75, 0.25))
    assert feature is not None
    assert feature.point == (0.5, 0.375)
    assert feature.confidence == 1.0

    one_eye = fuse_eye_features((0.1, 0.2), None)
    assert one_eye is not None
    assert one_eye.point == (0.1, 0.2)
    assert one_eye.confidence < 1.0


def test_normalize_iris_offset_centers_feature():
    point = normalize_iris_offset((110.0, 55.0), (100.0, 50.0), 40.0)
    assert point == (0.75, 0.625)
    assert normalize_iris_offset(None, (100.0, 50.0), 40.0) is None


def test_detect_pupil_centroid_finds_dark_blob():
    eye = np.full((80, 120, 3), 230, dtype=np.uint8)
    yy, xx = np.ogrid[:80, :120]
    mask = (xx - 72) ** 2 + (yy - 44) ** 2 <= 10 ** 2
    eye[mask] = 5

    point = detect_pupil_centroid(eye)
    assert point is not None
    assert abs(point[0] - 72 / 119) < 0.08
    assert abs(point[1] - 44 / 79) < 0.08


def test_classic_polynomial_calibration_maps_features_to_screen():
    cm = CalibrationModule(num_points=25, method="polynomial", max_residual_px=1.0)
    for y in np.linspace(0.1, 0.9, 5):
        for x in np.linspace(0.1, 0.9, 5):
            target = (x * 1000.0 + 20.0, y * 600.0 + 30.0)
            cm.add_calibration_point((float(x), float(y)), target)

    residual = cm.calibrate()
    assert residual < 1e-6
    mapped = cm.apply((0.42, 0.73))
    assert abs(mapped[0] - 440.0) < 1e-4
    assert abs(mapped[1] - 468.0) < 1e-4


def test_classic_kalman_smoother_converges_on_constant_point():
    smoother = ClassicKalmanSmoother()
    point = (0.0, 0.0)
    for _ in range(80):
        point = smoother.update((0.4, 0.7))

    assert abs(point[0] - 0.4) < 0.05
    assert abs(point[1] - 0.7) < 0.05
