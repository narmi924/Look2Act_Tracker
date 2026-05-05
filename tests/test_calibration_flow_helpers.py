from src.ui.calibration_page import (
    calibration_path_for_backend,
    min_samples_per_calibration_point,
    min_valid_points_for_calibration,
)


def test_calibration_path_is_backend_specific():
    assert calibration_path_for_backend("classic").name == "calibration_classic.json"
    assert calibration_path_for_backend("deep").name == "calibration_deep.json"
    assert calibration_path_for_backend("unknown").name == "calibration_classic.json"


def test_min_valid_points_for_calibration():
    assert min_valid_points_for_calibration(25, "polynomial") == 12
    assert min_valid_points_for_calibration(9, "polynomial") == 6
    assert min_valid_points_for_calibration(9, "affine") == 3


def test_min_samples_per_calibration_point():
    assert min_samples_per_calibration_point(45) == 15
    assert min_samples_per_calibration_point(12) == 6
