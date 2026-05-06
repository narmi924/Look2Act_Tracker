import numpy as np
import pandas as pd

from scripts.analyze_3d_geometry_contract import (
    analyze_processed_geometry,
    euler_to_rotation_matrix,
    implied_screen_distance_mm,
    project_gaze_to_norm,
    projection_metrics,
    synthetic_projection_check,
)
from src.data.preprocessing import compute_gaze_vector


def test_project_gaze_to_norm_inverts_synthetic_label_geometry():
    gx, gy, gz = compute_gaze_vector(0.75 * 1920, 0.25 * 1080, 1920, 1080)

    projected = project_gaze_to_norm(np.array([gx, gy, gz]))

    assert projected is not None
    assert np.allclose(projected, [0.75, 0.25], atol=1e-9)


def test_implied_screen_distance_recovers_synthetic_distance():
    gx, gy, gz = compute_gaze_vector(0.75 * 1920, 0.25 * 1080, 1920, 1080, distance_mm=650.0)

    distance = implied_screen_distance_mm(np.array([gx, gy, gz]), np.array([0.75, 0.25]))

    assert np.isclose(distance, 650.0, atol=1e-6)


def test_projection_metrics_detects_large_error():
    pred = np.array([[0.1, 0.1], [0.9, 0.9]], dtype=np.float64)
    target = np.array([[0.1, 0.1], [0.1, 0.1]], dtype=np.float64)

    metrics = projection_metrics(pred, target, screen_w=1000, screen_h=1000)

    assert metrics["valid_ratio"] == 1.0
    assert metrics["mean_pixel_error"] > 500.0


def test_euler_to_rotation_matrix_is_orthonormal():
    R = euler_to_rotation_matrix(10.0, 20.0, -30.0)

    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-9)


def test_analyze_processed_geometry_reports_camera_space_contract():
    rows = []
    for nx, ny in [(0.2, 0.2), (0.8, 0.8)]:
        gx, gy, gz = compute_gaze_vector(nx * 1920, ny * 1080, 1920, 1080)
        rows.append(
            {
                "gaze_x": gx,
                "gaze_y": gy,
                "gaze_z": gz,
                "norm_target_x": nx,
                "norm_target_y": ny,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
            }
        )
    summary = analyze_processed_geometry(pd.DataFrame(rows))

    camera = summary["projection_variants"]["camera_space_zero_origin"]
    implied = summary["projection_variants"]["camera_space_implied_screen_distance"]
    assert camera["valid_ratio"] == 1.0
    assert camera["mean_pixel_error"] < 1e-6
    assert implied["mean_pixel_error"] < 1e-6
    assert summary["approx_inverse_local_z_negative_ratio"] == 0.0


def test_synthetic_projection_check_passes():
    assert synthetic_projection_check()["passes"] is True
