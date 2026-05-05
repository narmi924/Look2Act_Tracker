import numpy as np
import pandas as pd

from scripts.audit_gaze_labels import (
    audit_split,
    describe_series,
    euler_to_rotation_matrix,
    flatten_for_csv,
)


def test_describe_series_ignores_non_numeric_values():
    stats = describe_series(pd.Series([1, 2, "bad", 3]))

    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0


def test_euler_to_rotation_matrix_identity():
    mat = euler_to_rotation_matrix(0.0, 0.0, 0.0)

    assert np.allclose(mat, np.eye(3))


def test_audit_split_reports_pose_and_target_risks():
    df = pd.DataFrame(
        {
            "gaze_x": [0.0, 0.0, 0.1],
            "gaze_y": [0.0, 0.1, 0.0],
            "gaze_z": [1.0, 0.995, 0.995],
            "norm_target_x": [0.5, 0.5, 1.2],
            "norm_target_y": [0.5, 0.5, 0.2],
            "head_yaw": [0.0, 10.0, -10.0],
            "head_pitch": [-170.0, -171.0, -169.0],
            "head_roll": [0.0, 0.0, 0.0],
        }
    )

    summary = audit_split("toy", df)
    csv_row = flatten_for_csv(summary)

    assert summary["rows"] == 3
    assert summary["target_duplicate_ratio"] > 0.0
    assert summary["out_of_bounds_target_ratio"] > 0.0
    assert summary["pose_abs_gt_90_ratio"]["head_pitch"] == 1.0
    assert csv_row["split"] == "toy"
    assert csv_row["head_pitch_abs_gt_90_ratio"] == 1.0
