import pandas as pd

from scripts.analyze_tracker_diagnostics import (
    analyze_file,
    bool_ratio,
    point_extent,
    point_jitter,
)


def test_bool_ratio_parses_common_true_values():
    df = pd.DataFrame({"valid": ["True", "false", "1", "yes", ""]})

    assert bool_ratio(df, "valid") == 0.6


def test_point_extent_and_jitter():
    df = pd.DataFrame(
        {
            "raw_x": [0.0, 3.0, 6.0],
            "raw_y": [0.0, 4.0, 8.0],
        }
    )

    extent = point_extent(df, "raw_x", "raw_y")
    jitter = point_jitter(df, "raw_x", "raw_y")

    assert extent["width"] == 6.0
    assert extent["height"] == 8.0
    assert jitter["mean"] == 5.0
    assert jitter["p95"] == 5.0


def test_analyze_file_summarizes_diagnostic_csv(tmp_path):
    csv_path = tmp_path / "diag.csv"
    pd.DataFrame(
        {
            "backend": ["classic", "classic"],
            "valid": [True, False],
            "face_detected": [True, True],
            "fps": [30.0, 20.0],
            "raw_x": [0.1, 0.2],
            "raw_y": [0.3, 0.4],
            "calibrated_x": [100.0, 110.0],
            "calibrated_y": [200.0, 210.0],
            "error": ["", "no face"],
        }
    ).to_csv(csv_path, index=False)

    summary = analyze_file(csv_path)

    assert summary["rows"] == 2
    assert summary["backend_counts"] == {"classic": 2}
    assert summary["valid_ratio"] == 0.5
    assert summary["error_counts"] == {"no face": 1}
