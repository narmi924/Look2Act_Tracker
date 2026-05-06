from scripts.analyze_calibration_topology import analyze_calibration


def _point(raw, target):
    return {"raw": list(raw), "target": list(target)}


def test_analyze_calibration_topology_detects_ordered_grid():
    data = {
        "method": "polynomial",
        "residual_mean_px": 12.0,
        "calibration_points": [
            _point((0, 0), (0, 0)),
            _point((1, 0), (1, 0)),
            _point((0, 1), (0, 1)),
            _point((1, 1), (1, 1)),
        ],
    }

    summary = analyze_calibration(data)

    assert summary["available"] is True
    assert summary["points"] == 4
    assert summary["corr"]["raw_x_vs_target_x"] > 0.9
    assert summary["corr"]["raw_y_vs_target_y"] > 0.9
    assert summary["monotonic"]["rows_raw_x_increases_with_target_x"] == 1.0
    assert summary["monotonic"]["cols_raw_y_increases_with_target_y"] == 1.0


def test_analyze_calibration_topology_detects_folded_grid():
    data = {
        "method": "polynomial",
        "residual_mean_px": 315.0,
        "calibration_points": [
            _point((0.0, 0.0), (0, 0)),
            _point((0.2, 0.1), (1, 0)),
            _point((0.1, 0.2), (2, 0)),
            _point((0.3, 0.4), (0, 1)),
            _point((0.1, 0.3), (1, 1)),
            _point((0.0, 0.2), (2, 1)),
        ],
    }

    summary = analyze_calibration(data)

    assert summary["raw_to_target_area_ratio"] < 0.1
    assert summary["monotonic"]["rows_raw_x_increases_with_target_x"] < 1.0
