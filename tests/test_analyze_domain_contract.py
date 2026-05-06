import json
from pathlib import Path

import pandas as pd

from scripts.analyze_domain_contract import (
    audit_calibration,
    audit_processed_dataset,
    audit_raw_dataset,
    build_summary,
)


def test_audit_raw_dataset_reports_target_and_pose_stats(tmp_path):
    session = tmp_path / "dataset_raw" / "1"
    session.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "session_id": "s1",
                "user_id": "u1",
                "screen_w": 1000,
                "screen_h": 500,
                "frame_w": 1280,
                "frame_h": 720,
                "target_x": 100,
                "target_y": 50,
                "head_yaw": 10,
                "head_pitch": 120,
                "head_roll": 0,
            },
            {
                "session_id": "s1",
                "user_id": "u1",
                "screen_w": 1000,
                "screen_h": 500,
                "frame_w": 1280,
                "frame_h": 720,
                "target_x": 900,
                "target_y": 450,
                "head_yaw": 20,
                "head_pitch": -130,
                "head_roll": 0,
            },
        ]
    ).to_csv(session / "labels.csv", index=False)

    summary = audit_raw_dataset(tmp_path / "dataset_raw")

    assert summary["available"] is True
    assert summary["rows"] == 2
    assert summary["sessions"] == 1
    assert summary["target_count_rounded"] == 2
    assert summary["pose_abs_gt_90_ratio"]["head_pitch"] == 1.0


def test_audit_processed_dataset_reports_balancing_stats(tmp_path):
    train = tmp_path / "dataset_processed" / "train"
    train.mkdir(parents=True)
    pd.DataFrame(
        [
            {"norm_target_x": 0.1, "norm_target_y": 0.2, "eye_img_path": "a.png", "right_eye_img_path": "b.png"},
            {"norm_target_x": 0.1, "norm_target_y": 0.2, "eye_img_path": "c.png", "right_eye_img_path": "d.png"},
            {"norm_target_x": 0.8, "norm_target_y": 0.9, "eye_img_path": "e.png", "right_eye_img_path": "f.png"},
        ]
    ).to_csv(train / "labels.csv", index=False)

    summary = audit_processed_dataset(tmp_path / "dataset_processed")

    assert summary["available"] is True
    assert summary["splits"]["train"]["rows"] == 3
    assert summary["splits"]["train"]["target_count"] == 2
    assert summary["splits"]["train"]["frames_per_target"]["max"] == 2.0
    assert summary["splits"]["train"]["has_right_eye"] is True


def test_audit_calibration_uses_topology_metrics(tmp_path):
    calibration = tmp_path / "calibration_deep_pog.json"
    calibration.write_text(
        json.dumps(
            {
                "method": "polynomial",
                "residual_mean_px": 20.0,
                "calibration_points": [
                    {"raw": [0, 0], "target": [0, 0]},
                    {"raw": [1, 0], "target": [1, 0]},
                    {"raw": [0, 1], "target": [0, 1]},
                    {"raw": [1, 1], "target": [1, 1]},
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = audit_calibration(calibration)

    assert summary["available"] is True
    assert summary["points"] == 4
    assert summary["corr"]["raw_x_vs_target_x"] > 0.9


def test_build_summary_handles_missing_paths(tmp_path):
    class Args:
        raw_dir = str(tmp_path / "missing_raw")
        processed_dir = str(tmp_path / "missing_processed")
        calibration = str(tmp_path / "missing_calibration.json")

    summary = build_summary(Args())

    assert summary["raw_dataset"]["available"] is False
    assert summary["processed_dataset"]["available"] is False
    assert summary["calibration"]["available"] is False
    assert "matches" in summary["crop_contract"]
