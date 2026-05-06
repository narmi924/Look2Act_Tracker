import numpy as np
import pandas as pd
import torch

from src.data.dataset import GazeDataset
from scripts.evaluate_pog_topology import analyze_prediction_topology
from scripts.train import (
    build_balanced_target_sampler,
    build_model,
    compute_training_loss,
    pog_topology_loss,
    target_mode_for_model,
)
from src.tracker.pipeline import SystemConfig, TrackerPipeline


def test_gaze_dataset_exposes_pog_target(tmp_path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    img_path = image_dir / "eye.png"
    import cv2

    cv2.imwrite(str(img_path), np.zeros((128, 128, 3), dtype=np.uint8))
    df = pd.DataFrame(
        [
            {
                "eye_img_path": "eye.png",
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "gaze_z": 1.0,
                "norm_target_x": 0.25,
                "norm_target_y": 0.75,
                "head_yaw": 0.0,
                "head_pitch": 0.0,
                "head_roll": 0.0,
            }
        ]
    )

    ds = GazeDataset(df, image_root=image_dir, model_version="v2", target_mode="pog2d")
    sample = ds[0]

    assert torch.allclose(sample["pog"], torch.tensor([0.25, 0.75]))
    assert sample["left_eye"].shape == (3, 128, 128)


def test_gaze_dataset_can_zero_head_pose_for_pog_baseline(tmp_path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    img_path = image_dir / "eye.png"
    import cv2

    cv2.imwrite(str(img_path), np.zeros((128, 128, 3), dtype=np.uint8))
    df = pd.DataFrame(
        [
            {
                "eye_img_path": "eye.png",
                "gaze_x": 0.0,
                "gaze_y": 0.0,
                "gaze_z": 1.0,
                "norm_target_x": 0.25,
                "norm_target_y": 0.75,
                "head_yaw": 12.0,
                "head_pitch": -170.0,
                "head_roll": 4.0,
            }
        ]
    )

    ds = GazeDataset(
        df,
        image_root=image_dir,
        model_version="v2",
        target_mode="pog2d",
        head_pose_mode="zero",
    )

    assert torch.allclose(ds[0]["head_pose"], torch.zeros(3))


def test_deep_pog_output_maps_norm_point_to_screen_pixels():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="deep_pog", smoother_type="none"),
    )

    class Screen:
        screen_w_px = 1920
        screen_h_px = 1080
        screen_w_mm = 344.0
        screen_h_mm = 194.0

    class Pose:
        yaw = 1.0
        pitch = 2.0
        roll = 3.0

    pipeline.screen_geometry = Screen()
    result = pipeline._process_deep_pog_output(np.array([0.25, 0.75]), Pose(), {})

    assert result.valid
    assert result.backend == "deep_pog"
    assert result.raw_point == (480.0, 810.0)
    assert result.gaze_point == (480.0, 810.0)
    assert result.debug["output_mode"] == "normalized_point_of_gaze"


def test_deep_ray_origin_zero_origin_experiment():
    pipeline = TrackerPipeline(
        model_path="",
        config=SystemConfig(tracker_backend="deep", deep_ray_origin="zero_origin"),
    )
    origin = pipeline._select_deep_ray_origin(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(origin, np.zeros(3))


def test_train_config_builds_pog_model():
    model = build_model(
        {
            "model": {
                "version": "pog_v1",
                "channels": [8, 8, 8, 8],
                "head_pose_dim": 3,
                "fusion_dim": 16,
                "dropout": 0.0,
                "output_activation": "linear",
            }
        }
    )
    assert type(model).__name__ == "GazeNetPoG"
    assert model.output_activation == "linear"
    assert target_mode_for_model("pog_v1") == "pog2d"


def test_pog_topology_loss_penalizes_reversed_order():
    targets = torch.tensor([[0.1, 0.5], [0.9, 0.5]], dtype=torch.float32)
    ordered = torch.tensor([[0.1, 0.5], [0.9, 0.5]], dtype=torch.float32)
    reversed_pred = torch.tensor([[0.9, 0.5], [0.1, 0.5]], dtype=torch.float32)

    assert pog_topology_loss(ordered, targets).item() < 0.001
    assert pog_topology_loss(reversed_pred, targets).item() > 0.1


def test_compute_training_loss_adds_topology_term():
    targets = torch.tensor([[0.1, 0.5], [0.9, 0.5]], dtype=torch.float32)
    preds = torch.tensor([[0.9, 0.5], [0.1, 0.5]], dtype=torch.float32)

    base = compute_training_loss(preds, targets, "pog2d")
    combined = compute_training_loss(
        preds,
        targets,
        "pog2d",
        {"topology_weight": 0.5, "topology_epsilon": 0.03, "topology_margin": 0.01},
    )

    assert combined > base


def test_balanced_target_sampler_weights_rare_targets_more(tmp_path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    img_path = image_dir / "eye.png"
    import cv2

    cv2.imwrite(str(img_path), np.zeros((128, 128, 3), dtype=np.uint8))
    df = pd.DataFrame(
        [
            {"eye_img_path": "eye.png", "gaze_x": 0, "gaze_y": 0, "gaze_z": 1, "norm_target_x": 0.1, "norm_target_y": 0.1},
            {"eye_img_path": "eye.png", "gaze_x": 0, "gaze_y": 0, "gaze_z": 1, "norm_target_x": 0.1, "norm_target_y": 0.1},
            {"eye_img_path": "eye.png", "gaze_x": 0, "gaze_y": 0, "gaze_z": 1, "norm_target_x": 0.8, "norm_target_y": 0.8},
        ]
    )
    ds = GazeDataset(df, image_root=image_dir, model_version="v2", target_mode="pog2d")

    sampler = build_balanced_target_sampler(ds)

    assert sampler is not None
    assert sampler.weights[2] > sampler.weights[0]


def test_analyze_prediction_topology_detects_ordered_predictions():
    targets = np.array(
        [
            [0.1, 0.1],
            [0.5, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.9],
            [0.9, 0.9],
        ],
        dtype=np.float32,
    )
    predictions = targets.copy()

    summary = analyze_prediction_topology(predictions, targets)

    assert summary["available"] is True
    assert summary["points"] == 6
    assert summary["corr"]["pred_x_vs_target_x"] > 0.9
    assert summary["corr"]["pred_y_vs_target_y"] > 0.9
    assert summary["monotonic"]["rows_pred_x_increases_with_target_x"] == 1.0


def test_analyze_prediction_topology_detects_collapsed_predictions():
    targets = np.array(
        [
            [0.1, 0.1],
            [0.5, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.9],
            [0.9, 0.9],
        ],
        dtype=np.float32,
    )
    predictions = np.full_like(targets, 0.5)

    summary = analyze_prediction_topology(predictions, targets)

    assert summary["pred_to_target_area_ratio"] < 0.01
    assert summary["corr"]["pred_x_vs_target_x"] == 0.0
