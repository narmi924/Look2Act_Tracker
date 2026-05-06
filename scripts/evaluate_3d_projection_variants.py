"""Evaluate 3D gaze checkpoint under runtime geometry-contract variants.

This script does not train. It asks which runtime contract turns a predicted
3D gaze vector into the most coherent screen point:

    camera-space/no rotation vs head-space/rotation
    zero origin vs approximate face translation origin
    fixed 500/720/900mm vs dataset median screen distance

Usage:
    conda run --no-capture-output -n gaze-env python scripts/evaluate_3d_projection_variants.py \
        --checkpoint checkpoints/best_model.pth \
        --processed-dir dataset_processed \
        --output-dir evaluation_results/3d_contract_runtime
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from data.dataset import GazeDataset
from models.gaze_net import GazeNet, GazeNetV2
from scripts.analyze_3d_geometry_contract import (
    DEFAULT_GEOMETRY,
    add_implied_distances,
    euler_to_rotation_matrix,
    project_many,
    projection_metrics,
    target_norm_from_frame,
    topology_metrics,
)


def load_config(path: str | None) -> dict:
    if path is None:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_from_checkpoint(checkpoint_path: Path, fallback_config: dict) -> tuple[torch.nn.Module, str, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        model = GazeNet()
        model.load_state_dict(checkpoint)
        model.eval()
        return model, "v1", fallback_config

    config = checkpoint.get("config", fallback_config)
    model_cfg = config.get("model", {})
    model_version = checkpoint.get("model_version", model_cfg.get("version", "v1"))
    channels = model_cfg.get("channels", [32, 64, 128, 256])
    if model_version == "v2":
        model = GazeNetV2(
            num_channels=channels,
            head_pose_dim=model_cfg.get("head_pose_dim", 3),
            fusion_dim=model_cfg.get("fusion_dim", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        model = GazeNet(num_channels=channels)
        model_version = "v1"
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_version, config


def predict_with_torch_model(model: torch.nn.Module, loader: DataLoader, model_version: str) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for batch in loader:
            if model_version == "v2":
                out = model(batch["left_eye"], batch["right_eye"], batch["head_pose"])
            else:
                out = model(batch["eye_img"])
            preds.append(out.cpu().numpy())
    arr = np.vstack(preds) if preds else np.empty((0, 3), dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return arr / norms


def predict_with_onnx(onnx_path: Path, loader: DataLoader) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_name = session.get_outputs()[0].name
    preds = []
    for batch in loader:
        if {"left_eye", "right_eye", "head_pose"}.issubset(set(input_names)):
            feed = {
                "left_eye": batch["left_eye"].numpy(),
                "right_eye": batch["right_eye"].numpy(),
                "head_pose": batch["head_pose"].numpy(),
            }
        else:
            eyes = torch.stack([batch["left_eye"], batch["right_eye"]], dim=1)
            b, two, c, h, w = eyes.shape
            feed = {input_names[0]: eyes.reshape(b * two, c, h, w).numpy()}
        out = session.run([output_name], feed)[0]
        if out.shape[0] == batch["left_eye"].shape[0] * 2:
            out = out.reshape(batch["left_eye"].shape[0], 2, -1).mean(axis=1)
        preds.append(out)
    arr = np.vstack(preds) if preds else np.empty((0, 3), dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return arr / norms


def rotated_predictions(preds: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    rows = []
    for vec, (_, row) in zip(preds, df.iterrows()):
        r = euler_to_rotation_matrix(
            float(row.get("head_yaw", 0.0)),
            float(row.get("head_pitch", 0.0)),
            float(row.get("head_roll", 0.0)),
        )
        rotated = r @ vec
        norm = np.linalg.norm(rotated)
        rows.append(rotated / norm if norm > 1e-12 else rotated)
    return np.asarray(rows, dtype=np.float64)


def approximate_face_origin(distance_mm: float, n: int) -> np.ndarray:
    """Offline approximation for face translation origin.

    Processed labels do not store PnP translation vectors. We therefore use a
    simple camera-axis origin at 80% of screen distance to test the direction
    of the effect without pretending this is the exact runtime tvec.
    """
    return np.tile(np.array([0.0, 0.0, distance_mm * 0.8], dtype=np.float64), (n, 1))


def project_with_origins(
    gaze_vectors: np.ndarray,
    distances: np.ndarray,
    origin_mode: str,
) -> np.ndarray:
    rows = []
    for vec, dist in zip(gaze_vectors, distances):
        if not np.isfinite(dist) or dist <= 0:
            rows.append([np.nan, np.nan])
            continue
        if origin_mode == "zero_origin":
            origin = np.zeros(3, dtype=np.float64)
        elif origin_mode == "face_translation":
            origin = np.array([0.0, 0.0, float(dist) * 0.8], dtype=np.float64)
        else:
            raise ValueError(origin_mode)
        projected = project_many(np.asarray([vec]), float(dist), ray_origin=origin, geometry=DEFAULT_GEOMETRY)[0]
        rows.append(projected.tolist())
    return np.asarray(rows, dtype=np.float64)


def distance_values(df: pd.DataFrame, mode: str, fixed_value: float | None = None) -> np.ndarray:
    if mode == "fixed":
        assert fixed_value is not None
        return np.full(len(df), fixed_value, dtype=np.float64)
    if mode == "dataset_median":
        return np.full(len(df), float(df["implied_screen_distance_mm"].median()), dtype=np.float64)
    raise ValueError(mode)


def evaluate_variants(preds: np.ndarray, df: pd.DataFrame) -> tuple[list[dict[str, object]], dict[str, object]]:
    df = add_implied_distances(df)
    target = target_norm_from_frame(df)
    distances = [
        ("fixed_500mm", "fixed", 500.0),
        ("fixed_720mm", "fixed", 720.0),
        ("fixed_900mm", "fixed", 900.0),
        ("dataset_median", "dataset_median", None),
    ]
    gaze_spaces = {
        "camera_no_rotation": preds,
        "head_space_with_rotation": rotated_predictions(preds, df),
    }
    rows: list[dict[str, object]] = []
    nested: dict[str, object] = {
        "num_samples": int(len(df)),
        "implied_screen_distance_mm": {
            "median": float(df["implied_screen_distance_mm"].median()),
            "mean": float(df["implied_screen_distance_mm"].mean()),
            "min": float(df["implied_screen_distance_mm"].min()),
            "max": float(df["implied_screen_distance_mm"].max()),
        },
        "variants": {},
    }
    for distance_name, distance_mode, fixed_value in distances:
        dist_arr = distance_values(df, distance_mode, fixed_value)
        for gaze_space, gaze_arr in gaze_spaces.items():
            for origin_mode in ("zero_origin", "face_translation"):
                projected = project_with_origins(gaze_arr, dist_arr, origin_mode)
                projection = projection_metrics(projected, target)
                topology = topology_metrics(projected, target)
                name = f"{gaze_space}+{origin_mode}+{distance_name}"
                nested["variants"][name] = {
                    "gaze_space": gaze_space,
                    "ray_origin": origin_mode,
                    "distance": distance_name,
                    "projection": projection,
                    "topology": topology,
                }
                corr = topology.get("corr", {}) if isinstance(topology, dict) else {}
                mono = topology.get("monotonic", {}) if isinstance(topology, dict) else {}
                rows.append(
                    {
                        "variant": name,
                        "gaze_space": gaze_space,
                        "ray_origin": origin_mode,
                        "distance": distance_name,
                        "valid_ratio": projection.get("valid_ratio", 0.0),
                        "in_screen_ratio": projection.get("in_screen_ratio", 0.0),
                        "mean_pixel_error": projection.get("mean_pixel_error", 0.0),
                        "median_pixel_error": projection.get("median_pixel_error", 0.0),
                        "p95_pixel_error": projection.get("p95_pixel_error", 0.0),
                        "area_ratio": topology.get("pred_to_target_area_ratio", 0.0),
                        "corr_x": corr.get("pred_x_vs_target_x", 0.0),
                        "corr_y": corr.get("pred_y_vs_target_y", 0.0),
                        "mono_x": mono.get("rows_pred_x_increases_with_target_x", 0.0),
                        "mono_y": mono.get("cols_pred_y_increases_with_target_y", 0.0),
                    }
                )
    rows.sort(key=lambda row: (float(row["valid_ratio"]) < 1.0, float(row["mean_pixel_error"])))
    return rows, nested


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 3D projection runtime variants")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--processed-dir", default="dataset_processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", default="evaluation_results/3d_contract_runtime")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    split_dir = processed_dir / args.split
    labels_path = split_dir / "labels.csv"
    if not labels_path.exists():
        print(f"[3d-variant] labels not found: {labels_path}")
        return 2
    model_path = Path(args.checkpoint)
    if not model_path.exists():
        print(f"[3d-variant] checkpoint/onnx not found: {model_path}")
        return 2

    df = pd.read_csv(labels_path)
    config = load_config(args.config)
    if model_path.suffix.lower() == ".onnx":
        dataset = GazeDataset(df, image_root=split_dir, model_version="v2", target_mode="gaze3d")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        preds = predict_with_onnx(model_path, loader)
        model_version = "onnx"
    else:
        model, model_version, _ = load_model_from_checkpoint(model_path, config)
        dataset = GazeDataset(df, image_root=split_dir, model_version=model_version, target_mode="gaze3d")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        preds = predict_with_torch_model(model, loader, model_version)

    rows, nested = evaluate_variants(preds, df)
    nested["checkpoint"] = str(model_path)
    nested["model_version"] = model_version
    nested["split"] = args.split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_summary.json").write_text(
        json.dumps(nested, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "variant_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print("[3d-variant] top variants by mean pixel error")
    for row in rows[:8]:
        print(
            "[3d-variant] "
            f"{row['variant']}: mean={row['mean_pixel_error']:.1f}px "
            f"p95={row['p95_pixel_error']:.1f}px valid={row['valid_ratio']:.3f} "
            f"corr=({row['corr_x']:.3f},{row['corr_y']:.3f}) "
            f"mono=({row['mono_x']:.3f},{row['mono_y']:.3f})"
        )
    print(f"[3d-variant] output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
