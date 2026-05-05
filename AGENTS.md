# Look2Act Agent Notes

## Project Context

Look2Act is a webcam gaze interaction project with two active goals:

- Research path: keep the deep `GazeNetV2 + head pose + screen geometry` pipeline for paper experiments.
- Experience path: restore a smooth demo-grade interaction flow with a classic image-processing tracker inspired by `Eye_Touch_Project\Eye_Touch`.

The current practical diagnosis is that deep-model offline metrics are useful, but real-time screen tracking can fail because model-space labels, head-pose rotation, screen geometry, and calibration are tightly coupled. Do not assume "more data" is the first fix.

## Working Rules

- All Look2Act Python commands must run in the `gaze-env` conda environment.
- Prefer `conda run --no-capture-output -n gaze-env python ...` for Codex-run checks so output is visible and dependencies match the project.
- Do not start long training, LOO evaluation, or long-running GUI sessions from Codex.
- When training or long evaluation is required, provide the exact command and ask the user to run it in a separate terminal.
- Common user-run commands live in `bin\常用命令.txt`.
- Do not modify or delete `dataset_raw`, `dataset_processed`, `checkpoints`, `paper-acm`, or old calibration files unless the user explicitly asks.
- Prefer short tests and static checks from Codex.

## Current Implementation Direction

- Default user-facing backend should be `classic`.
- Deep backend remains available as `deep` for research and paper experiments.
- Calibration files are separated:
  - `calibration_classic.json`
  - `calibration_deep.json`
- Legacy `calibration.json` may be inspected for diagnosis, but should not be the default tracking calibration.

## Important Commands

Use these only for short checks unless the user asks otherwise:

```powershell
conda run --no-capture-output -n gaze-env python -m pytest tests/test_calibration.py tests/test_smoother.py tests/test_tracker_pipeline.py
conda run --no-capture-output -n gaze-env python -m pytest tests/test_classic_tracker.py
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend classic --frames 300
```

Long commands for the user to run manually:

```powershell
conda activate gaze-env
python scripts/preprocess.py
python scripts/train.py --config configs/train_config.yaml
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output checkpoints/gaze_net.onnx
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
python scripts/exp_leave_one_out.py --epochs 50 --device xpu
```
