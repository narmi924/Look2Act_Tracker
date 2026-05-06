# Look2Act Agent Notes

## Project Context

Look2Act is a webcam gaze interaction project with two active goals:

- Research path: keep the deep `GazeNetV2 + head pose + screen geometry` pipeline for paper experiments.
- ML baseline path: add `deep_pog`, where a CNN predicts normalized screen points directly before calibration/smoothing.
- Experience path: restore a smooth demo-grade interaction flow with a classic image-processing tracker inspired by `Eye_Touch_Project\Eye_Touch`.

The current practical diagnosis is that deep-model offline metrics are useful, but real-time screen tracking can fail because model-space labels, head-pose rotation, screen geometry, and calibration are tightly coupled. Do not assume "more data" is the first fix.
Training uses Intel XPU + Intel Extension for PyTorch (IPEX). Runtime inference uses ONNX Runtime CPU for cross-platform deployment.

## Working Rules

- All Look2Act Python commands must run in the `gaze-env` conda environment.
- Prefer `conda run --no-capture-output -n gaze-env python ...` for Codex-run checks so output is visible and dependencies match the project.
- The user normally runs commands in Git Bash, not PowerShell. User-facing commands should prefer forward slashes (`configs/experiments/foo.yaml`) or quoted Windows paths. Avoid backslashes in Git Bash examples because `\` is treated as an escape character.
- Do not start long training, LOO evaluation, or long-running GUI sessions from Codex.
- When training or long evaluation is required, provide the exact command and ask the user to run it in a separate terminal.
- Common user-run commands live in `bin\常用命令.txt`.
- Do not modify or delete `dataset_raw`, `dataset_processed`, `checkpoints`, `paper-acm`, or old calibration files unless the user explicitly asks.
- Prefer short tests and static checks from Codex.

## Current Implementation Direction

- Default user-facing backend should be `classic`.
- Deep backend remains available as `deep` for research and paper experiments.
- `deep_pog` is the first ML path to make usable end-to-end: eye crops + head pose -> normalized screen point -> calibration -> smoothing -> fullscreen validation/interaction.
- 3D geometry research stays in `deep`: compare camera/head gaze space and ray-origin choices before any head-local relabeling.
- Calibration files are separated:
  - `calibration_classic.json`
  - `calibration_deep.json`
  - `calibration_deep_pog.json`
- Legacy `calibration.json` may be inspected for diagnosis, but should not be the default tracking calibration.

## Important Commands

Use these only for short checks unless the user asks otherwise:

```bash
conda run --no-capture-output -n gaze-env python -m pytest tests/test_calibration.py tests/test_smoother.py tests/test_tracker_pipeline.py
conda run --no-capture-output -n gaze-env python -m pytest tests/test_classic_tracker.py
conda run --no-capture-output -n gaze-env python scripts/audit_gaze_labels.py
conda run --no-capture-output -n gaze-env python scripts/evaluate_pog.py --checkpoint checkpoints/deep_pog_zero/best_model.pth
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend classic --frames 300
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend deep --deep-space camera --deep-pose-input zero --smoother none --frames 300 --csv diagnostics_deep.csv
conda run --no-capture-output -n gaze-env python scripts/analyze_tracker_diagnostics.py diagnostics_deep.csv
```

Long commands for the user to run manually:

```bash
conda activate gaze-env
python scripts/preprocess.py
python scripts/train.py --config configs/train_config.yaml
python scripts/train.py --config configs/train_pog_config.yaml
python scripts/export_onnx.py --checkpoint checkpoints/best_model.pth --output checkpoints/gaze_net.onnx
python scripts/export_onnx.py --checkpoint checkpoints/deep_pog_zero/best_model.pth --output checkpoints/gaze_pog_zero.onnx
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
python scripts/evaluate_pog.py --checkpoint checkpoints/deep_pog_zero/best_model.pth
python scripts/exp_leave_one_out.py --epochs 50 --device xpu
```
