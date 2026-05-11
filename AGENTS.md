# Look2Act Agent Notes

## Project Context

Look2Act Tracker is a Windows desktop gaze interaction project based on a standard webcam. The current product shape is a runnable PyQt6/QFluentWidgets application with startup language/mode selection, camera preview, calibration, fullscreen verification, and dwell-based interaction demos.

The project currently has two active routes:

- **Classic Demo**: the stable demonstration route. It uses Eye_Touch-style classic image-processing features, 25-point calibration, and screen-space smoothing. Use this as the default user-facing fallback for demos and manual experience checks.
- **Deep Research/Demo**: the research route. It uses GazeNetV2, ONNX Runtime CPU inference, and a 3D gaze-to-screen pipeline. It is useful for paper experiments and comparison, but it remains sensitive to head movement.

The current practical diagnosis is that deep-model offline metrics are useful, but real-time screen tracking can fail because model-space labels, head-pose rotation, screen geometry, eye input semantics, calibration, and smoothing are tightly coupled. Do not assume "more data" or "retraining" is the first fix.

Training uses Intel XPU + Intel Extension for PyTorch (IPEX). Runtime inference uses ONNX Runtime CPU for cross-platform Windows deployment.

The current best Deep Demo contract is camera-space gaze, zero pose input, zero ray origin, 720 mm screen plane, swapped eye input, and EMA smoothing.

## Current Facts

- Main app entry: `main.py`.
- Current public-facing configs: `configs/classic.yaml` and `configs/deep.yaml`.
- Recommended demo configs remain:
  - `configs/experiments/system_classic_demo.yaml`
  - `configs/experiments/system_deep_demo.yaml`
- Equivalent Deep research config:
  - `configs/experiments/system_deep_camera_zero_720_pose_zero_swap_ema.yaml`
- Main demonstration calibration setting is 25-point calibration. Treat 9-point calibration as an experimental comparison unless a task explicitly says otherwise.
- Calibration target generation is shared through `src/ui/calibration_points.py`.
- Classic calibration uses 5x5 polynomial fitting. Deep with 25 points also uses polynomial fitting; smaller Deep runs may use affine fitting.
- User-editable runtime configuration and calibration files should be stored in user-local runtime paths, not hard-coded project-root files, unless the existing code path explicitly does so.
- Legacy `calibration.json` may be inspected for diagnosis, but should not be the default tracking calibration.

## Paper And Artifact Context

- Final project paper files live under `docs/final-project-paper/`.
- Main Word paper path:
  - `docs/final-project-paper/论文/实时视线追踪技术研究与实现.docx`
- Paper writing guide:
  - `docs/final-project-paper/Look2Act_paper_master_guide.md`
- Internetware 2026 tool demonstration artifact lives under `look2act-tool-demo/`. It is a non-code artifact package with reviewer-facing docs, screenshots, schemas, synthetic sample logs, aggregate figures, and screencast material.
- The initial artifact boundary intentionally excludes source code, full dataset, raw participant images, standalone model weights, ONNX files, `.pth` files, and Windows executable packages.

When editing the Word paper or other `.docx` paper materials, use the Documents plugin because the work involves reading and manipulating Word documents. For this project, do **not** perform the plugin's usual final visual QA loop, page-by-page image rendering, or exhaustive screenshot comparison after the edit. The user will manually open and review the Word document after Codex finishes.

When modifying paper `.docx` files, do not change research content, descriptions, methods, claims, conclusions, terminology, or paper stance unless the user gives a specific command for those content changes. By default, future paper edits should focus on formatting, layout, style, pagination, headings, captions, references, tables, figures, and other document-structure work.

## Working Rules

- All Look2Act Python commands must run in the `gaze-env` conda environment.
- Prefer `conda run --no-capture-output -n gaze-env python ...` for Codex-run checks so output is visible and dependencies match the project.
- The user normally runs commands in Git Bash, not PowerShell. User-facing commands should prefer forward slashes (`configs/experiments/foo.yaml`) or quoted Windows paths. Avoid backslashes in Git Bash examples because `\` is treated as an escape character.
- Do not start long training, leave-one-user-out evaluation, or long-running GUI sessions from Codex.
- When training, LOO evaluation, or manual GUI verification is required, provide the exact command and ask the user to run it in a separate terminal.
- Common user-run commands live in `bin/常用命令.txt`.
- Do not modify or delete `dataset_raw`, `dataset_processed`, `checkpoints`, `paper-acm`, `evaluation_results`, large videos under `look2act-tool-demo/video`, or old calibration files unless the user explicitly asks.
- Prefer short tests and static checks from Codex.
- Be careful with the dirty worktree. Do not revert unrelated user changes.

## Current Implementation Direction

- Default user-facing backend should be `classic`.
- Deep backend remains available as `deep` for research, demo comparison, and paper experiments.
- `deep_pog` remains available as an experimental baseline, but it is not the current main demonstration path.
- 3D geometry research stays in `deep`: use facts from runtime topology and projection diagnostics before making paper claims.
- Current application flow is: startup language/mode dialog -> home -> camera preview -> calibration -> calibration result -> fullscreen verification -> fullscreen interaction.
- Dwell-based interaction includes the 3x3 launcher and Gomoku interaction demo.
- Settings should keep a basic/advanced separation; model paths, screen physical geometry, and deep research parameters should not be prominent for normal demo users.

## Research And Paper Claims

Use current facts instead of older head-local/PnP assumptions:

- Dataset/evaluation summary: 31 users, 16 devices, 7,395 valid samples.
- Fixed test mean angular error: about 2.82 deg.
- 31-user leave-one-user-out mean angular error: about 2.48 deg.
- 31-user leave-one-user-out mean screen-space error: about 321 px.
- Do not claim mouse-level precision, state-of-the-art accuracy, or robust infrared-eye-tracker replacement.
- Do not claim robust head-motion compensation for the current real-time Deep Demo.
- Describe the current system as low-cost webcam gaze visualization, large-target selection, and gaze-driven interaction prototyping.
- Treat Classic as an engineering/demo fallback, not the deep-model research contribution.
- Treat Deep as the model/geometry/research route whose main lesson is that gaze estimation must be connected carefully to runtime geometry, calibration, and interaction.

## Important Commands

Use these only for short checks unless the user asks otherwise:

```bash
conda run --no-capture-output -n gaze-env python -m pytest tests/test_calibration.py tests/test_smoother.py tests/test_tracker_pipeline.py
conda run --no-capture-output -n gaze-env python -m pytest tests/test_classic_tracker.py tests/test_classic_pipeline.py tests/test_calibration_flow_helpers.py
conda run --no-capture-output -n gaze-env python scripts/audit_gaze_labels.py
conda run --no-capture-output -n gaze-env python scripts/evaluate_pog.py --checkpoint checkpoints/deep_pog_zero/best_model.pth
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend classic --frames 300 --csv diagnostics_classic.csv
conda run --no-capture-output -n gaze-env python scripts/diagnose_tracker.py --backend deep --deep-space camera --deep-pose-input zero --smoother none --frames 300 --csv diagnostics_deep.csv
conda run --no-capture-output -n gaze-env python scripts/analyze_tracker_diagnostics.py diagnostics_deep.csv
conda run --no-capture-output -n gaze-env python scripts/analyze_3d_geometry_contract.py --processed-dir dataset_processed --split test
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
python scripts/exp_leave_one_out.py --epochs 50 --device xpu --resume --output evaluation_results/leave_one_out_resume
```

Demo commands for the user in Git Bash:

```bash
conda activate gaze-env
cd /d/Projects/Look2Act_Tracker_Project
python main.py
python main.py --config configs/classic.yaml
python main.py --config configs/deep.yaml
python main.py --config configs/experiments/system_classic_demo.yaml
python main.py --config configs/experiments/system_deep_demo.yaml
```
