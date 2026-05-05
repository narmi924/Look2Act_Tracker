# Look2Act Recovery Status

## Branch

- Working branch: `feat/look2act-diagnose-classic-flow`
- Remote branch: `origin/feat/look2act-diagnose-classic-flow`
- PR creation was not completed because `gh` is not authenticated locally.
- Browser PR URL:
  `https://github.com/narmi924/Look2Act_Tracker/pull/new/feat/look2act-diagnose-classic-flow`

## What Changed

- Default demo path is now `classic`.
- Deep research path remains available as `deep`.
- Calibration files are separated:
  - `calibration_classic.json`
  - `calibration_deep.json`
- Tracking diagnostics now expose backend, model version, ONNX inputs, raw/calibrated/pre-clamp/clamped points, timing, head pose where available, and calibration metadata.
- Classic tracker uses MediaPipe iris offsets when available and pupil centroid fallback from eye crops.
- Classic calibration uses 5x5 polynomial fitting; deep calibration uses 3x3 affine fitting.
- Calibration no longer hangs forever if a target gets no valid gaze samples; weak targets are skipped and final fitting checks the minimum valid point count.
- Interaction flow defaults to fullscreen verification and standalone interaction windows, with a gaze-playable Gomoku window.

## Manual Smoke Test

Run in a new terminal:

```powershell
cd D:\Projects\Look2Act_Tracker_Project
conda activate gaze-env
python main.py
```

Suggested order:

1. Confirm settings show `classic` backend and `kalman` smoother.
2. Start tracking.
3. Run 5x5 calibration and save it.
4. Load calibration on the tracking page.
5. Open fullscreen verification.
6. Open fullscreen interaction and launch Gomoku.
7. Toggle diagnostics if tracking looks wrong.

## CLI Diagnostics

Classic:

```powershell
conda activate gaze-env
python scripts/diagnose_tracker.py --backend classic --frames 300 --csv diagnostics_classic.csv
```

Deep camera-space experiment:

```powershell
conda activate gaze-env
python scripts/diagnose_tracker.py --backend deep --deep-space camera --smoother none --frames 300 --csv diagnostics_deep.csv
```

## Label Coordinate Audit

Run:

```powershell
conda activate gaze-env
python scripts/audit_gaze_labels.py
```

Current audit result on `dataset_processed`:

- Gaze labels are unit vectors across train/val/test.
- Normalized screen targets are in bounds.
- Target duplicate ratio is high because the dataset contains repeated frames per calibration target.
- `head_pitch` is almost always `abs(pitch) > 90°`.
- Treating stored labels as camera-space and converting with `R^-1 @ gaze` gives nearly all negative local z, which indicates the stored Euler angles should not be used directly for head-local label regeneration.
- Projecting stored labels back to screen with the fixed runtime plane already has large mean error, and a runtime-style ray origin worsens it. This points to distance/screen-geometry mismatch in addition to the rotation-space question.

Research implication: do not run the deep label-regeneration/retraining path until the head-pose coordinate convention and label/runtime screen geometry are fixed or explicitly modeled.

## Short Tests

Codex-run checks used:

```powershell
conda run --no-capture-output -n gaze-env python -m pytest tests\test_calibration.py tests\test_smoother.py tests\test_tracker_pipeline.py tests\test_classic_tracker.py tests\test_classic_pipeline.py tests\test_calibration_flow_helpers.py tests\test_tracking_page_unit.py tests\test_settings_page.py -q
```

Latest result before this document was added: `32 passed, 1 warning`.

## Next Decision After Manual Testing

- If classic works smoothly enough, tune dwell timings and verification/interaction visuals.
- If classic is unstable, inspect `diagnostics_classic.csv` for feature jumps, invalid frames, and calibration residuals.
- If deep output is still unusable, compare `deep_gaze_space=head` against `camera` before retraining labels.
- If retraining labels is needed, first resolve the head-pose Euler convention and the fixed-plane/distance mismatch flagged by `audit_gaze_labels.py`.
- Only run preprocess/train/export/evaluate manually in a separate terminal.
