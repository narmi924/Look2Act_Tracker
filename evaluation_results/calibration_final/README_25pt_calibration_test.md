# 25-point Deep Calibration Manual Test

1. Activate the project environment: `conda activate gaze-env`.
2. Start Deep mode: `python main.py --config configs/deep.yaml`.
3. Open camera preview and confirm the requested camera resolution, face detection, and eye crops.
4. Run the 25-point calibration and save the result.
5. Confirm the file exists at `%APPDATA%/Look2Act/calibration/calibration_deep.json`.
6. Run `python scripts/analyze_calibration_results.py` and copy the Deep row into the paper table.

Do not report the Deep values as measured until `calibration_deep.json` exists.
