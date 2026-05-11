# Preparation Report

## Files Created

- `README.md`
- `ARTIFACT_EVALUATION.md`
- `DEMO_SCRIPT.md`
- `SECURITY_AND_PRIVACY.md`
- `MODEL_AND_DATA_AVAILABILITY.md`
- `RELEASE_PLAN.md`
- `LICENSE_NOTICE.md`
- `LIMITATIONS.md`
- `FACTS.md`
- `TODO_ASSETS.md`
- `sample_schema/labels_schema.csv`
- `sample_schema/meta_schema.json`
- `sample_schema/calibration_log_example.json`
- `sample_logs/runtime_status_example.json`
- `sample_logs/evaluation_summary_example.md`
- `video/screencast_link.md`
- `release_notes/v0.1.0-initial-submission.md`
- `screenshots/.gitkeep`
- `figures/.gitkeep`
- `figures/fig_dataset_split_en.png`
- `figures/fig_fixed_test_angle_error_en.png`
- `figures/fig_loo_angle_error_en.png`
- `figures/fig_loo_screen_error_en.png`
- `figures/fig_calibration_strategy_en.png`

## Facts Extracted From The Private Project

- Existing private README and docs describe startup configuration, camera preview, calibration, fullscreen verification, and gaze interaction workflows.
- Private docs identify Classic and Deep modes.
- Private docs identify GazeNetV2 as the Deep model.
- Private docs and configs show Windows desktop assumptions and DirectShow camera configuration.
- Private configs show 25-point calibration for current Classic and Deep demo configs.
- Private calibration summary includes runtime 25-point calibration logs using polynomial mapping.
- Private fixed-test metrics confirm mean angular error 2.8206 deg on 1,390 test samples.
- Private 31-user leave-one-user-out summary confirms mean angular error 2.4788 deg, mean screen-space error 321.3820 px, 31 completed users, and 7,395 total samples.
- Private notes and thesis backup mention 31 users, 16 devices, and 7,395 valid samples.

## Items Intentionally Not Copied

- Source code.
- Full datasets.
- Raw eye-region images.
- Raw face images.
- Participant-level CSV files.
- Model weights.
- ONNX files.
- `.pth` checkpoint files.
- Executable binaries.
- Private configuration files with local paths.
- Existing UI screenshots with desktop background, personal content, Chinese UI text, or uncertain privacy status.

## Assets Still Needed From The User

See [TODO_ASSETS.md](TODO_ASSETS.md). The main manual additions are English screenshots for launch/mode selection, camera preview, 25-point calibration, real-time tracking, fullscreen validation, dwell interaction, the screencast link, and optional future Windows package link/checksum.

## Inconsistencies Or Risks Found

- 25-point vs 9-point: current demo configs and runtime summaries support 25-point calibration; a legacy 9-point calibration file exists and should be described only as an experimental or legacy comparison.
- 16 devices vs other device counts: fixed submission facts and thesis backup state 16 devices, but one private code-evidence note says current evaluation CSV independently resolves fewer device strings. Confirm the 16-device count from original metadata before final paper submission.
- Source release vs artifact-only release: private project includes source code and packaging files, but this public repository is intentionally artifact-only for the initial submission.
- Windows-only vs cross-platform claims: current docs should avoid cross-platform claims. The prototype should be described as Windows desktop only.
- Executable availability: packaging notes exist privately, but a public Windows package requires license and release-boundary review first.

## Suggested Next Steps Before Paper Writing

- Add English screenshots using the filenames in [TODO_ASSETS.md](TODO_ASSETS.md).
- Add screencast link to [video/screencast_link.md](video/screencast_link.md).
- Review generated English figures under `figures/`.
- Confirm the 16-device statistic from collection metadata.
- Finalize documentation-only license choice.
- Audit licenses before any Windows demonstration package is published.
