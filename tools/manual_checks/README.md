# Manual UI Checks

This directory stores ad-hoc PyQt/manual checks that are useful during local debugging but should not be collected by the normal pytest suite.

Run them from the repository root so imports and relative resource paths stay consistent.

```bash
conda run --no-capture-output -n gaze-env python tools/manual_checks/calibration_ui_check.py
```
