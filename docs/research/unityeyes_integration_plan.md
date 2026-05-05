# UnityEyes / Synthetic Data Integration Plan

## Current Position

UnityEyes-style synthetic data can be useful for Look2Act, but it should not be
used as a blind drop-in replacement for the current training set.

The current label audit shows two unresolved issues in the real-data deep path:

- stored head-pose Euler angles are not directly usable for head-local label regeneration;
- fixed runtime screen geometry does not match the label-generation geometry closely enough.

Therefore, synthetic data should be treated as a controlled pretraining and
coverage tool, not as the first fix for the real-time tracking failure.

## What Synthetic Data Can Help With

- Increase eye appearance diversity: eyelid shape, pupil position, iris color,
  illumination, shadow, partial occlusion, and gaze extremes.
- Provide dense labels for gaze direction under controlled rendering.
- Pretrain the eye-feature backbone before fine-tuning on Look2Act real data.
- Support ablation experiments about data coverage:
  - real only;
  - synthetic only;
  - synthetic pretrain + real fine-tune;
  - synthetic mixed with real using low sampling weight.

## What Synthetic Data Should Not Do

- Do not mix UnityEyes 2D eye parameters directly with Look2Act screen-point
  calibration targets.
- Do not claim synthetic data solves the live tracking failure until real-time
  geometry and calibration diagnostics pass.
- Do not train head-local labels from synthetic data while the real-data head
  pose convention remains unresolved.
- Do not report synthetic gains without a real validation/test split and a
  no-synthetic baseline.

## Recommended Label Contract

Use one explicit training target contract per experiment:

### Option A: Camera-Space Gaze Pretraining

- Synthetic adapter outputs left/right eye crops and a 3D gaze vector in a
  camera-like coordinate system.
- Real-data fine-tuning uses the current `gaze_x/y/z` labels as camera-space
  vectors.
- Runtime deep backend uses `deep_gaze_space=camera`.
- Screen projection must use a ray origin consistent with the label contract.

This is the safest first synthetic path because it avoids the currently suspect
Euler-angle conversion.

### Option B: Head-Local Gaze Pretraining

- Synthetic adapter outputs eye crops, head pose, and head-local gaze.
- Real labels must be regenerated with a verified rotation matrix, not the
  current raw Euler values.
- Runtime deep backend uses `deep_gaze_space=head`.

This path should wait until the head-pose audit is resolved.

## Practical Data Adapter Shape

Create a converter that emits the same processed schema as Look2Act:

```csv
eye_img_path,right_eye_img_path,gaze_x,gaze_y,gaze_z,head_yaw,head_pitch,head_roll,norm_target_x,norm_target_y,session_id,user_id
```

For synthetic rows where no real screen target exists:

- set `norm_target_x/norm_target_y` to empty or a sentinel only if the training
  code does not require them;
- otherwise generate a virtual target plane consistently from the gaze vector
  and synthetic camera geometry.

The adapter should write to a separate directory such as:

```text
dataset_synthetic_processed/
  train/
    images/
    labels.csv
  val/
    images/
    labels.csv
```

Do not merge synthetic rows into `dataset_processed` in-place.

## First Experiment Matrix

Use a small, finite experiment before any long LOO run:

| ID | Training data | Label space | Runtime backend | Goal |
| --- | --- | --- | --- | --- |
| R0 | Look2Act real only | current camera-like labels | deep camera | baseline |
| S0 | synthetic only | camera-space | deep camera | check domain gap |
| S1 | synthetic pretrain + real fine-tune | camera-space | deep camera | test coverage benefit |
| S2 | synthetic mixed low weight + real | camera-space | deep camera | test regularization |

Only after these pass should head-local variants be added.

## Success Criteria

- Offline angular error improves or stays stable on real validation/test users.
- Live diagnostic CSV shows raw deep points are not collapsed and do not jump
  wildly outside the screen before calibration.
- Calibration residual decreases after deep raw output becomes spatially
  meaningful.
- The classic tracker remains the demo fallback while synthetic experiments
  are being evaluated.

## Next Implementation Tasks

1. Keep `classic` as default for demo flow.
2. Fix or replace head-pose Euler convention before generating head-local labels.
3. Add a synthetic dataset adapter only after choosing the target contract.
4. Add a training config that can select `real`, `synthetic`, or `mixed` data
   without modifying existing datasets.
5. Report synthetic results as an ablation, not as the core system dependency.
