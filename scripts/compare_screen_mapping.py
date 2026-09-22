"""Small deterministic R1/R2 mathematical comparison; no camera, Qt or OS actions.

R1 control reproduces the Deep path in the recorded merged baseline:
world_to_screen(clamp=True) -> backend EMA (unless none) -> UI calibration
-> UI display clamp. It is a control only, not a selectable runtime pipeline.
"""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geometry.screen_geometry import ScreenGeometry
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult
from src.tracker.screen_mapping import ScreenMapper
from src.tracker.smoother import GazeSmoother

BASELINE = 'b01b8f1f6c860c3588f4c27dad703f13d930a1e2'
SIZE = (1000, 800)


def compare(raw_sequence, mapping, references, smoothing):
    config = SystemConfig(tracker_backend='deep', smoother_type=smoothing, smoother_alpha=.3)
    mapper = ScreenMapper()
    cal = SimpleNamespace(is_calibrated=True, apply=mapping)
    old_filter = GazeSmoother(.3)
    rows = []
    for raw, reference in zip(raw_sequence, references):
        # Exact R1 Deep order, including its unchanged W-1/H-1 display bounds.
        old_raw = tuple(float(np.clip(v, 0., extent - 1.)) for v, extent in zip(raw, SIZE))
        old_smoothed_raw = old_raw if smoothing == 'none' else old_filter.update(old_raw)
        old_calibrated = mapping(old_smoothed_raw)
        old_display = tuple(float(np.clip(v, 0., extent - 1.)) for v, extent in zip(old_calibrated, SIZE))
        output = mapper.process(TrackerResult(raw, True, 30., raw_point=raw, backend='deep'),
                                config, cal, SIZE)
        assert output.screen_rejection is None
        assert np.allclose(output.smoothed_point, reference, rtol=0, atol=1e-10)
        rows.append(dict(raw=raw, reference=reference,
                         r1=dict(backend_raw=old_raw, smoothed_raw=old_smoothed_raw,
                                 calibrated=old_calibrated, display=old_display),
                         r2=dict(raw=output.raw_point, calibrated=output.calibrated_point,
                                 smoothed=output.smoothed_point, display=output.display_point),
                         r1_error=float(np.linalg.norm(np.array(old_display) - reference)),
                         r2_error=float(np.linalg.norm(np.array(output.smoothed_point) - reference))))
    return rows


def run_comparison():
    # Known translation; early clipping collapses two distinct source values.
    a = compare([(-200., 100.), (-50., 100.)], lambda p: (p[0] + 300., p[1]),
                [(100., 100.), (250., 100.)], 'none')
    # Independent recurrence on squared inputs: 4; .3*16+.7*4; .3*36+.7*7.6.
    b = compare([(2., 100.), (4., 100.), (6., 100.)], lambda p: (p[0] ** 2, p[1]),
                [(4., 100.), (7.6, 100.), (16.12, 100.)], 'ema')
    geometry = ScreenGeometry(1000, 800, 500., 400.)
    geometry.setup_plane(np.array([0., 0., 720.]), np.array([0., 0., 1.]))
    hit = np.array([100., 100., 720.])
    raw = geometry.world_to_screen_px(hit, clamp=False)
    assert raw == geometry.world_to_screen_px(hit, clamp=True) == (200., 200.)
    # deep_pog still uses W/H (not W-1/H-1) to convert normalized model output.
    pipeline = TrackerPipeline('', SystemConfig(tracker_backend='deep_pog'))
    pipeline.screen_geometry = geometry
    pog = pipeline._process_deep_pog_output(np.array([.2, .25]),
                                            SimpleNamespace(yaw=0., pitch=0., roll=0.), {})
    assert pog.raw_point == (200., 200.)
    c = compare([raw], lambda p: (p[0] + 10., p[1] + 20.), [(210., 220.)], 'none')
    return dict(baseline=BASELINE, screen_px=SIZE, ema_alpha=.3, random_seed=None,
                units='screen pixels; B uses a synthetic quadratic calibration',
                A_translation_no_smoothing=a, B_nonlinear_ema=b,
                C_interior_no_smoothing=c, C_deep_pog_raw=pog.raw_point)


if __name__ == '__main__':
    print(json.dumps(run_comparison(), indent=2))
