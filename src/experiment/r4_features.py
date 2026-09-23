"""R4's fixed, offline numerical feature contract. No detector is run here."""
import math

import numpy as np

from src.tracker.classic import fuse_eye_features


MIN_EYE_WIDTH_PX = 2.0  # Fixed before fitting; reject collapsed/noisy corners.
EYE_CORNERS = {'left': ('33', '133'), 'right': ('362', '263')}
FEATURES = ('C0', 'F0', 'F1', 'F2', 'F3', 'H0', 'F2+H', 'F3+H')


def point(value):
    try:
        array = np.asarray(value, dtype=float)
        if array.shape == (2,) and np.isfinite(array).all():
            return array
    except (TypeError, ValueError):
        pass
    return None


def local_eye_point(a, b, q, *, min_width=MIN_EYE_WIDTH_PX):
    """(q-c) in the fixed a->b basis, divided by eye-corner width."""
    a, b, q = point(a), point(b), point(q)
    if a is None or b is None or q is None:
        return None, 'missing_or_nonfinite_point'
    width = float(np.linalg.norm(b - a))
    if not math.isfinite(width) or width < min_width:
        return None, 'eye_width_below_2px'
    ex = (b - a) / width
    ey = np.array([-ex[1], ex[0]])
    offset = q - (a + b) / 2
    return [float(np.dot(offset, ex) / width), float(np.dot(offset, ey) / width)], None


def extract_features(snapshot):
    """Return each feature and its rejection reason from one saved camera frame."""
    result = {name: None for name in FEATURES if name not in ('C0', 'H0', 'F2+H', 'F3+H')}
    reasons = {name: 'missing_snapshot' for name in result}
    if not isinstance(snapshot, dict) or snapshot.get('units') != 'camera_px':
        return result, {name: 'wrong_or_missing_coordinate_source' for name in result}
    size = snapshot.get('frame_size')
    try:
        width, height = (float(v) for v in size)
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError
    except (TypeError, ValueError):
        return result, {name: 'invalid_frame_size' for name in result}

    dark = snapshot.get('classic_dark_centroid') or {}
    landmarks = snapshot.get('eye_landmarks') or {}
    eyes = snapshot.get('eyes') or {}
    dark_points = {}
    for side in EYE_CORNERS:
        # ROI and absolute point must belong to this same saved camera frame.
        q = point(dark.get(side + '_frame'))
        roi = point(dark.get(side + '_roi'))
        origin = point((eyes.get(side) or {}).get('roi_origin'))
        if (dark.get('units') == 'camera_px' and q is not None and roi is not None
                and origin is not None and np.allclose(q, roi + origin, atol=1e-5, rtol=0)):
            dark_points[side] = q

    if dark_points:
        left, right = dark_points.get('left'), dark_points.get('right')
        fused = fuse_eye_features(None if left is None else tuple(left),
                                  None if right is None else tuple(right), int(width), int(height))
        result['F0'] = [fused.x, fused.y]
        reasons['F0'] = None
    else:
        reasons['F0'] = 'missing_dark_point_or_roi_mismatch'
    if len(dark_points) == 2:
        result['F1'] = [float(dark_points[s][i] / dim) for s in EYE_CORNERS
                        for i, dim in enumerate((width, height))]
        reasons['F1'] = None
    else:
        reasons['F1'] = 'one_or_both_dark_points_unavailable'

    for name, source in (('F2', 'dark'), ('F3', 'iris')):
        values, error = [], None
        for side, (a_key, b_key) in EYE_CORNERS.items():
            if source == 'dark':
                q = dark_points.get(side)
            else:
                eye = eyes.get(side) or {}
                q = eye.get('iris_center') if eye.get('iris_status') == 'available' else None
            if q is None:
                error = 'missing_' + source + '_' + side
                break
            local, problem = local_eye_point(landmarks.get(a_key), landmarks.get(b_key), q)
            if problem:
                error = problem + '_' + side
                break
            values.extend(local)
        if error is None:
            result[name], reasons[name] = values, None
        else:
            reasons[name] = error
    return result, reasons
