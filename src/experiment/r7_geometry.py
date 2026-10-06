"""R7 offline geometry check: head pose and eye position from geometry, eye rotation from the person.

Everything here is read-only numerical analysis on saved R3 sessions. No detector runs, no camera
is opened, and nothing is loaded into the online tracker. The constants below were fixed before
any real session was evaluated; the sensitivity sweep reports what changes when they move, it is
not used to pick a result.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.experiment.r4_analysis import (MAX_REPROJECTION_RMSE_PX, _stat, build_rows, fit_ridge,
                                        guarded_output, hash_file, metrics, predict, split_name)
from src.experiment.r4_features import extract_features, point
from src.experiment.recording import read_session, write_json
from src.vision.head_pose import _MODEL_POINTS_3D, _REQUIRED_KEYS

# --- fixed physical assumptions (engineering values, not subject measurements) ---------------------
MODEL_UNIT_MM = 0.2  # generic 6-point model has outer canthi 450 units apart; adult mean ~90 mm
EYEBALL_RADIUS_MM = 12.0
EYE_BACK_OFFSET_MM = 11.0  # rotation centre sits behind the plane of the eye corners
EYE_CORNER_HALF_WIDTH_MM = 15.0  # only sets each corner's depth under head rotation
# Eyeball rotation centres in the generic model frame (nose tip origin, model units):
# ~31 mm either side of the midline, ~34 mm above the nose tip, ~40 mm behind it.
EYE_CENTER_MODEL = {'left': np.array([-150., 170., -200.]), 'right': np.array([150., 170., -200.])}
EYE_CORNERS = {'left': ('33', '133'), 'right': ('362', '263')}
# Laptop screen plane in the camera frame: camera horizontally centred, 8 mm above the top edge,
# screen glass in the camera's own XY plane with no tilt. Nothing here is fitted.
SCREEN_X0_MM, SCREEN_Y0_MM, SCREEN_Z0_MM, SCREEN_TILT_RAD = 0.0, 8.0, 0.0, 0.0
FX_SWEEP = (1100.0, 900.0)  # recorded fx is always the primary; these are reported, never selected
ROTATION_SOURCES = ('pnp6', 'none')  # pre-registered primary first; 'none' was added after the first real run
MODELS = ('F2', 'G', 'GEO', 'GEO+gain', 'GEO+affine')
PROTOCOLS = {'P1_static_train': (('A_train',), ('A_holdout', 'B_natural', 'B_yaw', 'B_pitch')),
             'P2_with_natural_motion': (('A_train', 'B_natural'), ('A_holdout', 'B_yaw', 'B_pitch'))}
MAIN_PROTOCOL = 'P1_static_train'
MODEL_PARAMETERS = {'F2': 10, 'G': 10, 'GEO': 2, 'GEO+gain': 6, 'GEO+affine': 8}


def mm_per_logical_px(meta):
    """Screen scale from the recorded Qt physical DPI; config millimetres only as a fallback."""
    dpi = (meta.get('screen_scale') or {}).get('physical_dpi_reported')
    if isinstance(dpi, (int, float)) and math.isfinite(dpi) and dpi > 0:
        return 25.4 / float(dpi), 'qt_physical_dpi_reported_not_measured'
    config = meta.get('config') or {}
    width_mm, width_px = config.get('screen_w_mm'), (meta.get('screen_size') or [None])[0]
    if isinstance(width_mm, (int, float)) and width_px:
        return float(width_mm) / float(width_px), 'config_screen_w_mm_not_measured'
    raise ValueError('no screen scale available')


def camera_matrix(pnp_meta, fx=None):
    camera = np.asarray(pnp_meta['camera_matrix'], dtype=float)
    if camera.shape != (3, 3) or not np.isfinite(camera).all() or camera[0, 0] <= 0:
        raise ValueError('invalid camera matrix')
    if fx is not None:
        camera = camera.copy()
        camera[0, 0] = camera[1, 1] = float(fx)
    return camera


def solve_head_pose(snapshot, pnp_meta, camera):
    """solvePnP with the recorded 6-point contract; R4's 20 px reprojection guard; t in model units."""
    if list(pnp_meta.get('keys', [])) != list(_REQUIRED_KEYS):
        return None, 'pnp_contract_differs'
    model = np.asarray(pnp_meta.get('model_points_mm'), dtype=float)
    if model.shape != (6, 3) or not np.array_equal(model, _MODEL_POINTS_3D):
        return None, 'pnp_model_differs'
    saved = snapshot.get('pnp_points') or {}
    try:
        image = np.asarray([saved[k] for k in _REQUIRED_KEYS], dtype=float)
    except (KeyError, TypeError, ValueError):
        return None, 'pnp_points_missing'
    if image.shape != (6, 2) or not np.isfinite(image).all():
        return None, 'pnp_points_invalid'
    distortion = np.zeros((4, 1))
    ok, rvec, tvec = cv2.solvePnP(model, image, camera, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, 'solvepnp_failed'
    rotation, _ = cv2.Rodrigues(rvec)
    projection, _ = cv2.projectPoints(model, rvec, tvec, camera, distortion)
    rmse = float(np.sqrt(np.mean(np.sum((projection.reshape(-1, 2) - image) ** 2, axis=1))))
    if not np.isfinite(rotation).all() or not math.isfinite(rmse) or rmse > MAX_REPROJECTION_RMSE_PX:
        return None, 'reprojection_rejected'
    return dict(R=rotation, t_units=tvec.reshape(3), reprojection_rmse_px=rmse), None


def pupil_points(snapshot, source):
    """Same-frame pupil proxy per eye: R4's dark centroid contract, or MediaPipe iris centre."""
    out = {}
    eyes = snapshot.get('eyes') or {}
    if source == 'dark':
        dark = snapshot.get('classic_dark_centroid') or {}
        for side in EYE_CORNERS:
            q, roi = point(dark.get(side + '_frame')), point(dark.get(side + '_roi'))
            origin = point((eyes.get(side) or {}).get('roi_origin'))
            if (dark.get('units') == 'camera_px' and q is not None and roi is not None and origin is not None
                    and np.allclose(q, roi + origin, atol=1e-5, rtol=0)):
                out[side] = q
    elif source == 'iris':
        for side in EYE_CORNERS:
            eye = eyes.get(side) or {}
            q = point(eye.get('iris_center'))
            if eye.get('iris_status') == 'available' and q is not None:
                out[side] = q
    else:
        raise ValueError('pupil source must be dark or iris')
    return out


def ray_sphere_direction(camera, pixel, center_mm, radius_mm):
    """Direction from the eyeball centre to where the camera ray through `pixel` meets the sphere.

    Returns (unit direction in camera frame, hit). When the ray misses the sphere the closest
    point of the ray is used and hit=False; callers count these.
    """
    d = np.linalg.solve(camera, np.array([pixel[0], pixel[1], 1.0]))
    d /= np.linalg.norm(d)
    b = float(d @ center_mm)
    disc = b * b - float(center_mm @ center_mm) + radius_mm ** 2
    if disc >= 0:
        lam, hit = b - math.sqrt(disc), True  # near intersection: the visible front of the eye
    else:
        lam, hit = b, False
    p = lam * d
    a = p - center_mm
    norm = np.linalg.norm(a)
    if norm < 1e-9:
        return None, False
    return a / norm, hit


HEAD_FLIP = np.diag([1., -1., -1.])  # generic model: y up, +z out of the face; camera: y down, +z away


def tangent(direction):
    """Head-frame tangent coordinates; +z is out of the face in the generic model frame."""
    forward = direction[2]
    if forward <= 1e-6:
        return None
    return np.array([direction[0] / forward, direction[1] / forward])


def eye_center(camera, corners, R, t_mm, side):
    """Eyeball centre: the eye's own corner midpoint, back-projected at the PnP depth, pushed back.

    Lateral position comes from the eye landmarks (sub-pixel), not from the generic PnP model whose
    millimetre-level placement error would tilt a 12 mm sphere's axis by degrees. PnP supplies only
    depth and the direction of "back into the head".
    """
    back = R @ np.array([0., 0., -EYE_BACK_OFFSET_MM])
    points = []
    for corner_px, dx in zip(corners, (-EYE_CORNER_HALF_WIDTH_MM, EYE_CORNER_HALF_WIDTH_MM)):
        # Each corner at its own PnP-implied depth; under yaw the two corners are not equally deep.
        model = R @ (EYE_CENTER_MODEL[side] * MODEL_UNIT_MM + np.array([dx, 0., EYE_BACK_OFFSET_MM])) + t_mm
        ray = np.linalg.solve(camera, np.array([corner_px[0], corner_px[1], 1.0]))
        points.append(ray * (model[2] / ray[2]))
    return np.mean(points, axis=0) + back


def frame_geometry(snapshot, pnp_meta, camera, pupil='dark', rotation='pnp6'):
    """Per-frame geometry: head pose, eyeball centres (mm) and head-frame optical axes.

    rotation='pnp6' uses the saved generic-model PnP orientation (the pre-registered primary).
    rotation='none' keeps only PnP depth and treats the head as facing the camera, so the model
    compensates eyeball position (parallax) but not head rotation.
    """
    if rotation not in ROTATION_SOURCES:
        raise ValueError('rotation must be one of ' + ', '.join(ROTATION_SOURCES))
    if not isinstance(snapshot, dict) or snapshot.get('units') != 'camera_px':
        return None, 'wrong_or_missing_coordinate_source'
    pose, reason = solve_head_pose(snapshot, pnp_meta, camera)
    if pose is None:
        return None, reason
    pupils = pupil_points(snapshot, pupil)
    if len(pupils) != 2:
        return None, 'pupil_missing_' + pupil
    landmarks = snapshot.get('eye_landmarks') or {}
    R = pose['R'] if rotation == 'pnp6' else HEAD_FLIP
    t_mm = pose['t_units'] * MODEL_UNIT_MM
    eyes, misses, feats = {}, 0, []
    for side in ('left', 'right'):
        corners = [point(landmarks.get(k)) for k in EYE_CORNERS[side]]
        if any(c is None for c in corners):
            return None, 'eye_corners_missing'
        center = eye_center(camera, corners, R, t_mm, side)
        axis, hit = ray_sphere_direction(camera, pupils[side], center, EYEBALL_RADIUS_MM)
        if axis is None:
            return None, 'degenerate_axis'
        misses += not hit
        head_axis = R.T @ axis
        tan = tangent(head_axis)
        if tan is None:
            return None, 'axis_not_forward'
        eyes[side] = dict(center_mm=center, axis_cam=axis, axis_head=head_axis)
        feats.extend(tan.tolist())
    width = float(np.linalg.norm(point(landmarks['263']) - point(landmarks['33'])))
    return dict(R=R, R_pnp=pose['R'], t_mm=t_mm, reprojection_rmse_px=pose['reprojection_rmse_px'], eyes=eyes,
                sphere_misses=misses, G=feats, outer_corner_width_px=width), None


def screen_plane(mm_per_px, size_px):
    # The screen faces the user, so the user's "screen right" is the camera's -x.
    width_mm = mm_per_px * size_px[0]
    origin = np.array([SCREEN_X0_MM + width_mm / 2, SCREEN_Y0_MM, SCREEN_Z0_MM])
    ex = np.array([-1., 0., 0.])
    ey = np.array([0., math.cos(SCREEN_TILT_RAD), math.sin(SCREEN_TILT_RAD)])
    return dict(origin=origin, ex=ex, ey=ey, normal=np.cross(ex, ey), mm_per_px=mm_per_px)


def intersect_screen(origin_mm, direction, plane):
    denominator = float(direction @ plane['normal'])
    if abs(denominator) < 1e-9:
        return None
    lam = float((plane['origin'] - origin_mm) @ plane['normal']) / denominator
    if lam <= 0:
        return None
    hit = origin_mm + lam * direction - plane['origin']
    return np.array([hit @ plane['ex'], hit @ plane['ey']]) / plane['mm_per_px']


def unpack(params):
    """[kx, ky] is the pure model; [kx, ky, g11, g12, g21, g22] adds a head-frame measurement gain."""
    params = np.asarray(params, dtype=float)
    gain = params[2:6].reshape(2, 2) if params.size >= 6 else np.eye(2)
    return params[:2], gain


def geo_predict(geo, params, plane, size_px):
    """Visual axis = gain * measured optical tangent + personal constant, in the head frame.

    Head rotation and eyeball position come from PnP geometry and are never fitted; the ray
    starts at the eyeball centre and meets the fixed screen plane.
    """
    kappa, gain = unpack(params)
    points = []
    for eye in geo['eyes'].values():
        tan = tangent(eye['axis_head'])
        if tan is None:
            return None
        corrected = gain @ tan + kappa
        visual_head = np.array([corrected[0], corrected[1], 1.0])
        visual = geo['R'] @ (visual_head / np.linalg.norm(visual_head))
        hit = intersect_screen(eye['center_mm'], visual, plane)
        if hit is None:
            return None
        points.append(hit)
    return np.mean(points, axis=0) / np.asarray(size_px, dtype=float)


def presentation_weights(rows):
    groups = Counter((r['segment'], r['epoch']) for r in rows)
    return np.asarray([1 / (len(groups) * groups[r['segment'], r['epoch']]) for r in rows])


def fit_geo(rows, plane, size_px, with_gain=False):
    """Pure model: two numbers (visual-axis offset). With gain: six, the 2x2 measurement gain added.

    Everything else (head pose, eyeball position, screen plane, scale) stays fixed geometry.
    """
    weights = np.sqrt(presentation_weights(rows))
    targets = np.asarray([r['target_norm'] for r in rows], dtype=float)

    def residuals(params):
        out = np.empty((len(rows), 2))
        for i, row in enumerate(rows):
            pred = geo_predict(row['geo'], params, plane, size_px)
            out[i] = (pred - targets[i]) if pred is not None else 1.0
        return (out * weights[:, None]).ravel()

    x0 = np.array([0., 0., 1., 0., 0., 1.]) if with_gain else np.zeros(2)
    solution = least_squares(residuals, x0=x0, method='lm')
    if not np.isfinite(solution.x).all():
        raise ValueError('geometry fit failed')
    kappa, gain = unpack(solution.x)
    return dict(params=solution.x.tolist(), kappa=kappa.tolist(), gain=gain.tolist(), cost=float(solution.cost),
                n=len(rows), presentations=len({(r['segment'], r['epoch']) for r in rows}))


def fit_affine(predicted, targets, weights):
    """Weighted least squares residual map on the training support (6 parameters)."""
    design = np.hstack([predicted, np.ones((len(predicted), 1))]) * np.sqrt(weights)[:, None]
    response = targets * np.sqrt(weights)[:, None]
    coef = np.linalg.lstsq(design, response, rcond=None)[0]
    return coef


def apply_affine(coef, predicted):
    return np.hstack([predicted, np.ones((len(predicted), 1))]) @ coef


def geo_predictions(rows, params, plane, size_px):
    """Rows whose ray misses the plane stay non-finite; metrics count them, never drop them silently."""
    out = np.full((len(rows), 2), np.nan)
    for i, row in enumerate(rows):
        pred = geo_predict(row['geo'], params, plane, size_px)
        if pred is not None:
            out[i] = pred
    return out


def fit_models(train, plane, size_px):
    models = {'F2': fit_ridge(train, 'F2'), 'G': fit_ridge(train, 'G')}
    geo = fit_geo(train, plane, size_px)
    models['GEO'] = geo
    models['GEO+gain'] = fit_geo(train, plane, size_px, with_gain=True)
    base = geo_predictions(train, geo['params'], plane, size_px)
    coef = fit_affine(base, np.asarray([r['target_norm'] for r in train]), presentation_weights(train))
    models['GEO+affine'] = dict(params=geo['params'], kappa=geo['kappa'], affine=coef.tolist(), n=len(train))
    return models


def predict_model(name, model, rows, plane, size_px):
    if name in ('F2', 'G'):
        return predict(model, rows)
    base = geo_predictions(rows, model['params'], plane, size_px)
    if name in ('GEO', 'GEO+gain'):
        return base
    return apply_affine(np.asarray(model['affine']), base)


def decomposition(rows, predicted, size_px):
    """Per fixed-target contiguous piece: MSE = |bias|^2 + within-piece population variance."""
    size = np.asarray(size_px, dtype=float)
    pieces = defaultdict(list)
    for row, output in zip(rows, predicted):
        if np.isfinite(output).all():
            pieces[(row['segment'], row['epoch'], row['continuous_part'])].append(
                (np.asarray(output) * size, np.asarray(row['target_norm']) * size))
    bias2, disp2, biases, disps, counts = [], [], [], [], []
    for group in pieces.values():
        if len(group) < 5:
            continue
        xy = np.asarray([p for p, _ in group])
        target = group[0][1]
        mean = xy.mean(axis=0)
        bias_sq = float(np.sum((mean - target) ** 2))
        disp_sq = float(np.mean(np.sum((xy - mean) ** 2, axis=1)))
        bias2.append(bias_sq)
        disp2.append(disp_sq)
        biases.append(math.sqrt(bias_sq))
        disps.append(math.sqrt(disp_sq))
        counts.append(len(group))
    if not counts:
        return dict(pieces=0)
    total = float(np.mean(bias2) + np.mean(disp2))
    return dict(pieces=len(counts), frames=int(sum(counts)), bias_px=_stat(biases), within_rms_px=_stat(disps),
                pooled_rms_bias_px=math.sqrt(float(np.mean(bias2))), pooled_rms_within_px=math.sqrt(float(np.mean(disp2))),
                bias_share_of_mse=None if total <= 0 else float(np.mean(bias2) / total))


def pnp_yaw_consistency(rows):
    """Does the saved PnP yaw agree with the eye-width foreshortening it implies? (A_train, per target)

    A real yaw of psi shrinks the outer-corner width by (1 - cos psi). The generic model's depth
    structure is not the subject's, so PnP yaw can be inflated; this reports both numbers.
    """
    by_target = defaultdict(list)
    for row in rows:
        if row.get('geo') and row.get('split') == 'A_train':
            yaw = cv2.RQDecomp3x3(row['geo']['R_pnp'] @ HEAD_FLIP.T)[0][1]
            by_target[row['target_id']].append((yaw, row['geo']['outer_corner_width_px']))
    if not by_target:
        return {}
    table = {target: dict(n=len(v), pnp_yaw_deg=float(np.median([a for a, _ in v])),
                          outer_corner_width_px=float(np.median([w for _, w in v]))) for target, v in sorted(by_target.items())}
    yaws = [v['pnp_yaw_deg'] for v in table.values()]
    widths = [v['outer_corner_width_px'] for v in table.values()]
    half = math.radians((max(yaws) - min(yaws)) / 2)
    return dict(per_target=table, pnp_yaw_range_deg=max(yaws) - min(yaws),
                width_range_px=max(widths) - min(widths),
                width_change_implied_by_pnp_yaw_px=float(np.median(widths)) * (1 - math.cos(half)))


def head_motion_summary(rows):
    """PnP translation range (mm) and PnP rotation relative to the A_train mean, per split."""
    by_split = defaultdict(list)
    for row in rows:
        if row.get('geo') and row.get('split'):
            by_split[row['split']].append(row['geo'])
    reference = [g['R_pnp'] for g in by_split.get('A_train', [])]
    if not reference:
        return {}
    u, _, vt = np.linalg.svd(np.mean(reference, axis=0))
    R0 = u @ vt
    out = {}
    for split, geos in by_split.items():
        t = np.asarray([g['t_mm'] for g in geos])
        angles = [math.degrees(math.acos(min(1., max(-1., (np.trace(g['R_pnp'] @ R0.T) - 1) / 2)))) for g in geos]
        out[split] = dict(n=len(geos), translation_range_mm=(t.max(axis=0) - t.min(axis=0)).tolist(),
                          translation_std_mm=t.std(axis=0).tolist(), distance_mm=_stat(t[:, 2]),
                          pnp_rotation_from_train_deg=_stat(angles),
                          reprojection_rmse_px=_stat([g['reprojection_rmse_px'] for g in geos]))
    return out


def attach_geometry(rows, events, pnp_meta, camera, pupil='dark', rotation='pnp6'):
    snapshots = {e['event_id']: (e['result'].get('numeric_snapshot') or {}) for e in events if e['kind'] == 'producer'}
    reasons = Counter()
    for row in rows:
        row['geo'] = None
        row['features']['G'] = None
        if not row['producer_valid']:
            reasons['producer_invalid'] += 1
            continue
        geo, reason = frame_geometry(snapshots.get(row['event_id']), pnp_meta, camera, pupil, rotation)
        if geo is None:
            reasons[reason] += 1
            continue
        row['geo'] = geo
        row['features']['G'] = geo['G']
    return dict(reasons)


def common_support(rows, split):
    return [r for r in rows if r['split'] == split and r['features']['F2'] is not None and r['geo'] is not None]


def evaluate(rows, plane, size_px, protocol):
    """One protocol: identical training IDs and test IDs for every model."""
    train_splits, test_splits = PROTOCOLS[protocol]
    train = [r for s in train_splits for r in common_support(rows, s)]
    if not train:
        raise ValueError('no common training support')
    models = fit_models(train, plane, size_px)
    result = dict(protocol=protocol, train_splits=list(train_splits), train_n=len(train),
                  train_ids=[r['id'] for r in train],
                  train_presentations=len({(r['segment'], r['epoch']) for r in train}),
                  models={name: dict(parameters=MODEL_PARAMETERS[name]) for name in MODELS},
                  splits={}, predictions=[])
    result['models']['GEO']['kappa'] = models['GEO']['kappa']
    result['models']['GEO+gain'].update(kappa=models['GEO+gain']['kappa'], gain=models['GEO+gain']['gain'])
    for name in MODELS:
        fitted = predict_model(name, models[name], train, plane, size_px)
        result['models'][name]['train'] = metrics(train, fitted, size_px)
        result['models'][name]['train_decomposition'] = decomposition(train, fitted, size_px)
    for split in test_splits:
        support = common_support(rows, split)
        entry = dict(n=len(support), ids=[r['id'] for r in support], models={})
        outputs = {name: predict_model(name, models[name], support, plane, size_px) for name in MODELS}
        for name in MODELS:
            entry['models'][name] = dict(metrics=metrics(support, outputs[name], size_px),
                                         decomposition=decomposition(support, outputs[name], size_px))
            for row, output in zip(support, outputs[name]):
                result['predictions'].append(dict(id=row['id'], model=name, split=split,
                                                  predicted_norm=[float(v) for v in output],
                                                  target_norm=row['target_norm']))
        if support:
            targets = np.asarray([r['target_norm'] for r in support])
            errors = {name: np.linalg.norm((outputs[name] - targets) * np.asarray(size_px), axis=1) for name in MODELS}
            finite = np.all([np.isfinite(e) for e in errors.values()], axis=0)
            entry['paired_common_finite_n'] = int(finite.sum())
            entry['paired_mean_delta_px'] = {f'{name}_minus_F2': float(np.mean(errors[name][finite] - errors['F2'][finite]))
                                             for name in MODELS if name != 'F2'} if finite.any() else {}
        result['splits'][split] = entry
    result['fitted'] = {name: {k: v for k, v in models[name].items() if k not in ('presentation_weight_sums',)}
                        for name in MODELS}
    return result


def strip_private(result):
    """Aggregate view: no sample IDs, per-frame predictions or regression coefficients.

    The two to six geometric measurement numbers (kappa, gain) stay: they are the physical
    diagnostic the report is about, not a per-frame trace or an identity.
    """
    out = {k: v for k, v in result.items() if k not in ('train_ids', 'predictions', 'fitted')}
    out['splits'] = {split: {k: v for k, v in entry.items() if k != 'ids'} for split, entry in result['splits'].items()}
    return out


def _means(result):
    summary = {'train_n': result['train_n'], 'kappa': result['models']['GEO']['kappa'],
               'gain': result['models']['GEO+gain']['gain']}
    for split, entry in result['splits'].items():
        summary[split] = {name: entry['models'][name]['metrics']['error_px']['mean']
                          if entry['models'][name]['metrics'].get('error_px') else None for name in MODELS}
    return summary


def sensitivity(rows, events, pnp_meta, plane, size_px, rotation):
    """Fixed sweep, reported next to the primary run; the primary never changes because of it."""
    out = {}
    for fx in FX_SWEEP:
        attach_geometry(rows, events, pnp_meta, camera_matrix(pnp_meta, fx), 'dark', rotation)
        out[f'fx_{int(fx)}'] = _means(evaluate(rows, plane, size_px, MAIN_PROTOCOL))
    attach_geometry(rows, events, pnp_meta, camera_matrix(pnp_meta), 'iris', rotation)
    out['pupil_iris'] = _means(evaluate(rows, plane, size_px, MAIN_PROTOCOL))
    return out


def rows_for(meta, events, allow_synthetic=False):
    """R4 rows; a synthetic fixture passes only with the caller's explicit opt-in."""
    if meta.get('synthetic'):
        if not allow_synthetic or not meta.get('r7_synthetic_fixture'):
            raise ValueError('synthetic session refused without explicit opt-in')
        meta = {**meta, 'synthetic': False}
    return build_rows(meta, events)


def run_session(source, output, protected, code_commit, allow_synthetic=False):
    """`allow_synthetic` is for the selftest and tests only; real runs never set it."""
    start = time.perf_counter()
    output = guarded_output(output, source, protected)
    meta, events, issues = read_session(source)
    if (issues or not meta.get('complete') or (meta.get('synthetic') and not allow_synthetic) or meta.get('backend') != 'classic'
            or meta.get('plan', {}).get('selection') != 'AB' or len(meta['plan']['segments']) != 30):
        raise ValueError('requires complete, real Classic AB with two A rounds and B')
    if {e['segment'] for e in events if e['kind'] == 'target_painted'} != set(range(30)):
        raise ValueError('not all target presentations were painted')
    before = {name: hash_file(Path(source) / name) for name in ('session.json', 'events.jsonl')}
    size_px = meta['screen_size']
    scale, scale_source = mm_per_logical_px(meta)
    plane = screen_plane(scale, size_px)
    pnp_meta = meta.get('pnp') or {}
    camera = camera_matrix(pnp_meta)
    rows, counts = rows_for(meta, events, allow_synthetic)
    for row in rows:
        row['split'] = split_name(row, meta['plan'])
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(analysis_run=True, r7_geometry=True, synthetic_source=bool(meta.get('synthetic')),
                    source_session_sha256=before['session.json'],
                    source_events_sha256=before['events.jsonl'], source_session_local=str(Path(source).resolve()),
                    code_commit=code_commit, analysis_code_sha256=hash_file(Path(__file__)),
                    screen_size=size_px, mm_per_logical_px=scale, screen_scale_source=scale_source,
                    camera_fx_recorded=float(camera[0, 0]),
                    fixed=dict(model_unit_mm=MODEL_UNIT_MM, eyeball_radius_mm=EYEBALL_RADIUS_MM,
                               eye_back_offset_mm=EYE_BACK_OFFSET_MM,
                               eye_center_model_units={k: v.tolist() for k, v in EYE_CENTER_MODEL.items()},
                               screen_plane=dict(x0_mm=SCREEN_X0_MM, y0_mm=SCREEN_Y0_MM, z0_mm=SCREEN_Z0_MM,
                                                 tilt_rad=SCREEN_TILT_RAD),
                               max_reprojection_rmse_px=MAX_REPROJECTION_RMSE_PX, fx_sweep=list(FX_SWEEP),
                               protocols={k: dict(train=list(v[0]), test=list(v[1])) for k, v in PROTOCOLS.items()},
                               main_protocol=MAIN_PROTOCOL, model_parameters=MODEL_PARAMETERS,
                               rotation_sources=list(ROTATION_SOURCES), primary_rotation=ROTATION_SOURCES[0]),
                    pupil_source='dark', schema_version=1)
    write_json(output / 'manifest.json', manifest)  # frozen before any fit
    by_rotation, local, rejections = {}, {}, {}
    for rotation in ROTATION_SOURCES:
        rejections[rotation] = attach_geometry(rows, events, pnp_meta, camera, 'dark', rotation)
        protocols = {name: evaluate(rows, plane, size_px, name) for name in PROTOCOLS}
        by_rotation[rotation] = dict(protocols={name: strip_private(r) for name, r in protocols.items()},
                                     head_motion=head_motion_summary(rows),
                                     pnp_yaw_consistency=pnp_yaw_consistency(rows),
                                     sphere_misses=sum(r['geo']['sphere_misses'] for r in rows if r['geo']),
                                     sensitivity=sensitivity(rows, events, pnp_meta, plane, size_px, rotation))
        local[rotation] = {name: dict(train_ids=r['train_ids'], fitted=r['fitted'], predictions=r['predictions'])
                           for name, r in protocols.items()}
    attach_geometry(rows, events, pnp_meta, camera, 'dark', ROTATION_SOURCES[0])
    support = {split: dict(measurements=sum(r['split'] == split for r in rows),
                           f2=sum(r['split'] == split and r['features']['F2'] is not None for r in rows),
                           common=len(common_support(rows, split)))
               for split in ('A_train', 'A_holdout', 'B_natural', 'B_yaw', 'B_pitch')}
    aggregate = dict(analysis_run=True, source_counts=counts, geometry_rejections=rejections, support=support,
                     rotation=by_rotation, elapsed_s=time.perf_counter() - start)
    write_json(output / 'aggregate.json', aggregate)
    # Local-only: sample identities, per-frame predictions and personal coefficients.
    write_json(output / 'local_predictions.json', local)
    after = {name: hash_file(Path(source) / name) for name in ('session.json', 'events.jsonl')}
    if after != before:
        raise ValueError('source changed during analysis')
    return aggregate


def summarize_sessions(results):
    """Session-level means per rotation source; each session contributes one number per model and split."""
    out = dict(sessions=len(results), rotation={})
    for rotation in ROTATION_SOURCES:
        table = defaultdict(lambda: defaultdict(list))
        deltas = defaultdict(lambda: defaultdict(list))
        per_session = {}
        for alias, aggregate in results:
            main = aggregate['rotation'][rotation]['protocols'][MAIN_PROTOCOL]
            per_session[alias] = {split: {name: entry['models'][name]['metrics']['error_px']['mean']
                                          if entry['models'][name]['metrics'].get('error_px') else None
                                          for name in MODELS} for split, entry in main['splits'].items()}
            for split, entry in main['splits'].items():
                for name in MODELS:
                    value = entry['models'][name]['metrics'].get('error_px')
                    if value:
                        table[split][name].append(value['mean'])
                for key, value in (entry.get('paired_mean_delta_px') or {}).items():
                    deltas[split][key].append(value)
        out['rotation'][rotation] = dict(
            per_session=per_session,
            session_level_mean={split: {name: float(np.mean(v)) for name, v in names.items()} for split, names in table.items()},
            paired_mean_delta_px={split: {key: dict(mean=float(np.mean(v)), per_session=v) for key, v in keys.items()}
                                  for split, keys in deltas.items()})
    return out


# --- synthetic forward model for tests and the selftest -------------------------------------------
def project(camera, points_mm):
    points = np.atleast_2d(points_mm)
    return np.stack([camera[0, 0] * points[:, 0] / points[:, 2] + camera[0, 2],
                     camera[1, 1] * points[:, 1] / points[:, 2] + camera[1, 2]], axis=1)


def synthetic_snapshot(camera, R, t_mm, target_mm, kappa, frame_size=(1280, 720), noise_px=0.0, rng=None):
    """Observed landmarks for a subject whose visual axes meet `target_mm` (camera frame)."""
    rng = rng or np.random.default_rng(0)
    model_mm = _MODEL_POINTS_3D * MODEL_UNIT_MM
    pnp = project(camera, (R @ model_mm.T).T + t_mm)
    snapshot = dict(units='camera_px', frame_size=list(frame_size),
                    pnp_points={k: pnp[i].tolist() for i, k in enumerate(_REQUIRED_KEYS)},
                    eye_landmarks={}, eyes={}, classic_dark_centroid=dict(units='camera_px'))
    for side in ('left', 'right'):
        center = R @ (EYE_CENTER_MODEL[side] * MODEL_UNIT_MM) + t_mm
        visual = target_mm - center
        visual /= np.linalg.norm(visual)
        tan = tangent(R.T @ visual)
        optical_head = np.array([tan[0] - kappa[0], tan[1] - kappa[1], 1.0])
        optical = R @ (optical_head / np.linalg.norm(optical_head))
        pupil = project(camera, center + EYEBALL_RADIUS_MM * optical)[0]
        pupil = pupil + rng.normal(0, noise_px, 2) if noise_px else pupil
        corners = [R @ (EYE_CENTER_MODEL[side] * MODEL_UNIT_MM + np.array([dx, 0., EYE_BACK_OFFSET_MM])) + t_mm
                   for dx in (-EYE_CORNER_HALF_WIDTH_MM, EYE_CORNER_HALF_WIDTH_MM)]
        corner_px = project(camera, np.asarray(corners))
        for key, value in zip(EYE_CORNERS[side], corner_px):
            snapshot['eye_landmarks'][key] = value.tolist()
        origin = np.floor(corner_px.min(axis=0) - 4)
        snapshot['eyes'][side] = dict(roi_origin=origin.tolist(), iris_center=pupil.tolist(), iris_status='available')
        snapshot['classic_dark_centroid'][side + '_frame'] = pupil.tolist()
        snapshot['classic_dark_centroid'][side + '_roi'] = (pupil - origin).tolist()
    return snapshot


def head_pose_for(segment, phase):
    """Scenario: static first round; moved and turned second round; B as instructed motions."""
    base = np.array([0., 150., 600.])
    if segment['protocol'] == 'A' and segment['round'] == 0:
        return HEAD_FLIP, base
    if segment['protocol'] == 'A':
        return cv2.Rodrigues(np.array([0., math.radians(6.), 0.]))[0] @ HEAD_FLIP, base + np.array([35., 15., -50.])
    motion = segment['requested_motion']
    if motion == 'natural':
        return HEAD_FLIP, base + np.array([15. * math.sin(2 * math.pi * phase), 5., 0.])
    if motion == 'yaw':
        return cv2.Rodrigues(np.array([0., math.radians(18. * math.sin(2 * math.pi * phase)), 0.]))[0] @ HEAD_FLIP, base
    return cv2.Rodrigues(np.array([math.radians(10. * math.sin(2 * math.pi * phase)), 0., 0.]))[0] @ HEAD_FLIP, base


def write_synthetic_session(path, *, kappa=(0.02, -0.035), noise_px=0.0, mm_per_px=0.2359, fps=30,
                            frame_size=(1280, 720), screen=(1280, 800), seed=7):
    """Full R3-shaped AB session from the forward model; marked as a fixture, never as real data."""
    from src.experiment.protocol import make_plan
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    plan = make_plan(screen, 'AB')
    camera = np.array([[frame_size[0], 0., frame_size[0] / 2], [0., frame_size[0], frame_size[1] / 2], [0., 0., 1.]])
    pnp_meta = dict(keys=list(_REQUIRED_KEYS), model_points_mm=_MODEL_POINTS_3D.tolist(),
                    camera_matrix=camera.tolist(), dist_coeffs=[[0.], [0.], [0.], [0.]])
    meta = dict(schema_version=1, experiment_id='r7-fixture-' + str(seed), complete=True, synthetic=True,
                r7_synthetic_fixture=True, backend='classic', screen_size=list(screen), plan=plan, pnp=pnp_meta,
                screen_scale=dict(physical_dpi_reported=25.4 / mm_per_px), camera_actual=list(frame_size))
    plane = screen_plane(mm_per_px, screen)
    events, sequence = [], 0
    for segment in plan['segments']:
        at = 100. + segment['planned_offset_s']
        events.append(dict(kind='target_painted', at=at, event_id=len(events) + 1, epoch=0, **segment))
        px = np.asarray(segment['instructed_target'], dtype=float)
        target_mm = plane['origin'] + px[0] * mm_per_px * plane['ex'] + px[1] * mm_per_px * plane['ey']
        frames = int((segment['duration_s'] - .55) * fps)
        for j in range(frames):
            source_time = at + .55 + j / fps
            R, t = head_pose_for(segment, j / max(1, frames - 1))
            snapshot = synthetic_snapshot(camera, R, t, target_mm, kappa, frame_size, noise_px, rng)
            sequence += 1
            result = dict(observation=dict(session='fixture-tracker', sequence=sequence, timestamp=source_time,
                                           continuity=0), valid=True, point_kind='observed',
                          raw_point=[.5, .5], numeric_snapshot=snapshot)
            events.append(dict(kind='producer', at=source_time + .001, event_id=len(events) + 1, result=result))
    meta['written'] = len(events)
    write_json(path / 'session.json', meta)
    (path / 'events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events), encoding='utf-8')
    return meta, events
