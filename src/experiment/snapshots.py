"""Explicit numerical allowlists; never serialize a detector/debug object wholesale."""
import copy
from dataclasses import asdict
import re
import math


EYE_INDICES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
PNP_INDICES = dict(nose_tip=1, chin=152, left_eye_outer=33, right_eye_outer=263,
                   left_mouth=61, right_mouth=291)


def safe_reason(reason):
    if reason and (re.search(r'[A-Za-z]:[\\/]', reason) or '/' in reason or '\\' in reason):
        return 'source_error_details_redacted'
    return reason


def face_snapshot(face):
    landmarks = getattr(face, 'landmarks_68', None)
    points = {} if landmarks is None else {
        str(index): [float(v) for v in point]
        for index, point in zip(EYE_INDICES, landmarks[36:48])}
    eyes = {}
    for side in ('left', 'right'):
        roi = getattr(face, side + '_eye_roi', None)
        iris = getattr(face, side + '_iris_center', None)
        eyes[side] = dict(roi_origin=getattr(face, side + '_eye_origin', None),
                          roi_size=None if roi is None else [int(roi.shape[1]), int(roi.shape[0])],
                          iris_center=iris, iris_index=468 if side == 'left' else 473,
                          iris_status='available' if iris is not None else 'unavailable_not_emitted_by_detector')
    return copy.deepcopy(dict(eye_landmarks=points, eyes=eyes, units='camera_px',
                              pnp_points=getattr(face, 'pnp_points_2d', {}), pnp_indices=PNP_INDICES,
                              frame_size=getattr(face, 'frame_size', None),
                              head_pose=None, model_pose_input_deg=None,
                              head_pose_status='not_executed',
                              eye_convention='existing_code_left_33_right_263_no_swap_correction',
                              roi_convention='original_unscaled_ROI_origin_top_left_frame_pixels'))


def pose_snapshot(pose):
    def matrix(name):
        value = getattr(pose, name, None)
        return None if value is None else value.tolist()
    return dict(valid=bool(pose.valid and all(math.isfinite(v) for v in (pose.yaw, pose.pitch, pose.roll))),
                estimator_reported_valid=bool(pose.valid), yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll,
                rotation_matrix=matrix('rotation_matrix'), translation_vec=matrix('translation_vec'),
                units='degrees', origin='estimated_online')


def result_snapshot(result):
    return copy.deepcopy(dict(observation=asdict(result.observation) if result.observation else None,
                              published_at=result.published_at, backend=result.backend,
                              valid=bool(result.valid), face_detected=bool(result.face_detected),
                              point_kind=result.point_kind, error_message=safe_reason(result.error_message),
                              raw_point=result.raw_point, raw_units=result.raw_units,
                              gaze_point=result.gaze_point, fps=result.fps,
                              timings=result.timings, numeric_snapshot=result.numeric_snapshot))


def restore_result(data):
    from src.tracker.pipeline import TrackerResult
    from src.tracker.observation import Observation
    values = dict(data)
    values.pop('raw_units', None)
    stamp = values.pop('observation')
    return TrackerResult(**values, observation=Observation(**stamp) if stamp else None)


def observation_id(result):
    obs = getattr(result, 'observation', None)
    return None if obs is None else [obs.session, obs.sequence]
