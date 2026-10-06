"""Synthetic-only R8 checks: plan, moving-stimulus events, frame writer, pipeline wiring, offline check."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.experiment.frames import FrameWriter, eye_boxes, load_landmarks, read_frames_index
from src.experiment.protocol import Protocol, make_plan, target_at
from src.experiment.r8_protocol import STIMULI, make_r8_plan, marker_positions, path_position, stimulus_at
from src.experiment.r8_pursuit import check, write_synthetic_r8_session
from src.experiment.recording import read_session
from src.tracker.observation import Observation
from src.tracker.pipeline import SystemConfig, TrackerPipeline, TrackerResult
from src.vision.landmarks import landmarks_array

SIZE = (1280, 800)
_APP = None


@pytest.fixture(scope='module')
def qapp():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


class Clock:
    def __init__(self, now=100.):
        self.now = now

    def __call__(self):
        return self.now


def test_r8_plan_shape_and_bounds():
    plan = make_plan(SIZE, 'R8')
    segments = plan['segments']
    assert plan['selection'] == 'R8' and {s['stimulus'] for s in segments} == set(STIMULI)
    assert [s['protocol'] for s in segments[:9]] == ['A'] * 9 and all(s['round'] == 0 for s in segments[:9])
    assert [s['protocol'] for s in segments[-9:]] == ['A'] * 9 and all(s['round'] == 1 for s in segments[-9:])
    assert sum(s['stimulus'] == 'choice' for s in segments) == 20 and sum(s['stimulus'] == 'path' for s in segments) == 7
    offsets = [s['planned_offset_s'] for s in segments]
    assert offsets == sorted(offsets) and math.isclose(plan['duration_s'], sum(s['duration_s'] for s in segments))
    assert plan['parameters']['settling_s'] == .5 and plan['parameters']['transition_guard_s'] == .05
    for segment in segments:
        for phase in np.linspace(0, segment['duration_s'], 40):
            stimulus = stimulus_at(segment, phase, SIZE)
            points = [stimulus['dot']] if 'dot' in stimulus else stimulus['markers']
            for x, y in points:
                assert -1 <= x <= SIZE[0] and -1 <= y <= SIZE[1]
    choice = next(s for s in segments if s['stimulus'] == 'choice')['choice']
    assert 0 <= choice['instructed'] < len(choice['anchors']) and len(choice['phases']) == len(choice['anchors'])
    with pytest.raises(ValueError):
        make_r8_plan(SIZE, dict(unknown=1))


def test_path_and_marker_math_are_deterministic():
    path = dict(type='hsweep', y=.5, duration_s=8.)
    assert np.allclose(path_position(path, 0., SIZE), [SIZE[0] * .08, 400.])
    assert np.allclose(path_position(path, 4., SIZE), [SIZE[0] * .92, 400.])
    assert np.allclose(path_position(path, 8., SIZE), [SIZE[0] * .08, 400.])
    choice = dict(anchors=[[100., 100.], [300., 100.]], period_s=2., a_px=10., b_px=5., phases=[0., math.pi], directions=[1, -1])
    first, second = marker_positions(choice, 0.)
    assert np.allclose(first, [110., 100.]) and np.allclose(second, [290., 100.])
    with pytest.raises(ValueError):
        path_position(dict(type='spiral', duration_s=1.), 0., SIZE)


def test_protocol_phase_and_moved_follow_paint_and_pause():
    events = []
    clock = Clock()
    protocol = Protocol(make_plan(SIZE, 'R8'), lambda kind, at, **payload: events.append(dict(kind=kind, at=at, **payload)), clock)
    assert protocol.phase() is None and protocol.moved('target_moved', x=1., y=2.) is False
    protocol.start()
    clock.now += 18.5  # first path segment starts at 18 s
    segment = protocol.tick()
    assert segment['stimulus'] == 'path' and math.isclose(protocol.phase(), .5)
    assert protocol.moved('target_moved', phase_s=.5, x=1., y=2.) is False  # not painted yet
    protocol.painted()
    assert protocol.moved('target_moved', phase_s=.5, x=1., y=2.) is True
    moved = [e for e in events if e['kind'] == 'target_moved']
    assert moved[-1]['segment'] == segment['segment'] and moved[-1]['epoch'] == 0 and moved[-1]['at'] == clock.now
    protocol.pause()
    assert protocol.phase() is None and protocol.moved('target_moved', x=0., y=0.) is False
    clock.now += 5.
    protocol.resume()
    assert math.isclose(protocol.phase(), .5)  # pause time is shifted out of the phase
    assert protocol.moved('target_moved', x=0., y=0.) is False  # new epoch needs a new paint
    protocol.painted()
    assert protocol.moved('target_moved', x=0., y=0.) is True
    label = target_at(clock.now + .3, events, .5, .05)  # clear of the resume/paint guard
    assert label['segment'] == segment['segment'] and label['instructed_target'] is None


def _landmarks():
    points = np.zeros((478, 3), dtype=np.float32) + np.array([640., 400., 0.], dtype=np.float32)
    for index, (x, y) in {33: (560., 470.), 133: (606., 470.), 362: (664., 470.), 263: (712., 470.)}.items():
        points[index, :2] = (x, y)
    return points


def test_eye_boxes_follow_corner_width_and_reject_degenerate():
    boxes = eye_boxes(_landmarks(), (1280, 720))
    assert boxes['left'] == [532, 442, 102, 56] and boxes['right'][2] == 106  # 2.2 x corner width
    assert eye_boxes(None, (1280, 720)) is None
    collapsed = _landmarks()
    collapsed[133] = collapsed[33]
    assert eye_boxes(collapsed, (1280, 720)) is None
    clipped = eye_boxes(_landmarks(), (700, 460))
    assert clipped['right'][0] + clipped['right'][2] <= 700 and clipped['right'][2] < 106
    assert clipped['left'][1] + clipped['left'][3] <= 460
    assert eye_boxes(_landmarks(), (600, 460)) is None  # right eye entirely outside the frame


def test_frame_writer_writes_crops_landmarks_and_index(tmp_path):
    writer = FrameWriter(tmp_path / 'session', capacity=8)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :, 1] = 200
    for sequence in (1, 2, 3):
        result = SimpleNamespace(observation=SimpleNamespace(session='s', sequence=sequence, timestamp=10. + sequence),
                                 published_at=10.5 + sequence, valid=True)
        assert writer.submit(result, frame, _landmarks()) is True
    assert writer.submit(SimpleNamespace(observation=SimpleNamespace(session='s', sequence=4, timestamp=14.), published_at=14.5, valid=False),
                         frame, None) is False
    assert writer.submit(SimpleNamespace(observation=None), frame, _landmarks()) is False
    stats = writer.close()
    assert stats['written'] == 3 and stats['submitted'] == 4 and stats['skipped_no_landmarks'] == 1 and stats['dropped'] == 0
    assert stats['writer_error'] is None and not stats['writer_still_alive']
    index = read_frames_index(tmp_path / 'session')
    assert [r['observation'] for r in index] == [['s', 1], ['s', 2], ['s', 3]]
    assert all((tmp_path / 'session' / f).is_file() for r in index for f in r['files'].values())
    landmarks = load_landmarks(tmp_path / 'session')
    assert landmarks.shape == (3, 478, 3) and np.allclose(landmarks[0, 33, :2], (560., 470.))
    meta = json.loads((tmp_path / 'session' / 'landmarks_meta.json').read_text(encoding='utf-8'))
    assert meta['records'] == 3
    assert writer.submit(result, frame, _landmarks()) is False  # closed
    assert writer.close() == stats


def test_landmarks_array_uses_pixels_and_width_scaled_depth():
    lms = [SimpleNamespace(x=.5, y=.25, z=-.1), SimpleNamespace(x=0., y=1., z=.0)]
    array = landmarks_array(lms, 1280, 720)
    assert array.dtype == np.float32 and np.allclose(array, [[640., 180., -128.], [0., 720., 0.]])


def test_pipeline_frame_sink_receives_same_frame_and_landmarks():
    pipeline = TrackerPipeline('', SystemConfig(tracker_backend='classic'))
    pipeline._running = True
    seen = []
    pipeline.frame_sink = lambda result, frame, landmarks: seen.append((result, frame, landmarks))
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    marks = _landmarks()
    pipeline._frame_landmarks = marks
    result = TrackerResult((.5, .5), True, 30., backend='classic', raw_point=(.5, .5),
                           observation=Observation('s', 1, 1., 0, 'host_read_completed'))
    pipeline._publish_result(result, frame)
    assert len(seen) == 1 and seen[0][1] is frame and seen[0][2] is marks and seen[0][0].observation.sequence == 1
    pipeline.frame_sink = None
    pipeline._publish_result(result, frame)
    assert len(seen) == 1
    assert pipeline.collect_full_landmarks is False  # default stays off


class FakePipeline:
    def __init__(self, _, config):
        self.config = config
        self.session_id = 'fake-session'
        self.actual_camera_size = (1280, 720)
        self.record_sink = self.observation_sink = self.frame_sink = None
        self.collect_full_landmarks = False
        self.head_pose_estimator = self.screen_geometry = None
        self.model_version = 'classic'
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        return True

    def stop(self):
        self._running = False

    def get_latest_result(self):
        return None

    def get_dispatch_snapshot(self, *, clock):
        return dict(checked_at=clock(), running=self._running, worker_alive=self._running, calibrating=False,
                    session=self.session_id, continuity=0, candidate_continuity=0, latest_valid=False,
                    latest_candidate_valid=False, latest_sequence=0)


def test_window_records_moving_stimuli_and_frames_offscreen(qapp, tmp_path, monkeypatch):
    import src.ui.experiment_window as base
    from src.ui.r8_record_window import R8RecordWindow, SESSION_TYPE
    monkeypatch.setattr(base, 'OUTPUT_ROOT', tmp_path)
    clock = Clock()
    window = R8RecordWindow(SystemConfig(tracker_backend='classic'), clock=clock, pipeline_factory=FakePipeline)
    window.resize(*SIZE)
    window.start_recording()
    assert window.pipeline.collect_full_landmarks is True and window.pipeline.frame_sink is not None
    assert window.recorder.metadata['session_type'] == SESSION_TYPE
    window.tick()
    window.grab()  # paint the first fixation
    clock.now += 18.6  # first pursuit path
    window.tick()
    window.grab()
    window.grab()
    clock.now += 62.5  # choice trials start after 7 path segments (24 + 18 + 20 s)
    window.tick()
    window.grab()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = TrackerResult((.5, .5), True, 30., backend='classic', raw_point=(.5, .5),
                           observation=Observation('fake-session', 7, clock.now, 0, 'host_read_completed'), published_at=clock.now)
    window.pipeline.frame_sink(result, frame, _landmarks())
    directory = window.recorder.directory
    window.finish()
    meta, events, issues = read_session(directory)
    kinds = [e['kind'] for e in events]
    assert 'target_moved' in kinds and 'markers_moved' in kinds and 'target_painted' in kinds
    moved = next(e for e in events if e['kind'] == 'markers_moved')
    assert len(moved['positions']) in (4, 6) and 0 <= moved['instructed'] < len(moved['positions'])
    assert meta['session_type'] == SESSION_TYPE and meta['frames']['written'] == 1 and meta['complete'] is True
    assert (directory / 'frames.jsonl').is_file() and read_frames_index(directory)[0]['observation'] == ['fake-session', 7]
    assert window.pipeline.frame_sink is None
    window.close()


def test_synthetic_session_check_reports_feasibility(tmp_path):
    source = tmp_path / 'r8'
    write_synthetic_r8_session(source, noise_px=.3)
    report = check(source)
    integrity = report['integrity']
    assert integrity['complete'] and not report['issues'] and integrity['missing_png'] == 0
    assert integrity['landmark_index_consistent'] and integrity['image_records'] == integrity['valid_producers']
    assert integrity['segments']['painted'] == integrity['segments']['planned']
    choice = report['choice']['lag_150ms']
    assert choice['scored'] == 20 and choice['accuracy'] >= .9
    calibration = report['calibration']['lag_150ms']['train']
    assert calibration['pursuit_paths']['n'] > 500 and calibration['pursuit_paths']['holdout_mean_px'] < 60  # 0.3 px pupil noise is ~38 px here
    assert calibration['fixation_round0']['holdout_mean_px'] < 60
    assert (source / 'r8_check.json').is_file()
    with pytest.raises((ValueError, OSError)):
        check(tmp_path)  # not a session
