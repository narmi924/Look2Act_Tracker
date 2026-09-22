"""Small fixed software fixture, NOT human eye tracking data."""
from src.calibration.calibrator import CalibrationModule
from src.experiment.consumer import Consumer
from src.experiment.protocol import make_plan
from src.experiment.recording import Recorder
from src.experiment.session import VirtualClock, metadata, replay
from src.experiment.snapshots import result_snapshot
from src.tracker.observation import Observation
from src.tracker.pipeline import SystemConfig, TrackerResult


def record_synthetic(directory):
    clock = VirtualClock(10.)
    config = SystemConfig(tracker_backend='deep', smoother_type='ema')
    cal = CalibrationModule()
    cal.set_calibration_data(dict(method='affine', transform_matrix=[[1., 0., 300.], [0., 1., 0.]]))
    plan = make_plan((1000, 800), 'A', {'dwell_s': 1.2})
    plan['segments'] = plan['segments'][:2]
    plan['duration_s'] = 2.4
    meta = metadata(config, cal, (1000, 800), plan, clock(), synthetic=True)
    rec = Recorder(directory, meta)
    consumer = Consumer(config, cal, (1000, 800), clock, rec.emit)
    consumer.reset('synthetic', 'start')
    rec.emit('start', clock())
    rec.emit('target_painted', clock(), **plan['segments'][0], epoch=0, planned_at=10.)
    sequence = continuity = 0
    latest = None

    def publish(at, raw=(-200., 100.), valid=True):
        nonlocal sequence, continuity, latest
        clock.now = at
        sequence += 1
        if not valid:
            continuity += 1
        latest = TrackerResult(raw, valid, 30., raw_point=raw, backend='deep',
                               observation=Observation('synthetic', sequence, at, continuity, 'host_read_completed'),
                               point_kind='observed' if valid else 'held',
                               error_message=None if valid else 'synthetic_face_failure',
                               published_at=at + .005,
                               numeric_snapshot=dict(eye_landmarks={'33': [10., 20.], '133': [20., 20.]},
                                                     head_pose=None, units='camera_px'))
        rec.emit('producer', at + .005, result=result_snapshot(latest))
        clock.now = at + .01

    def state():
        return dict(checked_at=clock(), running=True, worker_alive=True, calibrating=False,
                    session='synthetic', continuity=continuity, latest_valid=latest.valid,
                    latest_sequence=sequence)

    def consume():
        return consumer.consume(latest, state)

    try:
        publish(10.125)
        consume()
        clock.now = 10.15
        consume()  # duplicate
        publish(10.25, valid=False)  # failure overwritten before consumer reads
        publish(10.375)
        consume()
        clock.now = 10.7
        consume()  # stalled source, no new invalid result
        publish(10.75)
        consume()  # long sampling gap
        publish(10.875)
        consume()  # recovery
        publish(11., (900., 100.))
        consume()  # calibrated out of bounds
        publish(11.125)
        consume()
        rec.emit('target_painted', 11.2, **plan['segments'][1], epoch=0, planned_at=11.2)
        for i in range(1, 13):
            publish(11.125 + i * .125, (-50., 100.))
            consume()
        rec.emit('end', clock(), reason='synthetic_complete')
    finally:
        rec.close()
    return replay(directory)[0]
