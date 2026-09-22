"""One numerical consumer for experiments, diagnostic polling and deterministic replay.

It never executes system actions. Dispatch here means permission at a recorded
check instant, not a click. Production TrackingPage keeps its existing UI flow.
"""
import time
from dataclasses import replace
from src.tracker.observation import ObservationGate, ObservationState, dispatch_rejection
from src.tracker.screen_mapping import ScreenMapper
from src.experiment.snapshots import observation_id, safe_reason


class Consumer:
    def __init__(self, config, calibrator, size, clock=time.perf_counter, sink=None):
        self.config, self.calibrator, self.size = config, calibrator, size
        self.clock, self.sink = clock, sink
        self.gate = ObservationGate(config.max_observation_age_ms, clock)
        self.mapper = ScreenMapper()
        self.active = True

    def reset(self, session, reason='context', active=True):
        self.active = active
        self.gate.reset(session)
        self.mapper.reset()
        if self.sink:
            self.sink('context', self.gate.boundary, session=session, reason=reason, active=active)

    def consume(self, result, state_provider):
        at = self.clock()
        if result is not None:
            result = replace(result, error_message=safe_reason(result.error_message))
        state = self.gate.consume(result, now=at) if self.active else self.gate.reject('paused')
        reset = self.gate.reset_required or state is ObservationState.INVALID
        if reset:
            self.mapper.reset()
        out = None
        event = dict(read_at=at, observation_id=observation_id(result), source_valid=None if result is None else result.valid,
                     source_time=None if result is None or result.observation is None else result.observation.timestamp,
                     source_time_source=None if result is None or result.observation is None else result.observation.time_source,
                     source_continuity=None if result is None or result.observation is None else result.observation.continuity,
                     published_at=None if result is None else result.published_at,
                     age_s=None if result is None or result.observation is None else at - result.observation.timestamp,
                     backend=None if result is None else result.backend,
                     raw_units=None if result is None else result.raw_units,
                     gate_state=state.value, gate_reason=self.gate.reason, reset=reset,
                     raw_point=None if result is None else result.raw_point,
                     calibrated_point=None, smoothed_point=None, display_point=None,
                     calibrated_in_bounds=None, smoothed_in_bounds=None, screen_rejection=None,
                     dispatch_state=None, dispatch_rejection=None, dispatch_allowed=False,
                     source_timings={} if result is None else dict(result.timings),
                     processing_timings={'calibration': None, 'smoothing': None},
                     processing_status={'calibration': 'not_executed', 'smoothing': 'not_executed'})
        if state is ObservationState.NEW:
            out = self.mapper.process(result, self.config, self.calibrator, self.size)
            out.screen_rejection = safe_reason(out.screen_rejection)
            for name in ('calibrated_point', 'smoothed_point', 'display_point', 'calibrated_in_bounds',
                         'smoothed_in_bounds', 'screen_rejection', 'processing_timings', 'processing_status'):
                event[name] = getattr(out, name)
            reason = out.screen_rejection
            if reason is None:
                snapshot = state_provider()
                event['dispatch_state'] = snapshot
                reason = dispatch_rejection(out.observation, self.gate.max_age, snapshot, snapshot['checked_at'])
                event['dispatch_rejection'] = reason
                event['dispatch_allowed'] = reason is None
            if reason is not None:
                self.gate.reject(reason)
                self.mapper.reset()
                event['reset'] = True
        event['processed_at'] = self.clock()
        if self.sink:
            self.sink('consume', at, **event)
        return event


class PollSchedule:
    """Consumption cadence independent of printing. No catch-up fabricated ticks."""
    def __init__(self, period=.033, print_interval=.2):
        if period <= 0 or print_interval < 0:
            raise ValueError('invalid polling/printing interval')
        self.period, self.print_interval = period, print_interval
        self.next_poll = self.next_print = float('-inf')

    def due(self, now):
        if now < self.next_poll:
            return False, False
        self.next_poll = now + self.period
        printing = now >= self.next_print
        if printing:
            self.next_print = now + self.print_interval
        return True, printing
