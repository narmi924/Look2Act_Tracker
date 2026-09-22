"""Identity and freshness rules shared by gaze consumers.

Times are host monotonic seconds (perf_counter), never sensor exposure times.
The 250 ms default is an engineering starting point, not a validated human limit.
"""
from dataclasses import dataclass
from enum import Enum
import math
import time


def validate_max_age(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("max_observation_age_ms must be finite and positive")
    return value / 1000.0


@dataclass(frozen=True)
class Observation:
    session: str
    sequence: int
    timestamp: float
    continuity: int
    time_source: str = "processing_entry"


class ObservationState(Enum):
    NEW = "new"
    DUPLICATE = "duplicate"
    INVALID = "invalid"


class ObservationGate:
    def __init__(self, max_age_ms=250.0, clock=time.perf_counter):
        self.max_age = validate_max_age(max_age_ms)
        self.clock = clock
        self.session = None
        self.last = None
        self.reason = "not_bound"
        self.interrupted = True
        self.reset_required = True
        self.boundary = clock()

    def reset(self, session):
        self.session = session
        self.last = None
        self.boundary = self.clock()
        self.interrupted = True

    def reject(self, reason):
        self.reason = reason
        self.interrupted = True
        self.reset_required = True
        return ObservationState.INVALID

    def consume(self, result):
        now = self.clock()
        obs = getattr(result, "observation", None)
        self.reset_required = False
        if not isinstance(obs, Observation) or not self.session or obs.session != self.session:
            return self.reject("missing_or_old_session")
        if (type(obs.sequence) is not int or obs.sequence < 1
                or type(obs.continuity) is not int or obs.continuity < 0
                or not isinstance(obs.timestamp, (int, float)) or isinstance(obs.timestamp, bool)
                or not math.isfinite(obs.timestamp) or not math.isfinite(now)
                or obs.timestamp <= self.boundary or obs.timestamp > now):
            return self.reject("invalid_identity_or_time")
        if now - obs.timestamp > self.max_age:
            return self.reject("expired")
        duplicate = False
        if self.last is not None:
            if obs.sequence < self.last.sequence or obs.continuity < self.last.continuity:
                return self.reject("out_of_order")
            duplicate = obs.sequence == self.last.sequence
            if duplicate and obs != self.last:
                return self.reject("reused_identity")
            if not duplicate and obs.timestamp <= self.last.timestamp:
                return self.reject("non_increasing_time")
        point = getattr(result, "gaze_point", None)
        raw = getattr(result, "raw_point", None)
        try:
            finite = point is not None and len(point) == 2 and all(math.isfinite(v) for v in point)
            finite = finite and (raw is None or (len(raw) == 2 and all(math.isfinite(v) for v in raw)))
        except (TypeError, ValueError):
            finite = False
        if (not getattr(result, "valid", False) or not getattr(result, "face_detected", False)
                or getattr(result, "point_kind", None) != "observed" or not finite):
            self.last = obs
            return self.reject(getattr(result, "error_message", None) or "invalid_observation")
        if duplicate:
            if self.interrupted:
                return self.reject("reused_identity")
            return ObservationState.DUPLICATE
        previous = self.last
        self.last = obs
        if previous is not None and obs.timestamp - previous.timestamp > self.max_age:
            return self.reject("sampling_gap")
        self.reset_required = self.interrupted or (previous is not None and obs.continuity != previous.continuity)
        self.interrupted = False
        self.reason = ""
        return ObservationState.NEW
