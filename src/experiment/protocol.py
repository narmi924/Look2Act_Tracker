"""Fixed A/B plans and actual paint history. Targets are instructions, not gaze truth."""
from bisect import bisect_right
import random
import math


DEFAULTS = dict(seed=924, dwell_s=2., settling_s=.5, transition_guard_s=.05,
                a_rounds=2, b_durations=[2., 4., 4., 2.])


def make_plan(size, selection='AB', parameters=None):
    if parameters and set(parameters) - set(DEFAULTS):
        raise ValueError('unknown protocol parameters')
    params = {**DEFAULTS, **(parameters or {})}
    if selection not in ('A', 'B', 'AB'):
        raise ValueError('protocol must be A, B or AB')
    if (len(params['b_durations']) != 4 or type(params['a_rounds']) is not int or params['a_rounds'] < 1
            or type(params['seed']) is not int or params['transition_guard_s'] < 0
            or not all(math.isfinite(v) for v in [params['settling_s'], params['dwell_s'],
                       params['transition_guard_s'], *params['b_durations']])
            or params['settling_s'] < 0 or params['dwell_s'] <= 0 or any(t <= 0 for t in params['b_durations'])):
        raise ValueError('invalid durations')
    width, height = size
    targets = [(round((width - 1) * x), round((height - 1) * y)) for y in (.2, .5, .8) for x in (.2, .5, .8)]
    rng = random.Random(params['seed'])
    plan = []
    if 'A' in selection:
        previous = None
        for round_id in range(params['a_rounds']):
            order = list(range(9))
            rng.shuffle(order)
            if order == previous:
                order = order[1:] + order[:1]
            previous = order
            for target in order:
                plan.append(dict(protocol='A', round=round_id, target_id=target,
                                 instructed_target=targets[target], instruction='自然保持头部，注视目标',
                                 requested_motion='stable', duration_s=params['dwell_s']))
    if 'B' in selection:
        for target in (4, 3, 5):
            for motion, prompt, duration in zip(('natural', 'yaw', 'pitch', 'natural'),
                ('自然保持，注视目标', '舒适范围内缓慢左右转头，仍看目标',
                 '舒适范围内缓慢抬头/低头，仍看目标', '自然保持，注视目标'), params['b_durations']):
                plan.append(dict(protocol='B', target_id=target, instructed_target=targets[target],
                                 instruction=prompt, requested_motion=motion, duration_s=duration))
    start = 0.
    for index, item in enumerate(plan):
        item.update(segment=index, planned_offset_s=start)
        start += item['duration_s']
    return dict(parameters=params, segments=plan, duration_s=start, selection=selection)


class Protocol:
    def __init__(self, plan, sink, clock):
        self.plan, self.sink, self.clock = plan, sink, clock
        self.started = None
        self.paused_at = None
        self.shift = 0.
        self.current = None
        self.drawn = None
        self.finished = False
        self.epoch = 0
        self.requested_at = None

    def _request(self, index, at, reason):
        self.requested_at = at
        self.sink('target_request', at, segment=index, epoch=self.epoch, reason=reason,
                  planned_at=self.started + self.shift + self.plan['segments'][index]['planned_offset_s'])

    def start(self):
        self.started = self.clock()
        self.sink('start', self.started)
        self.tick()

    def tick(self, _user_skip=None):
        if self.started is None or self.paused_at is not None or self.finished:
            return None
        now = self.clock()
        elapsed = now - self.started - self.shift
        offsets = [s['planned_offset_s'] for s in self.plan['segments']]
        index = bisect_right(offsets, elapsed) - 1
        if elapsed >= self.plan['duration_s']:
            index = len(offsets)
        if self.current != index:
            previous = -1 if self.current is None else self.current
            for skipped in range(previous, index):
                if skipped < 0:
                    continue
                user_skipped = _user_skip is not None and skipped == _user_skip[0]
                epoch = _user_skip[1] if user_skipped else self.epoch
                painted = _user_skip[2] if user_skipped else self.drawn == (skipped, epoch)
                outcome = ('user_skipped_after_paint' if painted else 'user_skipped_before_paint') if user_skipped else (
                    'completed_after_paint' if painted else 'deadline_without_paint')
                self.sink('target_closed', now, segment=skipped, epoch=epoch, outcome=outcome)
                if not painted:
                    reason = 'user_skip_before_paint' if user_skipped else 'not_painted_before_deadline'
                    self.sink('skipped', now, segment=skipped, epoch=epoch, reason=reason)
            self.current = index
            if index == len(offsets):
                self.end('user_skip_complete' if _user_skip is not None else 'protocol_complete')
                return None
            self._request(index, now, 'schedule')
        return self.plan['segments'][index]

    def painted(self):
        if self.paused_at is not None or self.finished or self.current is None:
            return
        key = (self.current, self.epoch)
        if key != self.drawn:
            segment = self.plan['segments'][self.current]
            self.sink('target_painted', self.clock(), **segment, epoch=self.epoch,
                      planned_at=self.started + self.shift + segment['planned_offset_s'],
                      requested_at=self.requested_at)
            self.drawn = key

    def pause(self):
        if not self.finished and self.started is not None and self.paused_at is None:
            self.paused_at = self.clock()
            self.sink('pause', self.paused_at)

    def resume(self):
        if self.paused_at is not None:
            now = self.clock()
            self.shift += now - self.paused_at
            self.paused_at = None
            self.epoch += 1
            self.sink('resume', now)
            if self.current is not None and self.current < len(self.plan['segments']):
                self._request(self.current, now, 'resume')

    def skip(self):
        if self.current is not None and not self.finished and self.paused_at is None:
            now = self.clock()
            closed = (self.current, self.epoch, self.drawn == (self.current, self.epoch))
            self.sink('skip', now, segment=self.current, epoch=self.epoch,
                      was_painted=closed[2])
            next_offset = self.plan['segments'][self.current]['planned_offset_s'] + self.plan['segments'][self.current]['duration_s']
            self.shift = now - self.started - next_offset
            self.epoch += 1
            self.tick(_user_skip=closed)

    def end(self, reason='user_end'):
        if not self.finished:
            if reason == 'user_end' and self.current is not None and self.current < len(self.plan['segments']):
                painted = self.drawn == (self.current, self.epoch)
                self.sink('target_closed', self.clock(), segment=self.current, epoch=self.epoch,
                          outcome='user_end_after_paint' if painted else 'user_end_before_paint')
            self.finished = True
            self.sink('end', self.clock(), reason=reason)


def target_at(source_time, events, settling_s=.5, guard_s=.05):
    history = sorted((e for e in events if e['kind'] in
                      ('target_painted', 'pause', 'resume', 'skip', 'end', 'gap')), key=lambda e: e['at'])
    if source_time is None:
        return dict(status='unavailable')
    previous = None
    for event in history:
        if abs(event['at'] - source_time) <= guard_s:
            return dict(status='time_uncertain')
        if event['at'] <= source_time:
            previous = event
    if previous is None or previous['kind'] != 'target_painted':
        return dict(status='unavailable')
    return dict(status='settling' if source_time - previous['at'] < settling_s else 'measurement',
                segment=previous['segment'], epoch=previous['epoch'],
                instructed_target=previous['instructed_target'],
                requested_motion=previous['requested_motion'], protocol=previous['protocol'])
