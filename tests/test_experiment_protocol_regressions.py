"""Observable protocol event and summary semantics with a controlled clock."""
import pytest

from src.experiment.protocol import Protocol, make_plan, target_at
from src.experiment.summary import summarize


class Clock:
    def __init__(self, now=10.):
        self.now = now

    def __call__(self):
        return self.now


def session(selection='A', parameters=None):
    clock = Clock()
    plan = make_plan((1000, 800), selection, parameters)
    events = []
    protocol = Protocol(plan, lambda kind, at, **data: events.append(dict(kind=kind, at=at, **data)), clock)
    protocol.start()
    return clock, plan, protocol, events


def report(plan, events):
    return summarize(dict(complete=True, backend='classic', calibration=None, plan=plan), events)['protocol']


def test_painted_target_user_skip_is_not_reported_unpainted():
    clock, plan, protocol, events = session()
    protocol.painted()
    clock.now = 10.75
    protocol.skip()
    assert [(e['kind'], e.get('segment')) for e in events if e['kind'] in ('skip', 'skipped')] == [('skip', 0)]
    assert next(e for e in events if e['kind'] == 'skip')['epoch'] == 0
    assert report(plan, events)['skipped'] == [0]
    assert [(e['segment'], e['epoch'], e['outcome']) for e in events if e['kind'] == 'target_closed'] == [
        (0, 0, 'user_skipped_after_paint')]
    assert report(plan, events)['outcome_counts'] == {'user_skipped_after_paint': 1}


def test_unpainted_target_user_skip_is_distinct_from_deadline_miss():
    clock, plan, protocol, events = session()
    clock.now = 10.75
    protocol.skip()
    skip, missing = (next(e for e in events if e['kind'] == kind) for kind in ('skip', 'skipped'))
    assert (skip['segment'], skip['epoch'], skip['was_painted']) == (0, 0, False)
    assert (missing['segment'], missing['epoch'], missing['reason']) == (0, 0, 'user_skip_before_paint')
    summary = report(plan, events)
    assert summary['skipped'] == [0]
    assert summary['user_skips'] == [dict(segment=0, epoch=0, was_painted=False)]
    assert summary['unpainted'] == [dict(segment=0, epoch=0, reason='user_skip_before_paint')]
    assert summary['outcome_counts'] == {'user_skipped_before_paint': 1}


def test_late_ui_tick_keeps_each_never_painted_deadline_miss():
    clock, plan, protocol, events = session()
    clock.now = 16.1
    protocol.tick()
    assert [(e['segment'], e['epoch'], e['reason']) for e in events if e['kind'] == 'skipped'] == [
        (i, 0, 'not_painted_before_deadline') for i in range(3)]
    assert report(plan, events)['skipped'] == [0, 1, 2]
    assert report(plan, events)['outcome_counts'] == {'deadline_without_paint': 3}


def test_skip_after_resume_identifies_new_display_epoch():
    clock, plan, protocol, events = session()
    protocol.painted()
    clock.now = 11.5
    protocol.pause()
    clock.now = 21.5
    protocol.resume()
    assert target_at(21.6, events)['status'] == 'unavailable'
    clock.now = 21.75
    protocol.skip()
    skip, missing = (next(e for e in reversed(events) if e['kind'] == kind) for kind in ('skip', 'skipped'))
    assert (skip['segment'], skip['epoch'], skip['was_painted']) == (0, 1, False)
    assert (missing['segment'], missing['epoch'], missing['reason']) == (0, 1, 'user_skip_before_paint')
    assert report(plan, events)['painted'] == 1
    assert report(plan, events)['skipped'] == [0]


def test_skip_last_painted_target_finishes_once():
    clock, plan, protocol, events = session(parameters={'a_rounds': 1})
    clock.now = 26.
    protocol.tick()
    protocol.painted()
    clock.now = 26.75
    protocol.skip()
    assert protocol.finished
    assert [e['reason'] for e in events if e['kind'] == 'end'] == ['user_skip_complete']
    assert not any(e['kind'] == 'skipped' and e['segment'] == 8 for e in events)
    assert report(plan, events)['skipped'][-1] == 8


def test_normal_deadline_completion_is_distinct_from_final_user_skip():
    clock, plan, protocol, events = session(parameters={'a_rounds': 1})
    protocol.painted()
    clock.now = 28.
    protocol.tick()
    assert protocol.finished
    assert [e['reason'] for e in events if e['kind'] == 'end'] == ['protocol_complete']
    assert not any(e['kind'] == 'skip' for e in events)
    assert report(plan, events)['outcome_counts']['completed_after_paint'] == 1


def test_user_end_after_paint_is_distinct_from_normal_completion():
    clock, plan, protocol, events = session()
    protocol.painted()
    clock.now = 10.75
    protocol.end()
    assert [e['reason'] for e in events if e['kind'] == 'end'] == ['user_end']
    assert report(plan, events)['outcome_counts'] == {'user_end_after_paint': 1}


def test_resume_immediate_repaint_has_zero_request_wait():
    clock, plan, protocol, events = session()
    protocol.painted()
    clock.now = 11.5
    protocol.pause()
    clock.now = 21.5
    protocol.resume()
    protocol.painted()
    repaint = [e for e in events if e['kind'] == 'target_painted'][-1]
    assert (repaint['epoch'], repaint['planned_at'], repaint['requested_at'], repaint['at']) == (1, 20., 21.5, 21.5)
    timing = report(plan, events)
    assert timing['paint_delay_s']['count'] == 2
    assert timing['paint_delay_s']['max'] == 0.
    assert timing['paint_plan_deviation_s']['max'] == 1.5
    assert target_at(21.6, events)['status'] == 'settling'


def test_resume_delayed_repaint_measures_twenty_millisecond_wait():
    clock, plan, protocol, events = session()
    protocol.painted()
    clock.now = 11.5
    protocol.pause()
    clock.now = 21.5
    protocol.resume()
    clock.now = 21.52
    protocol.painted()
    repaint = [e for e in events if e['kind'] == 'target_painted'][-1]
    assert repaint['requested_at'] == 21.5
    assert repaint['at'] - repaint['requested_at'] == pytest.approx(.02)
    assert report(plan, events)['paint_delay_s']['max'] == pytest.approx(.02)
    assert target_at(21.6, events)['status'] == 'settling'


def test_normal_switch_uses_request_to_paint_wait_not_plan_deviation():
    clock, plan, protocol, events = session()
    protocol.painted()
    clock.now = 12.01
    protocol.tick()
    request = [e for e in events if e['kind'] == 'target_request'][-1]
    clock.now = 12.03
    protocol.painted()
    painted = [e for e in events if e['kind'] == 'target_painted'][-1]
    assert (request['at'], painted['requested_at'], painted['planned_at']) == (12.01, 12.01, 12.)
    summary = report(plan, events)
    assert summary['paint_delay_s']['max'] == pytest.approx(.02)
    assert summary['paint_plan_deviation_s']['max'] == pytest.approx(.03)


def test_unpainted_request_has_no_fabricated_paint_time():
    _, plan, _, events = session()
    assert [e['kind'] for e in events] == ['start', 'target_request']
    summary = report(plan, events)
    assert summary['paint_delay_s'] == dict(count=0, status='unavailable', denominator=0, unavailable_count=0)


def test_legacy_paint_without_request_time_is_unknown_not_zero():
    _, plan, _, _ = session()
    events = [dict(kind='target_painted', at=10.5, segment=0, epoch=0, planned_at=10.)]
    summary = report(plan, events)
    assert summary['paint_delay_s'] == dict(count=0, status='unavailable', denominator=1, unavailable_count=1)
    assert summary['paint_plan_deviation_s']['mean'] == .5
