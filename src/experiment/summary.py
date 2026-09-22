"""Availability/quality only; instructed targets are not independent gaze truth."""
from collections import Counter, defaultdict
import numpy as np
from src.experiment.protocol import target_at


def stats(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return dict(count=0, status='unavailable')
    return dict(count=len(values), mean=float(np.mean(values)), min=float(np.min(values)),
                max=float(np.max(values)), p95=float(np.percentile(values, 95)))


def finite(point):
    return point is not None and len(point) == 2 and bool(np.all(np.isfinite(point)))


def summarize(meta, events):
    producers = [e for e in events if e['kind'] == 'producer']
    consumers = [e for e in events if e['kind'] == 'consume']
    independent, seen = [], set()
    for event in consumers:
        identity = event['observation_id']
        if identity is not None and tuple(identity) not in seen:
            seen.add(tuple(identity))
            independent.append(event)
    failures = sum(not e['result']['valid'] for e in producers)
    reasons = Counter()
    for event in consumers:
        reason = event['gate_reason'] if event['gate_state'] == 'invalid' else event['screen_rejection'] or event['dispatch_rejection']
        if reason:
            reasons[reason] += 1
    intervals, previous = [], None
    poses = defaultdict(list)
    for event in events:
        if event['kind'] == 'gap':
            previous = None
        if event['kind'] != 'producer':
            continue
        result = event['result']
        stamp = result['observation']
        if previous and stamp and stamp['session'] == previous['session'] and stamp['timestamp'] > previous['timestamp']:
            intervals.append(stamp['timestamp'] - previous['timestamp'])
        previous = stamp
        pose = (result.get('numeric_snapshot') or {}).get('head_pose')
        if pose and pose['valid']:
            for axis in ('yaw', 'pitch', 'roll'):
                poses[axis].append(pose[axis])
    denominator = len(independent)
    stages = {}
    for name in ('raw_point', 'calibrated_point', 'smoothed_point', 'display_point'):
        valid = sum(finite(e[name]) for e in independent)
        flag = name.replace('_point', '_in_bounds')
        tested = sum(e.get(flag) is not None for e in independent)
        stages[name] = dict(finite=valid, missing_or_nonfinite=denominator - valid, denominator=denominator,
                            out_of_bounds=sum(e.get(flag) is False for e in independent), bounds_denominator=tested)
        stages[name]['finite_ratio'] = valid / denominator if denominator else None
        stages[name]['out_of_bounds_ratio'] = stages[name]['out_of_bounds'] / tested if tested else None
        stages[name]['units'] = ('camera_normalized_feature' if meta['backend'] == 'classic' else 'screen_px') if name == 'raw_point' else 'screen_px'
    groups = defaultdict(list)
    unavailable_labels = 0
    continuity_segment = 0
    seen = set()
    parameters = meta['plan']['parameters']
    for event in events:
        if event['kind'] in ('gap', 'context', 'pause', 'skip'):
            continuity_segment += 1
        if event['kind'] != 'consume':
            continue
        if event['reset']:
            continuity_segment += 1
        identity = event['observation_id']
        if identity is None or tuple(identity) in seen:
            continue
        seen.add(tuple(identity))
        label = target_at(event['source_time'], events, parameters['settling_s'], parameters['transition_guard_s'])
        if label['status'] != 'measurement':
            unavailable_labels += 1
            continue
        key = (label['segment'], label['epoch'], continuity_segment)
        groups[key].append((event, label))
    target_stats = []
    for key, records in groups.items():
        record = dict(segment=key[0], epoch=key[1], continuous_part=key[2], denominator=len(records),
                      instructed_target=records[0][1]['instructed_target'],
                      requested_motion=records[0][1]['requested_motion'])
        for stage in ('calibrated_point', 'smoothed_point'):
            points = [e[stage] for e, _ in records if finite(e[stage])]
            # Identity/no-calibration Deep output is available, but not a calibrated accuracy claim.
            if meta['calibration'] is None:
                record[stage] = dict(status='unavailable_no_calibration', finite_count=len(points), denominator=len(records))
                continue
            error = [float(np.linalg.norm(np.array(p) - record['instructed_target'])) for p in points]
            record[stage] = dict(deviation_px=stats(error), missing=len(records) - len(points),
                                 denominator=len(records),
                                 dispersion_std_xy_px=np.std(points, axis=0).tolist() if len(points) >= 2 else None)
        target_stats.append(record)
    timing = {}
    for group in ('source_timings', 'processing_timings'):
        keys = set().union(*(e[group].keys() for e in independent)) if independent else set()
        timing[group] = {key: stats([e[group].get(key) for e in independent]) for key in keys}
    return dict(complete=meta.get('complete', False), producer_count=len(producers),
                independent_consumed=denominator, total_consumption_ticks=len(consumers),
                duplicate_ticks=sum(e['gate_state'] == 'duplicate' for e in consumers),
                source_failure=dict(count=failures, denominator=len(producers), ratio=failures / len(producers) if producers else None),
                rejection_reasons_per_tick=dict(reasons), rejection_denominator=len(consumers),
                write_lost=meta.get('write_lost', 0),
                write_unconfirmed=meta.get('write_unconfirmed', 0),
                source_interval_s=stats(intervals), age_s=stats([e['age_s'] for e in independent]),
                age_all_ticks_s=dict(stats([e['age_s'] for e in consumers]), denominator=len(consumers)),
                timings_ms=timing, stages=stages, target_segments=target_stats,
                unavailable_or_settling_labels=unavailable_labels,
                raw_units=sorted(set(e['result']['raw_units'] for e in producers)),
                head_pose=dict(source='estimated_online' if poses else 'unavailable',
                               ranges_deg={key: stats(values) for key, values in poses.items()}),
                protocol=dict(selection=meta['plan']['selection'], planned=len(meta['plan']['segments']),
                              painted=len(set(e['segment'] for e in events if e['kind'] == 'target_painted')),
                              paint_delay_s=stats([e['at'] - e['planned_at'] for e in events
                                                   if e['kind'] == 'target_painted' and 'planned_at' in e]),
                              skipped=[e.get('segment') for e in events if e['kind'] in ('skip', 'skipped')],
                              end_reasons=[e['reason'] for e in events if e['kind'] == 'end']),
                interpretation='instructed_target is not independently measured gaze; numerical replay only')
