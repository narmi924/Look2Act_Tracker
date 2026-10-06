"""R8 plan: still fixations, smooth-pursuit paths, pursuit-choice trials and short head motions.

Pure functions shared by the recording window and the offline check, so the painted stimulus and
the analysis use the same trajectory math. Positions are window logical pixels.
"""
import math
import random

from src.experiment.protocol import DEFAULTS

R8_DEFAULTS = dict(seed=924, fixation_s=2., settling_s=.5, transition_guard_s=.05,
                   hsweep_s=8., vsweep_s=6., lissajous_s=20., choice_trials=20, choice_s=3.,
                   orbit_period_s=1.8, orbit_a_px=64., orbit_b_px=30., b_durations=[2., 4., 4.])
STIMULI = ('fixation', 'path', 'choice')  # segment field is 'stimulus'; 'kind' is the event envelope
MARGIN = .08  # paths stay inside [8%, 92%] of the window


def _tri(u):
    """0 -> 1 -> 0 over u in [0, 1]."""
    u = u - math.floor(u)
    return 1. - abs(2. * u - 1.)


def path_position(path, phase_s, size):
    """Single moving dot for a 'path' segment at `phase_s` seconds into the segment."""
    width, height = size
    kind, duration = path['type'], path['duration_s']
    if kind == 'hsweep':
        return [width * (MARGIN + (1 - 2 * MARGIN) * _tri(phase_s / duration)), height * path['y']]
    if kind == 'vsweep':
        return [width * path['x'], height * (MARGIN + (1 - 2 * MARGIN) * _tri(phase_s / duration))]
    if kind == 'lissajous':
        return [width * (.5 + .40 * math.sin(2 * math.pi * phase_s / 10.)),
                height * (.5 + .35 * math.sin(2 * math.pi * phase_s / 7. + math.pi / 2))]
    raise ValueError('unknown path type ' + str(kind))


def marker_positions(choice, phase_s):
    """All orbiting markers of a 'choice' segment at `phase_s`; index `instructed` is the one to follow."""
    omega = 2 * math.pi / choice['period_s']
    out = []
    for (cx, cy), offset, direction in zip(choice['anchors'], choice['phases'], choice['directions']):
        out.append([cx + choice['a_px'] * math.cos(omega * phase_s + offset),
                    cy + direction * choice['b_px'] * math.sin(omega * phase_s + offset)])
    return out


def _layouts(width, height):
    return [[[width * x, height * .5] for x in (.2, .4, .6, .8)],
            [[width * x, height * y] for y in (.35, .65) for x in (.25, .5, .75)],
            [[width * x, height * y] for y in (.3, .7) for x in (.3, .7)]]


def make_r8_plan(size, parameters=None):
    if parameters and set(parameters) - set(R8_DEFAULTS):
        raise ValueError('unknown R8 protocol parameters')
    params = {**R8_DEFAULTS, **(parameters or {})}
    for key in ('fixation_s', 'hsweep_s', 'vsweep_s', 'lissajous_s', 'choice_s', 'orbit_period_s'):
        if not (math.isfinite(params[key]) and params[key] > 0):
            raise ValueError('invalid duration ' + key)
    if type(params['choice_trials']) is not int or params['choice_trials'] < 0 or len(params['b_durations']) != 3:
        raise ValueError('invalid R8 parameters')
    width, height = size
    targets = [(round((width - 1) * x), round((height - 1) * y)) for y in (.2, .5, .8) for x in (.2, .5, .8)]
    rng = random.Random(params['seed'])

    def fixations(round_id):
        order = list(range(9))
        rng.shuffle(order)
        return [dict(protocol='A', stimulus='fixation', round=round_id, target_id=t, instructed_target=targets[t],
                     instruction='头部自然保持不动，注视黄点', requested_motion='stable',
                     duration_s=params['fixation_s']) for t in order]

    plan = fixations(0)
    for y in (.2, .5, .8):
        plan.append(dict(protocol='P', stimulus='path', instructed_target=None, requested_motion='pursuit',
                         instruction='眼睛跟着移动的黄点，头尽量不动',
                         path=dict(type='hsweep', y=y, duration_s=params['hsweep_s']), duration_s=params['hsweep_s']))
    for x in (.2, .5, .8):
        plan.append(dict(protocol='P', stimulus='path', instructed_target=None, requested_motion='pursuit',
                         instruction='眼睛跟着移动的黄点，头尽量不动',
                         path=dict(type='vsweep', x=x, duration_s=params['vsweep_s']), duration_s=params['vsweep_s']))
    plan.append(dict(protocol='P', stimulus='path', instructed_target=None, requested_motion='pursuit',
                     instruction='眼睛跟着移动的黄点，头尽量不动',
                     path=dict(type='lissajous', duration_s=params['lissajous_s']), duration_s=params['lissajous_s']))
    layouts = _layouts(width, height)
    for trial in range(params['choice_trials']):
        anchors = layouts[trial % len(layouts)]
        k = len(anchors)
        plan.append(dict(protocol='C', stimulus='choice', instructed_target=None, requested_motion='choice',
                         instruction='只用眼睛跟着带圈的高亮小点转，其他点不要看',
                         choice=dict(anchors=anchors, instructed=rng.randrange(k), period_s=params['orbit_period_s'],
                                     a_px=params['orbit_a_px'], b_px=params['orbit_b_px'],
                                     phases=[i * 2 * math.pi / k for i in range(k)],
                                     directions=[1 if i % 2 == 0 else -1 for i in range(k)], trial=trial),
                         duration_s=params['choice_s']))
    for motion, prompt, duration in zip(('natural', 'yaw', 'pitch'),
                                        ('自然保持，注视目标', '舒适范围内缓慢左右转头，仍看目标', '舒适范围内缓慢抬头/低头，仍看目标'),
                                        params['b_durations']):
        plan.append(dict(protocol='B', stimulus='fixation', target_id=4, instructed_target=targets[4], instruction=prompt,
                         requested_motion=motion, duration_s=duration))
    plan.extend(fixations(1))
    start = 0.
    for index, item in enumerate(plan):
        item.update(segment=index, planned_offset_s=start)
        start += item['duration_s']
    parameters_out = {**DEFAULTS, 'settling_s': params['settling_s'], 'transition_guard_s': params['transition_guard_s'],
                      'dwell_s': params['fixation_s'], 'r8': params}
    return dict(parameters=parameters_out, segments=plan, duration_s=start, selection='R8')


def stimulus_at(segment, phase_s, size):
    """What the window paints for a segment at `phase_s`: dict(dot=[x, y]) or dict(markers=[...], instructed=i)."""
    if segment['stimulus'] == 'fixation':
        return dict(dot=list(segment['instructed_target']))
    if segment['stimulus'] == 'path':
        return dict(dot=path_position(segment['path'], phase_s, size))
    return dict(markers=marker_positions(segment['choice'], phase_s), instructed=segment['choice']['instructed'])
