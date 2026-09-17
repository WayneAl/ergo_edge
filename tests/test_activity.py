import math
from dataclasses import replace

import pytest

from linesafe.activity import ActivityTracker, ActivityParams
from linesafe.geometry import Angles, Measured, Missing, View

FPS = 30
def ang(trunk, arm=None):
    m = lambda v: Missing("x") if v is None else Measured(float(v), 0.9)
    return Angles(t=0.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=False, neck_flex=m(0),
                  upper_arm=(m(arm), Missing("x")), lower_arm=(m(90), m(90)), knee=(m(0), m(0)), legs_bilateral=True)

def arms(trunk, left, right):
    return replace(ang(trunk), upper_arm=(Measured(float(left), 0.9), Measured(float(right), 0.9)))

def run(fn, seconds, tracker=None, t0=0.0):
    tr = tracker or ActivityTracker(); out = None
    for i in range(int(seconds * FPS) + 1):
        t = t0 + i / FPS
        out = tr.update(t, fn(t))
    return tr, out

def test_static_after_one_minute_not_before():
    _, f = run(lambda t: ang(30), 59)
    assert f.static is False
    _, f = run(lambda t: ang(30), 61)
    assert f.static is True and f.reba_points >= 1

def test_gap_breaks_static():
    _, f = run(lambda t: None if 20 <= t < 22 else ang(30), 61)
    assert f.static is False

def test_repeated_fast_cycles():
    _, f = run(lambda t: ang(15 + 15 * math.sin(2 * math.pi * t / 10)), 70)   # 6 cycles/min, 30° p-p
    assert f.repeated is True

def test_slow_cycles_not_repeated():
    _, f = run(lambda t: ang(15 + 15 * math.sin(2 * math.pi * t / 30)), 70)   # 2 cycles/min
    assert f.repeated is False

def test_rapid_trunk_change_holds_two_seconds():
    tr = ActivityTracker()
    for i in range(0, 31):
        t = i / FPS; tr.update(t, ang(0))
    for i in range(31, 46):                                   # 0 -> 60 degrees in 0.5 s
        t = i / FPS; f = tr.update(t, ang(60 * (i - 30) / 15))
    assert f.rapid is True
    for i in range(46, 46 + int(1.5 * FPS)):                  # ends t = 3.0; last trigger was ~2.1
        f = tr.update(i / FPS, ang(60))
    assert f.rapid is True
    for i in range(46 + int(1.5 * FPS), 46 + int(3.0 * FPS)): # ends t = 4.5
        f = tr.update(i / FPS, ang(60))
    assert f.rapid is False

def test_rula_muscle_use_after_ten_minutes_static():
    _, f = run(lambda t: ang(30), 601)
    assert f.rula_muscle_use is True
    _, f = run(lambda t: ang(30), 300)
    assert f.rula_muscle_use is False


def test_mid_swing_start_counts_repetitions():
    # 28° peak-to-peak at 10 cycles/min, starting at the midpoint: no sample is 15° from the first one
    _, f = run(lambda t: ang(14 + 14 * math.sin(2 * math.pi * t / 6)), 70)
    assert f.repeated is True

def test_mid_swing_resume_after_gap_counts_repetitions():
    wave = lambda t: ang(14 + 14 * math.sin(2 * math.pi * (t - 11.5) / 6))   # resumes at its midpoint
    _, f = run(lambda t: None if 10 <= t < 11.5 else wave(t), 80)          # window [20, 80]: post-gap only
    assert f.repeated is True

def test_travel_of_exactly_rep_amp_counts():
    def tri(t):                                               # 0 <-> 15 triangle, 6 s period, exact turning values
        k = round(t * FPS) % 180
        return ang(k / 6 if k <= 90 else (180 - k) / 6)
    _, f = run(tri, 70)
    assert f.repeated is True

def test_arm_signal_is_the_larger_upper_arm():
    # left arm still at 10°, right arm swings 30..50°: only max(arms) has 20° travel (min is flat, mean has 10°)
    _, f = run(lambda t: arms(30, 10, 40 + 10 * math.cos(2 * math.pi * t / 6)), 70)
    assert f.repeated is True

def test_gap_clears_zig_zag_direction():
    # 0 -> ~19° climbs separated by 1.5 s gaps: nothing comes back down, so no reversal may be counted
    def climbs(t):
        phase = t % 2.5
        return ang(20 * phase) if phase < 1.0 else None
    _, f = run(climbs, 70)
    assert f.repeated is False and f.rula_muscle_use is False

def test_rula_muscle_use_at_exactly_rep_per_min_actions():
    # 5 rises and 4 falls of 20°, 1 s each: the first rise only sets the direction, so 8 reversals = 4 actions
    points = [0, 20] * 5
    values = [a + (b - a) * k / FPS for a, b in zip(points, points[1:]) for k in range(FPS)]
    tr = ActivityTracker()
    for i, v in enumerate(values):
        tr.update(i / FPS, ang(v))
    for i in range(len(values), len(values) + 10 * FPS):
        f = tr.update(i / FPS, ang(20))
    assert f.rula_muscle_use is True and f.repeated is False

def test_held_needs_a_recent_sample():
    _, f = run(lambda t: ang(30), 70)
    assert f.static is True
    _, f = run(lambda t: ang(30) if t <= 70 else None, 71.5)   # the run's last sample is 1.5 s old
    assert f.static is False

def test_time_must_increase_and_rejected_input_leaves_no_trace():
    tr = ActivityTracker()
    tr.update(1.0, ang(30))
    for t in (1.0, 0.5, float("nan")):
        with pytest.raises(ValueError, match="t must"):
            tr.update(t, ang(30))
    with pytest.raises(ValueError, match=r"angles\.trunk_flex"):
        tr.update(2.0, ang(float("nan")))
    tr.update(2.0, ang(30))                                   # the rejected call at t = 2.0 consumed nothing
