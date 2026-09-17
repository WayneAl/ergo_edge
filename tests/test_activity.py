import math
from linesafe.activity import ActivityTracker, ActivityParams
from linesafe.geometry import Angles, Measured, Missing, View

FPS = 30
def ang(trunk, arm=None):
    m = lambda v: Missing("x") if v is None else Measured(float(v), 0.9)
    return Angles(t=0.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=False, neck_flex=m(0),
                  upper_arm=(m(arm), Missing("x")), lower_arm=(m(90), m(90)), knee=(m(0), m(0)), legs_bilateral=True)

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
