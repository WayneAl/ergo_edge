from dataclasses import replace

import pytest

from linesafe.events import EventDetector
from linesafe.reba import RebaScore, Band

FPS = 30
BASE = RebaScore(t=0.0, total=9, band=Band.HIGH, score_a=8, score_b=6, table_c=9, activity=0, parts={}, side="right",
                 drivers=("trunk flexion 64°",), partial=False, missing=(), assumed=())
def sc(total, drivers=("trunk flexion 64°",)):
    return replace(BASE, total=total, drivers=drivers)

def feed(det, seq):
    closed = []
    t = 0.0
    for seconds, total in seq:
        for _ in range(int(round(seconds * FPS))):
            e = det.update(t, None if total is None else sc(total))
            if e: closed.append((t, e))
            t += 1 / FPS
    return closed

def test_short_high_never_opens():
    d = EventDetector("S1")
    assert feed(d, [(2.9, 9), (5, 5)]) == [] and not d.active

def test_event_opens_and_closes_after_three_seconds_below_high():
    d = EventDetector("S1")
    closed = feed(d, [(5, 9), (3.2, 7)])
    assert len(closed) == 1
    t_close, e = closed[0]
    assert e.station == "S1" and e.t_start == 0.0 and abs(e.t_end - (5 - 1 / FPS)) < 1e-6
    assert e.peak == 9 and e.band is Band.HIGH and abs(e.duration_s - e.t_end) < 1e-9
    assert abs(t_close - (5 + 3)) < 0.05

def test_short_dip_keeps_one_event():
    d = EventDetector("S1")
    closed = feed(d, [(4, 9), (1, 5), (4, 11), (4, 3)])
    assert len(closed) == 1 and closed[0][1].peak == 11 and closed[0][1].band is Band.VERY_HIGH

def test_none_counts_as_below():
    d = EventDetector("S1")
    assert len(feed(d, [(4, 9), (3.2, None)])) == 1

def test_flush_closes_active_only():
    d = EventDetector("S1"); feed(d, [(4, 9)])
    assert d.active and d.flush().peak == 9 and not d.active
    d2 = EventDetector("S1"); feed(d2, [(1, 9)])
    assert d2.flush() is None

def test_total_exactly_enter_total_opens():
    d = EventDetector("S1"); feed(d, [(3.5, 8)])
    assert d.active

def test_peak_keeps_drivers_of_first_occurrence():
    d = EventDetector("S1"); t = 0.0; closed = []
    for seconds, total, drivers in [(1, 8, ("low",)), (2, 9, ("first",)), (2, 9, ("second",)), (3.2, 5, ("x",))]:
        for _ in range(int(round(seconds * FPS))):
            e = d.update(t, sc(total, drivers))
            if e: closed.append(e)
            t += 1 / FPS
    assert len(closed) == 1 and closed[0].peak == 9 and closed[0].drivers == ("first",)

def test_total_out_of_range_raises_before_state_change():
    d = EventDetector("S1"); feed(d, [(4, 9)])                # active; last t = 119/30
    for bad in (0, 16, True, 9.0):
        with pytest.raises(ValueError, match=r"score\.total"):
            d.update(4.0, replace(BASE, total=bad))
    assert d.active and d.update(4.0, sc(9)) is None and d.active   # t = 4.0 was not consumed
    idle = EventDetector("S1")
    with pytest.raises(ValueError, match=r"score\.total"):
        idle.update(0.0, replace(BASE, total=16))
    feed(idle, [(3.5, 5)])
    assert not idle.active and idle.flush() is None

def test_time_must_strictly_increase():
    d = EventDetector("S1")
    d.update(1.0, sc(9))
    for t in (1.0, 0.5, float("inf")):
        with pytest.raises(ValueError, match="t must"):
            d.update(t, sc(9))
    assert d.update(1.1, sc(9)) is None
