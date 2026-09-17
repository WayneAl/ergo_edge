from dataclasses import replace
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
