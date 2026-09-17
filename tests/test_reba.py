import pytest
from linesafe.config import StationConfig
from linesafe.geometry import Angles, Measured, Missing, View
from linesafe import reba as R

def angles(trunk=0.0, twisted=False, neck=0.0, ua=(0.0, 0.0), la=(0.0, 0.0), knee=(0.0, 0.0), bilateral=True):
    m = lambda v: v if isinstance(v, Missing) else Measured(float(v), 0.9)
    return Angles(t=1.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=twisted, neck_flex=m(neck),
                  upper_arm=(m(ua[0]), m(ua[1])), lower_arm=(m(la[0]), m(la[1])), knee=(m(knee[0]), m(knee[1])),
                  legs_bilateral=bilateral)

CFG = StationConfig(station_id="S1")

@pytest.mark.parametrize("f,exp", [(0, 1), (5, 1), (5.01, 2), (20, 2), (20.01, 3), (60, 3), (60.01, 4),
                                   (-5, 1), (-5.01, 2), (-20, 2), (-20.01, 3)])
def test_trunk_bands(f, exp):
    assert R.trunk_score(f, False) == exp and R.trunk_score(f, True) == exp + 1

@pytest.mark.parametrize("n,exp", [(0, 1), (20, 1), (20.01, 2), (-5, 1), (-5.01, 2)])
def test_neck_bands(n, exp):
    assert R.neck_score(n) == exp

@pytest.mark.parametrize("u,sup,exp", [(-20, False, 1), (20, False, 1), (20.01, False, 2), (45, False, 2),
                                       (45.01, False, 3), (90, False, 3), (90.01, False, 4), (-20.01, False, 2),
                                       (0, True, 1), (60, True, 2)])
def test_upper_arm_bands(u, sup, exp):
    assert R.upper_arm_score(u, sup) == exp

@pytest.mark.parametrize("l,exp", [(59.99, 2), (60, 1), (100, 1), (100.01, 2), (0, 2)])
def test_lower_arm_bands(l, exp):
    assert R.lower_arm_score(l) == exp

@pytest.mark.parametrize("bi,k,exp", [(True, None, 1), (False, None, 2), (True, 29.99, 1), (True, 30, 2),
                                      (True, 60, 2), (True, 60.01, 3), (False, 70, 4)])
def test_legs(bi, k, exp):
    assert R.legs_score(bi, k) == exp

@pytest.mark.parametrize("total,band", [(1, R.Band.NEGLIGIBLE), (2, R.Band.LOW), (3, R.Band.LOW), (4, R.Band.MEDIUM),
                                        (7, R.Band.MEDIUM), (8, R.Band.HIGH), (10, R.Band.HIGH),
                                        (11, R.Band.VERY_HIGH), (15, R.Band.VERY_HIGH)])
def test_bands(total, band):
    assert R.reba_band(total) is band

@pytest.mark.parametrize("bad", [0, 16])
def test_band_out_of_range(bad):
    with pytest.raises(ValueError):
        R.reba_band(bad)

def test_upright_neutral_is_negligible():
    s = R.score_reba(angles(), CFG)
    # trunk1 neck1 legs1 -> A=1; upper1 lower2 (straight arm) wrist1 -> B=1; C[1][1]=1
    assert (s.score_a, s.score_b, s.table_c, s.total, s.band) == (1, 1, 1, 1, R.Band.NEGLIGIBLE)
    assert s.partial is False and s.drivers == ()

def test_lifting_box_is_high():
    cfg = StationConfig(station_id="S1", reba_load=2, reba_coupling=1, reba_wrist=2)
    s = R.score_reba(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90), knee=(40, 35)), cfg)
    # trunk4 neck2 legs1+1=2 -> A table 6 + load 2 = 8; right arm: upper4 lower1 wrist2 -> 5 + coupling 1 = 6
    assert s.parts == {"trunk": 4, "neck": 2, "legs": 2, "upper_arm": 4, "lower_arm": 1, "wrist": 2, "load": 2, "coupling": 1}
    assert (s.score_a, s.score_b, s.table_c, s.total, s.band, s.side) == (8, 6, 10, 10, R.Band.HIGH, "right")
    assert s.drivers == ("trunk flexion 64°", "neck flexion 30°", "upper arm right 95°", "knee flexion 40°",
                         "load (station)", "coupling (station)")

def test_activity_adds_to_total():
    cfg = StationConfig(station_id="S1", reba_load=2, reba_coupling=1, reba_wrist=2)
    s = R.score_reba(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90), knee=(40, 35)), cfg, activity=3)
    assert s.total == 13 and s.band is R.Band.VERY_HIGH and s.drivers[-1] == "activity +3"

def test_missing_trunk_gives_no_score():
    assert R.score_reba(angles(trunk=Missing("facing unknown")), CFG) is None

def test_missing_neck_is_partial():
    s = R.score_reba(angles(neck=Missing("no confident ear")), CFG)
    assert s.parts["neck"] == 1 and s.partial is True and "neck" in s.missing

def test_missing_twist_is_listed_not_partial():
    s = R.score_reba(angles(twisted=Missing("hips not confident")), CFG)
    assert "trunk twist" in s.missing and s.partial is False

def test_no_upper_arm_is_partial_side_none():
    s = R.score_reba(angles(ua=(Missing("x"), Missing("y"))), CFG)
    assert s.side == "none" and s.parts["upper_arm"] == 1 and s.partial is True

def test_activity_out_of_range():
    with pytest.raises(ValueError):
        R.score_reba(angles(), CFG, activity=4)

def test_assumed_is_listed():
    assert R.score_reba(angles(), CFG).assumed == R.ASSUMED_SIDE_VIEW
