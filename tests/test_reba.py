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

# --- Fix round 1: rulings R1-R3 and review minors M2-M5, M7, M8 ---

def test_chosen_side_lower_arm_missing_is_partial():
    s = R.score_reba(angles(ua=(10, 95), la=(90, Missing("wrist"))), CFG)
    assert s.side == "right" and s.partial is True and "lower arm right" in s.missing

def test_unchosen_side_lower_arm_missing_is_not_partial():
    s = R.score_reba(angles(ua=(10, 95), la=(Missing("wrist"), 90)), CFG)
    assert s.side == "right" and s.partial is False and "lower arm left" not in s.missing

def test_trunk_driver_uses_base_flexion_score():
    s = R.score_reba(angles(trunk=10, twisted=True), CFG)
    assert s.parts["trunk"] == 3 and "trunk twist" in s.drivers and "trunk flexion 10°" not in s.drivers

def test_neck_flexion_and_extension_drivers():
    assert "neck flexion 30°" in R.score_reba(angles(neck=30), CFG).drivers
    assert "neck extension 10°" in R.score_reba(angles(neck=-10), CFG).drivers

@pytest.mark.parametrize("knee", [(40, float("nan")), (float("nan"), 40)])
def test_nan_knee_raises_either_order(knee):
    with pytest.raises(ValueError, match="knee flex_deg"):
        R.score_reba(angles(knee=knee), CFG)

def test_trunk_bands_follow_upright_tolerance(monkeypatch):
    monkeypatch.setattr(R, "UPRIGHT_TOL_DEG", 3.0)
    assert R.trunk_score(4.0, False) == 2 and R.trunk_score(-4.0, False) == 2 and R.trunk_score(3.0, False) == 1

def test_flags_must_be_bool():
    with pytest.raises(ValueError, match="trunk twisted"):
        R.trunk_score(10.0, Missing("hips not confident"))
    with pytest.raises(ValueError, match="legs bilateral"):
        R.legs_score(Missing("ankle not confident"), None)

@pytest.mark.parametrize("fn,name", [(lambda v: R.trunk_score(v, False), "trunk flex_deg"),
                                     (R.neck_score, "neck flex_deg"),
                                     (lambda v: R.upper_arm_score(v, False), "upper_arm flex_deg"),
                                     (R.lower_arm_score, "lower_arm flex_deg"),
                                     (lambda v: R.legs_score(True, v), "knee flex_deg")])
def test_non_finite_angle_names_the_segment(fn, name):
    with pytest.raises(ValueError, match=name):
        fn(float("nan"))

def test_missing_legs_is_partial():
    s = R.score_reba(angles(bilateral=Missing("ankle not confident")), CFG)
    assert s.parts["legs"] == 1 and "legs" in s.missing and s.partial is True

def test_both_knees_missing_is_listed_not_partial():
    s = R.score_reba(angles(knee=(Missing("a"), Missing("b"))), CFG)
    assert "knee" in s.missing and s.partial is False

@pytest.mark.parametrize("ua,side", [((10, 15), "right"), ((15, 10), "left"), ((10, 10), "left")])
def test_arm_tie_breaks(ua, side):
    # both arms upper 1, lower 1, wrist 1 -> Table B 1 on both sides
    assert R.score_reba(angles(ua=ua, la=(90, 90)), CFG).side == side

def test_shock_adds_to_load():
    s = R.score_reba(angles(), StationConfig(station_id="S1", reba_shock=True))
    assert s.parts["load"] == 1 and s.score_a == 2 and "load (station)" in s.drivers

def test_supported_arm_floor_in_score():
    s = R.score_reba(angles(ua=(0, 10)), StationConfig(station_id="S1", arm_supported=True))
    assert s.parts["upper_arm"] == 1

def test_extension_and_one_leg_stance_drivers():
    assert R.score_reba(angles(trunk=-30, bilateral=False), CFG).drivers == ("trunk extension 30°", "one-leg stance")

def test_one_upper_arm_missing_is_listed_not_partial():
    s = R.score_reba(angles(ua=(Missing("x"), 100)), CFG)
    assert s.side == "right" and "upper arm left" in s.missing and s.partial is False

def test_scenario_supported_extension_knee():
    cfg = StationConfig(station_id="S1", arm_supported=True)
    s = R.score_reba(angles(trunk=-25, neck=0, ua=(30, 60), la=(120, 80), knee=(70, Missing("k")), bilateral=False), cfg)
    # trunk3 neck1 legs2+2=4 -> A 6; left upper2-1=1 lower2 -> B 1, right upper3-1=2 lower1 -> B 1, tie -> 60 > 30 right; C[6][1]=6
    assert (s.total, s.band, s.side) == (6, R.Band.MEDIUM, "right")
    assert s.drivers == ("trunk extension 25°", "knee flexion 70°")

def test_scenario_supported_extension_one_leg():
    cfg = StationConfig(station_id="S1", arm_supported=True)
    s = R.score_reba(angles(trunk=-25, neck=0, ua=(30, 60), la=(120, 80), knee=(20, Missing("k")), bilateral=False), cfg)
    # trunk3 neck1 legs2 -> A 4; B 1; C[4][1]=3
    assert (s.total, s.band, s.side) == (3, R.Band.LOW, "right")
    assert s.drivers == ("trunk extension 25°", "one-leg stance")

def test_scenario_left_arm_missing_elbow():
    s = R.score_reba(angles(ua=(100, 50), la=(Missing("elbow"), 80)), CFG)
    # A 1; left upper4 lower2 wrist1 -> 5, right upper3 lower1 -> 3; C[1][5]=3
    assert (s.total, s.band, s.side) == (3, R.Band.LOW, "left")
    assert (s.parts["upper_arm"], s.parts["lower_arm"]) == (4, 2)
    assert s.partial is True and s.missing == ("lower arm left",) and s.drivers == ("upper arm left 100°",)

def test_scenario_worst_case_all_drivers():
    cfg = StationConfig(station_id="S1", reba_load=2, reba_shock=True, reba_coupling=3, reba_wrist=3)
    s = R.score_reba(angles(trunk=30, twisted=True, neck=25, ua=(95, 40), la=(110, 90), knee=(45, 50)), cfg, activity=3)
    # trunk3+1=4 neck2 legs1+1=2 -> A 6 + load 3 = 9; left upper4 lower2 wrist3 -> 7 + 3 = 10; C[9][10]=12; +3 = 15
    assert (s.total, s.band) == (15, R.Band.VERY_HIGH)
    assert s.drivers == ("trunk flexion 30°", "trunk twist", "neck flexion 25°", "upper arm left 95°",
                         "knee flexion 50°", "load (station)", "coupling (station)", "activity +3")
