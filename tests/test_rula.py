import pytest
from linesafe.config import StationConfig
from linesafe.geometry import Angles, Measured, Missing, View
from linesafe import reba as R
from linesafe import rula as U

def angles(trunk=0.0, twisted=False, neck=0.0, ua=(0.0, 0.0), la=(0.0, 0.0), bilateral=True):
    m = lambda v: v if isinstance(v, Missing) else Measured(float(v), 0.9)
    return Angles(t=1.0, view=View.SIDE, facing=1, trunk_flex=m(trunk), trunk_twisted=twisted, neck_flex=m(neck),
                  upper_arm=(m(ua[0]), m(ua[1])), lower_arm=(m(la[0]), m(la[1])), knee=(m(0), m(0)),
                  legs_bilateral=bilateral)

@pytest.mark.parametrize("n,exp", [(0, 1), (10, 1), (10.01, 2), (20, 2), (20.01, 3), (-5, 1), (-5.01, 4)])
def test_neck(n, exp):
    assert U.rula_neck_score(n) == exp

@pytest.mark.parametrize("t,exp", [(0, 1), (5, 1), (5.01, 2), (20, 2), (20.01, 3), (60, 3), (60.01, 4), (-10, 2)])
def test_trunk(t, exp):
    assert U.rula_trunk_score(t, False) == exp and U.rula_trunk_score(t, True) == exp + 1

@pytest.mark.parametrize("total,level", [(1, U.RulaLevel.ACCEPTABLE), (2, U.RulaLevel.ACCEPTABLE),
    (3, U.RulaLevel.INVESTIGATE), (4, U.RulaLevel.INVESTIGATE), (5, U.RulaLevel.CHANGE_SOON),
    (6, U.RulaLevel.CHANGE_SOON), (7, U.RulaLevel.CHANGE_NOW)])
def test_levels(total, level):
    assert U.rula_level(total) is level

def test_upright_is_acceptable():
    s = U.score_rula(angles(), StationConfig(station_id="S1"))
    # upper1 lower2 wrist1 twist1 -> A 2; neck1 trunk1 legs1 -> B 1; C[2][1] = 2
    assert (s.score_a, s.score_b, s.total, s.level) == (2, 1, 2, U.RulaLevel.ACCEPTABLE)

def test_lifting_box_is_change_now():
    cfg = StationConfig(station_id="S1", rula_wrist=2, rula_force=3)
    s = U.score_rula(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90)), cfg)
    # right arm: upper4 lower1 wrist2 twist1 -> A 4 + 0 + 3 = 7; neck3 trunk4 legs1 -> B 5 + 0 + 3 = 8; C[7][7+] = 7
    assert (s.score_a, s.score_b, s.total, s.level, s.side) == (7, 8, 7, U.RulaLevel.CHANGE_NOW, "right")

def test_muscle_use_adds_to_both_groups():
    s0 = U.score_rula(angles(), StationConfig(station_id="S1"))
    s1 = U.score_rula(angles(), StationConfig(station_id="S1"), muscle_use=True)
    assert (s1.score_a, s1.score_b) == (s0.score_a + 1, s0.score_b + 1)

def test_missing_trunk_is_none():
    assert U.score_rula(angles(trunk=Missing("facing unknown")), StationConfig(station_id="S1")) is None

# --- Added beyond the excerpt: bands, validation, Missing paths, side selection ---

CFG = StationConfig(station_id="S1")

@pytest.mark.parametrize("l,exp", [(59.99, 2), (60, 1), (100, 1), (100.01, 2), (0, 2)])
def test_lower_arm(l, exp):
    assert U.rula_lower_arm_score(l) == exp

@pytest.mark.parametrize("bi,exp", [(True, 1), (False, 2)])
def test_legs(bi, exp):
    assert U.rula_legs_score(bi) == exp

@pytest.mark.parametrize("bad", [0, 8, 2.0, True])
def test_level_rejects_bad_total(bad):
    with pytest.raises(ValueError, match="total"):
        U.rula_level(bad)

@pytest.mark.parametrize("fn,name", [(lambda v: U.rula_trunk_score(v, False), "trunk flex_deg"),
                                     (U.rula_neck_score, "neck flex_deg"),
                                     (U.rula_lower_arm_score, "lower_arm flex_deg")])
def test_non_finite_angle_names_the_segment(fn, name):
    with pytest.raises(ValueError, match=name):
        fn(float("nan"))

def test_flags_must_be_bool():
    with pytest.raises(ValueError, match="trunk twisted"):
        U.rula_trunk_score(10.0, Missing("hips not confident"))
    with pytest.raises(ValueError, match="legs bilateral"):
        U.rula_legs_score(Missing("ankle not confident"))
    with pytest.raises(ValueError, match="muscle_use"):
        U.score_rula(angles(), CFG, muscle_use=1)

def test_lifting_box_parts():
    cfg = StationConfig(station_id="S1", rula_wrist=2, rula_force=3)
    s = U.score_rula(angles(trunk=64, neck=30, ua=(30, 95), la=(90, 90)), cfg)
    assert s.parts == {"upper_arm": 4, "lower_arm": 1, "wrist": 2, "wrist_twist": 1, "neck": 3, "trunk": 4,
                       "legs": 1, "muscle": 0, "force": 3}
    assert s.t == 1.0 and s.partial is False and s.missing == ()

def test_upright_parts_and_tie_side():
    s = U.score_rula(angles(), CFG)
    assert s.parts == {"upper_arm": 1, "lower_arm": 2, "wrist": 1, "wrist_twist": 1, "neck": 1, "trunk": 1,
                       "legs": 1, "muscle": 0, "force": 0}
    assert s.side == "left" and s.partial is False and s.missing == ()

def test_muscle_use_part():
    assert U.score_rula(angles(), CFG, muscle_use=True).parts["muscle"] == 1

def test_twisted_trunk_adds_in_score():
    assert U.score_rula(angles(trunk=10, twisted=True), CFG).parts["trunk"] == 3

def test_supported_arm_lowers_upper_arm():
    s = U.score_rula(angles(ua=(0, 60)), StationConfig(station_id="S1", arm_supported=True))
    assert s.side == "right" and s.parts["upper_arm"] == 2

def test_wrist_twist_from_station():
    s1 = U.score_rula(angles(la=(90, 90)), CFG)
    s2 = U.score_rula(angles(la=(90, 90)), StationConfig(station_id="S1", rula_wrist_twist=2))
    # upper1 lower1 wrist1: twist1 -> A 1, twist2 -> A 2
    assert (s1.score_a, s2.score_a, s2.parts["wrist_twist"]) == (1, 2, 2)

def test_missing_neck_is_partial():
    s = U.score_rula(angles(neck=Missing("no confident ear")), CFG)
    assert s.parts["neck"] == 1 and s.partial is True and s.missing == ("neck",)

def test_missing_legs_is_partial():
    s = U.score_rula(angles(bilateral=Missing("ankle not confident")), CFG)
    assert s.parts["legs"] == 1 and s.partial is True and s.missing == ("legs",)

def test_missing_twist_is_listed_not_partial():
    s = U.score_rula(angles(twisted=Missing("hips not confident")), CFG)
    assert s.parts["trunk"] == 1 and "trunk twist" in s.missing and s.partial is False

def test_no_upper_arm_is_partial_side_none():
    s = U.score_rula(angles(ua=(Missing("x"), Missing("y")), la=(90, 90)), CFG)
    assert s.side == "none" and (s.parts["upper_arm"], s.parts["lower_arm"]) == (1, 2) and s.score_a == 2
    assert s.partial is True and s.missing == ("upper arm",)

def test_chosen_side_lower_arm_missing_is_partial():
    s = U.score_rula(angles(ua=(10, 95), la=(90, Missing("wrist"))), CFG)
    assert s.side == "right" and s.parts["lower_arm"] == 2
    assert s.partial is True and s.missing == ("lower arm right",)

def test_unchosen_side_lower_arm_missing_is_not_partial():
    s = U.score_rula(angles(ua=(10, 95), la=(Missing("wrist"), 90)), CFG)
    assert s.side == "right" and s.parts["lower_arm"] == 1 and s.partial is False and s.missing == ()

@pytest.mark.parametrize("ua,la,side,missing", [((Missing("x"), 100), (90, 90), "right", "upper arm left"),
                                                ((100, Missing("x")), (90, 90), "left", "upper arm right")])
def test_one_upper_arm_missing_is_listed_not_partial(ua, la, side, missing):
    s = U.score_rula(angles(ua=ua, la=la), CFG)
    assert s.side == side and s.parts["upper_arm"] == 4 and s.missing == (missing,) and s.partial is False

@pytest.mark.parametrize("ua,side", [((10, 15), "right"), ((15, 10), "left"), ((10, 10), "left")])
def test_arm_tie_breaks_on_upper_arm_flexion(ua, side):
    # both arms upper 1, lower 1, wrist 1, twist 1 -> Table A 1 on both sides
    assert U.score_rula(angles(ua=ua, la=(90, 90)), CFG).side == side

@pytest.mark.parametrize("ua,la,side", [((40, 44), (30, 90), "left"), ((44, 40), (90, 30), "right")])
def test_larger_table_a_beats_larger_flexion(ua, la, side):
    # 40 deg: upper2 lower2 -> A 3; 44 deg: upper2 lower1 -> A 2
    s = U.score_rula(angles(ua=ua, la=la), CFG)
    assert s.side == side and (s.parts["upper_arm"], s.parts["lower_arm"], s.score_a) == (2, 2, 3)

# --- Fix round 1: C2 shared upright tolerance, I1 one-leg scenario ---

def test_trunk_bands_follow_reba_upright_tolerance(monkeypatch):
    monkeypatch.setattr(R, "UPRIGHT_TOL_DEG", 3.0)
    assert U.rula_trunk_score(4.0, False) == 2 and U.rula_trunk_score(-4.0, False) == 2
    assert U.rula_trunk_score(3.0, False) == 1 and U.rula_trunk_score(-3.0, False) == 1

def test_scenario_one_leg_twisted_muscle_use():
    cfg = StationConfig(station_id="S1", rula_wrist=2, rula_force=1)
    s = U.score_rula(angles(trunk=10, twisted=True, neck=15, ua=(50, 55), la=(120, 70), bilateral=False), cfg,
                     muscle_use=True)
    # left upper3 lower2 wrist2 twist1 -> A 4; right upper3 lower1 -> A 4; tie -> 55 > 50 right; 4 + 1 + 1 = 6
    # neck2 trunk2+1=3 legs2 -> B 5 + 1 + 1 = 7; C[6][7] = 7
    assert (s.score_a, s.score_b, s.total, s.side) == (6, 7, 7, "right")
    assert s.parts == {"upper_arm": 3, "lower_arm": 1, "wrist": 2, "wrist_twist": 1, "neck": 2, "trunk": 3,
                       "legs": 2, "muscle": 1, "force": 1}
