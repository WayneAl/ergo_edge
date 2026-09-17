import math
import numpy as np
import pytest
from linesafe import keypoints as K
from linesafe.pose import PoseFrame
from linesafe.geometry import (GeometryParams, Measured, Missing, View, LEFT, RIGHT,
                               classify_view, compute_angles, facing)

def pose(points, conf=0.9, low=()):
    k = np.zeros((17, 2), np.float32); c = np.full(17, conf, np.float32)
    for i, (x, y) in points.items():
        k[i] = (x, y)
    for i in low:
        c[i] = 0.0
    return PoseFrame(t=0.0, kpts=k, conf=c, bbox=(0.0, 0.0, 400.0, 600.0))

def upright(over=()):
    p = {K.NOSE: (115, 178), K.L_EAR: (100, 180), K.R_EAR: (100, 180),
         K.L_SHOULDER: (102, 200), K.R_SHOULDER: (98, 200), K.L_HIP: (102, 300), K.R_HIP: (98, 300),
         K.L_ELBOW: (102, 250), K.R_ELBOW: (98, 250), K.L_WRIST: (102, 300), K.R_WRIST: (98, 300),
         K.L_KNEE: (102, 400), K.R_KNEE: (98, 400), K.L_ANKLE: (102, 500), K.R_ANKLE: (98, 500)}
    p.update(over); return p

def deg(a):
    assert isinstance(a, Measured), a
    return a.deg

def test_upright_is_all_zero():
    a = compute_angles(pose(upright()))
    assert a.view is View.SIDE and a.facing == 1
    assert deg(a.trunk_flex) == pytest.approx(0, abs=0.01)
    assert deg(a.neck_flex) == pytest.approx(0, abs=0.01)
    for side in (LEFT, RIGHT):
        assert deg(a.upper_arm[side]) == pytest.approx(0, abs=0.01)
        assert deg(a.lower_arm[side]) == pytest.approx(0, abs=0.01)
        assert deg(a.knee[side]) == pytest.approx(0, abs=0.01)
    assert a.legs_bilateral is True and a.trunk_twisted is False

def lean(deg_, facing_=1):
    dx, dy = 100 * math.sin(math.radians(deg_)) * facing_, -100 * math.cos(math.radians(deg_))
    sh = (100 + dx, 300 + dy)
    ear = (sh[0], sh[1] - 20); nose = (ear[0] + 15 * facing_, ear[1] - 2)
    return upright({K.L_SHOULDER: sh, K.R_SHOULDER: sh, K.L_HIP: (100, 300), K.R_HIP: (100, 300),
                      K.L_EAR: ear, K.R_EAR: ear, K.NOSE: nose})

def test_trunk_flexion_45_facing_right():
    assert deg(compute_angles(pose(lean(45))).trunk_flex) == pytest.approx(45, abs=0.01)

def test_trunk_flexion_45_facing_left_is_still_positive():
    a = compute_angles(pose(lean(45, facing_=-1)))
    assert a.facing == -1 and deg(a.trunk_flex) == pytest.approx(45, abs=0.01)

def test_trunk_extension_is_negative():
    assert deg(compute_angles(pose(lean(-20))).trunk_flex) == pytest.approx(-20, abs=0.01)

def test_upper_arm_forward_90_and_back_30():
    p = upright({K.L_ELBOW: (152, 200), K.R_ELBOW: (98 - 25, 200 + 43.30127)})
    a = compute_angles(pose(p))
    assert deg(a.upper_arm[LEFT]) == pytest.approx(90, abs=0.01)
    assert deg(a.upper_arm[RIGHT]) == pytest.approx(-30, abs=0.01)

def test_upper_arm_is_relative_to_trunk():
    p = lean(45)
    sh = p[K.L_SHOULDER]
    p[K.L_ELBOW] = (sh[0], sh[1] + 50)            # hanging straight down in the image
    assert deg(compute_angles(pose(p)).upper_arm[LEFT]) == pytest.approx(45, abs=0.01)

def test_elbow_90():
    p = upright({K.L_SHOULDER: (100, 200), K.L_ELBOW: (100, 250), K.L_WRIST: (150, 250)})
    assert deg(compute_angles(pose(p)).lower_arm[LEFT]) == pytest.approx(90, abs=0.01)

def test_knee_60():
    p = upright({K.L_HIP: (100, 300), K.L_KNEE: (100, 400), K.L_ANKLE: (186.60254, 450)})
    assert deg(compute_angles(pose(p)).knee[LEFT]) == pytest.approx(60, abs=0.01)

def test_neck_flexion_30():
    ear = (100 + 20 * math.sin(math.radians(30)), 200 - 20 * math.cos(math.radians(30)))
    p = upright({K.L_SHOULDER: (100, 200), K.R_SHOULDER: (100, 200), K.L_HIP: (100, 300),
                   K.R_HIP: (100, 300), K.L_EAR: ear, K.R_EAR: ear, K.NOSE: (ear[0] + 15, ear[1])})
    assert deg(compute_angles(pose(p)).neck_flex) == pytest.approx(30, abs=0.01)

def test_missing_elbow_is_missing_with_reason():
    a = compute_angles(pose(upright(), low=(K.L_ELBOW,)))
    assert isinstance(a.upper_arm[LEFT], Missing) and "l_elbow" in a.upper_arm[LEFT].reason
    assert isinstance(a.upper_arm[RIGHT], Measured)

def test_front_view_detected_and_blocks_flexion():
    p = upright({K.L_SHOULDER: (60, 200), K.R_SHOULDER: (140, 200)})
    assert classify_view(pose(p)) is View.FRONT
    a = compute_angles(pose(p), view_ok=False)
    assert isinstance(a.trunk_flex, Missing) and "front" in a.trunk_flex.reason
    assert isinstance(a.knee[LEFT], Missing)
    for angle in (a.trunk_flex, a.neck_flex, *a.upper_arm, *a.lower_arm, *a.knee):
        assert isinstance(angle, Missing) and "front" in angle.reason
    assert not isinstance(a.trunk_twisted, Missing) and not isinstance(a.legs_bilateral, Missing)

def test_facing_unknown_blocks_signed_angles():
    a = compute_angles(pose(upright(), low=(K.NOSE,)))
    assert a.facing is None
    assert isinstance(a.trunk_flex, Missing) and "facing" in a.trunk_flex.reason
    assert isinstance(a.lower_arm[LEFT], Measured)       # unsigned angles do not need facing

def test_one_foot_raised_is_not_bilateral():
    a = compute_angles(pose(upright({K.R_ANKLE: (98, 440)})))
    assert a.legs_bilateral is False

def test_twist_proxy():
    p = upright({K.L_SHOULDER: (60, 200), K.R_SHOULDER: (140, 200), K.L_HIP: (95, 300), K.R_HIP: (105, 300)})
    assert compute_angles(pose(p)).trunk_twisted is True

def test_facing_needs_nose_offset():
    p = upright({K.NOSE: (101, 178)})
    assert facing(pose(p)) is None

def test_zero_length_segments_are_missing_not_zero():
    p = upright({K.L_ELBOW: (102, 200), K.L_ANKLE: (102, 400), K.L_EAR: (100, 200), K.R_EAR: (100, 200)})
    a = compute_angles(pose(p))
    for angle in (a.upper_arm[LEFT], a.lower_arm[LEFT], a.knee[LEFT], a.neck_flex):
        assert isinstance(angle, Missing) and "degenerate" in angle.reason

def test_no_usable_torso_propagates_to_legs_and_twist():
    p = upright({K.L_HIP: (102, 200.5), K.R_HIP: (98, 200.5)})    # torso length 0.5 px
    a = compute_angles(pose(p))
    assert a.trunk_twisted == Missing("degenerate torso") and a.legs_bilateral == Missing("degenerate torso")

@pytest.mark.parametrize("field, value", [("conf_min", 0.0), ("conf_min", 1.01), ("front_ratio_min", 0.0),
                                          ("legs_level_frac", -0.1), ("twist_frac", -0.1),
                                          ("neck_offset_deg", math.nan)])
def test_bad_geometry_params_raise_naming_the_field(field, value):
    with pytest.raises(ValueError, match=field):
        GeometryParams(**{field: value})

def test_pose_frame_rejects_bad_input():
    k, c, box = np.zeros((17, 2), np.float32), np.ones(17, np.float32), (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="PoseFrame.t"):
        PoseFrame(t=math.nan, kpts=k, conf=c, bbox=box)
    with pytest.raises(ValueError, match="PoseFrame.kpts"):
        PoseFrame(t=0.0, kpts=k[:16], conf=c, bbox=box)
    with pytest.raises(ValueError, match="PoseFrame.bbox"):
        PoseFrame(t=0.0, kpts=k, conf=c, bbox=box[:3])

def test_facing_left_signs_neck_and_arms():
    # trunk 30 deg flexed facing left: u = (-0.5, -0.866), f = (-0.866, 0.5)
    sh = (300 - 100 * math.sin(math.radians(30)), 400 - 100 * math.cos(math.radians(30)))   # (250, 313.397)
    ear = (sh[0] - 5, sh[1] - 23.40)
    p = upright({K.L_HIP: (300, 400), K.R_HIP: (300, 400), K.L_SHOULDER: sh, K.R_SHOULDER: sh,
                 K.L_ELBOW: (sh[0] - 50, sh[1]), K.L_WRIST: (sh[0] - 50, sh[1] - 40),
                 K.R_ELBOW: (sh[0], sh[1] + 50), K.R_WRIST: (sh[0], sh[1] + 100),
                 K.L_EAR: ear, K.R_EAR: ear, K.NOSE: (ear[0] - 15, ear[1] - 2)})
    a = compute_angles(pose(p))
    assert a.facing == -1
    assert deg(a.trunk_flex) == pytest.approx(30, abs=0.01)
    assert deg(a.neck_flex) == pytest.approx(-17.94, abs=0.01)
    assert deg(a.upper_arm[LEFT]) == pytest.approx(120, abs=0.01)
    assert deg(a.upper_arm[RIGHT]) == pytest.approx(30, abs=0.01)
    assert deg(a.lower_arm[LEFT]) == pytest.approx(90, abs=0.01)

@pytest.mark.parametrize("true_deg", [190, -60, 180])
def test_upper_arm_range_is_minus90_to_270(true_deg):
    r = math.radians(true_deg)
    p = upright({K.L_ELBOW: (102 + 50 * math.sin(r), 200 + 50 * math.cos(r))})
    assert deg(compute_angles(pose(p)).upper_arm[LEFT]) == pytest.approx(true_deg, abs=0.01)

def test_pose_frame_rejects_non_finite_kpts_and_conf():
    k, c, box = np.zeros((17, 2), np.float32), np.ones(17, np.float32), (0.0, 0.0, 1.0, 1.0)
    bad_k, bad_c = k.copy(), c.copy()
    bad_k[K.L_ELBOW, 0], bad_c[K.NOSE] = math.nan, math.inf
    with pytest.raises(ValueError, match="PoseFrame.kpts"):
        PoseFrame(t=0.0, kpts=bad_k, conf=c, bbox=box)
    with pytest.raises(ValueError, match="PoseFrame.conf"):
        PoseFrame(t=0.0, kpts=k, conf=bad_c, bbox=box)

def test_sub_pixel_segments_are_degenerate():
    p = upright({K.L_ELBOW: (102, 200.5), K.L_ANKLE: (102, 400.5), K.L_EAR: (100, 199.5), K.R_EAR: (100, 199.5)})
    a = compute_angles(pose(p))
    for angle in (a.upper_arm[LEFT], a.lower_arm[LEFT], a.knee[LEFT], a.neck_flex):
        assert isinstance(angle, Missing) and "degenerate" in angle.reason

@pytest.mark.parametrize("low, field, reason", [
    ((K.L_SHOULDER, K.R_SHOULDER), "trunk_flex", "no confident shoulders"),
    ((K.L_HIP, K.R_HIP), "trunk_flex", "no confident hips"),
    ((K.L_EAR, K.R_EAR), "neck_flex", "no confident ear"),
    ((K.L_ANKLE,), "legs_bilateral", "ankle not confident"),
    ((K.L_HIP,), "trunk_twisted", "shoulders or hips not confident"),
])
def test_missing_reasons(low, field, reason):
    assert getattr(compute_angles(pose(upright(), low=low)), field) == Missing(reason)

def test_facing_falls_back_to_nose_vs_shoulders():
    assert facing(pose(upright({K.NOSE: (97, 178)}), low=(K.L_EAR, K.R_EAR))) == -1
