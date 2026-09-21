"""RULA scoring from side-view angles (McAtamney & Corlett, Applied Ergonomics 24(2) (1993) 91-99).

Segment boundaries are fixed here; table lookups go through ``linesafe.tables``. The
upper arm uses ``reba.upper_arm_score`` (the bands are identical). Wrist, wrist twist
and force come from ``StationConfig``. A side view cannot see a raised shoulder, an
abducted upper arm, an arm working across the midline or out to the side, or neck and
trunk side-bend and twist (other than the trunk twist flag); those adjustments are
assumed absent.

Muscle use is "static > 10 min or repeated >= 4/min" (ErgoPlus and IEH worksheets
agree). It is computed upstream in ``activity.py``; ``score_rula`` only receives the
boolean.

Missing angles mirror ``score_reba``: no trunk flexion means no score (``None``). Any
other missing angle is scored from the available parts, listed in ``missing`` and,
when it can change the total, marks the score ``partial``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import reba
from .config import StationConfig
from .geometry import LEFT, RIGHT, Angles, Measured, Missing
from .reba import _finite, _flag, lower_arm_score, upper_arm_score
from .tables import rula_table_a, rula_table_b, rula_table_c


class RulaLevel(str, Enum):
    ACCEPTABLE = "acceptable"
    INVESTIGATE = "investigate"
    CHANGE_SOON = "change soon"
    CHANGE_NOW = "change now"


def rula_level(total: int) -> RulaLevel:
    """1-2 | 3-4 | 5-6 | 7."""
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError(f"total must be an int, got {total!r}")
    if 1 <= total <= 2:
        return RulaLevel.ACCEPTABLE
    if 3 <= total <= 4:
        return RulaLevel.INVESTIGATE
    if 5 <= total <= 6:
        return RulaLevel.CHANGE_SOON
    if total == 7:
        return RulaLevel.CHANGE_NOW
    raise ValueError(f"total must be 1..7, got {total}")


def rula_neck_score(flex_deg: float) -> int:
    n = _finite("neck flex_deg", flex_deg)
    if -5 <= n <= 10:
        return 1
    if 10 < n <= 20:
        return 2
    if n > 20:
        return 3
    return 4  # n < -5: extension


def rula_trunk_score(flex_deg: float, twisted: bool) -> int:
    f = _finite("trunk flex_deg", flex_deg)
    twisted = _flag("trunk twisted", twisted)
    tol = reba.UPRIGHT_TOL_DEG  # shared with REBA so a calibrated tolerance moves both
    if abs(f) <= tol:
        s = 1
    elif tol < f <= 20 or f < -tol:
        s = 2
    elif 20 < f <= 60:
        s = 3
    else:  # f > 60
        s = 4
    return s + int(twisted)


def rula_legs_score(bilateral: bool) -> int:
    return 1 if _flag("legs bilateral", bilateral) else 2


def rula_lower_arm_score(flex_deg: float) -> int:
    """60 <= l <= 100 -> 1, else 2 (same bands as REBA)."""
    return lower_arm_score(flex_deg)


@dataclass(frozen=True)
class RulaScore:
    t: float
    total: int  # 1-7
    level: RulaLevel
    score_a: int  # table A + muscle + force
    score_b: int  # table B + muscle + force
    parts: dict[str, int]  # upper_arm, lower_arm, wrist, wrist_twist, neck, trunk, legs, muscle, force
    side: str  # "left" | "right" | "none" — arm used for score A
    partial: bool
    missing: tuple[str, ...]


_SIDE_NAMES = {LEFT: "left", RIGHT: "right"}


def score_rula(angles: Angles, cfg: StationConfig, muscle_use: bool = False) -> RulaScore | None:
    """RULA score for one frame; ``None`` when trunk flexion is Missing."""
    if not isinstance(muscle_use, bool):
        raise ValueError(f"muscle_use must be a bool, got {muscle_use!r}")
    if isinstance(angles.trunk_flex, Missing):
        return None

    missing: list[str] = []
    partial = False

    # Group B body parts: trunk, neck, legs
    if isinstance(angles.trunk_twisted, Missing):
        twisted = False
        missing.append("trunk twist")
    else:
        twisted = angles.trunk_twisted
    trunk = rula_trunk_score(angles.trunk_flex.deg, twisted)

    if isinstance(angles.neck_flex, Missing):
        neck = 1
        missing.append("neck")
        partial = True
    else:
        neck = rula_neck_score(angles.neck_flex.deg)

    if isinstance(angles.legs_bilateral, Missing):
        legs = 1
        missing.append("legs")
        partial = True
    else:
        legs = rula_legs_score(angles.legs_bilateral)

    muscle = int(muscle_use)
    force = cfg.rula_force
    wrist = cfg.rula_wrist
    twist = cfg.rula_wrist_twist

    # Group A: the arm with the larger Table A value; ties -> larger upper-arm flexion -> left.
    best: tuple[int, float, int, int, int, bool] | None = None  # (table_a, u, side, upper, lower, lower_missing)
    for side in (LEFT, RIGHT):
        ua = angles.upper_arm[side]
        if not isinstance(ua, Measured):
            if isinstance(angles.upper_arm[1 - side], Measured):
                missing.append(f"upper arm {_SIDE_NAMES[side]}")
            continue
        upper = upper_arm_score(ua.deg, cfg.arm_supported)
        la = angles.lower_arm[side]
        lower_missing = not isinstance(la, Measured)
        lower = 2 if lower_missing else rula_lower_arm_score(la.deg)
        ta = rula_table_a(int(upper), int(lower), int(wrist), int(twist))
        if best is None or (ta, ua.deg) > (best[0], best[1]):
            best = (ta, ua.deg, side, upper, lower, lower_missing)

    if best is None:
        upper, lower, side_name = 1, 2, "none"
        table_a = rula_table_a(upper, lower, int(wrist), int(twist))
        missing.append("upper arm")
        partial = True
    else:
        table_a, _, side_i, upper, lower, lower_missing = best
        side_name = _SIDE_NAMES[side_i]
        if lower_missing:
            missing.append(f"lower arm {side_name}")
            partial = True

    score_a = table_a + muscle + force
    score_b = rula_table_b(int(min(neck, 6)), int(min(trunk, 6)), int(legs)) + muscle + force
    total = rula_table_c(int(score_a), int(score_b))
    assert 1 <= total <= 7, total

    return RulaScore(
        t=angles.t,
        total=total,
        level=rula_level(total),
        score_a=score_a,
        score_b=score_b,
        parts={
            "upper_arm": upper,
            "lower_arm": lower,
            "wrist": wrist,
            "wrist_twist": twist,
            "neck": neck,
            "trunk": trunk,
            "legs": legs,
            "muscle": muscle,
            "force": force,
        },
        side=side_name,
        partial=partial,
        missing=tuple(missing),
    )
