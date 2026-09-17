"""REBA scoring from side-view angles (Hignett & McAtamney, Applied Ergonomics 31 (2000) 201-205).

Segment boundaries are fixed here; table lookups go through ``linesafe.tables``.
A side view cannot see trunk side-bend, neck twist or side-bend, a raised shoulder
or an abducted upper arm; those adjustments are assumed absent and listed in
``RebaScore.assumed``.

Missing angles: no trunk flexion means no score (``None``). Any other missing angle
is scored from the available parts, listed in ``missing`` and, when it can change
the total, marks the score ``partial``.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from enum import Enum

from .config import StationConfig
from .geometry import LEFT, RIGHT, Angles, Measured, Missing
from .tables import reba_table_a, reba_table_b, reba_table_c


class Band(str, Enum):
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very high"


def reba_band(total: int) -> Band:
    """1 | 2-3 | 4-7 | 8-10 | 11-15."""
    if isinstance(total, bool) or not isinstance(total, int):
        raise ValueError(f"total must be an int, got {total!r}")
    if total == 1:
        return Band.NEGLIGIBLE
    if 2 <= total <= 3:
        return Band.LOW
    if 4 <= total <= 7:
        return Band.MEDIUM
    if 8 <= total <= 10:
        return Band.HIGH
    if 11 <= total <= 15:
        return Band.VERY_HIGH
    raise ValueError(f"total must be 1..15, got {total}")


UPRIGHT_TOL_DEG = 5.0


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def trunk_score(flex_deg: float, twisted: bool) -> int:
    f = _finite("flex_deg", flex_deg)
    if abs(f) <= UPRIGHT_TOL_DEG:
        s = 1
    elif 5 < f <= 20 or -20 <= f < -5:
        s = 2
    elif 20 < f <= 60 or f < -20:
        s = 3
    else:  # f > 60
        s = 4
    return s + int(bool(twisted))


def neck_score(flex_deg: float) -> int:
    n = _finite("flex_deg", flex_deg)
    return 1 if -5 <= n <= 20 else 2


def legs_score(bilateral: bool, knee_flex_deg: float | None) -> int:
    s = 1 if bilateral else 2
    if knee_flex_deg is None:
        return s
    k = _finite("knee_flex_deg", knee_flex_deg)
    if 30 <= k <= 60:
        return s + 1
    if k > 60:
        return s + 2
    return s


def upper_arm_score(flex_deg: float, supported: bool) -> int:
    u = _finite("flex_deg", flex_deg)
    if -20 <= u <= 20:
        s = 1
    elif 20 < u <= 45 or u < -20:
        s = 2
    elif 45 < u <= 90:
        s = 3
    else:  # u > 90
        s = 4
    return max(1, s - 1) if supported else s


def lower_arm_score(flex_deg: float) -> int:
    lo = _finite("flex_deg", flex_deg)
    return 1 if 60 <= lo <= 100 else 2


ASSUMED_SIDE_VIEW: tuple[str, ...] = (
    "trunk side-bend",
    "neck twist",
    "neck side-bend",
    "shoulder raised",
    "upper arm abducted",
)


@dataclass(frozen=True)
class RebaScore:
    t: float
    total: int  # 1-15
    band: Band
    score_a: int  # table A + load
    score_b: int  # table B + coupling
    table_c: int
    activity: int  # 0-3
    parts: dict[str, int]  # keys: trunk, neck, legs, upper_arm, lower_arm, wrist, load, coupling
    side: str  # "left" | "right" | "none" — arm used for group B
    drivers: tuple[str, ...]
    partial: bool
    missing: tuple[str, ...]
    assumed: tuple[str, ...]  # always ASSUMED_SIDE_VIEW in v1


_SIDE_NAMES = {LEFT: "left", RIGHT: "right"}


def score_reba(angles: Angles, cfg: StationConfig, activity: int = 0) -> RebaScore | None:
    """REBA score for one frame; ``None`` when trunk flexion is Missing."""
    if isinstance(activity, bool) or not isinstance(activity, int) or not 0 <= activity <= 3:
        raise ValueError(f"activity must be an int 0..3, got {activity!r}")
    if isinstance(angles.trunk_flex, Missing):
        return None

    missing: list[str] = []
    partial = False

    # Group A
    f = angles.trunk_flex.deg
    if isinstance(angles.trunk_twisted, Missing):
        twisted = False
        missing.append("trunk twist")
    else:
        twisted = angles.trunk_twisted
    trunk = trunk_score(f, twisted)

    if isinstance(angles.neck_flex, Missing):
        n = None
        neck = 1
        missing.append("neck")
        partial = True
    else:
        n = angles.neck_flex.deg
        neck = neck_score(n)

    if isinstance(angles.legs_bilateral, Missing):
        bilateral = True
        missing.append("legs")
        partial = True
    else:
        bilateral = angles.legs_bilateral
    knees = [a.deg for a in angles.knee if isinstance(a, Measured)]
    k = max(knees) if knees else None
    if k is None:
        missing.append("knee")
    legs = legs_score(bilateral, k)
    knee_added = k is not None and k >= 30

    load = cfg.reba_load + int(cfg.reba_shock)
    score_a = reba_table_a(int(neck), int(trunk), int(legs)) + load

    # Group B: the arm with the larger Table B value; ties -> larger upper-arm flexion -> left.
    wrist = cfg.reba_wrist
    best: tuple[int, float, int, int, int, bool] | None = None  # (table_b, u, side, upper, lower, lower_missing)
    for side in (LEFT, RIGHT):
        ua = angles.upper_arm[side]
        if not isinstance(ua, Measured):
            continue
        upper = upper_arm_score(ua.deg, cfg.arm_supported)
        la = angles.lower_arm[side]
        lower_missing = not isinstance(la, Measured)
        lower = 2 if lower_missing else lower_arm_score(la.deg)
        tb = reba_table_b(int(upper), int(lower), int(wrist))
        if best is None or (tb, ua.deg) > (best[0], best[1]):
            best = (tb, ua.deg, side, upper, lower, lower_missing)

    if best is None:
        u = None
        upper, lower, side_name = 1, 2, "none"
        table_b = reba_table_b(upper, lower, int(wrist))
        missing.append("upper arm")
        partial = True
    else:
        table_b, u, side_i, upper, lower, lower_missing = best
        side_name = _SIDE_NAMES[side_i]
        if lower_missing:
            missing.append(f"lower arm {side_name}")
            partial = True

    coupling = cfg.reba_coupling
    score_b = table_b + coupling

    table_c = reba_table_c(int(min(score_a, 12)), int(min(score_b, 12)))
    total = table_c + activity
    assert 1 <= total <= 15, total

    drivers: list[str] = []
    if trunk >= 3:
        drivers.append(f"trunk flexion {f:.0f}°" if f >= 0 else f"trunk extension {-f:.0f}°")
    if twisted:
        drivers.append("trunk twist")
    if neck == 2:
        drivers.append(f"neck flexion {n:.0f}°")
    if upper >= 3:
        drivers.append(f"upper arm {side_name} {u:.0f}°")
    if legs >= 2:
        drivers.append(f"knee flexion {k:.0f}°" if knee_added else "one-leg stance")
    if load > 0:
        drivers.append("load (station)")
    if coupling > 0:
        drivers.append("coupling (station)")
    if activity > 0:
        drivers.append(f"activity +{activity}")

    return RebaScore(
        t=angles.t,
        total=total,
        band=reba_band(total),
        score_a=score_a,
        score_b=score_b,
        table_c=table_c,
        activity=activity,
        parts={
            "trunk": trunk,
            "neck": neck,
            "legs": legs,
            "upper_arm": upper,
            "lower_arm": lower,
            "wrist": wrist,
            "load": load,
            "coupling": coupling,
        },
        side=side_name,
        drivers=tuple(drivers),
        partial=partial,
        missing=tuple(missing),
        assumed=ASSUMED_SIDE_VIEW,
    )
