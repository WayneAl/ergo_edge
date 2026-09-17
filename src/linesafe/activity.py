"""Activity flags over time: REBA static / repeated / rapid, RULA muscle use.

Two signals are followed, both in degrees:
  * trunk = ``angles.trunk_flex.deg`` when measured;
  * arm = the larger measured upper-arm flexion, when either arm is measured.

A frame with ``angles is None`` or a missing signal adds no sample to that signal
(a gap). Every per-frame operation is O(1) amortised: no history is rescanned.

  * *held(signal, span)*: the signal has stayed within ``static_band_deg`` for at
    least ``span`` seconds without a gap longer than ``max_gap_s``, and its last
    sample is no older than ``max_gap_s``.
  * *actions(signal)*: reversals of at least ``rep_amp_deg`` travel within the last
    ``window_s`` seconds, halved (a flex-and-return is one action).
  * *rapid*: the trunk ranged by at least ``rapid_deg`` within ``rapid_window_s``
    at some point in the last ``rapid_hold_s`` seconds.

Time is in seconds and must not go backwards.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .geometry import Angles, Measured


def _check_float(name: str, value: float, *, positive: bool) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (value <= 0 if positive else value < 0)
    ):
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"ActivityParams.{name} must be finite and {bound}, got {value!r}")


@dataclass(frozen=True)
class ActivityParams:
    window_s: float = 60.0  # REBA static / repetition window
    static_band_deg: float = 10.0  # held = max - min within this band
    rep_amp_deg: float = 15.0  # a reversal needs this much travel from the running extremum
    rep_per_min: int = 4  # REBA: more than this many actions per minute -> +1
    rapid_window_s: float = 1.0
    rapid_deg: float = 45.0  # trunk range within rapid_window_s at or above this -> rapid
    rapid_hold_s: float = 2.0  # the rapid flag stays on this long after the last trigger
    rula_static_s: float = 600.0  # RULA muscle use: static longer than 10 min (worksheet)
    max_gap_s: float = 1.0  # a gap longer than this breaks "held"

    def __post_init__(self) -> None:
        _check_float("window_s", self.window_s, positive=True)
        _check_float("static_band_deg", self.static_band_deg, positive=False)
        _check_float("rep_amp_deg", self.rep_amp_deg, positive=True)
        n = self.rep_per_min
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"ActivityParams.rep_per_min must be an int >= 0, got {n!r}")
        _check_float("rapid_window_s", self.rapid_window_s, positive=True)
        _check_float("rapid_deg", self.rapid_deg, positive=True)
        _check_float("rapid_hold_s", self.rapid_hold_s, positive=False)
        _check_float("rula_static_s", self.rula_static_s, positive=True)
        _check_float("max_gap_s", self.max_gap_s, positive=False)


@dataclass(frozen=True)
class ActivityFlags:
    static: bool
    repeated: bool
    rapid: bool
    rula_muscle_use: bool

    @property
    def reba_points(self) -> int:
        """REBA activity score: +1 each for static, repeated and rapid."""
        return int(self.static) + int(self.repeated) + int(self.rapid)


class _Signal:
    """Run (held) and zig-zag (reversal) state of one angle signal."""

    def __init__(self) -> None:
        self.last_t: float | None = None
        # run
        self.run_start = 0.0
        self.run_min = 0.0
        self.run_max = 0.0
        # zig-zag
        self.dir = 0  # 0 unknown, +1 rising, -1 falling
        self.ref = 0.0
        self.ext = 0.0
        self.reversals: deque[float] = deque()

    def add(self, t: float, v: float, p: ActivityParams) -> None:
        gap = self.last_t is None or t - self.last_t > p.max_gap_s

        if gap or max(self.run_max, v) - min(self.run_min, v) > p.static_band_deg:
            self.run_start = t
            self.run_min = self.run_max = v
        else:
            self.run_min = min(self.run_min, v)
            self.run_max = max(self.run_max, v)

        if gap:
            self.dir = 0
            self.ref = v
        if self.dir == 0:
            if v - self.ref >= p.rep_amp_deg:
                self.dir, self.ext = 1, v
            elif self.ref - v >= p.rep_amp_deg:
                self.dir, self.ext = -1, v
        elif self.dir == 1:
            if v > self.ext:
                self.ext = v
            elif self.ext - v >= p.rep_amp_deg:
                self.reversals.append(t)
                self.dir, self.ext = -1, v
        else:
            if v < self.ext:
                self.ext = v
            elif v - self.ext >= p.rep_amp_deg:
                self.reversals.append(t)
                self.dir, self.ext = 1, v

        self.last_t = t

    def held(self, t: float, span: float, p: ActivityParams) -> bool:
        return self.last_t is not None and self.last_t >= t - p.max_gap_s and self.run_start <= t - span

    def actions(self, t: float, p: ActivityParams) -> int:
        while self.reversals and self.reversals[0] < t - p.window_s:
            self.reversals.popleft()
        return len(self.reversals) // 2


class _RangeWindow:
    """Min and max of the samples in ``[t - window, t]`` via monotonic deques."""

    def __init__(self) -> None:
        self._hi: deque[tuple[float, float]] = deque()  # values decreasing front to back
        self._lo: deque[tuple[float, float]] = deque()  # values increasing front to back

    def add(self, t: float, v: float) -> None:
        while self._hi and self._hi[-1][1] <= v:
            self._hi.pop()
        self._hi.append((t, v))
        while self._lo and self._lo[-1][1] >= v:
            self._lo.pop()
        self._lo.append((t, v))

    def evict_before(self, t_min: float) -> None:
        while self._hi and self._hi[0][0] < t_min:
            self._hi.popleft()
        while self._lo and self._lo[0][0] < t_min:
            self._lo.popleft()

    def span(self) -> float:
        return self._hi[0][1] - self._lo[0][1] if self._hi else 0.0


def _deg(name: str, angle: object) -> float | None:
    if not isinstance(angle, Measured):
        return None
    if not math.isfinite(angle.deg):
        raise ValueError(f"{name}.deg must be finite, got {angle.deg!r}")
    return float(angle.deg)


class ActivityTracker:
    def __init__(self, params: ActivityParams = ActivityParams()) -> None:
        self.params = params
        self._last_t: float | None = None
        self._trunk = _Signal()
        self._arm = _Signal()
        self._rapid = _RangeWindow()
        self._last_rapid_t: float | None = None

    def update(self, t: float, angles: Angles | None) -> ActivityFlags:
        """Add one frame at time ``t`` (seconds) and return the current flags."""
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError(f"ActivityTracker.update t must be finite, got {t!r}")
        if self._last_t is not None and t < self._last_t:
            raise ValueError(f"ActivityTracker.update t must not go backwards: {self._last_t} -> {t}")
        self._last_t = t
        p = self.params

        trunk = arm = None
        if angles is not None:
            trunk = _deg("angles.trunk_flex", angles.trunk_flex)
            arms = [_deg(f"angles.upper_arm[{i}]", a) for i, a in enumerate(angles.upper_arm)]
            arm = max((d for d in arms if d is not None), default=None)

        if trunk is not None:
            self._trunk.add(t, trunk, p)
            self._rapid.add(t, trunk)
        if arm is not None:
            self._arm.add(t, arm, p)

        self._rapid.evict_before(t - p.rapid_window_s)
        if self._rapid.span() >= p.rapid_deg:
            self._last_rapid_t = t

        actions = max(self._trunk.actions(t, p), self._arm.actions(t, p))
        return ActivityFlags(
            static=self._trunk.held(t, p.window_s, p) or self._arm.held(t, p.window_s, p),
            repeated=actions > p.rep_per_min,
            rapid=self._last_rapid_t is not None and t - self._last_rapid_t <= p.rapid_hold_s,
            rula_muscle_use=(
                self._trunk.held(t, p.rula_static_s, p)
                or self._arm.held(t, p.rula_static_s, p)
                or actions >= p.rep_per_min
            ),
        )
