"""High-risk events from a stream of REBA scores.

A sample is *above* when it has a score whose total is at or above
``enter_total`` (REBA High); a ``None`` score counts as below.

States:
  * IDLE -> PENDING when a sample is above (the event's start).
  * PENDING -> ACTIVE once still above ``enter_s`` seconds after the start;
    PENDING -> IDLE on any sample below (the run was too short).
  * ACTIVE stays open through dips shorter than ``exit_s``; after ``exit_s``
    seconds below it closes, :meth:`EventDetector.update` returns the
    :class:`Event`, and the detector is IDLE again.

The event ends at the last sample above; peak and drivers are tracked from the
start of PENDING. Time is in seconds and must strictly increase.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .reba import Band, RebaScore, reba_band

_IDLE, _PENDING, _ACTIVE = "idle", "pending", "active"


@dataclass(frozen=True)
class EventParams:
    enter_total: int = 8  # REBA High
    enter_s: float = 3.0
    exit_s: float = 3.0  # below High this long closes the event

    def __post_init__(self) -> None:
        total = self.enter_total
        if isinstance(total, bool) or not isinstance(total, int) or not 1 <= total <= 15:
            raise ValueError(f"EventParams.enter_total must be an int 1..15, got {total!r}")
        for name in ("enter_s", "exit_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"EventParams.{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True)
class Event:
    station: str
    t_start: float
    t_end: float  # last sample at or above enter_total
    peak: int
    band: Band  # reba_band(peak)
    drivers: tuple[str, ...]  # drivers of the first sample that reached peak
    duration_s: float  # t_end - t_start


class EventDetector:
    def __init__(self, station: str, params: EventParams = EventParams()) -> None:
        if not isinstance(station, str) or not station.strip():
            raise ValueError(f"EventDetector.station must be a non-blank str, got {station!r}")
        self.station = station
        self.params = params
        self._last_t: float | None = None
        self._reset()

    def _reset(self) -> None:
        self._state = _IDLE
        self._t_start = 0.0
        self._last_above = 0.0
        self._below_since: float | None = None
        self._peak: int | None = None
        self._drivers: tuple[str, ...] = ()

    @property
    def active(self) -> bool:
        return self._state == _ACTIVE

    def _track(self, t: float, score: RebaScore) -> None:
        self._last_above = t
        if self._peak is None or score.total > self._peak:
            self._peak = score.total
            self._drivers = tuple(score.drivers)

    def _close(self) -> Event:
        assert self._peak is not None
        event = Event(
            station=self.station,
            t_start=self._t_start,
            t_end=self._last_above,
            peak=self._peak,
            band=reba_band(self._peak),
            drivers=self._drivers,
            duration_s=self._last_above - self._t_start,
        )
        self._reset()
        return event

    def update(self, t: float, score: RebaScore | None) -> Event | None:
        """Add one sample at time ``t`` (seconds); returns the event when it closes."""
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t):
            raise ValueError(f"EventDetector.update t must be finite, got {t!r}")
        if self._last_t is not None and t <= self._last_t:
            raise ValueError(f"EventDetector.update t must increase: {self._last_t} -> {t}")
        if score is not None:
            total = score.total
            if isinstance(total, bool) or not isinstance(total, int) or not 1 <= total <= 15:
                raise ValueError(f"score.total must be an int 1..15, got {total!r}")
        self._last_t = t  # all input checked; state changes from here on
        above = score is not None and score.total >= self.params.enter_total

        if self._state == _IDLE:
            if not above:
                return None
            self._state = _PENDING
            self._t_start = t

        if self._state == _PENDING:
            if not above:
                self._reset()
                return None
            self._track(t, score)
            if t - self._t_start >= self.params.enter_s:
                self._state = _ACTIVE
            return None

        # ACTIVE
        if above:
            self._track(t, score)
            self._below_since = None
            return None
        if self._below_since is None:
            self._below_since = t
        if t - self._below_since >= self.params.exit_s:
            return self._close()
        return None

    def flush(self) -> Event | None:
        """End of stream: close and return an active event; a pending run is dropped (``None``)."""
        if self._state == _ACTIVE:
            return self._close()
        self._reset()
        return None
