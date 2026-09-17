"""One frame through the whole chain: backend -> tracker -> view and angles -> REBA/RULA
-> activity -> events -> store.

The same :class:`Pipeline` runs live (``linesafe run``) and from a keypoints file
(``linesafe replay``). Time passed to :meth:`Pipeline.step` is the pipeline clock in
seconds and must strictly increase (live: ``time.monotonic()``; replay: ``idx / fps``).
Everything written to the store is shifted by ``wall_offset`` so a live session lands on
wall-clock time while the pipeline itself never sees a clock jump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .activity import ActivityFlags, ActivityParams, ActivityTracker
from .backends.base import PoseBackend
from .config import StationConfig
from .events import Event, EventDetector, EventParams
from .geometry import Angles, GeometryParams, View, classify_view, compute_angles
from .pose import PoseFrame
from .reba import RebaScore, score_reba
from .rula import RulaScore, score_rula
from .store import EventStore
from .track import Tracker, TrackerParams


@dataclass(frozen=True)
class PipelineParams:
    tracker: TrackerParams = TrackerParams()
    geometry: GeometryParams = GeometryParams()
    activity: ActivityParams = ActivityParams()
    events: EventParams = EventParams()
    wrong_view_s: float = 2.0  # FRONT continuously this long -> wrong_view
    status_every_s: float = 0.5  # store.set_status at most this often

    def __post_init__(self) -> None:
        for name in ("wrong_view_s", "status_every_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"PipelineParams.{name} must be finite and >= 0, got {value!r}")


@dataclass(frozen=True)
class FrameResult:
    t: float
    pose: PoseFrame | None
    view: View | None
    wrong_view: bool
    angles: Angles | None
    reba: RebaScore | None
    rula: RulaScore | None
    activity: ActivityFlags
    event_active: bool
    closed_event: Event | None  # in pipeline time (not shifted by wall_offset)


class Pipeline:
    def __init__(
        self,
        backend: PoseBackend,
        cfg: StationConfig,
        params: PipelineParams = PipelineParams(),
        store: EventStore | None = None,
        wall_offset: float = 0.0,
    ) -> None:
        if isinstance(wall_offset, bool) or not isinstance(wall_offset, (int, float)) or not math.isfinite(
            wall_offset
        ):
            raise ValueError(f"Pipeline.wall_offset must be a finite number, got {wall_offset!r}")
        self.backend = backend
        self.cfg = cfg
        self.params = params
        self.store = store
        self.wall_offset = float(wall_offset)
        self._tracker = Tracker(params.tracker, roi=cfg.roi)
        self._activity = ActivityTracker(params.activity)
        self._events = EventDetector(cfg.station_id, params.events)
        self._front_since: float | None = None  # t of the first frame of the current FRONT run
        self._last_status_t: float | None = None

    def _store_event(self, e: Event) -> None:
        if self.store is not None:
            off = self.wall_offset
            self.store.add_event(replace(e, t_start=e.t_start + off, t_end=e.t_end + off))

    def step(self, t: float, frame_bgr: np.ndarray | None, idx: int, fps: float = 0.0) -> FrameResult:
        """Process frame ``idx`` taken at pipeline time ``t`` (seconds)."""
        p = self.params
        dets = self.backend.infer(frame_bgr, idx)
        pose = self._tracker.update(t, dets)

        view: View | None = None
        wrong_view = False
        angles: Angles | None = None
        if pose is None:
            self._front_since = None  # no person breaks a FRONT run
        else:
            view = classify_view(pose, p.geometry)
            if view is View.FRONT:
                if self._front_since is None:
                    self._front_since = t
                wrong_view = t - self._front_since >= p.wrong_view_s
            else:
                self._front_since = None
            angles = compute_angles(pose, p.geometry, view_ok=not wrong_view)

        flags = self._activity.update(t, angles)
        reba = score_reba(angles, self.cfg, flags.reba_points) if angles is not None else None
        rula = score_rula(angles, self.cfg, flags.rula_muscle_use) if angles is not None else None
        closed = self._events.update(t, reba)

        if self.store is not None:
            if closed is not None:
                self._store_event(closed)
            if self._last_status_t is None or t - self._last_status_t >= p.status_every_s:
                self.store.set_status(
                    self.cfg.station_id,
                    t + self.wall_offset,
                    reba.total if reba is not None else None,
                    reba.band.value if reba is not None else None,
                    rula.total if rula is not None else None,
                    reba.drivers if reba is not None else (),
                    reba.partial if reba is not None else False,
                    fps,
                )
                self._last_status_t = t

        return FrameResult(
            t=t,
            pose=pose,
            view=view,
            wrong_view=wrong_view,
            angles=angles,
            reba=reba,
            rula=rula,
            activity=flags,
            event_active=self._events.active,
            closed_event=closed,
        )

    def close(self) -> Event | None:
        """End of stream: flush the event detector and store an event it closes.

        Does not close the backend or the store; their owner does.
        """
        e = self._events.flush()
        if e is not None:
            self._store_event(e)
        return e
