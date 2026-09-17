"""Streaming single-person tracker with per-keypoint 1€ smoothing.

Frame by frame, :class:`Tracker` picks one person out of a pose backend's
detections, holds on to them by bounding-box overlap, and smooths their
keypoints into a :class:`~linesafe.pose.PoseFrame`.

Lock rules:
  * Not locked: seed on the largest bbox whose both shoulders are confident and
    whose centre lies inside the ROI (if one is set); ties by score, then list order.
  * Locked: follow the detection with the highest IoU against the last matched
    bbox, if that IoU reaches ``iou_min`` — a bigger bystander never steals the
    lock while the target still overlaps. The ROI is not applied to matching.
  * No match for longer than ``lost_s``: drop the lock, reset all smoothing
    state, and re-seed from the same frame's detections.

Units: pixels (x right, y down) for keypoints and bbox, seconds for time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import keypoints as K
from .detections import Detection
from .pose import PoseFrame
from .smoothing import OneEuroFilter


@dataclass(frozen=True)
class TrackerParams:
    conf_min: float = 0.3
    iou_min: float = 0.3
    lost_s: float = 0.5  # no matching detection for longer than this -> drop the lock and re-seed
    hold_s: float = 0.1  # a keypoint below conf_min keeps its last smoothed value this long
    min_cutoff: float = 1.5
    beta: float = 0.05
    freq_hint: float = 30.0  # OneEuroFilter(freq=...) before timestamps exist

    def __post_init__(self) -> None:
        if not 0.0 < self.conf_min <= 1.0:
            raise ValueError(
                f"TrackerParams.conf_min must be in (0, 1], got {self.conf_min}"
            )
        if not 0.0 < self.iou_min <= 1.0:
            raise ValueError(
                f"TrackerParams.iou_min must be in (0, 1], got {self.iou_min}"
            )
        if not (math.isfinite(self.lost_s) and self.lost_s >= 0):
            raise ValueError(
                f"TrackerParams.lost_s must be finite and >= 0, got {self.lost_s}"
            )
        if not (math.isfinite(self.hold_s) and self.hold_s >= 0):
            raise ValueError(
                f"TrackerParams.hold_s must be finite and >= 0, got {self.hold_s}"
            )
        if not self.min_cutoff > 0:
            raise ValueError(
                f"TrackerParams.min_cutoff must be > 0, got {self.min_cutoff}"
            )
        if not self.beta >= 0:
            raise ValueError(f"TrackerParams.beta must be >= 0, got {self.beta}")
        if not self.freq_hint > 0:
            raise ValueError(
                f"TrackerParams.freq_hint must be > 0, got {self.freq_hint}"
            )


def _iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection over union of two ``(x1, y1, x2, y2)`` boxes; 0 if the union is empty."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, iw) * max(0.0, ih)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Follows one person across frames and smooths their keypoints.

    Args:
        params: thresholds and 1€ settings, see :class:`TrackerParams`.
        roi: optional ``(x1, y1, x2, y2)`` pixel region; only detections whose
            bbox centre lies inside it (edges included) can seed the lock.
    """

    def __init__(
        self,
        params: TrackerParams = TrackerParams(),
        roi: tuple[int, int, int, int] | None = None,
    ) -> None:
        if roi is not None:
            roi = tuple(float(v) for v in roi)
            if len(roi) != 4:
                raise ValueError(f"Tracker.roi must have 4 values, got {len(roi)}")
            if not (roi[2] > roi[0] and roi[3] > roi[1]):
                raise ValueError(
                    f"Tracker.roi must have x2 > x1 and y2 > y1, got {roi}"
                )
        self.params = params
        self.roi = roi
        self._filters = [
            [
                OneEuroFilter(
                    freq=params.freq_hint,
                    min_cutoff=params.min_cutoff,
                    beta=params.beta,
                )
                for _ in range(2)
            ]
            for _ in range(K.N_KPTS)
        ]
        self._locked = False
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_matched_t: float | None = None
        self._last_t: float | None = None  # last t passed to update, any call
        self._reset_keypoints()

    @property
    def locked(self) -> bool:
        return self._locked

    def _reset_keypoints(self) -> None:
        for pair in self._filters:
            for f in pair:
                f.reset()
        self._last_xy = np.zeros((K.N_KPTS, 2), np.float32)
        self._last_conf = np.zeros(K.N_KPTS, np.float32)
        self._last_good_t = np.zeros(K.N_KPTS, np.float64)
        self._has_value = np.zeros(K.N_KPTS, bool)

    def _in_roi(self, d: Detection) -> bool:
        if self.roi is None:
            return True
        x1, y1, x2, y2 = d.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx1, ry1, rx2, ry2 = self.roi
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _seed(self, dets: list[Detection]) -> Detection | None:
        cm = self.params.conf_min
        best: Detection | None = None
        for d in dets:
            if not (d.conf[K.L_SHOULDER] >= cm and d.conf[K.R_SHOULDER] >= cm):
                continue
            if not self._in_roi(d):
                continue
            if best is None or (d.area, d.score) > (best.area, best.score):
                best = d
        return best

    def _match(self, dets: list[Detection]) -> Detection | None:
        assert self._last_bbox is not None
        best: Detection | None = None
        best_iou = -1.0
        for d in dets:
            iou = _iou(self._last_bbox, d.bbox)
            if iou > best_iou:
                best, best_iou = d, iou
        if best is None or best_iou < self.params.iou_min:
            return None
        return best

    def update(self, t: float, dets: list[Detection]) -> PoseFrame | None:
        """Process one frame's detections at time ``t`` (seconds).

        Returns the tracked person's smoothed pose, or ``None`` when nobody is
        tracked this frame.

        Raises ``ValueError``, before any tracker state changes, when ``t`` is
        not finite or not strictly greater than the ``t`` of the previous call
        (any call, with or without detections); when any detection's bbox is
        not finite or has ``x2 < x1`` or ``y2 < y1`` (zero width or height is
        allowed); or when the detection chosen this frame has a non-finite
        confidence, or a non-finite keypoint that would be used (confident, or
        output raw because nothing is held for it). A call that raised leaves
        the tracker exactly as it was, so the next valid call behaves as if the
        bad one never happened.
        """
        t = float(t)
        if not math.isfinite(t):
            raise ValueError(f"Tracker.update t must be finite, got {t}")
        if self._last_t is not None and t <= self._last_t:
            raise ValueError(
                f"Tracker.update t must strictly increase: {self._last_t} -> {t}"
            )
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            finite = all(math.isfinite(v) for v in d.bbox)
            if not (finite and x1 <= x2 and y1 <= y2):
                raise ValueError(
                    "Detection.bbox must be finite with x1 <= x2 and y1 <= y2, "
                    f"got {d.bbox}"
                )

        p = self.params
        # Decide what this frame does without touching state.
        chosen: Detection | None = None
        lost = False
        if self._locked:
            chosen = self._match(dets)
            if chosen is None:
                assert self._last_matched_t is not None
                lost = t - self._last_matched_t > p.lost_s
        if chosen is None and (lost or not self._locked):
            chosen = self._seed(dets)
        if chosen is not None:
            if not np.isfinite(chosen.conf).all():
                raise ValueError(
                    "Detection.conf must be finite for the tracked detection"
                )
            has_value = np.zeros(K.N_KPTS, bool) if lost else self._has_value
            used = (chosen.conf >= p.conf_min) | ~has_value
            if not np.isfinite(chosen.kpts[used]).all():
                raise ValueError(
                    "Detection.kpts must be finite for every keypoint the tracker uses"
                )

        # Validated: state changes from here on.
        self._last_t = t
        if lost:
            self._locked = False
            self._last_bbox = None
            self._last_matched_t = None
            self._reset_keypoints()
        if chosen is None:
            return None

        kpts = np.empty((K.N_KPTS, 2), np.float32)
        conf = np.empty(K.N_KPTS, np.float32)
        for k in range(K.N_KPTS):
            c = float(chosen.conf[k])
            if c >= p.conf_min:
                for axis in range(2):
                    kpts[k, axis] = self._filters[k][axis](
                        float(chosen.kpts[k, axis]), t
                    )
                conf[k] = c
                self._last_xy[k] = kpts[k]
                self._last_conf[k] = c
                self._last_good_t[k] = t
                self._has_value[k] = True
            elif self._has_value[k]:
                kpts[k] = self._last_xy[k]
                if t - self._last_good_t[k] <= p.hold_s:
                    conf[k] = self._last_conf[k]
                else:
                    conf[k] = 0.0
            else:
                kpts[k] = chosen.kpts[k]
                conf[k] = 0.0

        self._locked = True
        self._last_bbox = chosen.bbox
        self._last_matched_t = t
        return PoseFrame(t, kpts, conf, bbox=chosen.bbox)
