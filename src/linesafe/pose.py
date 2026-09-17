"""The single tracked, smoothed person that geometry works on.

Units: pixels (x right, y down) for keypoints and bbox, seconds for time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .detections import _check_kpts_conf


@dataclass(frozen=True)
class PoseFrame:
    """One tracked person at time ``t``."""

    t: float  # seconds
    kpts: np.ndarray  # (17, 2) float32, pixels
    conf: np.ndarray  # (17,) float32, 0..1
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    def __post_init__(self) -> None:
        t = float(self.t)
        if not math.isfinite(t):
            raise ValueError(f"PoseFrame.t must be finite, got {self.t}")
        kpts, conf = _check_kpts_conf(self.kpts, self.conf, "PoseFrame")
        bbox = tuple(float(v) for v in self.bbox)
        if len(bbox) != 4:
            raise ValueError(f"PoseFrame.bbox must have 4 values, got {len(bbox)}")
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "kpts", kpts)
        object.__setattr__(self, "conf", conf)
        object.__setattr__(self, "bbox", bbox)
