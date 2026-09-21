"""The pose-backend contract and the factory that builds one by name."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from ..detections import Detection


class PoseBackend(Protocol):
    """Anything that can turn one BGR frame into COCO-17 detections."""

    #: Human-readable backend id, recorded in keypoints files and reports.
    name: str

    def infer(self, frame_bgr: np.ndarray, idx: int) -> list[Detection]:
        """Detections for frame ``idx``; empty list when nobody is detected."""
        ...

    def close(self) -> None:
        """Release any device or file handles. Safe to call more than once."""
        ...


def make_backend(name: str, **kwargs: Any) -> PoseBackend:
    """Build a backend by name.

    ``"ultralytics"`` runs yolov8-pose locally, ``"replay"`` replays a
    committed keypoints file (needs ``keypoints_path=``), ``"hailo"`` targets
    the UGen300 NPU (unverified adapter). Any other name is a ``ValueError`` —
    an unknown backend is never silently substituted.
    """
    # Imported lazily so that `make_backend("replay")` never drags in torch.
    if name == "ultralytics":
        from .ultralytics_backend import UltralyticsBackend

        return UltralyticsBackend(**kwargs)
    if name == "replay":
        from .replay import ReplayBackend

        return ReplayBackend(**kwargs)
    if name == "hailo":
        from .hailo_backend import HailoBackend

        return HailoBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")
