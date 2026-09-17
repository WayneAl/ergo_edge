"""Pose backends: everything that turns a BGR frame into COCO-17 detections.

Every backend emits :class:`linesafe.detections.Detection` objects in the COCO-17
layout of :mod:`linesafe.keypoints`, so the rest of the pipeline never learns
which one produced them. Heavy or hardware-specific imports (ultralytics/torch,
pyHailoRT) happen inside the concrete backends, so importing this package — and
running the test suite — stays dependency-free.
"""

from __future__ import annotations

from .base import PoseBackend, make_backend

__all__ = ["PoseBackend", "make_backend"]
