"""Replay backend: serves detections from a committed keypoints file.

This is what makes the whole pipeline testable without a GPU, an NPU or even a
video file: dump a clip once with a real backend, commit the keypoints JSON,
and every later run reproduces exactly the same detections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..detections import Detection, RawDetections, load_keypoints


class ReplayBackend:
    """Replays the detections stored in a version-1 keypoints file."""

    name = "replay"

    def __init__(self, keypoints_path: Path):
        self.keypoints_path = Path(keypoints_path)
        self.raw: RawDetections = load_keypoints(self.keypoints_path)

    def infer(self, frame_bgr: np.ndarray, idx: int) -> list[Detection]:
        """Detections recorded for frame ``idx``; ``frame_bgr`` is ignored.

        Raises ``IndexError`` past the end of the file rather than returning an
        empty list, so a video longer than its keypoints dump fails loudly.
        """
        n = len(self.raw.frames)
        if idx < 0 or idx >= n:
            raise IndexError(
                f"frame {idx} out of range: {self.keypoints_path.name} "
                f"has {n} frames"
            )
        return self.raw.frames[idx]

    def close(self) -> None:
        """Nothing to release."""
