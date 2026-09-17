"""Pose-backend detections and the keypoints JSON format.

Everything downstream of a pose backend speaks these types. They validate at
construction and raise ``ValueError`` naming the problem — nothing here silently
coerces or clamps bad data.

Units: pixels for raw detections, seconds for time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .keypoints import N_KPTS

#: Version of the keypoints file format written by :func:`save_keypoints`.
KEYPOINTS_FORMAT_VERSION = 1

_XY_DP = 2  # decimal places for pixel coordinates
_CONF_DP = 3  # decimal places for confidences / scores


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    return int(value)


def _check_kpts_conf(
    kpts: Any, conf: Any, owner: str
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a (17, 2) / (17,) keypoint pair and return it as float32."""
    kpts = np.asarray(kpts, dtype=np.float32)
    if kpts.shape != (N_KPTS, 2):
        raise ValueError(
            f"{owner}.kpts must have shape ({N_KPTS}, 2), got {kpts.shape}"
        )
    conf = np.asarray(conf, dtype=np.float32)
    if conf.shape != (N_KPTS,):
        raise ValueError(
            f"{owner}.conf must have shape ({N_KPTS},), got {conf.shape}"
        )
    return kpts, conf


# --- per-frame detections ---------------------------------------------------


@dataclass(frozen=True)
class Detection:
    """One detected person in one frame, straight out of a pose backend."""

    kpts: np.ndarray  # (17, 2) float32, pixels
    conf: np.ndarray  # (17,) float32, 0..1
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    score: float  # detection confidence 0..1

    def __post_init__(self) -> None:
        kpts, conf = _check_kpts_conf(self.kpts, self.conf, "Detection")
        bbox = tuple(float(v) for v in self.bbox)
        if len(bbox) != 4:
            raise ValueError(f"Detection.bbox must have 4 values, got {len(bbox)}")
        object.__setattr__(self, "kpts", kpts)
        object.__setattr__(self, "conf", conf)
        object.__setattr__(self, "bbox", bbox)
        object.__setattr__(self, "score", float(self.score))

    @property
    def area(self) -> float:
        """Bounding-box area in square pixels."""
        x1, y1, x2, y2 = self.bbox
        return abs(x2 - x1) * abs(y2 - y1)


# --- raw backend output -----------------------------------------------------


@dataclass
class RawDetections:
    """Everything a pose backend saw in a clip: the keypoints-file payload."""

    source: str
    fps: float
    width: int
    height: int
    backend: str
    frames: list[list[Detection]]

    def __post_init__(self) -> None:
        if not self.fps > 0:
            raise ValueError(f"RawDetections.fps must be > 0, got {self.fps}")
        if not self.width > 0:
            raise ValueError(f"RawDetections.width must be > 0, got {self.width}")
        if not self.height > 0:
            raise ValueError(
                f"RawDetections.height must be > 0, got {self.height}"
            )

    @property
    def n(self) -> int:
        return len(self.frames)


# --- JSON I/O ---------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def save_keypoints(path: Path, raw: RawDetections) -> None:
    """Write ``raw`` as a version-1 keypoints file (see the plan's format)."""
    frames = [
        [
            {
                "kpts": [
                    [round(float(x), _XY_DP), round(float(y), _XY_DP)]
                    for x, y in det.kpts
                ],
                "conf": [round(float(c), _CONF_DP) for c in det.conf],
                "bbox": [round(float(v), _XY_DP) for v in det.bbox],
                "score": round(float(det.score), _CONF_DP),
            }
            for det in dets
        ]
        for dets in raw.frames
    ]
    _write_json(
        Path(path),
        {
            "version": KEYPOINTS_FORMAT_VERSION,
            "source": raw.source,
            "fps": round(float(raw.fps), _CONF_DP),
            "width": int(raw.width),
            "height": int(raw.height),
            "backend": raw.backend,
            "frames": frames,
        },
    )


def load_keypoints(path: Path) -> RawDetections:
    """Read a version-1 keypoints file. Raises ``ValueError`` on any other."""
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        d = json.load(fh)
    version = d.get("version")
    if version != KEYPOINTS_FORMAT_VERSION:
        raise ValueError(
            f"unsupported keypoints file version {version!r} in {path}, "
            f"expected {KEYPOINTS_FORMAT_VERSION}"
        )
    for key in ("source", "fps", "width", "height", "backend", "frames"):
        if key not in d:
            raise ValueError(f"keypoints file {path} is missing key {key!r}")
    frames = [
        [
            Detection(
                kpts=np.asarray(det["kpts"], dtype=np.float32),
                conf=np.asarray(det["conf"], dtype=np.float32),
                bbox=tuple(float(v) for v in det["bbox"]),
                score=float(det["score"]),
            )
            for det in dets
        ]
        for dets in d["frames"]
    ]
    return RawDetections(
        source=d["source"],
        fps=float(d["fps"]),
        width=_as_int(d["width"], "keypoints width"),
        height=_as_int(d["height"], "keypoints height"),
        backend=d["backend"],
        frames=frames,
    )
