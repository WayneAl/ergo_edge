"""Synthetic people and sessions for pipeline tests (a helper module, not a test module).

``side_pose`` builds a COCO-17 side-view person from joint angles so that
``geometry.compute_angles`` returns those angles; ``front_pose`` builds a person facing
the camera; ``session`` strings poses into a ``RawDetections`` clip.

Units: degrees, seconds, pixels with x right and y down.
"""

from __future__ import annotations

import math

import numpy as np

from linesafe import keypoints as K
from linesafe.detections import Detection, RawDetections

CONF = 0.9
SCORE = 0.9
PAD_PX = 20.0
WIDTH, HEIGHT = 1280, 720
LR_PX = 2.0  # left side at x - 2, right side at x + 2: overlapped, still a side view

_SIDE_DEFAULTS = {
    "arm_deg": 0.0,
    "elbow_deg": 0.0,
    "knee_deg": 0.0,
    "facing": 1,
    "x0": 640.0,
    "y_hip": 500.0,
    "torso": 200.0,
}


def _pair(kpts: np.ndarray, left: int, right: int, xy: np.ndarray, dx: float) -> None:
    kpts[left] = (xy[0] - dx, xy[1])
    kpts[right] = (xy[0] + dx, xy[1])


def side_pose(
    trunk_deg: float,
    arm_deg: float = 0.0,
    elbow_deg: float = 0.0,
    knee_deg: float = 0.0,
    facing: int = 1,
    x0: float = 640.0,
    y_hip: float = 500.0,
    torso: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A side-view person whose trunk, upper arm, elbow and knee angles are the given ones.

    Hip mid at ``(x0, y_hip)``; neck flexion 0; both arms and both legs identical.
    Returns ``(kpts (17, 2) float32, conf (17,) float32)``.
    """
    if facing not in (1, -1):
        raise ValueError(f"facing must be 1 or -1, got {facing!r}")
    s = float(facing)
    tr = math.radians(trunk_deg)
    up = np.array([s * math.sin(tr), -math.cos(tr)])  # hip -> shoulder, unit
    fwd = np.array([s * math.cos(tr), math.sin(tr)])  # perpendicular to the trunk, towards the face

    hip = np.array([x0, y_hip])
    shoulder = hip + torso * up
    ear = shoulder + 0.25 * torso * up  # on the trunk line: neck flexion 0
    nose = ear + np.array([30.0 * s, 0.0])
    eye = ear + np.array([18.0 * s, -6.0])

    a = math.radians(arm_deg)
    e = math.radians(arm_deg + elbow_deg)
    elbow = shoulder + 0.55 * torso * (math.cos(a) * -up + math.sin(a) * fwd)
    wrist = elbow + 0.5 * torso * (math.cos(e) * -up + math.sin(e) * fwd)

    half = math.radians(knee_deg) / 2.0
    knee = hip + 0.9 * torso * np.array([s * math.sin(half), math.cos(half)])
    ankle = knee + 0.9 * torso * np.array([-s * math.sin(half), math.cos(half)])

    kpts = np.zeros((K.N_KPTS, 2), np.float64)
    kpts[K.NOSE] = nose
    for left, right, xy in (
        (K.L_EYE, K.R_EYE, eye),
        (K.L_EAR, K.R_EAR, ear),
        (K.L_SHOULDER, K.R_SHOULDER, shoulder),
        (K.L_ELBOW, K.R_ELBOW, elbow),
        (K.L_WRIST, K.R_WRIST, wrist),
        (K.L_HIP, K.R_HIP, hip),
        (K.L_KNEE, K.R_KNEE, knee),
        (K.L_ANKLE, K.R_ANKLE, ankle),
    ):
        _pair(kpts, left, right, xy, LR_PX)
    return kpts.astype(np.float32), np.full(K.N_KPTS, CONF, np.float32)


def front_pose(
    x0: float = 640.0, y_hip: float = 500.0, torso: float = 200.0
) -> tuple[np.ndarray, np.ndarray]:
    """A person facing the camera: ``classify_view`` is FRONT and ``facing`` is None."""
    y_sh = y_hip - torso
    y_ear = y_sh - 0.25 * torso
    kpts = np.zeros((K.N_KPTS, 2), np.float64)
    kpts[K.NOSE] = (x0, y_ear + 5.0)
    # The person's left is on the image right.
    for left, right, dx, y in (
        (K.L_EYE, K.R_EYE, 12.0, y_ear - 3.0),
        (K.L_EAR, K.R_EAR, 25.0, y_ear),
        (K.L_SHOULDER, K.R_SHOULDER, 70.0, y_sh),
        (K.L_ELBOW, K.R_ELBOW, 80.0, y_sh + 0.55 * torso),
        (K.L_WRIST, K.R_WRIST, 80.0, y_sh + 1.05 * torso),
        (K.L_HIP, K.R_HIP, 40.0, y_hip),
        (K.L_KNEE, K.R_KNEE, 40.0, y_hip + 0.9 * torso),
        (K.L_ANKLE, K.R_ANKLE, 40.0, y_hip + 1.8 * torso),
    ):
        kpts[left] = (x0 + dx, y)
        kpts[right] = (x0 - dx, y)
    return kpts.astype(np.float32), np.full(K.N_KPTS, CONF, np.float32)


def _detection(kpts: np.ndarray, conf: np.ndarray) -> Detection:
    x, y = kpts[:, 0], kpts[:, 1]
    bbox = (
        float(x.min()) - PAD_PX,
        float(y.min()) - PAD_PX,
        float(x.max()) + PAD_PX,
        float(y.max()) + PAD_PX,
    )
    return Detection(kpts=kpts, conf=conf, bbox=bbox, score=SCORE)


def session(
    segments: list[tuple[float, dict]], fps: float = 30.0, ramp_s: float = 0.5
) -> RawDetections:
    """One detection per frame for each ``(seconds, kwargs)`` segment.

    ``kwargs`` go to ``side_pose``, or ``{"front": True}`` (plus optional ``front_pose``
    kwargs) selects ``front_pose``. Entering a side segment right after another side
    segment, every numeric kwarg moves linearly from the last emitted pose to the new
    one over the first ``ramp_s`` seconds (``facing`` switches at once).
    """
    frames: list[list[Detection]] = []
    prev: dict | None = None  # kwargs of the last side pose emitted
    for seconds, kwargs in segments:
        n = int(round(seconds * fps))
        if kwargs.get("front"):
            front_kwargs = {k: v for k, v in kwargs.items() if k != "front"}
            frames.extend([_detection(*front_pose(**front_kwargs))] for _ in range(n))
            prev = None
            continue
        target = {**_SIDE_DEFAULTS, **kwargs}
        start = prev
        for i in range(n):
            w = 1.0 if start is None or ramp_s <= 0 else min(1.0, (i / fps) / ramp_s)
            cur = {
                k: v if k == "facing" else start[k] + w * (v - start[k])
                for k, v in target.items()
            } if w < 1.0 else target
            frames.append([_detection(*side_pose(**cur))])
            prev = cur
    return RawDetections(
        source="synth", fps=fps, width=WIDTH, height=HEIGHT, backend="synth", frames=frames
    )
