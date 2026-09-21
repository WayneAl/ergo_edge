"""COCO-17 keypoint layout — the single contract shared by every pose backend.

Index order is the standard COCO person keypoint order emitted by ultralytics
yolov8-pose and by the Hailo pose models. Nothing downstream may remap it.

Image coordinates are OpenCV's: x to the right, y **down**.
"""

from __future__ import annotations

# --- COCO-17 indices -------------------------------------------------------
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16

N_KPTS = 17

NAMES: list[str] = [
    "nose",
    "l_eye",
    "r_eye",
    "l_ear",
    "r_ear",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]

#: Index permutation that swaps every left/right pair; NOSE maps to itself.
#: Unused in LineSafe v1: no mirroring is performed, so left and right always
#: mean the person's own sides as the pose model reports them.
LR_SWAP: list[int] = [
    NOSE,
    R_EYE,
    L_EYE,
    R_EAR,
    L_EAR,
    R_SHOULDER,
    L_SHOULDER,
    R_ELBOW,
    L_ELBOW,
    R_WRIST,
    L_WRIST,
    R_HIP,
    L_HIP,
    R_KNEE,
    L_KNEE,
    R_ANKLE,
    L_ANKLE,
]

#: Drawing edges (19) used by the overlay: the standard COCO person skeleton,
#: with no edges added or removed.
SKELETON: list[tuple[int, int]] = [
    (NOSE, L_EYE),
    (NOSE, R_EYE),
    (L_EYE, L_EAR),
    (R_EYE, R_EAR),
    (L_EAR, L_SHOULDER),
    (R_EAR, R_SHOULDER),
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_ELBOW),
    (R_SHOULDER, R_ELBOW),
    (L_ELBOW, L_WRIST),
    (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (R_HIP, R_KNEE),
    (L_KNEE, L_ANKLE),
    (R_KNEE, R_ANKLE),
    (L_EYE, R_EYE),
]

# --- Lead / trail aliases ---------------------------------------------------
# Unused in LineSafe v1. These are fixed aliases (lead = LEFT, trail = RIGHT)
# carrying no handedness guarantee: no mirroring is performed, so nothing makes
# the "lead" side the one a person actually leads with.
LEAD_SHOULDER, LEAD_ELBOW, LEAD_WRIST, LEAD_HIP, LEAD_KNEE, LEAD_ANKLE = (
    L_SHOULDER,
    L_ELBOW,
    L_WRIST,
    L_HIP,
    L_KNEE,
    L_ANKLE,
)
TRAIL_SHOULDER, TRAIL_ELBOW, TRAIL_WRIST, TRAIL_HIP, TRAIL_KNEE, TRAIL_ANKLE = (
    R_SHOULDER,
    R_ELBOW,
    R_WRIST,
    R_HIP,
    R_KNEE,
    R_ANKLE,
)
