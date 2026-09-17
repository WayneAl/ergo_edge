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
#: Used when mirroring a left-handed golfer so that all downstream code can
#: assume a right-handed swing.
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

#: Drawing edges (19) used by the renderer. These are the COCO edges with one
#: golf-specific substitution: COCO's (L_EYE, R_EYE) edge is dropped — it adds
#: nothing at swing scale — and (L_WRIST, R_WRIST) takes its place, because both
#: hands hold the club, so that segment is the closest thing to a shaft line the
#: pose model gives us.
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
    (L_WRIST, R_WRIST),
]

# --- Lead / trail aliases ---------------------------------------------------
# After build_track the golfer is always right-handed (left-handers are mirrored
# and L/R swapped), so the lead side (closer to the target) is always LEFT and
# the trail side is always RIGHT.
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
