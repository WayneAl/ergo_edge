"""Side-view joint angles from one tracked pose.

Angles are in degrees, measured in the image plane of a side-view camera.
Image coordinates are OpenCV's: x to the right, y **down**.

Conventions:
  * trunk, neck and upper arm are signed: + flexion (forward), - extension.
    The sign needs the direction the person faces, so these are ``Missing``
    when facing is unknown.
  * neck and upper arm are relative to the trunk, not to the image vertical.
  * elbow (``lower_arm``) and knee are unsigned flexion, 0 = straight.

A keypoint is *confident* when ``conf >= conf_min``. Anything that cannot be
measured is ``Missing(reason)``, never 0 or NaN. ``Measured.conf`` is the
minimum confidence of the keypoints whose coordinates enter the angle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import keypoints as K
from .pose import PoseFrame


class View(str, Enum):
    SIDE = "side"
    FRONT = "front"


@dataclass(frozen=True)
class Measured:
    deg: float
    conf: float  # min conf of the keypoints used


@dataclass(frozen=True)
class Missing:
    reason: str


Angle = Measured | Missing

LEFT, RIGHT = 0, 1


@dataclass(frozen=True)
class GeometryParams:
    conf_min: float = 0.3
    front_ratio_min: float = 0.55  # shoulder width / torso length at or above this = FRONT
    legs_level_frac: float = 0.10  # ankle height difference <= this * torso length = both feet down
    twist_frac: float = 0.20  # |shoulder width - hip width| / torso length above this = twisted
    neck_offset_deg: float = 0.0  # subtracted from neck flexion; calibrated in the pilot check

    def __post_init__(self) -> None:
        if not 0.0 < self.conf_min <= 1.0:
            raise ValueError(
                f"GeometryParams.conf_min must be in (0, 1], got {self.conf_min}"
            )
        if not self.front_ratio_min > 0:
            raise ValueError(
                f"GeometryParams.front_ratio_min must be > 0, got {self.front_ratio_min}"
            )
        if not self.legs_level_frac >= 0:
            raise ValueError(
                f"GeometryParams.legs_level_frac must be >= 0, got {self.legs_level_frac}"
            )
        if not self.twist_frac >= 0:
            raise ValueError(
                f"GeometryParams.twist_frac must be >= 0, got {self.twist_frac}"
            )
        if not math.isfinite(self.neck_offset_deg):
            raise ValueError(
                f"GeometryParams.neck_offset_deg must be finite, got {self.neck_offset_deg}"
            )


@dataclass(frozen=True)
class Angles:
    t: float
    view: View
    facing: int | None  # +1 facing +x (right), -1 facing left, None unknown
    trunk_flex: Angle  # + flexion, - extension
    trunk_twisted: bool | Missing
    neck_flex: Angle  # relative to trunk; + flexion, - extension
    upper_arm: tuple[Angle, Angle]  # (left, right); relative to trunk; + flexion, - extension
    lower_arm: tuple[Angle, Angle]  # elbow flexion, 0 = straight
    knee: tuple[Angle, Angle]  # knee flexion, 0 = straight
    legs_bilateral: bool | Missing


_FRONT_VIEW = "front view on a side-view station"
_MIN_SEGMENT_PX = 1.0  # torso and limb segments shorter than this are degenerate

# (shoulder, elbow, wrist, hip, knee, ankle), indexed by LEFT / RIGHT
_SIDES = (
    (K.L_SHOULDER, K.L_ELBOW, K.L_WRIST, K.L_HIP, K.L_KNEE, K.L_ANKLE),
    (K.R_SHOULDER, K.R_ELBOW, K.R_WRIST, K.R_HIP, K.R_KNEE, K.R_ANKLE),
)


@dataclass(frozen=True)
class _Trunk:
    shoulder_mid: np.ndarray  # float64 (2,)
    torso: np.ndarray  # shoulder_mid - hip_mid
    length: float  # L = |torso|, >= 1
    up: np.ndarray  # u = torso / L
    conf: float  # min conf of the confident shoulders and hips


def _points(pose: PoseFrame) -> np.ndarray:
    return pose.kpts.astype(np.float64)


def _confident(pose: PoseFrame, params: GeometryParams, i: int) -> bool:
    return bool(pose.conf[i] >= params.conf_min)


def _mean_confident(
    pose: PoseFrame, params: GeometryParams, idxs: tuple[int, ...]
) -> tuple[np.ndarray, float] | None:
    """Mean of the confident keypoints in ``idxs`` and their min conf; None if none."""
    used = [i for i in idxs if _confident(pose, params, i)]
    if not used:
        return None
    pts = _points(pose)[used]
    return pts.mean(axis=0), float(min(pose.conf[i] for i in used))


def _trunk(pose: PoseFrame, params: GeometryParams) -> _Trunk | Missing:
    sh = _mean_confident(pose, params, (K.L_SHOULDER, K.R_SHOULDER))
    if sh is None:
        return Missing("no confident shoulders")
    hp = _mean_confident(pose, params, (K.L_HIP, K.R_HIP))
    if hp is None:
        return Missing("no confident hips")
    torso = sh[0] - hp[0]
    length = float(math.hypot(torso[0], torso[1]))
    if length < _MIN_SEGMENT_PX:
        return Missing("degenerate torso")
    return _Trunk(sh[0], torso, length, torso / length, min(sh[1], hp[1]))


def _sign(v: float) -> int:
    return 1 if v > 0 else -1


def classify_view(pose: PoseFrame, params: GeometryParams = GeometryParams()) -> View:
    """FRONT when the shoulders look wide relative to the torso, else SIDE."""
    if _confident(pose, params, K.L_SHOULDER) and _confident(pose, params, K.R_SHOULDER):
        trunk = _trunk(pose, params)
        if isinstance(trunk, _Trunk):
            p = _points(pose)
            width = abs(p[K.L_SHOULDER, 0] - p[K.R_SHOULDER, 0])
            return View.FRONT if width / trunk.length >= params.front_ratio_min else View.SIDE
    return View.SIDE


def facing(pose: PoseFrame, params: GeometryParams = GeometryParams()) -> int | None:
    """+1 facing +x (right), -1 facing -x (left), None when the nose gives no cue."""
    if not _confident(pose, params, K.NOSE):
        return None
    nose_x = float(_points(pose)[K.NOSE, 0])
    ears = _mean_confident(pose, params, (K.L_EAR, K.R_EAR))
    if ears is not None and abs(nose_x - ears[0][0]) >= 2:
        return _sign(nose_x - ears[0][0])
    sh = _mean_confident(pose, params, (K.L_SHOULDER, K.R_SHOULDER))
    if sh is not None and abs(nose_x - sh[0][0]) >= 2:
        return _sign(nose_x - sh[0][0])
    return None


def _not_confident(
    pose: PoseFrame, params: GeometryParams, idxs: tuple[int, ...]
) -> Missing | None:
    for i in idxs:
        if not _confident(pose, params, i):
            return Missing(f"{K.NAMES[i]} not confident")
    return None


def _flexion(pose: PoseFrame, params: GeometryParams, a: int, b: int, c: int) -> Angle:
    """180 - interior angle at ``b`` between (a - b) and (c - b); 0 = straight."""
    missing = _not_confident(pose, params, (a, b, c))
    if missing is not None:
        return missing
    p = _points(pose)
    va, vc = p[a] - p[b], p[c] - p[b]
    na, nc = math.hypot(va[0], va[1]), math.hypot(vc[0], vc[1])
    if na < _MIN_SEGMENT_PX:
        return Missing(f"degenerate {K.NAMES[a]}-{K.NAMES[b]}")
    if nc < _MIN_SEGMENT_PX:
        return Missing(f"degenerate {K.NAMES[b]}-{K.NAMES[c]}")
    cos = float(np.clip(np.dot(va, vc) / (na * nc), -1.0, 1.0))
    interior = math.degrees(math.acos(cos))
    conf = float(min(pose.conf[a], pose.conf[b], pose.conf[c]))
    return Measured(180.0 - interior, conf)


def compute_angles(
    pose: PoseFrame, params: GeometryParams = GeometryParams(), view_ok: bool = True
) -> Angles:
    """All side-view angles for one pose. ``view_ok=False`` blocks every flexion angle."""
    p = _points(pose)
    view = classify_view(pose, params)
    s = facing(pose, params)
    trunk = _trunk(pose, params)

    # Computed regardless of view_ok.
    if all(_confident(pose, params, i) for i in (K.L_SHOULDER, K.R_SHOULDER, K.L_HIP, K.R_HIP)):
        if isinstance(trunk, Missing):
            trunk_twisted: bool | Missing = trunk
        else:
            sh_w = abs(p[K.L_SHOULDER, 0] - p[K.R_SHOULDER, 0])
            hip_w = abs(p[K.L_HIP, 0] - p[K.R_HIP, 0])
            trunk_twisted = bool(abs(sh_w - hip_w) / trunk.length > params.twist_frac)
    else:
        trunk_twisted = Missing("shoulders or hips not confident")

    if _confident(pose, params, K.L_ANKLE) and _confident(pose, params, K.R_ANKLE):
        if isinstance(trunk, Missing):
            legs_bilateral: bool | Missing = trunk
        else:
            dy = abs(p[K.L_ANKLE, 1] - p[K.R_ANKLE, 1])
            legs_bilateral = bool(dy <= params.legs_level_frac * trunk.length)
    else:
        legs_bilateral = Missing("ankle not confident")

    if not view_ok:
        m = Missing(_FRONT_VIEW)
        return Angles(
            pose.t, view, s, m, trunk_twisted, m, (m, m), (m, m), (m, m), legs_bilateral
        )

    # Signed, trunk-relative angles.
    if isinstance(trunk, Missing):
        trunk_flex: Angle = trunk
        neck_flex: Angle = trunk
        upper_arm: tuple[Angle, Angle] = (trunk, trunk)
    elif s is None:
        m = Missing("facing unknown")
        trunk_flex, neck_flex, upper_arm = m, m, (m, m)
    else:
        u = trunk.up
        f = np.array([-u[1] * s, u[0] * s])
        trunk_flex = Measured(
            math.degrees(math.atan2(s * trunk.torso[0], -trunk.torso[1])), trunk.conf
        )

        ears = _mean_confident(pose, params, (K.L_EAR, K.R_EAR))
        if ears is None:
            neck_flex = Missing("no confident ear")
        else:
            h = ears[0] - trunk.shoulder_mid
            if math.hypot(h[0], h[1]) < _MIN_SEGMENT_PX:
                neck_flex = Missing("degenerate neck")
            else:
                neck = math.degrees(math.atan2(np.dot(h, f), np.dot(h, u)))
                neck_flex = Measured(neck - params.neck_offset_deg, min(trunk.conf, ears[1]))

        arms: list[Angle] = []
        for side in (LEFT, RIGHT):
            sh_i, el_i = _SIDES[side][:2]
            missing = _not_confident(pose, params, (sh_i, el_i))
            if missing is not None:
                arms.append(missing)
                continue
            v = p[el_i] - p[sh_i]
            if math.hypot(v[0], v[1]) < _MIN_SEGMENT_PX:
                arms.append(Missing(f"degenerate {K.NAMES[sh_i]}-{K.NAMES[el_i]}"))
                continue
            conf = float(min(trunk.conf, pose.conf[sh_i], pose.conf[el_i]))
            arm = math.degrees(math.atan2(np.dot(v, f), np.dot(v, -u)))
            if arm <= -90:
                arm += 360  # range (-90, 270]: an overhead reach past 180 stays flexion
            arms.append(Measured(arm, conf))
        upper_arm = (arms[0], arms[1])

    # Unsigned angles: need neither trunk nor facing.
    lower_arm = (
        _flexion(pose, params, *_SIDES[LEFT][0:3]),
        _flexion(pose, params, *_SIDES[RIGHT][0:3]),
    )
    knee = (
        _flexion(pose, params, *_SIDES[LEFT][3:6]),
        _flexion(pose, params, *_SIDES[RIGHT][3:6]),
    )

    return Angles(
        t=pose.t,
        view=view,
        facing=s,
        trunk_flex=trunk_flex,
        trunk_twisted=trunk_twisted,
        neck_flex=neck_flex,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        knee=knee,
        legs_bilateral=legs_bilateral,
    )
