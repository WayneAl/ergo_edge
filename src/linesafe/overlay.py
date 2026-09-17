"""OpenCV overlay for one :class:`~linesafe.pipeline.FrameResult`.

Skeleton, angle labels, a score panel, a REBA ruler and the loud states (event border,
partial score, wrong camera view, no person, low FPS). Headless: drawing calls only,
never a window. Pixels are image coordinates, x right and y down.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import keypoints as K
from .config import StationConfig
from .geometry import LEFT, RIGHT, Angle, Measured
from .pipeline import FrameResult
from .reba import Band, reba_band

BAND_BGR: dict[Band, tuple[int, int, int]] = {
    Band.NEGLIGIBLE: (87, 139, 46),
    Band.LOW: (87, 139, 46),
    Band.MEDIUM: (13, 147, 217),
    Band.HIGH: (47, 105, 228),
    Band.VERY_HIGH: (27, 55, 200),
}

CONF_MIN = 0.3  # skeleton segments need both ends at or above this
FPS_WARN = 15.0  # 0 < fps < this is drawn red
BORDER_PX = 12
WRONG_VIEW_TEXT = "WRONG CAMERA VIEW - station expects a side view"
NO_PERSON_TEXT = "no person in view"

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)
_GREY = (128, 128, 128)
_DARK = (20, 20, 20)
_LIMB = (235, 235, 235)
_JOINT = (0, 200, 255)

# (shoulder, elbow, knee) per LEFT / RIGHT
_SIDE_KPTS = ((K.L_SHOULDER, K.L_ELBOW, K.L_KNEE), (K.R_SHOULDER, K.R_ELBOW, K.R_KNEE))
_SIDE_INDEX = {"left": LEFT, "right": RIGHT}
_MAX_PX = 100_000  # keeps wild coordinates inside cv2's int range


def _pt(xy) -> tuple[int, int]:
    return (
        int(round(min(max(float(xy[0]), -_MAX_PX), _MAX_PX))),
        int(round(min(max(float(xy[1]), -_MAX_PX), _MAX_PX))),
    )


def _text(img, text, org, scale, color, thick, outline=True) -> tuple[int, int]:
    """Draw ``text`` with its baseline-left at ``org``; returns its (width, height)."""
    if outline:
        cv2.putText(img, text, _pt(org), _FONT, scale, _BLACK, thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, _pt(org), _FONT, scale, color, thick, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    return tw, th


def _text_deg(img, deg, org, scale, color, thick) -> None:
    """An integer angle with '°' on a shaded box (OpenCV 5 fonts render '°')."""
    text = f"{int(round(deg))}°"
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    x, y = _pt(org)
    pad = max(2, th // 4)
    _shade(img, x - pad, y - th - pad, x + tw + pad, y + base + pad, keep=0.25)
    _text(img, text, (x, y), scale, color, thick)


def _shade(img, x1, y1, x2, y2, keep=0.4) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        roi[:] = (roi.astype(np.float32) * keep).astype(np.uint8)


def _fit(text, scale, thick, max_w) -> float:
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thick)
    return scale * max_w / tw if tw > max_w > 0 else scale


def _pick(pair: tuple[Angle, Angle], preferred: int | None) -> int | None:
    """Side to label: ``preferred`` when measured, else the measured side with the larger angle."""
    if preferred is not None and isinstance(pair[preferred], Measured):
        return preferred
    measured = [s for s in (LEFT, RIGHT) if isinstance(pair[s], Measured)]
    return max(measured, key=lambda s: pair[s].deg) if measured else None


def _skeleton(img, result: FrameResult, s: float) -> None:
    pose = result.pose
    ok = pose.conf >= CONF_MIN
    lw, r = max(1, round(3 * s)), max(2, round(5 * s))
    for a, b in K.SKELETON:
        if ok[a] and ok[b]:
            cv2.line(img, _pt(pose.kpts[a]), _pt(pose.kpts[b]), _LIMB, lw, cv2.LINE_AA)
    for i in range(K.N_KPTS):
        if ok[i]:
            cv2.circle(img, _pt(pose.kpts[i]), r, _JOINT, -1, cv2.LINE_AA)


def _angle_labels(img, result: FrameResult, s: float) -> None:
    pose, a = result.pose, result.angles
    ok = pose.conf >= CONF_MIN
    scale, thick = 0.7 * s, max(1, round(2 * s))
    dx, dy = 12 * s, -12 * s

    def label(angle: Angle, idxs: tuple[int, ...]) -> None:
        use = [i for i in idxs if ok[i]]
        if isinstance(angle, Measured) and use:
            x, y = pose.kpts[use].astype(np.float64).mean(axis=0)
            _text_deg(img, angle.deg, (x + dx, y + dy), scale, _WHITE, thick)

    label(a.trunk_flex, (K.L_HIP, K.R_HIP))
    arm_pref = _SIDE_INDEX.get(result.reba.side) if result.reba is not None else None
    arm = _pick(a.upper_arm, arm_pref)
    if arm is not None:
        label(a.upper_arm[arm], (_SIDE_KPTS[arm][0],))
    elbow = _pick(a.lower_arm, arm if arm is not None else arm_pref)
    if elbow is not None:
        label(a.lower_arm[elbow], (_SIDE_KPTS[elbow][1],))
    knee = _pick(a.knee, None)
    if knee is not None:
        label(a.knee[knee], (_SIDE_KPTS[knee][2],))


def _panel(img, result: FrameResult, cfg: StationConfig, s: float) -> None:
    reba, rula = result.reba, result.rula
    x0, y0, pad = round(24 * s), round(24 * s), round(10 * s)
    scale, thick = 0.75 * s, max(1, round(2 * s))
    small = 0.6 * s
    line_h = round(36 * s)

    # (text, scale, colour, chip colour or None)
    lines: list[tuple[str, float, tuple[int, int, int], tuple[int, int, int] | None]] = [
        (f"Station {cfg.station_id}", scale, _WHITE, None)
    ]
    if reba is not None:
        lines.append((f"REBA {reba.total} {reba.band.value}", scale, None, BAND_BGR[reba.band]))
    else:
        lines.append(("REBA --", scale, _WHITE, _GREY))
    lines.append((f"RULA {rula.total} {rula.level.value}" if rula is not None else "RULA --", scale, _WHITE, None))
    if reba is not None:
        lines.extend((d, small, _WHITE, None) for d in reba.drivers[:2])
        if reba.partial:
            lines.append((f"PARTIAL: {', '.join(reba.missing)}", small, _YELLOW, None))
    elif result.angles is not None and not isinstance(result.angles.trunk_flex, Measured):
        lines.append((f"no score: {result.angles.trunk_flex.reason}", small, _YELLOW, None))

    width = max(cv2.getTextSize(t, _FONT, sc, thick)[0][0] for t, sc, _, _ in lines) + 2 * pad
    height = line_h * len(lines) + pad
    _shade(img, x0 - pad, y0 - pad, x0 + width, y0 + height)

    y = y0
    for text, sc, colour, chip in lines:
        base = y + line_h - round(10 * s)
        if chip is not None:
            (tw, th), _ = cv2.getTextSize(text, _FONT, sc, thick)
            cv2.rectangle(img, (x0, y + round(2 * s)), (x0 + tw + 2 * pad, y + line_h - round(2 * s)), chip, -1)
            if colour is None:
                colour = _DARK if chip == BAND_BGR[Band.MEDIUM] else _WHITE
            _text(img, text, (x0 + pad, base), sc, colour, thick, outline=False)
        else:
            _text(img, text, (x0, base), sc, colour, thick)
        y += line_h


def _ruler(img, result: FrameResult, s: float) -> None:
    h, w = img.shape[:2]
    margin = round(24 * s)
    cell_h = round(34 * s)
    y1, y2 = h - margin - cell_h, h - margin
    cell_w = (w - 2 * margin) / 15.0
    scale, thick = 0.55 * s, max(1, round(1.5 * s))
    for total in range(1, 16):
        x1, x2 = round(margin + (total - 1) * cell_w), round(margin + total * cell_w)
        band = reba_band(total)
        cv2.rectangle(img, (x1, y1), (x2, y2), BAND_BGR[band], -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), _BLACK, 1)
        text = str(total)
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
        colour = _DARK if band is Band.MEDIUM else _WHITE
        _text(img, text, ((x1 + x2 - tw) / 2, (y1 + y2 + th) / 2), scale, colour, thick, outline=False)
    if result.reba is not None:
        t = result.reba.total
        x1, x2 = round(margin + (t - 1) * cell_w), round(margin + t * cell_w)
        o = round(4 * s)
        cv2.rectangle(img, (x1 - o, y1 - o), (x2 + o, y2 + o), _BLACK, max(1, round(6 * s)))
        cv2.rectangle(img, (x1 - o, y1 - o), (x2 + o, y2 + o), _WHITE, max(1, round(3 * s)))


def _centre_box(img, text, colour, fill, s: float, y_mid: int) -> None:
    h, w = img.shape[:2]
    thick = max(1, round(2 * s))
    scale = _fit(text, 1.0 * s, thick, w - round(80 * s))
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    half = round(32 * s)
    if fill is None:
        _shade(img, (w - tw) // 2 - round(16 * s), y_mid - half, (w + tw) // 2 + round(16 * s), y_mid + half)
    else:
        cv2.rectangle(img, (0, y_mid - half), (w - 1, y_mid + half), fill, -1)
    _text(img, text, ((w - tw) / 2, y_mid + th / 2), scale, colour, thick, outline=fill is None)


def _fps(img, fps: float, s: float) -> None:
    w = img.shape[1]
    text = f"FPS {fps:.1f}"
    scale, thick = 0.75 * s, max(1, round(2 * s))
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    margin, pad = round(24 * s), round(8 * s)
    x = w - margin - tw
    _shade(img, x - pad, margin - pad, w - margin + pad, margin + th + 2 * pad)
    colour = _RED if 0 < fps < FPS_WARN else _WHITE
    _text(img, text, (x, margin + th + pad // 2), scale, colour, thick)


def draw(frame_bgr: np.ndarray, result: FrameResult, cfg: StationConfig, fps: float) -> np.ndarray:
    """Draw ``result`` onto ``frame_bgr`` in place and return the same array."""
    if not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3 or (
        frame_bgr.shape[2] != 3
    ):
        raise ValueError("frame_bgr must be a uint8 array of shape (height, width, 3)")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps):
        raise ValueError(f"fps must be a finite number, got {fps!r}")
    img = frame_bgr
    h, w = img.shape[:2]
    s = h / 720.0

    if result.pose is not None:
        _skeleton(img, result, s)
        if result.angles is not None:
            _angle_labels(img, result, s)
    _panel(img, result, cfg, s)
    if result.wrong_view:
        _centre_box(img, WRONG_VIEW_TEXT, _WHITE, _RED, s, h // 2)
    if result.pose is None:
        _centre_box(img, NO_PERSON_TEXT, _WHITE, None, s, h // 2)
    _ruler(img, result, s)
    _fps(img, fps, s)
    if result.event_active:
        red = np.array(_RED, np.uint8)
        b = BORDER_PX
        img[:b, :] = red
        img[-b:, :] = red
        img[:, :b] = red
        img[:, -b:] = red
    return img
