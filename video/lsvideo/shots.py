from __future__ import annotations

import re
import tomllib
from pathlib import Path

from .model import Clip, Inset, Script, Shot, Still, VideoError

ID_RE = re.compile(r"^[a-z0-9_]+$")
TOP_KEYS = {"voice", "max_total_s", "shot"}
SHOT_KEYS = {"id", "slide", "clip", "in_s", "out_s", "crop", "pip", "pip_in_s", "pip_out_s", "pip_crop",
             "hold_s", "lead_s", "narration"}
CLIP_ONLY_KEYS = ("in_s", "out_s", "crop", "pip_in_s", "pip_out_s", "pip_crop")
PIP_KEYS = ("pip_in_s", "pip_out_s", "pip_crop")


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load(path: Path, root: Path) -> Script:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise VideoError(f"{path}: cannot read the shot list: {e}") from e
    unknown = set(data) - TOP_KEYS
    if unknown:
        raise VideoError(f"{path}: unknown top-level keys {sorted(unknown)}")
    voice = data.get("voice")
    if not isinstance(voice, str) or not voice:
        raise VideoError(f'{path}: set voice to an Edge TTS voice name, e.g. voice = "en-US-AndrewNeural"')
    max_total_s = data.get("max_total_s", 180.0)
    if not _is_num(max_total_s) or max_total_s <= 0:
        raise VideoError(f"{path}: max_total_s must be a number > 0")
    raw_shots = data.get("shot")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise VideoError(f"{path}: add at least one [[shot]] table")
    shots: list[Shot] = []
    seen: set[str] = set()
    for n, raw in enumerate(raw_shots, start=1):
        shot = _shot(n, raw, root)
        if shot.id in seen:
            raise VideoError(f"shot {shot.id}: duplicate id; every shot id must be unique")
        seen.add(shot.id)
        shots.append(shot)
    return Script(voice, float(max_total_s), tuple(shots))


def _shot(n: int, raw: object, root: Path) -> Shot:
    if not isinstance(raw, dict):
        raise VideoError(f"shot #{n}: each shot must be a [[shot]] table")
    sid = raw.get("id")
    if not isinstance(sid, str) or not ID_RE.fullmatch(sid):
        raise VideoError(f"shot #{n}: id must be a string matching ^[a-z0-9_]+$ (got {sid!r})")
    unknown = set(raw) - SHOT_KEYS
    if unknown:
        raise VideoError(f"shot {sid}: unknown keys {sorted(unknown)}")
    if ("slide" in raw) == ("clip" in raw):
        raise VideoError(f"shot {sid}: set exactly one of slide or clip")
    lead_s = raw.get("lead_s", 0.5)
    if not _is_num(lead_s) or lead_s < 0:
        raise VideoError(f"shot {sid}: lead_s must be a number >= 0")
    narration = raw.get("narration", "")
    if not isinstance(narration, str):
        raise VideoError(f"shot {sid}: narration must be a string")
    narration = " ".join(narration.split())

    if "slide" in raw:
        page = raw["slide"]
        if not _is_int(page) or page < 1:
            raise VideoError(f"shot {sid}: slide must be a 1-based page number (got {page!r})")
        if "pip" in raw:
            raise VideoError(f"shot {sid}: pip needs a clip")
        stray = [k for k in CLIP_ONLY_KEYS if k in raw]
        if stray:
            raise VideoError(f"shot {sid}: {stray} apply to clip shots only")
        hold_s = raw.get("hold_s", 1.0)
        if not _is_num(hold_s) or hold_s < 0:
            raise VideoError(f"shot {sid}: hold_s must be a number >= 0")
        still = Still(page=page, png=root / "video/build/slides" / f"{page:02d}.png")
        return Shot(sid, still, narration, float(lead_s), float(hold_s))

    if "hold_s" in raw:
        raise VideoError(f"shot {sid}: hold_s applies to slide shots only")
    main = _clip(sid, raw, root, "clip", "")
    if "pip" not in raw:
        stray = [k for k in PIP_KEYS if k in raw]
        if stray:
            raise VideoError(f"shot {sid}: {stray} need pip")
        return Shot(sid, main, narration, float(lead_s), 0.0)
    pip = _clip(sid, raw, root, "pip", "pip_")
    return Shot(sid, Inset(main, pip), narration, float(lead_s), 0.0)


def _clip(sid: str, raw: dict, root: Path, key: str, prefix: str) -> Clip:
    rel = raw[key]
    if not isinstance(rel, str) or not rel:
        raise VideoError(f"shot {sid}: {key} must be a path relative to the repo root")
    in_s, out_s = raw.get(f"{prefix}in_s"), raw.get(f"{prefix}out_s")
    if not (_is_num(in_s) and _is_num(out_s) and 0 <= in_s < out_s):
        raise VideoError(f"shot {sid}: need 0 <= {prefix}in_s < {prefix}out_s "
                         f"(got {prefix}in_s={in_s!r}, {prefix}out_s={out_s!r})")
    crop = raw.get(f"{prefix}crop")
    if crop is not None:
        if not (isinstance(crop, list) and len(crop) == 4 and all(_is_int(c) for c in crop)
                and crop[0] >= 0 and crop[1] >= 0 and crop[2] > 0 and crop[3] > 0):
            raise VideoError(f"shot {sid}: {prefix}crop must be [x, y, w, h] integers "
                             f"with x, y >= 0 and w, h > 0 (got {crop!r})")
        crop = tuple(crop)
    return Clip(root / rel, float(in_s), float(out_s), crop)
