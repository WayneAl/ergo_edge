from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from .model import FPS, Clip, Inset, Placed, Shot, Still, VideoError, Voice, frame_ceil


def place(shots: Sequence[Shot], voices: Mapping[str, Voice], max_total_s: float) -> list[Placed]:
    placed: list[Placed] = []
    frames = 0   # accumulate whole frames so t0 and the total carry no float drift
    for shot in shots:
        if shot.narration:
            if shot.id not in voices:
                raise VideoError(f"shot {shot.id}: no voice synthesized")
            v = voices[shot.id].dur_s
        else:
            v = 0.0
        visual = shot.visual
        if isinstance(visual, Still):
            raw = shot.lead_s + v + shot.hold_s
        else:
            if isinstance(visual, Clip):
                raw = visual.length
            elif isinstance(visual, Inset):
                raw = visual.main.length
                if visual.pip.length < visual.main.length:
                    raise VideoError(f"shot {shot.id}: pip clip {visual.pip.length:.2f}s is shorter than "
                                     f"main clip {visual.main.length:.2f}s")
            else:
                raise VideoError(f"shot {shot.id}: unknown visual {visual!r}")
            if shot.lead_s + v > raw:
                raise VideoError(f"shot {shot.id}: narration {v:.2f}s starting at {shot.lead_s:.2f}s runs past "
                                 f"the clip end at {raw:.2f}s — lengthen the clip or shorten the narration")
        dur = frame_ceil(raw)
        placed.append(Placed(shot, frames / FPS, dur))
        frames += round(dur * FPS)
    total = frames / FPS
    if total > max_total_s:
        raise VideoError(f"total {total:.2f}s exceeds {max_total_s:.0f}s: "
                         + ", ".join(f"{p.shot.id}={p.dur:.2f}s" for p in placed))
    return placed


def check_sources(shots: Sequence[Shot], lengths: Mapping[Path, float]) -> None:
    """Fail before rendering when a shot's out point lies past the end of its footage (ffmpeg would cut it short)."""
    for shot in shots:
        visual = shot.visual
        if isinstance(visual, Clip):
            clips = [("clip", visual)]
        elif isinstance(visual, Inset):
            clips = [("clip", visual.main), ("pip", visual.pip)]
        else:
            continue
        for role, clip in clips:
            if clip.path not in lengths:
                continue
            length = lengths[clip.path]
            if clip.out_s > length + 1 / FPS:
                raise VideoError(f"shot {shot.id}: {role} footage {clip.path.name} is {length:.2f}s "
                                 f"but out_s is {clip.out_s:.2f}s")
