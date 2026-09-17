from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path

FPS = 30
W, H = 1920, 1080

class VideoError(Exception):
    """A build input or constraint is wrong; the message names the shot and the fix."""

def frame_ceil(seconds: float) -> float:
    return math.ceil(seconds * FPS - 1e-9) / FPS

@dataclass(frozen=True)
class WordTiming:
    text: str
    start: float   # seconds from the start of this shot's narration audio
    end: float

@dataclass(frozen=True)
class Still:
    page: int      # 1-based page of deck/dist/proposal.pdf
    png: Path      # <root>/video/build/slides/<page:02d>.png

@dataclass(frozen=True)
class Clip:
    path: Path
    in_s: float
    out_s: float
    crop: tuple[int, int, int, int] | None = None   # x, y, w, h in source pixels

    @property
    def length(self) -> float:
        return self.out_s - self.in_s

@dataclass(frozen=True)
class Inset:
    main: Clip
    pip: Clip

Visual = Still | Clip | Inset

@dataclass(frozen=True)
class Shot:
    id: str
    visual: Visual
    narration: str      # whitespace-collapsed; "" means a silent shot
    lead_s: float
    hold_s: float       # stills only; 0.0 for clips

@dataclass(frozen=True)
class Script:
    voice: str
    max_total_s: float
    shots: tuple[Shot, ...]

@dataclass(frozen=True)
class Voice:
    shot_id: str
    mp3: Path
    words: tuple[WordTiming, ...]
    dur_s: float

@dataclass(frozen=True)
class Placed:
    shot: Shot
    t0: float
    dur: float

@dataclass(frozen=True)
class Cue:
    t0: float
    t1: float
    text: str     # display lines joined by "\n"
