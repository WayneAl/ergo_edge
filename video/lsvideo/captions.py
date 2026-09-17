from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from .model import Cue, Placed, VideoError, Voice, WordTiming

Aligned = NamedTuple("Aligned", [("word", WordTiming), ("start", int), ("end", int), ("punct", str)])

TRAILING = ".,;:!?…\"'’”)"
PUNCT = ".,;:!?…"
SENTENCE_END = ".!?…"
CLAUSE_END = ",;:"

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Default,{font},{fontsize},&H00FFFFFF,&H60000000,&H60000000,0,3,10,0,2,160,160,40

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def align(words: Sequence[WordTiming], text: str) -> list[Aligned]:
    """Locate each Edge token in the script so captions show the script's own characters."""
    out: list[Aligned] = []
    cursor = 0
    for word in words:
        if not word.text:
            continue
        pattern = re.escape(word.text)
        if re.match(r"\w", word.text[0]):
            pattern = r"(?<!\w)" + pattern
        if re.match(r"\w", word.text[-1]):
            pattern = pattern + r"(?!\w)"
        m = re.compile(pattern).search(text, cursor)
        if m is None:
            continue
        end = m.end()
        while end < len(text) and text[end] in TRAILING:
            end += 1
        punct = "".join(c for c in text[m.end():end] if c in PUNCT)
        out.append(Aligned(word, m.start(), end, punct))
        cursor = end
    return out


def group_lines(aligned: Sequence[Aligned], text: str, max_chars: int = 42) -> list[list[Aligned]]:
    rows: list[list[Aligned]] = []
    buf: list[Aligned] = []
    for a in aligned:
        if buf and len(text[buf[0].start : a.end]) > max_chars:
            rows.append(buf)
            buf = []
        buf.append(a)
        if any(c in a.punct for c in SENTENCE_END):
            rows.append(buf)
            buf = []
        elif any(c in a.punct for c in CLAUSE_END) and len(text[buf[0].start : a.end]) >= int(max_chars * 0.7):
            rows.append(buf)
            buf = []
    if buf:
        rows.append(buf)
    return rows


def cues(placed: Sequence[Placed], voices: Mapping[str, Voice], max_chars: int = 42, lines_per_cue: int = 2,
         bridge_s: float = 0.5) -> list[Cue]:
    raw: list[Cue] = []
    for p in placed:
        shot = p.shot
        if not shot.narration:
            continue
        if shot.id not in voices:
            raise VideoError(f"shot {shot.id}: no voice synthesized")
        aligned = align(voices[shot.id].words, shot.narration)
        if not aligned:
            raise VideoError(f"shot {shot.id}: no narration word timing matched the script; re-synthesize the voice")
        offset = p.t0 + shot.lead_s
        rows = group_lines(aligned, shot.narration, max_chars)
        for i in range(0, len(rows), lines_per_cue):
            block = rows[i : i + lines_per_cue]
            lines = "\n".join(shot.narration[r[0].start : r[-1].end] for r in block)
            raw.append(Cue(offset + block[0][0].word.start, offset + block[-1][-1].word.end, lines))
    raw.sort(key=lambda c: c.t0)
    out: list[Cue] = []
    for i, c in enumerate(raw):
        t1 = c.t1
        if i + 1 < len(raw):
            nxt = raw[i + 1].t0
            # Overlap (nxt < t1) or a short pause (0 <= nxt - t1 < bridge_s): hold this cue until the next one.
            if nxt < t1 or 0 <= nxt - t1 < bridge_s:
                t1 = nxt
        out.append(Cue(c.t0, t1, c.text))
    return out


def to_ass(cues: Sequence[Cue], font: str = "IBM Plex Sans", fontsize: int = 54) -> str:
    lines = [ASS_HEADER.format(font=font, fontsize=fontsize).rstrip("\n")]
    for c in cues:
        text = c.text.replace("{", "(").replace("}", ")").replace("\n", "\\N")
        lines.append(f"Dialogue: 0,{format_ts_ass(c.t0)},{format_ts_ass(c.t1)},Default,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"


def format_ts_ass(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
