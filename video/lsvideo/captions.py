from __future__ import annotations

import math
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


def _length(items: Sequence[Aligned], text: str) -> int:
    return len(text[items[0].start : items[-1].end])


def _clause_end(a: Aligned) -> bool:
    return any(c in a.punct for c in CLAUSE_END)


def layout(aligned: Sequence[Aligned], text: str, max_chars: int = 42) -> list[list[list[Aligned]]]:
    """Cues -> lines -> words. A cue never crosses a sentence end; a cue has at most two lines; lines are balanced."""
    out: list[list[list[Aligned]]] = []
    sentence: list[Aligned] = []
    for i, a in enumerate(aligned):
        sentence.append(a)
        if any(c in a.punct for c in SENTENCE_END) or i == len(aligned) - 1:
            out.extend(_layout_sentence(sentence, text, max_chars))
            sentence = []
    return out


def _layout_sentence(items: list[Aligned], text: str, max_chars: int) -> list[list[list[Aligned]]]:
    n = min(math.ceil(_length(items, text) / (2 * max_chars)), len(items))
    while n < len(items):
        laid = [_lines(chunk, text, max_chars) for chunk in _chunks(items, n, text)]
        if all(lines is not None for lines in laid):
            return laid
        n += 1
    # n reached the item count: every item is its own one-line cue (an item longer than max_chars can never fit).
    return [[[a]] for a in items]


def _chunks(items: list[Aligned], n: int, text: str) -> list[list[Aligned]]:
    """Cut the sentence into n non-empty chunks, each cut nearest its k/n mark and pulled toward a clause end."""
    length = _length(items, text)
    cuts: list[int] = []
    prev = -1
    for k in range(1, n):
        target = items[0].start + k * length / n
        # Cut after item i: after the previous cut, leaving at least one item for each remaining chunk.
        candidates = range(prev + 1, len(items) - (n - k))
        prev = min(candidates, key=lambda i: (abs(items[i].end - target) - (12 if _clause_end(items[i]) else 0), i))
        cuts.append(prev)
    bounds = [-1, *cuts, len(items) - 1]
    return [items[bounds[m] + 1 : bounds[m + 1] + 1] for m in range(n)]


def _lines(chunk: list[Aligned], text: str, max_chars: int) -> list[list[Aligned]] | None:
    """One line if it fits, else the most balanced two-line split with both lines fitting; None if there is none."""
    if _length(chunk, text) <= max_chars:
        return [chunk]
    best: tuple[int, int] | None = None
    for j in range(len(chunk) - 1):
        first, second = _length(chunk[: j + 1], text), _length(chunk[j + 1 :], text)
        if first <= max_chars and second <= max_chars:
            score = max(first, second) - (6 if _clause_end(chunk[j]) else 0)
            if best is None or score < best[0]:
                best = (score, j)
    if best is None:
        return None
    return [chunk[: best[1] + 1], chunk[best[1] + 1 :]]


def cues(placed: Sequence[Placed], voices: Mapping[str, Voice], max_chars: int = 42,
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
        for cue in layout(aligned, shot.narration, max_chars):
            lines = "\n".join(shot.narration[line[0].start : line[-1].end] for line in cue)
            raw.append(Cue(offset + cue[0][0].word.start, offset + cue[-1][-1].word.end, lines))
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
