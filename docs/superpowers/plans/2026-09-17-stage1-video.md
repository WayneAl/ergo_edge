# Stage I demo video — implementation plan

**Goal:** `make video` builds `video/dist/linesafe_stage1.mp4` (≤180 s, 1920×1080, Edge TTS narration, burned-in captions)
from `video/shots.toml`, deck slide PNGs and captured footage; `make video DRAFT=1` builds before any footage exists.
**Architecture:** a small pure core (`shots` → `timeline` → `captions`) in `video/lsvideo/`, wrapped by I/O modules (`tts`,
`slides`, `fonts`, `render`) and one entry script `video/build.py`. ffmpeg does all media work; nothing is imported from
yt_audiobook — the Edge TTS streaming loop and the caption grouping rules are copied and adapted.
**Spec:** `docs/superpowers/specs/2026-09-17-stage1-video-design.md` — executors read both.
**Stack:** Python 3.12 via `uv run --no-project --python 3.12 --with …` (same pattern as the deck's `make check`; there is
no pyproject on `main`), edge-tts ≥ 7.2 (7.2.8 verified), pymupdf, fonttools + brotli, pytest, ffmpeg 8.1 (Homebrew, has
libass `subtitles`, `drawtext`, `amix`, `adelay`, `overlay`, `apad`).

## Global constraints
- Output 1920×1080, 30 fps, H.264 yuv420p + AAC; total ≤ `max_total_s` (180) or the build fails listing every shot.
- Every shot duration is rounded **up** to a whole frame: `math.ceil(raw * 30 - 1e-9) / 30`.
- Footage is never sped up and narration is never cut: narration that runs past its clip is a `VideoError` naming the shot.
- All user-facing failures raise `lsvideo.model.VideoError` with a message that names the shot id (when there is one) and
  the fix. `build.py` catches `VideoError`, prints it to stderr, exits 1. No bare `except:`; no silent fallbacks.
- Edge TTS: `edge_tts.Communicate(text, voice, boundary="WordBoundary")` — edge-tts 7 defaults to sentence boundaries,
  so the argument is mandatory. Zero WordBoundary events → `VideoError`.
- Caption font is `IBM Plex Sans` from `deck/fonts/*.woff2`, converted to TTF under `video/build/fonts/`; the final
  ffmpeg run uses `-loglevel verbose` and the build fails unless every `fontselect: (IBM Plex Sans, …) -> X` has X
  starting with `IBMPlexSans` (libass silently falls back to Helvetica on woff2).
- Generated files live only in gitignored `video/build/`, `video/dist/`, `video/footage/`. Never commit media.
- No network in tests. Tests run with `make video-test`.
- Minimal diff: touch only the files in the file map. Do not edit `deck/`, `docs/evidence/task10-session-guide.md`, or
  anything on the `prototype` branch.

## File map
| File | Responsibility |
|---|---|
| `video/lsvideo/__init__.py` | empty package marker |
| `video/lsvideo/model.py` | dataclasses, `FPS`, `W`, `H`, `VideoError`, `frame_ceil` |
| `video/lsvideo/shots.py` | `load(path, root) -> Script`: parse + validate `shots.toml` |
| `video/lsvideo/timeline.py` | `place(shots, voices, max_total_s) -> list[Placed]` |
| `video/lsvideo/captions.py` | `align`, `group_lines`, `cues`, `to_ass`, `format_ts_ass` |
| `video/lsvideo/tts.py` | Edge TTS synthesis + on-disk cache → `Voice` |
| `video/lsvideo/slides.py` | render deck PDF pages to 1920×1080 PNGs |
| `video/lsvideo/fonts.py` | woff2 → ttf conversion |
| `video/lsvideo/render.py` | ffmpeg command builders, `run`, `probe_duration`, `check_font` |
| `video/build.py` | entry script: the whole build, prints the shot table |
| `video/capture.sh` | screen capture of the live overlay window into `video/footage/` |
| `video/shots.toml` | the narration script and shot list (content below) |
| `video/tests/conftest.py` | puts `video/` on `sys.path` |
| `video/tests/test_shots.py`, `test_timeline.py`, `test_captions.py`, `test_render.py` | unit tests |
| `video/tests/fixtures/edge_words.json` | **already committed** — a real Edge TTS WordBoundary dump |
| `docs/evidence/video-shoot-guide.md` | what Wayne films, in which order, and the privacy checklist |
| `Makefile` | `video`, `video-test` targets |
| `.gitignore` | `video/build/`, `video/dist/`, `video/footage/` |

---

### Task 1: Pure core — model, shots, timeline, captions (logic, TDD)

**Files:** Create `video/lsvideo/__init__.py`, `video/lsvideo/model.py`, `video/lsvideo/shots.py`,
`video/lsvideo/timeline.py`, `video/lsvideo/captions.py`, `video/tests/conftest.py`, `video/tests/test_shots.py`,
`video/tests/test_timeline.py`, `video/tests/test_captions.py` · Modify `Makefile` (add `video-test` only) · Modify
`.gitignore`.

**Interfaces — Produces (Task 2 relies on these exact names):**
```python
# video/lsvideo/model.py
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
```
- `shots.load(path: Path, root: Path) -> Script`
- `timeline.place(shots: Sequence[Shot], voices: Mapping[str, Voice], max_total_s: float) -> list[Placed]`
- `captions.align(words: Sequence[WordTiming], text: str) -> list[Aligned]` where
  `Aligned = NamedTuple("Aligned", [("word", WordTiming), ("start", int), ("end", int), ("punct", str)])`
- `captions.group_lines(aligned: Sequence[Aligned], text: str, max_chars: int = 42) -> list[list[Aligned]]`
- `captions.cues(placed: Sequence[Placed], voices: Mapping[str, Voice], max_chars: int = 42, lines_per_cue: int = 2, bridge_s: float = 0.5) -> list[Cue]`
- `captions.to_ass(cues: Sequence[Cue], font: str = "IBM Plex Sans", fontsize: int = 54) -> str`
- `captions.format_ts_ass(seconds: float) -> str`

**Behaviour to implement:**

`shots.load` — TOML schema (top level `voice` required string, `max_total_s` float default 180, array `[[shot]]`):
- `id` required, matches `^[a-z0-9_]+$`, unique → else `VideoError("shot <id>: …")` (use the index when id is missing).
- Exactly one of `slide` (int ≥ 1) or `clip` (path string, relative to `root`) → else
  `VideoError(f"shot {id}: set exactly one of slide or clip")`.
- `slide = N` → `Still(page=N, png=root / "video/build/slides" / f"{N:02d}.png")`.
- `clip` requires `in_s` and `out_s`, `0 <= in_s < out_s` → else `VideoError(f"shot {id}: need 0 <= in_s < out_s")`.
  Optional `crop = [x, y, w, h]` (4 ints, w and h > 0).
- Optional `pip` (path) requires `pip_in_s`, `pip_out_s` (same rule), optional `pip_crop` → `Inset(main, pip)`. `pip` on a
  slide shot → `VideoError(f"shot {id}: pip needs a clip")`.
- `hold_s` allowed on slide shots only (default 1.0); set on a clip shot → `VideoError(f"shot {id}: hold_s applies to
  slide shots only")`. Clip shots get `hold_s=0.0`.
- `lead_s` default 0.5, must be ≥ 0.
- `narration` default `""`; stored as `" ".join(narration.split())`.
- Unknown keys in a shot → `VideoError(f"shot {id}: unknown keys {sorted(keys)}")` (typos must not pass silently).

`timeline.place` — walk shots in order, `t0` starts at 0.0 and accumulates rounded durations:
- `v = voices[shot.id].dur_s` if `shot.narration` else `0.0`; narrated shot with no voice →
  `VideoError(f"shot {id}: no voice synthesized")`.
- Still: `raw = lead_s + v + hold_s`. Clip: `raw = clip.length`. Inset: `raw = main.length`; if
  `pip.length < main.length` → `VideoError(f"shot {id}: pip clip {pip.length:.2f}s is shorter than main clip {main.length:.2f}s")`.
- Clip/Inset: if `lead_s + v > raw` → `VideoError(f"shot {id}: narration {v:.2f}s starting at {lead_s:.2f}s runs past
  the clip end at {raw:.2f}s — lengthen the clip or shorten the narration")`.
- `dur = frame_ceil(raw)`.
- After all shots: if `total > max_total_s` → `VideoError(f"total {total:.2f}s exceeds {max_total_s:.0f}s: " + ", ".join(f"{p.shot.id}={p.dur:.2f}s" for p in placed))`.

`captions.align` — Edge tokens carry timing but no punctuation, and `REBA/RULA` arrives as `REBA`, `/`, `RULA`:
- Cursor over `text`. For each word, search from the cursor with a regex built from `re.escape(word.text)`, adding
  `(?<!\w)` in front only if `word.text[0]` is a word char and `(?!\w)` behind only if `word.text[-1]` is a word char.
- Not found → skip the word (do not move the cursor). Empty `word.text` → skip.
- Found at `[s, e)`: then consume trailing characters while they are in `".,;:!?…\"'’”)"`; `punct` = the consumed
  characters that are in `".,;:!?…"`; `end` = index after the consumed characters. Cursor = `end`.

`captions.group_lines` — line display text is `text[line[0].start : line[-1].end]`:
- For each aligned `a`: if the buffer is non-empty and `len(text[buf[0].start : a.end]) > max_chars` → flush first.
- Append `a`. Then if `a.punct` contains any of `.!?…` → flush; elif it contains any of `,;:` and
  `len(text[buf[0].start : a.end]) >= int(max_chars * 0.7)` → flush.
- Flush the remainder at the end. A single token longer than `max_chars` stays alone on its line.

`captions.cues` — for each placed shot with narration: `offset = p.t0 + p.shot.lead_s`; rows =
`group_lines(align(voice.words, shot.narration), shot.narration, max_chars)`; pack `lines_per_cue` rows per cue:
`Cue(offset + block[0][0].word.start, offset + block[-1][-1].word.end, "\n".join(line texts))`. Then, over the whole cue
list in time order: if the next cue starts before this one ends, set `t1 = next.t0`; if `0 <= next.t0 - t1 < bridge_s`,
set `t1 = next.t0` (no caption flicker in short pauses).

`captions.to_ass` — exactly this header (values filled), then one `Dialogue` per cue; in the text replace `{`→`(`,
`}`→`)`, `\n`→`\N`:
```
[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV
Style: Default,{font},{fontsize},&H00FFFFFF,&H60000000,&H60000000,0,3,10,0,2,160,160,40

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,{format_ts_ass(t0)},{format_ts_ass(t1)},Default,,0,0,0,,{text}
```
`format_ts_ass` (copied from yt_audiobook `subtitles.py`):
```python
def format_ts_ass(seconds: float) -> str:
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360_000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"
```

**Steps:**
- [ ] 1. `.gitignore`: append `video/build/`, `video/dist/`, `video/footage/`. `Makefile`: add `video-test` to `.PHONY` and
  ```make
  video-test:
  	uv run --no-project --python 3.12 --with pytest python -m pytest video/tests -q
  ```
  Create `video/tests/conftest.py`:
  ```python
  import sys
  from pathlib import Path

  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  ```
  → verify: `make video-test` runs (no tests collected yet is fine: exit code 5 is expected at this step).
- [ ] 2. Write `video/tests/test_shots.py` (below) and `video/lsvideo/model.py` (verbatim above) with an empty
  `shots.py` → verify: `make video-test`, expect failures from missing `load`.
- [ ] 3. Implement `shots.load` → verify: `make video-test` passes `test_shots.py`.
- [ ] 4. Write `video/tests/test_timeline.py` (below) → verify: fails on missing `place`. Implement → passes.
- [ ] 5. Write `video/tests/test_captions.py` (below) → verify: fails on missing functions. Implement → passes; full
  suite green, no warnings.
- [ ] 6. Commit: `git add .gitignore Makefile video/lsvideo video/tests && git commit -m "feat(video): shot list, timeline and captions core"`

```python
# video/tests/test_shots.py
from pathlib import Path

import pytest

from lsvideo.model import Clip, Inset, Still, VideoError
from lsvideo.shots import load

HEAD = 'voice = "en-US-AndrewNeural"\n'


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "shots.toml"
    p.write_text(HEAD + body)
    return p


def test_slide_shot_resolves_png_under_build_slides(tmp_path):
    s = load(write(tmp_path, '[[shot]]\nid = "problem"\nslide = 2\nnarration = "Hi."\n'), tmp_path)
    shot = s.shots[0]
    assert shot.visual == Still(page=2, png=tmp_path / "video/build/slides/02.png")
    assert (shot.lead_s, shot.hold_s) == (0.5, 1.0)
    assert s.max_total_s == 180


def test_clip_with_pip_becomes_inset(tmp_path):
    body = ('[[shot]]\nid = "lift"\nclip = "video/footage/a.mp4"\nin_s = 1.0\nout_s = 9.0\ncrop = [0, 50, 1280, 720]\n'
            'pip = "video/footage/b.mp4"\npip_in_s = 2.0\npip_out_s = 12.0\n')
    shot = load(write(tmp_path, body), tmp_path).shots[0]
    assert shot.visual == Inset(
        main=Clip(tmp_path / "video/footage/a.mp4", 1.0, 9.0, (0, 50, 1280, 720)),
        pip=Clip(tmp_path / "video/footage/b.mp4", 2.0, 12.0, None),
    )
    assert shot.hold_s == 0.0


def test_shot_with_both_slide_and_clip_is_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\nclip = "a.mp4"\nin_s = 0\nout_s = 1\n'
    with pytest.raises(VideoError, match="shot x: set exactly one of slide or clip"):
        load(write(tmp_path, body), tmp_path)


def test_hold_s_on_clip_is_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nclip = "a.mp4"\nin_s = 0\nout_s = 1\nhold_s = 2\n'
    with pytest.raises(VideoError, match="shot x: hold_s applies to slide shots only"):
        load(write(tmp_path, body), tmp_path)


def test_duplicate_ids_are_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\n[[shot]]\nid = "x"\nslide = 2\n'
    with pytest.raises(VideoError, match="x"):
        load(write(tmp_path, body), tmp_path)


def test_in_s_must_be_before_out_s(tmp_path):
    body = '[[shot]]\nid = "x"\nclip = "a.mp4"\nin_s = 5\nout_s = 5\n'
    with pytest.raises(VideoError, match=r"shot x: need 0 <= in_s < out_s"):
        load(write(tmp_path, body), tmp_path)


def test_unknown_key_is_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\nnaration = "typo"\n'
    with pytest.raises(VideoError, match="shot x: unknown keys"):
        load(write(tmp_path, body), tmp_path)


def test_narration_whitespace_is_collapsed(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\nnarration = """\nOne  two.\n   Three.\n"""\n'
    assert load(write(tmp_path, body), tmp_path).shots[0].narration == "One two. Three."
```

```python
# video/tests/test_timeline.py
from pathlib import Path

import pytest

from lsvideo.model import Clip, Inset, Shot, Still, VideoError, Voice
from lsvideo.timeline import place

P = Path("x")


def still(id, narration="Hi.", lead=0.5, hold=1.0):
    return Shot(id, Still(1, P), narration, lead, hold)


def clip(id, length, narration="Hi.", lead=0.5):
    return Shot(id, Clip(P, 10.0, 10.0 + length), narration, lead, 0.0)


def voice(id, dur):
    return Voice(id, P, (), dur)


def test_still_duration_is_lead_plus_voice_plus_hold_rounded_up_to_a_frame():
    [p] = place([still("a")], {"a": voice("a", 4.128)}, 180)
    assert p.t0 == 0.0
    assert p.dur == pytest.approx(169 / 30)   # 5.628 s -> 168.84 frames -> 169


def test_clip_duration_is_the_clip_length():
    [p] = place([clip("a", 55.0)], {"a": voice("a", 20.0)}, 180)
    assert p.dur == pytest.approx(55.0)


def test_t0_accumulates_rounded_durations():
    placed = place([still("a"), clip("b", 3.0)], {"a": voice("a", 4.128), "b": voice("b", 1.0)}, 180)
    assert placed[1].t0 == pytest.approx(169 / 30)


def test_narration_running_past_the_clip_end_names_the_shot():
    with pytest.raises(VideoError, match="shot lift: narration 9.80s starting at 0.50s runs past the clip end at 10.00s"):
        place([clip("lift", 10.0)], {"lift": voice("lift", 9.8)}, 180)


def test_pip_shorter_than_main_names_the_shot():
    shot = Shot("lift", Inset(Clip(P, 0.0, 10.0), Clip(P, 0.0, 9.0)), "", 0.5, 0.0)
    with pytest.raises(VideoError, match="shot lift: pip clip 9.00s is shorter than main clip 10.00s"):
        place([shot], {}, 180)


def test_total_over_the_limit_lists_every_shot():
    with pytest.raises(VideoError, match=r"total 12.00s exceeds 10s: a=6.00s, b=6.00s"):
        place([clip("a", 6.0, ""), clip("b", 6.0, "")], {}, 10)


def test_silent_shot_needs_no_voice():
    [p] = place([still("a", narration="")], {}, 180)
    assert p.dur == pytest.approx(1.5)


def test_narrated_shot_without_a_voice_is_rejected():
    with pytest.raises(VideoError, match="shot a: no voice synthesized"):
        place([still("a")], {}, 180)
```

```python
# video/tests/test_captions.py
import json
from pathlib import Path

import pytest

from lsvideo.captions import align, cues, format_ts_ass, group_lines, to_ass
from lsvideo.model import Cue, Placed, Shot, Still, Voice, WordTiming

FIXTURE = json.loads((Path(__file__).parent / "fixtures/edge_words.json").read_text())
TEXT = FIXTURE["text"]
WORDS = tuple(WordTiming(w["text"], w["start"], w["end"]) for w in FIXTURE["words"])


def texts(rows, text):
    return [text[r[0].start : r[-1].end] for r in rows]


def test_align_recovers_trailing_punctuation_from_the_script():
    by_text = {}
    for a in align(WORDS, TEXT):
        by_text.setdefault(a.word.text, a)
    assert by_text["keypoints"].punct == "."
    assert by_text["angles"].punct == ","
    assert by_text["UGen300"].punct == ""


def test_align_matches_the_slash_token_inside_reba_rula():
    aligned = align(WORDS, TEXT)
    i = [a.word.text for a in aligned].index("/")
    assert TEXT[aligned[i - 1].start : aligned[i + 1].end] == "REBA/RULA"


def test_align_skips_a_token_missing_from_the_script():
    words = WORDS[:8] + (WordTiming("seventeen", 2.0, 2.4),) + WORDS[8:]
    aligned = align(words, TEXT)
    assert len(aligned) == len(WORDS)
    assert [a.word.text for a in aligned][8:10] == ["17", "body"]


def test_lines_wrap_at_max_chars_and_break_after_a_sentence():
    rows = texts(group_lines(align(WORDS, TEXT), TEXT, 42), TEXT)
    assert rows[0] == "A camera feeds a pose model that finds 17"
    assert rows[1] == "body keypoints."
    assert rows[2].startswith("Plain geometry")


def test_every_line_is_a_substring_of_the_script_within_max_chars():
    for row in texts(group_lines(align(WORDS, TEXT), TEXT, 42), TEXT):
        assert row in TEXT
        assert len(row) <= 42


def test_cue_times_are_offset_by_shot_t0_plus_lead():
    shot = Shot("a", Still(1, Path("x")), TEXT, 0.5, 1.0)
    out = cues([Placed(shot, 10.0, 25.0)], {"a": Voice("a", Path("x"), WORDS, 19.0)})
    assert out[0].t0 == pytest.approx(10.5 + WORDS[0].start)
    assert out[0].text == "A camera feeds a pose model that finds 17\nbody keypoints."


def two_sentence_voice(gap):
    words = (WordTiming("One", 0.0, 0.3), WordTiming("two", 0.35, 0.6),
             WordTiming("Three", 0.6 + gap, 0.9 + gap), WordTiming("four", 0.95 + gap, 1.2 + gap))
    shot = Shot("a", Still(1, Path("x")), "One two. Three four.", 0.0, 0.0)
    return [Placed(shot, 0.0, 5.0)], {"a": Voice("a", Path("x"), words, 2.0)}


def test_a_short_pause_between_cues_is_bridged():
    placed, voices = two_sentence_voice(0.3)
    first, second = cues(placed, voices, lines_per_cue=1)
    assert first.t1 == pytest.approx(second.t0)


def test_a_long_pause_between_cues_is_kept():
    placed, voices = two_sentence_voice(0.8)
    first, _ = cues(placed, voices, lines_per_cue=1)
    assert first.t1 == pytest.approx(0.6)


def test_to_ass_formats_timestamps_and_escapes_newlines():
    ass = to_ass([Cue(1.0, 62.5, "Line {one}\nLine two")])
    assert "PlayResX: 1920" in ass
    assert "Style: Default,IBM Plex Sans,54," in ass
    assert "Dialogue: 0,0:00:01.00,0:01:02.50,Default,,0,0,0,,Line (one)\\NLine two" in ass
    assert format_ts_ass(3725.456) == "1:02:05.46"
```

**Wayne-workflow rules for this task:**
- RED first, watched: run each new test file before implementing and confirm it fails because the feature is missing,
  not because of a typo or import error.
- Fixtures come from real data: `edge_words.json` is a real Edge TTS dump — do not regenerate or hand-edit it.
- Smallest diff that does the job; no speculative options beyond the signatures above.
- Commit with explicit paths only; never `git add -A`. Do not push (the orchestrator pushes).

---

### Task 2: I/O, render, build entry, script content, shoot guide (glue + one real end-to-end sample)

**Files:** Create `video/lsvideo/tts.py`, `video/lsvideo/slides.py`, `video/lsvideo/fonts.py`, `video/lsvideo/render.py`,
`video/build.py`, `video/capture.sh`, `video/shots.toml`, `video/tests/test_render.py`,
`docs/evidence/video-shoot-guide.md` · Modify `Makefile` (add `video`).

**Interfaces — Consumes:** everything Task 1 produces (names above). **Produces:**
- `tts.voice_for(shot: Shot, voice: str, cache_dir: Path) -> Voice | None`
- `slides.ensure_pngs(pdf: Path, pages: Iterable[int], out_dir: Path) -> None`
- `fonts.ensure_ttf(src_dir: Path, dst_dir: Path) -> Path`
- `render.run(cmd: list[str], cwd: Path | None = None) -> str` (returns stderr), `render.probe_duration(path: Path) -> float`,
  `render.check_font(log: str, family: str = "IBM Plex Sans", expect_prefix: str = "IBMPlexSans") -> None`,
  `render.still_cmd`, `render.clip_cmd`, `render.inset_cmd`, `render.placeholder_cmd`, `render.concat_cmd`,
  `render.final_cmd` (signatures below).

**Behaviour to implement:**

`tts.py` — the streaming loop is copied from `~/Documents/GitHub/yt_audiobook/src/yt_audiobook/providers/tts.py`
(`EdgeTTSProvider._run`); keep a one-line comment naming that source:
```python
async def _stream(text: str, voice: str) -> tuple[bytes, list[WordTiming]]:
    comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio = bytearray()
    words: list[WordTiming] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"] / 10_000_000
            words.append(WordTiming(chunk["text"], start, start + chunk["duration"] / 10_000_000))
    return bytes(audio), words
```
`voice_for`: return `None` for an empty narration. Key = `hashlib.sha1(f"{voice}\n{shot.narration}".encode()).hexdigest()[:16]`;
cache files `<key>.mp3` and `<key>.json` (`{"voice", "text", "dur_s", "words": [[text, start, end], …]}`). On a cache miss:
synthesize with `asyncio.run(_stream(...))`; any exception → `VideoError(f"shot {id}: Edge TTS failed and no cached
audio: {exc}")`; no words → `VideoError(f"shot {id}: Edge TTS returned no WordBoundary events")`; write the mp3, measure
`dur_s` with `render.probe_duration`, write the json last (a json without its mp3 never exists). Print one line per shot:
`tts <id>: cached` or `tts <id>: synthesized <dur>s`.

`slides.py` — `import pymupdf`. Missing PDF → `VideoError(f"{pdf} missing — run make pdf")`. Page outside
`1..page_count` → `VideoError(f"slide {n}: the deck has {page_count} pages")`. Re-render a PNG when it is missing or older
than the PDF: `page.get_pixmap(matrix=pymupdf.Matrix(W / page.rect.width, H / page.rect.height))`, saved as
`<out_dir>/<n:02d>.png`; assert the pixmap is exactly 1920×1080.

`fonts.py` — for every `*.woff2` in `src_dir`, write `<dst_dir>/<stem>.ttf` if missing via
`TTFont(src); font.flavor = None; font.save(dst)`. Return `dst_dir`.

`render.py` — every command starts `["ffmpeg", "-hide_banner", "-y"]`; parts are encoded identically
(`-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30 -an`) so `concat -c copy` works. The shared fit chain:
`FIT = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=30,format=yuv420p"`;
a crop `(x, y, w, h)` prepends `crop={w}:{h}:{x}:{y},`.
- `still_cmd(png: Path, dur: float, out: Path)`: `-loop 1 -framerate 30 -i png -t dur -vf FIT …`
- `clip_cmd(clip: Clip, dur: float, out: Path)`: `-ss in_s -i path -t dur -vf [crop,]FIT …`
- `inset_cmd(inset: Inset, dur: float, out: Path)`: `-ss main.in_s -i main.path -ss pip.in_s -i pip.path -t dur
  -filter_complex "[0:v][crop,]FIT[m];[1:v][crop,]scale=560:360:force_original_aspect_ratio=decrease,pad=iw+8:ih+8:4:4:color=white,fps=30[p];[m][p]overlay=40:40:shortest=1,format=yuv420p[v]" -map [v] …`
- `placeholder_cmd(label_file: Path, dur: float, out: Path, fontfile: Path)`: `-f lavfi -i color=c=0x404040:s=1920x1080:r=30:d={dur}
  -vf "drawtext=fontfile={fontfile}:textfile={label_file}:fontcolor=white:fontsize=64:x=(w-text_w)/2:y=(h-text_h)/2,format=yuv420p" …`
  Label text: `PLACEHOLDER — {shot id}\n{footage path relative to root}`.
- `concat_cmd(list_file: Path, out: Path)`: `-f concat -safe 0 -i list_file -c copy out`.
- `final_cmd(video: Path, voices: Sequence[tuple[Path, float]], ass_name: str, fonts_dir_name: str, total: float, out: Path)`:
  inputs `-i video` then `-i mp3` per voice; filter_complex: per voice `[{i}:a]adelay=delays={round(offset*1000)}:all=1[a{i}]`,
  then `[a1]…[an]amix=inputs=n:normalize=0:dropout_transition=0,apad[a]` (with zero voices use
  `anullsrc=r=48000:cl=mono` as `[a]`), and `[0:v]subtitles=filename={ass_name}:fontsdir={fonts_dir_name}[v]`; map `[v]` and
  `[a]`; `-t total -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -c:a aac -b:a 160k -ar 48000 -movflags +faststart
  -loglevel verbose out`. The caller runs it with `cwd=video/build` so the ass and fonts paths need no escaping.
- `run`: `subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)`; non-zero exit →
  `VideoError("ffmpeg failed: " + " ".join(cmd[:6]) + " …\n" + last 20 stderr lines)`. Returns stderr.
- `probe_duration`: `ffprobe -v error -show_entries format=duration -of csv=p=0 path` → float.
- `check_font(log, family, expect_prefix)`: find all `fontselect: ({family}, …) -> {target},` matches; none →
  `VideoError(f"no fontselect line for {family} in the ffmpeg log")`; any target not starting with `expect_prefix` →
  `VideoError(f"libass fell back to {target} for {family}")`.

`build.py` — `argparse`: `--draft`, `--shots` (default `video/shots.toml`). `root = Path(__file__).resolve().parents[1]`;
`sys.path.insert(0, str(root / "video"))`. Steps, each printing one line:
1. `script = shots.load(...)`.
2. `slides.ensure_pngs(root/"deck/dist/proposal.pdf", sorted pages of Still visuals, root/"video/build/slides")`.
3. `voices = {id: v}` via `tts.voice_for(shot, script.voice, root/"video/build/tts")` (skip `None`).
4. `placed = timeline.place(script.shots, voices, script.max_total_s)`; print a table `id  t0  dur  narration_s`
   and the total.
5. Footage check: every `Clip`/`Inset` path must exist; missing and not `--draft` → `VideoError(f"shot {id}: missing
   footage {path relative to root} — see docs/evidence/video-shoot-guide.md, or build with DRAFT=1")`; missing with
   `--draft` → placeholder part for that shot (an `Inset` with only the pip missing is also a placeholder).
6. Render each part to `video/build/parts/{index:02d}_{id}.mp4` with `dur = placed.dur`; write `video/build/parts.txt`
   (`file 'parts/NN_id.mp4'` lines); concat → `video/build/video_only.mp4`.
7. `fonts.ensure_ttf(root/"deck/fonts", root/"video/build/fonts")`; write `video/build/captions.ass` from
   `captions.to_ass(captions.cues(placed, voices))`.
8. Final → `video/dist/linesafe_stage1_DRAFT.mp4` with `--draft`, else `video/dist/linesafe_stage1.mp4`; voice offsets
   are `placed.t0 + shot.lead_s`; `render.check_font(stderr)`.
9. `probe_duration(out)` must be ≤ `script.max_total_s + 0.1` else `VideoError`; print the output path and duration.
`main()` wraps everything: `except VideoError as e: print(e, file=sys.stderr); sys.exit(1)`.

`Makefile` — add `video` to `.PHONY` and:
```make
VIDEO_PY := uv run --no-project --python 3.12 --with 'edge-tts>=7.2' --with pymupdf --with fonttools --with brotli python

video: $(PDF)
	$(VIDEO_PY) video/build.py $(if $(DRAFT),--draft)
```

`video/capture.sh` (make it executable):
```bash
#!/usr/bin/env bash
# Screen capture of the live overlay for the demo video: video/capture.sh <name> <seconds>
set -euo pipefail
name=${1:?usage: video/capture.sh <name> <seconds>}
seconds=${2:?usage: video/capture.sh <name> <seconds>}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/video/footage"
out="$root/video/footage/$name.mp4"
devices=$(ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true)
idx=$(printf '%s\n' "$devices" | sed -n 's/.*\[\([0-9][0-9]*\)\] Capture screen 0.*/\1/p' | head -1)
if [ -z "$idx" ]; then
  echo "no 'Capture screen 0' device — allow Screen Recording for this terminal in System Settings" >&2
  exit 1
fi
ffmpeg -hide_banner -y -f avfoundation -capture_cursor 0 -framerate 30 -i "${idx}:none" -t "$seconds" \
  -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$out"
echo "$out"
```

`video/shots.toml` — verbatim (in/out values are first guesses; they are tuned after the shoot):
```toml
# LineSafe Stage I demo video. Edit narration here; `make video` rebuilds.
# Every number spoken must be a row in deck/claims.md. Footage lives in video/footage/ (gitignored).
voice = "en-US-AndrewNeural"
max_total_s = 180

[[shot]]
id = "problem"
slide = 2
narration = """
In most plants, ergonomic risk is still assessed with a clipboard. A safety manager watches one station, scores each
posture by eye on a paper REBA or RULA sheet, and types it into a monthly report. Two assessors give two scores, the
risky posture happens when nobody is watching, and nothing is left to show what changed.
"""

[[shot]]
id = "approach"
slide = 5
narration = """
LineSafe turns the station PC into a continuous assessor. A camera feeds a pose model that finds 17 body keypoints.
Plain geometry turns them into joint angles, and the published REBA and RULA tables turn the angles into a score. The
same posture always gives the same score, and every score names the angle behind it.
"""

[[shot]]
id = "lift_intro"
clip = "video/footage/overlay_lift.mp4"
in_s = 2.0
out_s = 14.0
pip = "video/footage/phone_lift.mp4"
pip_in_s = 2.0
pip_out_s = 14.0
lead_s = 1.0
narration = """
This is the prototype, running live on a laptop. In Stage Two, the same pipeline moves onto the ASUS UGen300.
"""

[[shot]]
id = "lift_climb"
clip = "video/footage/overlay_lift.mp4"
in_s = 14.0
out_s = 36.0
pip = "video/footage/phone_lift.mp4"
pip_in_s = 14.0
pip_out_s = 36.0
narration = """
Upright, the score is Low. As the trunk bends to reach the box, it climbs to Medium, then to High.
"""

[[shot]]
id = "lift_alert"
clip = "video/footage/overlay_lift.mp4"
in_s = 36.0
out_s = 56.0
pip = "video/footage/phone_lift.mp4"
pip_in_s = 36.0
pip_out_s = 56.0
narration = """
Held in High for a few seconds, the posture raises an alert, and an event is stored with the angles that drove it.
"""

[[shot]]
id = "offline"
clip = "video/footage/overlay_offline.mp4"
in_s = 2.0
out_s = 20.0
pip = "video/footage/phone_offline.mp4"
pip_in_s = 2.0
pip_out_s = 20.0
narration = """
Now the laptop goes offline. Scoring carries on, because nothing ever needs to leave the room.
"""

[[shot]]
id = "dashboard"
clip = "video/footage/phone_dashboard.mp4"
in_s = 1.0
out_s = 25.0
narration = """
Back on the plant network, the safety manager opens the dashboard on a phone. Each event shows the station, the time,
the peak score and the angles behind it. No video is stored and no one is identified. The system rates the task, not
the person.
"""

[[shot]]
id = "stage_two"
slide = 16
narration = """
As a finalist, we bring the pipeline up on the UGen300, measure throughput per stream, and check the scores against
certified human assessors.
"""

[[shot]]
id = "close"
slide = 1
hold_s = 2.0
narration = """
LineSafe. Continuous ergonomic risk scoring at the workstation, offline, on an ASUS UGen300.
"""
```

`docs/evidence/video-shoot-guide.md` — ≤60 lines, for Wayne, written after Task 10 step 4 in the same sitting. Must cover:
setup (System Settings → Privacy & Security → Screen Recording: allow the terminal; Do Not Disturb on; close other
windows; only the `LineSafe` window and nothing identifying on screen); phone propped for a side-on wide shot that shows
the person and the laptop screen; start both recordings, then **one clap in view** so the two files can be lined up by
`in_s`/`pip_in_s`; the three takes and their file names — (1) `overlay_lift` via `video/capture.sh overlay_lift 75` +
phone `phone_lift.mp4`: upright 12 s, walk to the box, bend and lift, hold the bend ~10 s so the score reaches High and
the alert fires, stand up with the box, put it down; (2) `overlay_offline` via `video/capture.sh overlay_offline 30` +
`phone_offline.mp4`: turn Wi-Fi off on the laptop *in the phone shot only*, then bend a few times so the score visibly
keeps updating; turn Wi-Fi back on; (3) `phone_dashboard.mp4`: phone screen recording opening the dashboard and
scrolling the events — the address bar will be cropped out with `crop`; AirDrop the phone files into `video/footage/`;
then tell Claude, who tunes `in_s`/`out_s`/`crop` in `shots.toml`, adjusts any narration line that does not match what
the overlay actually shows, builds, and watches the result end to end.

`video/tests/test_render.py`:
```python
import pytest

from lsvideo.model import VideoError
from lsvideo.render import check_font

OK = "[Parsed_subtitles_0 @ 0x141806a10] fontselect: (IBM Plex Sans, 400, 0) -> IBMPlexSans-Regular, 0, IBMPlexSans-Regular"
FALLBACK = ("[Parsed_subtitles_0 @ 0x13f6151c0] fontselect: (IBM Plex Sans, 400, 0) -> "
            "/System/Library/Fonts/Helvetica.ttc, -1, Helvetica")


def test_font_resolved_to_plex_passes():
    check_font("noise\n" + OK + "\nmore noise")


def test_font_fallback_to_helvetica_fails():
    with pytest.raises(VideoError, match="libass fell back to /System/Library/Fonts/Helvetica.ttc for IBM Plex Sans"):
        check_font(OK + "\n" + FALLBACK)


def test_missing_fontselect_line_fails():
    with pytest.raises(VideoError, match="no fontselect line for IBM Plex Sans"):
        check_font("frame=  100 fps=30")
```
(The two log lines are real ffmpeg 8.1 output captured on this machine.)

**Steps:**
- [ ] 1. Write `test_render.py` → verify fails; implement `render.check_font` → `make video-test` passes.
- [ ] 2. Implement `tts.py`, `slides.py`, `fonts.py`, the rest of `render.py`, `build.py`, the Makefile `video` target,
  `shots.toml`, `capture.sh` → verify: `bash -n video/capture.sh` exits 0; `make video-test` still green.
- [ ] 3. Real end-to-end sample (gate 2): `make video DRAFT=1` → expect the shot table, total ≤ 180 s, and
  `video/dist/linesafe_stage1_DRAFT.mp4`. Then `ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate -of compact video/dist/linesafe_stage1_DRAFT.mp4`
  → h264 1920×1080 30/1 + aac, duration ≤ 180. Extract one frame from the middle of every shot
  (`ffmpeg -ss <t0 + dur/2> -i … -frames:v 1 video/build/check_<id>.png`) and look at each PNG: slide shots show the
  slide with a caption box at the bottom in IBM Plex Sans, footage shots show the labelled grey placeholder.
  Run `make video DRAFT=1` a second time → every shot prints `tts <id>: cached`.
- [ ] 4. If the total exceeds 180 s or a narration overruns its clip, stop and report the numbers — do not edit the
  narration text or the in/out values to make it fit.
- [ ] 5. Commit: `git add Makefile video/lsvideo video/build.py video/capture.sh video/shots.toml video/tests/test_render.py docs/evidence/video-shoot-guide.md && git commit -m "feat(video): Edge TTS narration, ffmpeg render, draft build and shoot guide"`

**Wayne-workflow rules for this task:**
- Code whose real path crosses an external system (Edge TTS, ffmpeg) is not done without one real end-to-end sample:
  step 3 is mandatory and its output (the shot table, the ffprobe line, what each check PNG shows) goes in the report.
- No "done / passing" without fresh verification output from this run.
- Fail loudly: every failure path above raises `VideoError` naming the shot; no silent fallback to a system font, a
  default voice, or a skipped shot.
- Commit with explicit paths only; never `git add -A`; never commit anything under `video/build/`, `video/dist/`,
  `video/footage/`. Do not push.
- Do not upload anything anywhere (YouTube upload is Wayne's, after he watches the final build).
