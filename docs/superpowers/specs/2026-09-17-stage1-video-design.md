# Stage I demo video — design

Context: optional YouTube field on the one-shot Stage I form — ≤3 min, English, concept + operation, unlisted link.
It goes in with the final deck. Narration the yt_audiobook way: Edge TTS + WordBoundary timings + ASS captions.

## Goal / non-goals
Goal: `make video` → `video/dist/linesafe_stage1.mp4`, 1920×1080, ≤180 s, Edge TTS narration with burned-in captions,
built from `video/shots.toml` + captured footage (gitignored `video/footage/`) + deck PNGs.
Non-goals: YouTube upload (Wayne), music, motion graphics, Hailo footage (no UGen300 before finals), an editing GUI.

Beats (from slide 12): 1 problem (slide 2) · 2 approach (slide 5) · 3 live demo — box lift, score Low → Medium → High,
alert, network cable pulled and scoring continues · 4 dashboard on the phone · 5 privacy, Stage II, GitHub link.

## Data structures
- `Shot(id, visual, narration, lead_s, hold_s)`; visual = `Still(page, png)` | `Clip(path, in_s, out_s, crop)` |
  `Inset(main: Clip, pip: Clip)`. One TOML row per beat; editing the video means editing that file. One footage file
  can feed several consecutive shots (different in/out), which is how narration lines up with the action.
- `Voice(shot_id, mp3, words: list[WordTiming], dur_s)`, cached at `video/build/tts/<sha1(voice+text)>` — rebuilds stay
  offline; changing one line re-synthesizes one shot.
- `Placed(shot, t0, dur)`: narration starts at t0 + lead_s. Still → dur = lead_s + voice + hold_s; Clip → dur = out_s −
  in_s. Every dur is rounded up to a whole frame at 30 fps so video and narration timelines cannot drift.
- `Cue(t0, t1, text)` → ASS, ≤42 chars per line (yt_audiobook LANDSCAPE). Cue text is the script's own substring —
  Edge drops punctuation and splits `REBA/RULA` into three tokens, so tokens only supply the timing.

## Interfaces (`video/`, run like the deck: `uv run --no-project --with edge-tts python`)
- `tts.synthesize(text, voice) -> (bytes, list[WordTiming])` — copied from yt_audiobook `EdgeTTSProvider`, not imported.
- `timeline.place(shots, voices) -> list[Placed]` — pure.
- `captions.cues(placed, voices) -> list[Cue]`, `captions.ass(cues) -> str` — adapted from yt_audiobook `subtitles.py`
  (punctuation re-alignment + grouping); plain white-on-shadow instead of karaoke.
- `render.shot(placed) -> Path` (still loop / trim + scale + pad / picture-in-picture), `render.final(parts, voices, ass)`
  (concat, place each narration at its t0, burn the ASS).
- `video/capture.sh <name> <seconds>` — `ffmpeg -f avfoundation` screen capture of the live `linesafe run` window.

## Invariants & failure modes
- Total ≤ 180 s, else fail listing every shot's duration.
- Narration longer than its clip → fail naming the shot. Footage is never sped up; narration is never cut.
- Footage is real time: the overlay FPS is whatever was captured. Narration says the demo runs on the laptop today and on
  the UGen300 in Stage II — never implies the accelerator is in the shot.
- A replay shot (`site_replay`, public footage via `video/site_replay.sh`) is labelled on screen as an offline replay,
  hides the replay HUD's FPS figure, and needs the uploader's written permission before the video is published.
- Every number spoken is a row in `deck/claims.md`; 🔴 rows are spoken as "we estimate".
- Missing footage → fail naming the shot; `DRAFT=1` renders a labelled grey placeholder instead and writes
  `linesafe_stage1_DRAFT.mp4`, so a draft is never mistaken for the upload.
- Captions use the deck font. libass cannot open the deck's woff2 files and silently falls back to Helvetica, so the
  build converts them to TTF and fails if ffmpeg's fontselect does not resolve to IBM Plex Sans.
- Edge TTS unreachable → cache; no cache → fail. Zero WordBoundary events → fail (edge-tts 7 defaults to sentences).
- Nothing identifying on screen besides the builder: no terminal history, Wi-Fi name, or LAN IP in captured frames.

## Verification
- Unit: `place` durations and overflow errors; cue grouping on a fixed WordTiming list.
- Real sample (gate 2): `make video DRAFT=1` — slide stills + placeholders through real Edge TTS and ffmpeg; ffprobe
  ≤180 s; one extracted frame per shot viewed; captions spot-checked against the audio.
- Final: full build with real footage, watched end to end once before upload.

## Open decisions
A. Voice: a) `en-US-AndrewNeural` (yt_audiobook en_public) · b) `en-US-AvaNeural` (en_summary).
B. Demo footage: a) screen-capture the live `run` window + phone wide shot as inset (shows the real live FPS) ·
   b) record raw, render the overlay offline (cleaner, but not live; needs a `run --out` change on `prototype`).
C. Filming: a) same sitting as Task 10, after step 4 · b) a separate session.
D. Captions: a) burned in · b) SRT uploaded to YouTube.
