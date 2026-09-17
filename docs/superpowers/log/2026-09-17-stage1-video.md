# 2026-09-17 — Stage I demo video pipeline (completion log)

Plan: `docs/superpowers/plans/2026-09-17-stage1-video.md` · Spec: `docs/superpowers/specs/2026-09-17-stage1-video-design.md`
Merged: `a957858` (branch `video`: c2bf82d, de4f380, 761dc11, 2ac7b0b, 015ae1f).

## What landed
- `make video` / `make video DRAFT=1` → `video/dist/linesafe_stage1[_DRAFT].mp4`; `make video-test` (35 tests, no network).
  Runs as `uv run --no-project --python 3.12 --with 'edge-tts>=7.2' --with pymupdf --with fonttools --with brotli`.
- `video/shots.toml` is the whole video: `voice`, `max_total_s`, `[[shot]]` rows with `slide = N` or `clip` (+ `in_s`,
  `out_s`, `crop`, optional `pip`, `pip_in_s`, `pip_out_s`, `pip_crop`), `lead_s`, `hold_s` (slides only), `narration`.
  Unknown keys fail. Current script: 9 shots, 160.90 s, `en-US-AndrewNeural`.
- `video/lsvideo/`: `model` (dataclasses, `FPS=30`, `frame_ceil`, `VideoError`) · `shots.load` · `timeline.place` (whole-frame
  t0/dur, narration-past-clip and 180 s checks) + `timeline.check_sources` · `captions` (`align` script substrings to Edge
  tokens, `layout` sentence-bounded balanced ≤2-line cues, `cues`, `to_ass`) · `tts.voice_for` (Edge TTS copied from
  yt_audiobook, cache `video/build/tts/<sha1>.{mp3,json}`) · `slides` (PDF page → 1920×1080 PNG, edge colour) · `fonts`
  (woff2 → ttf) · `render` (ffmpeg builders, `check_font`, `check_part`) · `video/build.py` entry.
- Slides render at 1680×945 on their own edge colour so captions sit in a band below the content.
- `video/capture.sh <name> <seconds>` (avfoundation screen capture) and `docs/evidence/video-shoot-guide.md`.

## Lessons that changed the plan
- libass cannot open woff2 and silently falls back to Helvetica → fonts converted to TTF, build fails on any other
  `fontselect` target.
- Edge TTS (7.2.8) strips punctuation and emits `/` as its own token; captions show script substrings, tokens only time them.
- Greedy 42-char wrap + two rows per cue orphaned words and crossed sentences in the real draft → replaced by `layout`.
- Full-bleed slides put captions over slide text → caption band.
- Whole-branch review: ffmpeg silently renders a short part when footage ends before `out_s` (overlay `shortest=1`,
  `apad`, `-t total` hide it) → pre-render source-length check + post-render part-duration check, repro before/after.

## Backlog
- Footage not shot yet: `overlay_lift`, `phone_lift`, `overlay_offline`, `phone_offline`, `phone_dashboard` (guide above,
  same sitting as Task 10). After the shoot: tune in/out/crop, align narration with what the overlay really shows,
  full build, watch end to end, Wayne uploads unlisted to YouTube.
- `site_replay` (18 s, source 0:41–0:59 of YouTube 27N3I5kvbjk): overlay fades to skeleton only; publish only after
  the uploader's written permission (Wayne is handling it). Source and composite are gitignored under `video/footage/`.
- Captions: leading punctuation and a dash at a line break are not shown; `\` is not escaped (none in the script).
- `check_sources` uses container duration; a phone file with longer audio than video is caught only by `check_part`.
