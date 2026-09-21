# LineSafe

Continuous, offline REBA/RULA ergonomic risk scoring from pose estimation.

A camera watches one workstation. A pose model finds 17 body keypoints, plain geometry turns them
into joint angles, and the published REBA and RULA worksheets turn the angles into a score — every
frame, every shift, on the machine in the room. Sustained high-risk postures become events in a
local SQLite file that a phone on the same LAN can open. The video never leaves the device and is
never stored.

Built as the entry for the ASUS UGen AI League hackathon (Battlefield Lightning — Workplace AI),
targeting an ASUS UGen300 (Hailo-10H) USB accelerator. The code runs on an ordinary laptop today.

## Status

Prototype. Honest state of each part:

| Part | State |
|---|---|
| REBA / RULA scoring, angle geometry, activity and event logic | Done, 338 unit tests, tables checked cell by cell against two independent sources |
| `extract` → `replay` → SQLite → dashboard | Done, verified end to end on recorded footage |
| Live camera path (`run`, `record`) | Implemented, not yet validated in a live scripted session |
| Hailo backend | Written against the pyHailoRT 5.3 API shape, **unverified on hardware** — no UGen300 unit yet |
| Accuracy study | Single-rater pilot check only. No inter-rater agreement study, no certification |

## How it works

```
camera / video ─► pose backend ─► tracker ─► angles ─► REBA + RULA ─► events ─► SQLite ─► web
                  YOLOv8-pose      one         plain     published      enter at   local     LAN
                  (ultralytics     person,     geometry  tables         REBA ≥ 8   file      dashboard
                   · Hailo         smoothed              (data, not     for 3 s,
                   · replay)                             heuristics)    close 3 s below
```

Design rules the code keeps:

- **The tables are data.** REBA and RULA lookups are transcribed from the published worksheets and
  every cell is unit-tested. Totals are never clamped or fudged; an out-of-range score raises.
- **Nothing visual is guessed.** Inputs a camera cannot see — load band, coupling, wrist score, arm
  support — are set per station in a TOML file, so every score is auditable.
- **Missing is not zero.** If an angle cannot be measured, the score is computed from the parts that
  can be, marked `partial`, and the reason is named on the overlay and in the JSON. A side view
  cannot see trunk side-bend or a raised shoulder; those assumptions are listed in `assumed`.
- **A score names its drivers.** Every score carries the per-segment breakdown and the angles behind it.

REBA bands: 1 negligible · 2–3 low · 4–7 medium · 8–10 high · 11–15 very high.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync                              # core: scoring, replay, CLI
uv sync --extra pose --extra web     # + ultralytics (pose inference) and the dashboard
```

The core package has no torch dependency — `replay` scores a keypoints file on any machine.

## Quick start

Score a video clip, then open the dashboard:

```sh
# 1. video -> keypoints (needs the `pose` extra; --device is mps | cuda | cpu)
uv run linesafe extract clip.mp4 --out keypoints.json --device mps

# 2. keypoints -> scores, events and overlay frames
uv run linesafe replay --keypoints keypoints.json --station stations/example.toml \
    --db linesafe.db --out frames.jsonl --events-out events.json --render-dir render

# 3. dashboard for phones on the LAN (needs the `web` extra)
uv run linesafe web --db linesafe.db --port 8080
```

Live camera with the overlay window (`s` saves a frame, `q` or ESC quits):

```sh
uv run linesafe run --source 0 --station stations/example.toml --db linesafe.db
```

## Commands

| Command | What it does |
|---|---|
| `run` | Score a camera or video file live, draw the overlay, write events |
| `replay` | Same pipeline from a keypoints file — no camera, no torch |
| `extract` | Run a pose backend over a video and save a keypoints file |
| `record` | Capture raw camera frames on their measured timeline (no inference) |
| `web` | Serve the LAN dashboard over an event store |

Exit codes: `1` on backend or input-file errors, `2` on a bad station file or bad options.

## Station configuration

One TOML file per workstation — the non-visual scoring inputs, fixed at calibration
(`stations/example.toml`):

```toml
[station]
id = "S1"
reba_load = 1        # 0: <5 kg · 1: 5-10 kg · 2: >10 kg
reba_coupling = 1    # 0 good · 1 fair · 2 poor · 3 unacceptable
reba_wrist = 2       # final REBA wrist score 1-3
rula_wrist = 2       # 1-4
rula_force = 1       # 0-3
# reba_shock, arm_supported, rula_wrist_twist and roi = [x1, y1, x2, y2] are also accepted
```

## Dashboard

`linesafe web` serves one self-contained page (inline CSS and JS, no external URL) plus JSON:

- `GET /api/status` — latest status per station
- `GET /api/events?limit=50&since=<epoch>` — events, newest first
- `GET /api/summary?days=7` — counts over the last N days

## Repo layout

```
src/linesafe/     scoring engine, pose backends, CLI, dashboard
  reba.py rula.py tables.py   the worksheets, as data
  geometry.py keypoints.py    angles from COCO-17 keypoints
  track.py smoothing.py       one-person tracking, One Euro filter
  events.py activity.py       hysteresis and the static/repeat adjustments
  store.py web.py             SQLite events, FastAPI dashboard
  backends/                   ultralytics · hailo (unverified) · replay
stations/         per-station calibration TOML
tests/            338 unit tests + synthetic pose fixtures
deck/             Stage I proposal deck (HTML -> PDF) and its claims ledger
video/            demo-video build (narration, captions, TTS, composition)
docs/             design specs, plans and status notes
```

## Tests

```sh
uv run pytest            # 338 tests, no camera and no torch needed
make video-test          # the video builder's own tests
```

## Deck and demo video

```sh
make pdf     # deck/dist/proposal.pdf via headless Chrome
make check   # every number on a slide must have a row in deck/claims.md
make video   # video/dist/linesafe_stage1.mp4 (needs footage in video/footage/, gitignored)
```

`deck/claims.md` is a claims ledger: no number appears on a slide without a row naming its source and
a grade — official/public, press or paper, or team assumption. Assumptions are marked on the slide too.

## Scoring method

- REBA: Hignett & McAtamney, *Applied Ergonomics* 31 (2000) 201–205.
- RULA: McAtamney & Corlett, *Applied Ergonomics* 24(2) (1993) 91–99.

Only the numeric lookup tables and the published segment boundaries are reproduced, in
`src/linesafe/tables.py`; the worksheets themselves are not redistributed.

## Scope and limitations

This is a research prototype, not a certified assessment tool and not a medical device. It does not
replace a qualified ergonomist. Specifically:

- Single-camera, single-person, side view. Front-view stations are detected and refused loudly rather
  than scored wrongly.
- Wrist scores come from the station config, not from hand keypoints.
- Scoring accuracy has had a single-rater pilot check only — no inter-rater agreement study.
- The Hailo path has never run on real hardware.

## Privacy

Frames are processed in memory and discarded. What persists is the event row — station, time span,
peak score, band, drivers — in a local SQLite file. No video, no images, no identity, no cloud.

## License

MIT — see [LICENSE](LICENSE).
