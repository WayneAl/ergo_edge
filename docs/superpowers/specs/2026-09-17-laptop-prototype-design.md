# LineSafe laptop prototype (sub-project 2) — design

Context: Stage I closes 2026-10-14; no UGen300 until the finalist unit ships (26 Oct). This prototype supplies deck
slide 12's own evidence and the 3-min video. Code in `src/linesafe/`, uv, Python 3.11 (same stack as golf_coach).

## Goal / non-goals
Goal: `linesafe run --source 0 --station S1` → live REBA (and RULA) per frame from a side-view webcam, overlay
window with skeleton + angles + REBA ruler + alert, High events to SQLite, a LAN web page a phone can open.
Non-goals: running on Hailo (adapter copied, unverified), wrist from hand keypoints, multi-person stations, LLM
weekly report (static mock page only), installer, 3D.

## Data structures
- Reused from golf_coach, copied not imported: `Detection`, `PoseBackend` protocol (`infer(frame_bgr, idx) ->
  list[Detection]`; ultralytics / replay / hailo), COCO-17 constants, `OneEuroFilter` (causal, fine live).
  golf's `select_person` and `fill_gaps` look at future frames → rewritten as a streaming `Tracker`.
- `PoseFrame(t, kpts[17,2], conf[17], bbox)` — one tracked person, smoothed.
- `Angles(t, view, trunk_flex, trunk_side, trunk_twist, neck_flex, upper_arm[L,R], shoulder_raised[L,R],
  lower_arm[L,R], knee[L,R], legs_bilateral)` — each a `Measured(deg, conf)` or `Missing(reason)`. Trick: the
  view (side vs front, golf's shoulder-width/torso ratio) decides which angles are measurable; side view gives
  sagittal flexion, twist is a width-ratio proxy.
- `StationConfig(id, load_band, coupling, wrist_score, arm_supported, roi)` — the inputs a camera cannot see,
  set at calibration (TOML). Idea: everything non-visual is explicit and auditable, never a hidden default.
- `RebaScore(t, total 1–15, band, table_a, table_b, parts{segment: score}, drivers[str], partial: bool,
  missing[str])` and `RulaScore(t, total 1–7, …)`. Worse arm side is scored.
- `Event(station, t_start, t_end, peak, band, drivers, duration_s)` — SQLite row.

## Interfaces (pure core; I/O only in capture, backend, store, ui)
`capture.frames(source) -> Iterator[(t, frame)]` → `backend.infer` → `Tracker.update(t, dets) -> PoseFrame|None`
→ `angles.compute(pose) -> Angles` → `reba.score(angles, cfg) -> RebaScore` (+ `rula.score`) →
`Activity.update(angles) -> flags` (REBA: +1 static > 1 min, +1 repeat > 4/min, +1 rapid change;
RULA muscle use: static > 10 min or repeat ≥ 4/min — both worksheets agree) →
`Events.update(score) -> Event|None` (enter High ≥ 3 s, leave below High ≥ 3 s) → `store.add(event)`;
`ui.overlay.draw(frame, pose, angles, score)`; `web` serves events JSON + one static page on the LAN.
CLI: `linesafe run`, `linesafe replay --keypoints f.json` (same pipeline, no camera), `linesafe web`.

## Invariants & failure modes
- REBA/RULA tables are data, transcribed once from the 2000/1993 papers; totals always in range or raise.
- A needed angle Missing → score still computed from available parts, `partial=True`, reason shown on the
  overlay; never a silent zero. No person in the ROI → no score (not score 1).
- Front view detected on a side-view station → loud banner "wrong camera view", flexion angles Missing.
- Tracker loses the person > 0.5 s → re-seed on the largest bbox with confident shoulders inside the ROI.
- End-to-end FPS logged every second; below 15 FPS the overlay says so (video must not look fake-smooth).

## Verification
- Unit: every table cell lookup; angle geometry on hand-built poses with known degrees; activity and event
  hysteresis on synthetic score sequences; tracker re-seed.
- Golden: one recorded scripted session → keypoints JSON → `replay` → committed scores/events (no torch in CI).
- Live sample (gate 2): 20 scripted postures held for 5 s, side view; angles measured on the frame with an
  on-screen protractor → per-angle error; manual REBA from the same frames vs engine → exact-match rate.
  Single, uncertified rater: labelled "pilot check" on slide 12; the κ study stays Stage II.
- End-to-end FPS on the M4 Pro with a 1080p webcam.

## Decisions (locked 2026-09-17, Wayne: 全照建議)
A. Side-view single camera. B. REBA first, RULA after events. C. Wrist = station default.
D. FastAPI + one static page. E. Pilot check rated by Wayne on photo-measured angles, labelled "pilot check".
Tables: verified cell by cell against two independent sources each → `tests/fixtures/worksheet_tables.py`.
Amended: events close after 3 s below High (below-Medium would keep a High event open through long Medium work).
