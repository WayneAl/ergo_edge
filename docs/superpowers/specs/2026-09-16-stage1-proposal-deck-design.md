# Stage I proposal deck — design

Context: ASUS UGen AI League, Battlefield Lightning (UGen300 / Hailo-10H), theme Workplace AI. Topic = Part 1
of `hackathon-topics.md` (REBA/RULA automation). Stage I closes 2026-10-14 17:00 Asia/Taipei; one submission
per team, the site states no edit window. Judging: innovation 30 · business potential & feasibility 35 ·
practical value 25 · storytelling 10.

## Goal / non-goals
Goal: English PDF (≤20 body pages + appendix, ≤15 MB) covering the five required sections — problem, solution
& methodology, HW/SW architecture, expected benefits, references + GitHub link — built from source in this repo.
Non-goals: prototype code and the 3-min video (sub-project 2, own spec); filling the submission form.

## Data structures
- `deck/slides.html` + `deck/deck.css` — one `<section class="slide">` per page at 1920×1080, inline SVG
  diagrams, fonts vendored locally. Idea: the deck is code — diffable, deterministic rebuild, fast iteration.
- `deck/claims.md` — ledger of every number on a slide: value · source URL · grade 🟢/🟡/🔴 · slide #.
  Invariant: no number on a slide without a row; 🔴 rows carry a visible "team estimate" label on the slide.
- Storyline (18 body pages): 1 Title · 2 Problem (paper REBA/RULA spot checks: subjective, sampled, no
  history) · 3 Why now (OSH Act ergonomic duty, insurance experience rating, labour shortage) · 4 Who pays
  (EHS buyer, budget line) · 5 Solution in one picture · 6 Methodology (pose → joint angles → REBA/RULA tables
  → activity score over time → events) · 7 Rules, not an LLM, on the live path · 8 Deployment (camera → host
  + UGen300 → LAN → phone dashboard, fully offline) · 9 Software architecture · 10 Why UGen300 (2.5 W
  always-on, ready `.hef` models, per-stream FPS budget) · 11 Privacy by design (skeleton only, no raw video,
  no face ID, station-level events) · 12 Feasibility evidence (prototype numbers) · 13 Validation plan (FPS
  ≥25, Cohen's κ ≥0.7 vs 2 raters, 7×24 run) · 14 ROI: 3 scenarios + 50 % sensitivity · 15 Competition &
  differentiation · 16 Stage II roadmap + risks · 17 Team · 18 References + GitHub.
  Appendix: COCO-17 → REBA/RULA angle mapping, ROI formulas, claim grades.

## Interfaces
- `make pdf` → headless Chrome `--print-to-pdf`, `@page { size: 1920px 1080px; margin: 0 }` → `deck/dist/proposal.pdf`.
- `make check` → PDF pages == `.slide` count (catches overflow), body pages ≤20, size ≤15 MB, no CJK glyphs,
  no `TODO` left; `make png` → one PNG per page for visual review.

## Invariants & failure modes
- A slide that overflows prints as an extra page → `make check` fails with expected vs actual page count.
- English only (contest rule) → CJK check fails loud; Chinese stays in `claims.md` notes only.
- Technical honesty on slides 6/12: 2D COCO-17 cannot see wrist flexion, load or coupling. Wrist → crop +
  `hand_landmark_lite` (has a Hailo `.hef`) or a station default; load/coupling → set per station at
  calibration; trunk twist → shoulder/hip width ratio (golf_coach turn proxy); 2D foreshortening → camera
  placement guideline + κ validation. Stated as limits, not hidden.
- ASUS-specific claims (e.g. an ASUS line as pilot site) go on a slide only after a fact check; until then
  they are phrased as the Stage II ask.
- Slide 12 depends on sub-project 2; it carries a visible TODO so `make check` blocks a final build without it.

## Verification
`make check` green on the final build; every page PNG reviewed for overflow and legibility at 50 % zoom;
one pass mapping each rubric criterion → the slides that earn it.

## Open decisions
A. Toolchain: HTML/CSS → Chrome PDF (rec.) vs Marp Markdown vs Google Slides (hand-edited).
B. Evidence: build the laptop prototype before locking the deck, so slide 12 shows real FPS and
   REBA-vs-manual numbers (rec.; the video needs it anyway) vs a deck with a plan only.
C. Reuse: copy golf_coach's pose front half (backends incl. Hailo adapter, smoothing, keypoints, gap-fill)
   into this repo (rec.) vs a shared package.
D. ASUS angle: fact-check ASUS's own manufacturing footprint, then pitch an ASUS line pilot as the Stage II ask.
E. Product name (placeholder "ErgoEdge").
F. Submission form: Onsite track (rec.) vs Global Online; UGen300 USB (rec.) vs M.2 — locked after submit.
