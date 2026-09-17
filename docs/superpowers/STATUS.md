# STATUS — ergo_edge

## Plans
- 🟡 DRAFT DONE — Stage I proposal deck: `deck/dist/proposal.pdf` via `make pdf` (18 body + 3 appendix), reviewed and fact-checked. `make check` blocks final only on slide 12's prototype-evidence TODO. No UGen300 before Stage I (finalist unit only); slide 12 cites Hailo's published Hailo-10H FPS.
- 🟡 BUILT, AWAITING LIVE SAMPLE — laptop prototype (sub-project 2): branch `prototype` (worktree `.claude/worktrees/prototype`, head 6ca8b14) — 9 plan tasks done, each reviewed + fix rounds, final whole-branch review fixed; 352 tests; real e2e sample done on a GolfDB clip (extract with ultralytics → replay → SQLite → web API). NOT merged: the camera path (`run`, `record`) needs Task 10 with Wayne — guide `docs/evidence/task10-session-guide.md`. Ledger `.claude/sdd/2026-09-17-laptop-prototype/progress.md`.

## What's next
- Sub-project 1: Stage I proposal deck (English PDF, ≤20 body pages).
- Sub-project 2: laptop prototype — pose (copied from `~/Documents/GitHub/golf_coach`) → REBA/RULA rule engine → live overlay; feeds deck slide 12 and the 3-min video. Target: evidence + video recorded by ~2026-10-10 to leave submission buffer.
- Deadline: Stage I 2026-10-14 17:00 Asia/Taipei. Submission needs deck PDF + signed IP/portrait consent PDF; GitHub link and YouTube video are optional fields but listed in the rules.

## Resumption recipe
1. Read this file, the spec above, and `hackathon-topics.md` Part 0, Part 1, Appendix B.
2. Trusted state on `/clear`: this file + `git log`.
