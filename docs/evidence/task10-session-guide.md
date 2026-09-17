# Task 10 — live session guide (Wayne)

Everything runs from the prototype worktree on the laptop. About 30 minutes including setup. You need: the webcam,
a box of 5–10 kg, a clear space where the camera sees your whole body from the SIDE at 2–3 m, and your phone on the
same Wi-Fi as the laptop.

## 0. Setup (once)
```bash
cd ~/Documents/GitHub/ergo_edge/.claude/worktrees/prototype
uv sync --all-extras
ln -s ~/Documents/GitHub/golf_coach/yolov8m-pose.pt .      # reuse the downloaded weights (gitignored)
```
macOS will ask for camera permission for your terminal the first time — allow it, then re-run the command.

## 1. Camera check and FPS (2 min)
```bash
uv run linesafe run --source 0 --station stations/example.toml
```
Stand side-on, whole body in frame. Check the skeleton follows you and read the FPS (top right). Press `q` to quit.
Tell Claude the steady FPS.

## 2. Scripted recording (2 min) — becomes the golden test fixture
```bash
uv run linesafe record --source 0 --out session.mp4 --seconds 90
```
Side-on, in this order (count in your head): upright 10 s → bend and pick up the box 10 s → hold bent with the box 10 s →
upright holding the box 10 s → arms raised overhead 10 s → squat 10 s → stand on one leg 5 s → upright until it stops.

## 3. Pilot check (10 min) — 20 held postures
```bash
uv run linesafe run --source 0 --station stations/example.toml
```
For each posture: get into it, hold still, press `s`, then move on. Each press saves `pilot/*_raw.png` (no overlay — this is
the one you will measure), the overlay PNG, and a JSON with the system's angles and scores.
1. upright · 2. trunk bent ~20° · 3. trunk bent ~45° · 4. trunk bent ~70° · 5. trunk leaning back ~15° ·
6. head bowed ~30° · 7. head tilted back · 8. arm forward ~45° · 9. arm forward ~90° · 10. arm forward ~135° ·
11. arm straight overhead · 12. elbow bent 90°, upper arm hanging · 13. half squat (knee ~30°) · 14. deep squat (knee ~60°+) ·
15. one-leg stance · 16. bent ~45° reaching forward at waist height · 17. bent ~45° with arm at 90° · 18. bent + twisted ·
19. reaching forward at shoulder height · 20. picking the box up off the floor
Measuring the raw PNGs: Claude will give you a click-to-measure page (click three points per angle) so every angle is
measured the same way.

## 4. Dashboard on the phone (3 min)
While step 3's `run` is going (or right after — it reads the same `linesafe.db`), in a second terminal:
```bash
cd ~/Documents/GitHub/ergo_edge/.claude/worktrees/prototype
ipconfig getifaddr en0                      # the laptop's LAN IP
uv run linesafe web --db linesafe.db --port 8080
```
On the phone open `http://<that IP>:8080` and take a screenshot. Stop the server with Ctrl-C.

## Hand-off
Tell Claude when done. Claude then: extracts keypoints from `session.mp4`, commits the golden fixture and test, analyses the
pilot check into `docs/evidence/2026-10-pilot-check.md`, fills deck slide 12, and merges the prototype branch.
Do not commit `session.mp4`, `pilot/` or `linesafe.db` (they are gitignored).
