# Demo video — shoot guide (Wayne)

> Superseded 2026-09-17: no filming. The demo uses `video/site_replay.sh` and `video/dashboard_view.sh` instead.

Same sitting as the Task 10 session, right after step 4. About 15 minutes. You need the Task 10 setup (webcam, box,
side-on space), the phone on a stand or propped up, and two terminals: one in the prototype worktree for `linesafe run`,
one in `~/Documents/GitHub/ergo_edge/.claude/worktrees/video` for `video/capture.sh`.

## 0. Setup (once)
- System Settings → Privacy & Security → Screen Recording: allow your terminal app, then quit and reopen it.
- Do Not Disturb on (laptop and phone). Close every other window: mail, chat, browser, Finder.
- Only the `LineSafe` window on screen, as large as it goes. Nothing identifying: no terminal history, Wi-Fi name, LAN
  IP, notifications or desktop files in view. Put the capture terminal behind the `LineSafe` window.
- Prop the phone for a side-on wide shot that shows you, the box and the laptop screen in one frame. Landscape.

## 1. Every take starts the same way
1. Phone: start recording. 2. Laptop: start `video/capture.sh <name> <seconds>` and bring `LineSafe` to the front.
3. **One clap in view** — both files see it, so Claude can line them up with `in_s` / `pip_in_s`. 4. Then act.

## 2. Take 1 — the lift (`overlay_lift` + `phone_lift.mp4`)
```bash
uv run linesafe run --source 0 --station stations/example.toml        # prototype terminal
video/capture.sh overlay_lift 75                                       # video terminal, 75 s
```
After the clap: stand upright 12 s → walk to the box → bend and lift, and **hold the bend ~10 s** so the score reaches
High and the alert fires → stand up with the box → put it down. Keep going until the capture stops.

## 3. Take 2 — offline (`overlay_offline` + `phone_offline.mp4`)
```bash
video/capture.sh overlay_offline 30
```
After the clap: turn Wi-Fi off on the laptop **in the phone shot only** (reach over to the laptop so the phone sees it;
the screen capture must not show the Wi-Fi menu or its network name). Then bend a few times so the score visibly keeps
updating. When the capture stops, turn Wi-Fi back on.

## 4. Take 3 — dashboard (`phone_dashboard.mp4`)
Start `linesafe web` as in Task 10 step 4. On the phone, start a screen recording, open the dashboard, scroll through the
events slowly, stop. The address bar shows the LAN IP — that is fine, Claude crops it out with `crop` in `shots.toml`.

## 5. Files
- `video/capture.sh` already wrote `video/footage/overlay_lift.mp4` and `overlay_offline.mp4`.
- AirDrop the three phone videos to the laptop and move them into `video/footage/` as `phone_lift.mp4`,
  `phone_offline.mp4` and `phone_dashboard.mp4` (renaming a `.mov` to `.mp4` is fine).
- Never commit anything in `video/footage/` (it is gitignored).

## Hand-off
Tell Claude. Claude then tunes `in_s` / `out_s` / `crop` in `video/shots.toml`, adjusts any narration line that does not
match what the overlay actually shows, runs `make video`, and watches the result end to end before you upload it.
