#!/usr/bin/env bash
# Footage for the `dashboard` shot: the real LAN dashboard over the site replay's events, rendered at phone width by
# headless Chrome and scrolled from "Today" to "This week".
# Usage: video/dashboard_view.sh   (run video/site_replay.sh first: it caches the keypoints)
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
work="$root/video/build/site"
out="$root/video/footage/dashboard_phone.mp4"
linesafe=${LINESAFE:-$root/.claude/worktrees/prototype/.venv/bin/linesafe}
station=${STATION:-$root/.claude/worktrees/prototype/stations/example.toml}
chrome=${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}
port=${PORT:-8765}
length=${LENGTH:-22}

[ -f "$work/keypoints.json" ] || { echo "missing $work/keypoints.json — run video/site_replay.sh first" >&2; exit 1; }
rm -f "$work/dash.db"
(cd "$work" && "$linesafe" replay --keypoints keypoints.json --station "$station" --db dash.db)

"$linesafe" web --db "$work/dash.db" --host 127.0.0.1 --port "$port" >"$work/web.log" 2>&1 &
web_pid=$!
trap 'kill "$web_pid" 2>/dev/null || true' EXIT
for _ in $(seq 1 40); do curl -sf -o /dev/null "http://127.0.0.1:$port/api/status" && break; sleep 0.25; done
curl -sf -o /dev/null "http://127.0.0.1:$port/api/status" || { echo "dashboard did not start; see $work/web.log" >&2; exit 1; }

# 500 CSS px is the narrowest window headless Chrome lays out; still the single-column phone layout (< 640 px).
"$chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=500,1500 --force-device-scale-factor=3 \
  --virtual-time-budget=4000 --screenshot="$work/dash_full.png" "http://127.0.0.1:$port/" 2>/dev/null
[ -f "$work/dash_full.png" ] || { echo "no screenshot written" >&2; exit 1; }

# Phone viewport 500×620 CSS px (1500×1860 image px), starting at "Today" (y 880), scrolled 1400 px from 3 s to 15 s.
ffmpeg -hide_banner -loglevel error -y -loop 1 -framerate 30 -i "$work/dash_full.png" -t "$length" -filter_complex "\
[0:v]crop=1500:1860:0:'880+1400*min(max((t-3)/12\,0)\,1)',scale=704:873,pad=712:881:4:4:color=0x3a3f45,\
pad=1920:1080:604:20:color=0x111418,format=yuv420p" -r 30 -c:v libx264 -crf 18 -pix_fmt yuv420p "$out"
echo "$out"
