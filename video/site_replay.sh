#!/usr/bin/env bash
# Footage for the `site_replay` shot: the pose overlay on public construction-site footage, fading to skeleton only.
# Usage: video/site_replay.sh
# Needs video/footage/site_source.mp4, the prototype's linesafe CLI (LINESAFE) and the caption fonts from `make video`.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
src="$root/video/footage/site_source.mp4"
out="$root/video/footage/site_replay.mp4"
work="$root/video/build/site"
fonts="$root/video/build/fonts"
linesafe=${LINESAFE:-$root/.claude/worktrees/prototype/.venv/bin/linesafe}
station=${STATION:-$root/.claude/worktrees/prototype/stations/example.toml}
weights=${WEIGHTS:-$HOME/Documents/GitHub/golf_coach/yolov8m-pose.pt}
start=${START:-41}      # excerpt start in the source, seconds
length=${LENGTH:-18}    # excerpt length, seconds
fade_at=${FADE_AT:-5}   # camera image fades out here, over 1.5 s

[ -f "$src" ] || { echo "missing $src" >&2; exit 1; }
[ -x "$linesafe" ] || { echo "missing linesafe CLI at $linesafe (set LINESAFE)" >&2; exit 1; }
[ -f "$fonts/IBMPlexSans-500.ttf" ] || { echo "missing $fonts/IBMPlexSans-500.ttf — run make video DRAFT=1 once" >&2; exit 1; }
mkdir -p "$work"

# Pose over the whole source, so the activity score has the same history it would have live. Cached.
if [ ! -f "$work/keypoints.json" ]; then
  ln -sf "$weights" "$work/yolov8m-pose.pt"
  (cd "$work" && "$linesafe" extract "$src" --out keypoints.json --device mps)
fi
rm -rf "$work/render"
(cd "$work" && "$linesafe" replay --keypoints keypoints.json --station "$station" \
  --render-dir render --render-every 1 --events-out events.json)

fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$src")
first=$(python3 -c "from fractions import Fraction as F; print(round($start * F('$fps')))")

# The replay HUD shows the file's frame rate, not live throughput: black it out (colorkey then removes it).
hide_fps="drawbox=x=1150:y=25:w=115:h=32:color=black:t=fill"
label="drawtext=fontfile=$fonts/IBMPlexSans-500.ttf:text='Offline replay of public construction-site footage':fontcolor=white:fontsize=34:x=(w-text_w)/2:y=46:box=1:boxcolor=black@0.55:boxborderw=12"
ffmpeg -hide_banner -loglevel error -y \
  -ss "$start" -t "$length" -i "$src" \
  -framerate "$fps" -start_number "$first" -t "$length" -i "$work/render/frame_%05d.png" \
  -filter_complex "\
[1:v]$hide_fps,scale=1640:922,setsar=1,split[s1][s2];\
[s1]colorkey=black:0.08:0.0[k];\
[0:v]scale=1640:922,setsar=1,eq=brightness=-0.12:saturation=0.8[v];\
[v][k]overlay=0:0,format=yuv420p,trim=0:$(python3 -c "print($fade_at + 1.5)"),setpts=PTS-STARTPTS,settb=AVTB[ov];\
[s2]format=yuv420p,trim=start=$fade_at,setpts=PTS-STARTPTS,settb=AVTB[sk];\
[ov][sk]xfade=transition=fade:duration=1.5:offset=$fade_at,pad=1920:1080:140:10:color=black,$label,format=yuv420p" \
  -r 30 -c:v libx264 -crf 18 -pix_fmt yuv420p -an "$out"
echo "$out"
