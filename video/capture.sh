#!/usr/bin/env bash
# Screen capture of the live overlay for the demo video: video/capture.sh <name> <seconds>
set -euo pipefail
name=${1:?usage: video/capture.sh <name> <seconds>}
seconds=${2:?usage: video/capture.sh <name> <seconds>}
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/video/footage"
out="$root/video/footage/$name.mp4"
devices=$(ffmpeg -hide_banner -f avfoundation -list_devices true -i "" 2>&1 || true)
idx=$(printf '%s\n' "$devices" | sed -n 's/.*\[\([0-9][0-9]*\)\] Capture screen 0.*/\1/p' | head -1)
if [ -z "$idx" ]; then
  echo "no 'Capture screen 0' device — allow Screen Recording for this terminal in System Settings" >&2
  exit 1
fi
ffmpeg -hide_banner -y -f avfoundation -capture_cursor 0 -framerate 30 -i "${idx}:none" -t "$seconds" \
  -c:v libx264 -preset ultrafast -crf 18 -pix_fmt yuv420p "$out"
echo "$out"
