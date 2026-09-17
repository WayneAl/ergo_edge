from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .model import Clip, Inset, VideoError

FFMPEG = ["ffmpeg", "-hide_banner", "-y"]
# Every part is encoded identically so `concat -c copy` can join them.
PART_ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "30", "-an"]
FIT = ("scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
       "setsar=1,fps=30,format=yuv420p")


def _s(seconds: float) -> str:
    # ffmpeg truncates time strings to microseconds, so 9 decimals keep a frame-exact duration from rounding up
    # into the next frame (".6f" would turn 362/30 s into 12.066667 and admit a 363rd frame).
    return f"{seconds:.9f}"


def _crop(clip: Clip) -> str:
    if clip.crop is None:
        return ""
    x, y, w, h = clip.crop
    return f"crop={w}:{h}:{x}:{y},"


def still_cmd(png: Path, dur: float, out: Path, pad_color: str) -> list[str]:
    # The slide shrinks to 1680x945 near the top so the caption box sits in a band of the slide's own colour below it.
    vf = f"scale=1680:945,pad=1920:1080:120:10:color={pad_color},setsar=1,fps=30,format=yuv420p"
    return FFMPEG + ["-loop", "1", "-framerate", "30", "-i", str(png), "-t", _s(dur), "-vf", vf] + PART_ENC + [str(out)]


def clip_cmd(clip: Clip, dur: float, out: Path) -> list[str]:
    return FFMPEG + ["-ss", _s(clip.in_s), "-i", str(clip.path), "-t", _s(dur), "-vf", _crop(clip) + FIT] + PART_ENC + [
        str(out)]


def inset_cmd(inset: Inset, dur: float, out: Path) -> list[str]:
    graph = (f"[0:v]{_crop(inset.main)}{FIT}[m];"
             f"[1:v]{_crop(inset.pip)}scale=560:360:force_original_aspect_ratio=decrease,"
             f"pad=iw+8:ih+8:4:4:color=white,fps=30[p];"
             f"[m][p]overlay=40:40:shortest=1,format=yuv420p[v]")
    return FFMPEG + ["-ss", _s(inset.main.in_s), "-i", str(inset.main.path),
                     "-ss", _s(inset.pip.in_s), "-i", str(inset.pip.path),
                     "-t", _s(dur), "-filter_complex", graph, "-map", "[v]"] + PART_ENC + [str(out)]


def placeholder_cmd(label_file: Path, dur: float, out: Path, fontfile: Path) -> list[str]:
    # label_file and fontfile go inside the filtergraph unescaped: pass paths relative to the run's cwd.
    vf = (f"drawtext=fontfile={fontfile}:textfile={label_file}:fontcolor=white:fontsize=64:"
          f"x=(w-text_w)/2:y=(h-text_h)/2,format=yuv420p")
    return FFMPEG + ["-f", "lavfi", "-i", f"color=c=0x404040:s=1920x1080:r=30:d={_s(dur)}", "-vf", vf] + PART_ENC + [
        str(out)]


def concat_cmd(list_file: Path, out: Path) -> list[str]:
    return FFMPEG + ["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(out)]


def final_cmd(video: Path, voices: Sequence[tuple[Path, float]], ass_name: str, fonts_dir_name: str, total: float,
              out: Path) -> list[str]:
    """Place each narration at its offset, burn the captions. Run with cwd=video/build so ass/fonts need no escaping."""
    cmd = FFMPEG + ["-i", str(video)]
    graph: list[str] = []
    for i, (mp3, offset) in enumerate(voices, start=1):
        cmd += ["-i", str(mp3)]
        graph.append(f"[{i}:a]adelay=delays={round(offset * 1000)}:all=1[a{i}]")
    if voices:
        labels = "".join(f"[a{i}]" for i in range(1, len(voices) + 1))
        graph.append(f"{labels}amix=inputs={len(voices)}:normalize=0:dropout_transition=0,apad[a]")
    else:
        graph.append("anullsrc=r=48000:cl=mono[a]")
    graph.append(f"[0:v]subtitles=filename={ass_name}:fontsdir={fonts_dir_name}[v]")
    return cmd + ["-filter_complex", ";".join(graph), "-map", "[v]", "-map", "[a]", "-t", _s(total),
                  "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart",
                  "-loglevel", "verbose", str(out)]


def run(cmd: list[str], cwd: Path | None = None) -> str:
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise VideoError(f"{cmd[0]} not found — install ffmpeg (brew install ffmpeg)") from e
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.splitlines()[-20:])
        raise VideoError("ffmpeg failed: " + " ".join(cmd[:6]) + " …\n" + tail)
    return proc.stderr


def probe_duration(path: Path) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise VideoError("ffprobe not found — install ffmpeg (brew install ffmpeg)") from e
    try:
        if proc.returncode != 0:
            raise ValueError(proc.stderr.strip())
        return float(proc.stdout.strip())
    except ValueError as e:
        raise VideoError(f"ffprobe cannot read the duration of {path}: {e}") from e


def check_font(log: str, family: str = "IBM Plex Sans", expect_prefix: str = "IBMPlexSans") -> None:
    """libass silently falls back to a system font; fail unless every fontselect for `family` resolved to it."""
    targets = re.findall(r"fontselect: \(" + re.escape(family) + r", [^)]*\) -> ([^,\n]+),", log)
    if not targets:
        raise VideoError(f"no fontselect line for {family} in the ffmpeg log")
    for target in targets:
        if not target.startswith(expect_prefix):
            raise VideoError(f"libass fell back to {target} for {family}")
