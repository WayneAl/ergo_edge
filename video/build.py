"""Build the LineSafe Stage I demo video from video/shots.toml: `make video` (or `make video DRAFT=1`)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "video"))

from lsvideo import captions, fonts, render, shots, slides, timeline, tts  # noqa: E402
from lsvideo.model import Clip, Inset, Still, VideoError  # noqa: E402

GUIDE = "docs/evidence/video-shoot-guide.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build(shots_path: Path, draft: bool) -> None:
    build_dir = root / "video/build"

    script = shots.load(shots_path, root)
    print(f"shots: {len(script.shots)} from {rel(shots_path)}, voice {script.voice}")

    pages = sorted({s.visual.page for s in script.shots if isinstance(s.visual, Still)})
    slides.ensure_pngs(root / "deck/dist/proposal.pdf", pages, build_dir / "slides")
    print(f"slides: pages {pages} -> video/build/slides")

    voices = {}
    for shot in script.shots:
        v = tts.voice_for(shot, script.voice, build_dir / "tts")
        if v is not None:
            voices[shot.id] = v

    placed = timeline.place(script.shots, voices, script.max_total_s)
    width = max(len(p.shot.id) for p in placed)
    print(f"{'id':<{width}}  {'t0':>7}  {'dur':>6}  {'narration_s':>11}")
    for p in placed:
        v = voices.get(p.shot.id)
        print(f"{p.shot.id:<{width}}  {p.t0:7.2f}  {p.dur:6.2f}  {(v.dur_s if v else 0.0):11.2f}")
    total = placed[-1].t0 + placed[-1].dur
    print(f"total {total:.2f}s (max {script.max_total_s:.0f}s)")

    missing: dict[str, Path] = {}
    for p in placed:
        visual = p.shot.visual
        if isinstance(visual, Clip):
            paths = [visual.path]
        elif isinstance(visual, Inset):
            paths = [visual.main.path, visual.pip.path]
        else:
            paths = []
        for path in paths:
            if not path.exists():
                if not draft:
                    raise VideoError(f"shot {p.shot.id}: missing footage {rel(path)} — see {GUIDE}, "
                                     f"or build with DRAFT=1")
                missing.setdefault(p.shot.id, path)
    print(f"footage: DRAFT placeholders for {list(missing)}" if missing else "footage: all present")

    # Placeholders draw their label in the deck font, so the TTFs are needed before the parts.
    fonts_dir = fonts.ensure_ttf(root / "deck/fonts", build_dir / "fonts")
    print(f"fonts: {rel(fonts_dir)}")

    parts_dir = build_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    listing: list[str] = []
    for i, p in enumerate(placed, start=1):
        name = f"{i:02d}_{p.shot.id}"
        out = parts_dir / f"{name}.mp4"
        visual = p.shot.visual
        if p.shot.id in missing:
            label = parts_dir / f"{name}.txt"
            label.write_text(f"PLACEHOLDER — {p.shot.id}\n{rel(missing[p.shot.id])}")
            # Relative paths: they sit inside the filtergraph and need no escaping with cwd=video/build.
            cmd = render.placeholder_cmd(Path("parts") / label.name, p.dur, out, Path("fonts/IBMPlexSans-400.ttf"))
            kind = "placeholder"
        elif isinstance(visual, Still):
            cmd, kind = render.still_cmd(visual.png, p.dur, out), f"slide {visual.page}"
        elif isinstance(visual, Clip):
            cmd, kind = render.clip_cmd(visual, p.dur, out), "clip"
        elif isinstance(visual, Inset):
            cmd, kind = render.inset_cmd(visual, p.dur, out), "inset"
        else:
            raise VideoError(f"shot {p.shot.id}: unknown visual {visual!r}")
        render.run(cmd, cwd=build_dir)
        print(f"part {name}: {kind} {p.dur:.2f}s")
        listing.append(f"file 'parts/{name}.mp4'")
    list_file = build_dir / "parts.txt"
    list_file.write_text("\n".join(listing) + "\n")
    video_only = build_dir / "video_only.mp4"
    render.run(render.concat_cmd(list_file, video_only), cwd=build_dir)
    print(f"concat: {len(listing)} parts -> {rel(video_only)}")

    (build_dir / "captions.ass").write_text(captions.to_ass(captions.cues(placed, voices)))
    print("captions: video/build/captions.ass")

    dist = root / "video/dist"
    dist.mkdir(parents=True, exist_ok=True)
    out = dist / ("linesafe_stage1_DRAFT.mp4" if draft else "linesafe_stage1.mp4")
    offsets = [(voices[p.shot.id].mp3, p.t0 + p.shot.lead_s) for p in placed if p.shot.id in voices]
    log = render.run(render.final_cmd(video_only, offsets, "captions.ass", "fonts", total, out), cwd=build_dir)
    try:
        render.check_font(log)
        dur = render.probe_duration(out)
        if dur > script.max_total_s + 0.1:
            raise VideoError(f"{rel(out)}: duration {dur:.2f}s exceeds {script.max_total_s:.0f}s")
    except VideoError:
        out.unlink(missing_ok=True)   # never leave a video that failed its checks where it could be uploaded
        raise
    print(f"wrote {rel(out)} ({dur:.2f}s)")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)   # progress lines stay in order with errors when piped
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", action="store_true", help="render labelled grey placeholders for missing footage")
    parser.add_argument("--shots", default="video/shots.toml", help="shot list, relative to the repo root")
    args = parser.parse_args()
    try:
        build(root / args.shots, args.draft)
    except VideoError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
