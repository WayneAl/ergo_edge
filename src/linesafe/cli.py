"""LineSafe command line.

``record`` captures raw camera frames on their measured timeline, ``run`` scores a live
camera (or a video file) and shows the overlay window, ``replay`` runs the same pipeline from
a keypoints file (no camera, no torch), ``extract`` turns a video into a keypoints file,
``web`` serves the LAN dashboard over the event store. Heavy imports happen inside the
commands; OpenCV windows are opened only by ``run``.

Exit codes: 1 on backend or input-file errors, 2 on a bad station file or bad options.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Iterable
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    import numpy as np

app = typer.Typer(no_args_is_help=True)

LIVE_BACKENDS = ("ultralytics", "hailo")
FPS_EMA_ALPHA = 0.1
WINDOW = "LineSafe"
PILOT_DIR = Path("pilot")
TIMESTAMPS_SUFFIX = ".timestamps.json"


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _load_station(path: Path):
    from .config import load_station

    try:
        return load_station(path)
    except (OSError, ValueError) as exc:  # tomllib.TOMLDecodeError is a ValueError
        raise _fail(f"bad station file {path}: {exc}", 2) from exc


def _check_out_writable(out: Path) -> None:
    """Exit 1 naming ``--out`` unless ``out`` can be written: not a directory, not read-only, and its
    nearest existing ancestor is a writable directory."""
    try:
        if out.is_dir() or (out.exists() and not os.access(out, os.W_OK)):
            raise _fail(f"--out {out} is not a writable file", 1)
        parent = out.parent
        while not parent.exists():  # save_keypoints creates the missing directories
            parent = parent.parent
        writable = parent.is_dir() and os.access(parent, os.W_OK | os.X_OK)
    except OSError as exc:  # e.g. PermissionError from exists() under an unreadable directory
        raise _fail(f"cannot check --out {out}: {exc}", 1) from exc
    if not writable:
        raise _fail(f"--out {out}: {parent} is not a writable directory", 1)


def _make_live_backend(name: str, device: str | None):
    if name not in LIVE_BACKENDS:
        raise _fail(f"--backend must be one of {', '.join(LIVE_BACKENDS)}, got {name!r}", 2)
    if device is not None and name != "ultralytics":
        raise _fail("--device applies to --backend ultralytics only", 2)
    from .backends.base import make_backend

    kwargs = {"device": device} if device is not None else {}
    try:
        return make_backend(name, **kwargs)
    except ImportError as exc:
        raise _fail(f"backend {name} unavailable: {exc} (install the pose extra: uv sync --extra pose)", 1) from exc
    except Exception as exc:
        raise _fail(f"backend {name} failed to start: {exc}", 1) from exc


def _angle_json(angle):
    from .geometry import Measured

    return round(angle.deg, 1) if isinstance(angle, Measured) else angle.reason


def _flag_json(value):
    from .geometry import Missing

    return value.reason if isinstance(value, Missing) else value


def _angles_json(angles) -> dict | None:
    if angles is None:
        return None
    return {
        "view": angles.view.value,
        "facing": angles.facing,
        "trunk_flex": _angle_json(angles.trunk_flex),
        "trunk_twisted": _flag_json(angles.trunk_twisted),
        "neck_flex": _angle_json(angles.neck_flex),
        "upper_arm_left": _angle_json(angles.upper_arm[0]),
        "upper_arm_right": _angle_json(angles.upper_arm[1]),
        "lower_arm_left": _angle_json(angles.lower_arm[0]),
        "lower_arm_right": _angle_json(angles.lower_arm[1]),
        "knee_left": _angle_json(angles.knee[0]),
        "knee_right": _angle_json(angles.knee[1]),
        "legs_bilateral": _flag_json(angles.legs_bilateral),
    }


def _event_json(e) -> dict:
    return {
        "station": e.station,
        "t_start": e.t_start,
        "t_end": e.t_end,
        "peak": e.peak,
        "band": e.band.value,
        "drivers": list(e.drivers),
        "duration_s": e.duration_s,
    }


def _save_pilot(raw, drawn, result, cfg, wall_offset: float, out_dir: Path = PILOT_DIR) -> Path | None:
    """Key ``s``: the raw frame (the pilot rater measures on it), the drawn frame, angles and scores.

    Writes ``<stem>_raw.png``, ``<stem>.png`` and ``<stem>.json`` and returns ``out_dir / stem``.
    A failed save is reported on stderr and returns ``None``, so a live session keeps going.
    """
    import cv2

    now = datetime.now()
    station = re.sub(r"[^A-Za-z0-9_-]", "_", cfg.station_id)
    stem = out_dir / f"{station}_{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}"
    reba, rula = result.reba, result.rula
    payload = {
        "t": result.t + wall_offset,
        "angles": _angles_json(result.angles),
        "reba": None
        if reba is None
        else {
            "total": reba.total,
            "band": reba.band.value,
            "parts": reba.parts,
            "partial": reba.partial,
            "missing": list(reba.missing),
        },
        "rula": None
        if rula is None
        else {
            "total": rula.total,
            "level": rula.level.value,
            "parts": rula.parts,
            "partial": rula.partial,
            "missing": list(rula.missing),
        },
    }
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for suffix, image in (("_raw.png", raw), (".png", drawn)):
            path = stem.with_name(stem.name + suffix)
            if not cv2.imwrite(str(path), image):
                raise OSError(f"could not write {path}")
        stem.with_name(stem.name + ".json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, cv2.error) as exc:
        typer.echo(f"pilot save failed: {exc}", err=True)
        return None
    typer.echo(f"saved {stem}_raw.png, {stem.name}.png, {stem.name}.json")
    return stem


def write_timed_video(frames: Iterable[tuple[float, np.ndarray]], out: Path) -> tuple[int, float]:
    """Write ``(t, frame)`` pairs as an mp4 at the measured rate, plus a timestamps sidecar.

    The rate is unknown until the last frame, so frames first go to a Motion-JPEG
    ``<out>.part.avi`` (bounded disk, not memory), which is then re-encoded to ``out`` (mp4v)
    at ``(n - 1) / (t_last - t_first)`` fps. ``<out>.timestamps.json`` lists every frame's
    time in seconds from the first frame. Returns ``(n, fps)``.

    Raises ``ValueError`` on no frames, a single frame, non-finite or non-increasing times,
    or a frame that is not uint8 BGR or changes shape. Nothing half-written is left behind.
    """
    import cv2
    import numpy as np

    out = Path(out)
    part = out.with_name(out.name + ".part.avi")
    sidecar = out.with_name(out.name + TIMESTAMPS_SUFFIX)
    times: list[float] = []
    shape: tuple[int, ...] | None = None
    started_out = False
    try:
        writer = None
        try:
            for t, frame in frames:
                t = float(t)
                if not math.isfinite(t):
                    raise ValueError(f"frame time must be finite, got {t}")
                if times and t <= times[-1]:
                    raise ValueError(f"frame times must strictly increase: {times[-1]} -> {t}")
                if not (isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3
                        and frame.shape[2] == 3):
                    raise ValueError("frame must be a uint8 array of shape (height, width, 3)")
                if shape is None:
                    shape = frame.shape
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (shape[1], shape[0]))
                    if not writer.isOpened():
                        raise OSError(f"cannot open {part} for writing")
                elif frame.shape != shape:
                    raise ValueError(f"frame shape changed from {shape} to {frame.shape}")
                writer.write(frame)
                times.append(t)
        finally:
            if writer is not None:
                writer.release()
        if not times:
            raise ValueError("no frames to write")
        if len(times) < 2:
            raise ValueError(f"need at least 2 frames to measure the rate, got {len(times)}")
        fps = (len(times) - 1) / (times[-1] - times[0])

        started_out = True
        cap = cv2.VideoCapture(str(part))
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shape[1], shape[0]))
        copied = 0
        try:
            if not cap.isOpened():
                raise OSError(f"cannot read back {part}")
            if not writer.isOpened():
                raise OSError(f"cannot open {out} for writing")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                copied += 1
        finally:
            cap.release()
            writer.release()
        if copied != len(times):
            raise OSError(f"re-encoded {copied} of {len(times)} frames into {out}")
        sidecar.write_text(json.dumps([round(t - times[0], 6) for t in times]) + "\n", encoding="utf-8")
    except BaseException:
        if started_out:
            out.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
        raise
    finally:
        part.unlink(missing_ok=True)
    return len(times), fps


@app.command()
def record(
    source: str = typer.Option("0", help="Camera index."),
    out: Path = typer.Option(..., help="mp4 to write; frame times go to <out>.timestamps.json."),
    seconds: float = typer.Option(..., help="How long to record."),
) -> None:
    """Record raw camera frames (no pose inference) on their measured timeline."""
    if not source.isdigit():
        raise _fail(f"--source must be a camera index, got {source!r}", 2)
    if not (math.isfinite(seconds) and seconds > 0):
        raise _fail(f"--seconds must be > 0, got {seconds}", 2)
    import cv2

    with ExitStack() as stack:
        cap = cv2.VideoCapture(int(source))
        stack.callback(cap.release)
        if not cap.isOpened():
            raise _fail(f"cannot open camera {source}", 1)

        def frames():
            t_stop = last = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                t = time.monotonic()
                if last is not None and t <= last:
                    t = last + 1e-6  # times must strictly increase
                if t_stop is None:
                    t_stop = t + seconds
                elif t > t_stop:
                    return
                last = t
                yield t, frame

        try:
            n, fps = write_timed_video(frames(), out)
        except (OSError, ValueError) as exc:
            raise _fail(f"record from camera {source} failed: {exc}", 1) from exc
    typer.echo(f"frames={n} fps={fps:.3f} -> {out} (+ {out.name}{TIMESTAMPS_SUFFIX})")


@app.command()
def run(
    source: str = typer.Option("0", help="Camera index or video path."),
    station: Path = typer.Option(..., help="Station TOML file."),
    backend: str = typer.Option("ultralytics", help="ultralytics | hailo"),
    device: str | None = typer.Option(None, help="Torch device for ultralytics: mps | cuda | cpu."),
    db: Path = typer.Option(Path("linesafe.db"), help="SQLite event store."),
) -> None:
    """Score a camera (or a video file) with the overlay window. Keys: s saves a pilot frame, q or ESC quits."""
    import cv2

    from . import overlay
    from .pipeline import Pipeline
    from .store import EventStore

    cfg = _load_station(station)
    camera = source.isdigit()
    with ExitStack() as stack:  # every cleanup runs even if an earlier one raises
        pose_backend = _make_live_backend(backend, device)
        stack.callback(pose_backend.close)
        cap = cv2.VideoCapture(int(source) if camera else source)
        stack.callback(cap.release)
        stack.callback(cv2.destroyAllWindows)
        if not cap.isOpened():
            raise _fail(f"cannot open source {source!r}", 1)
        file_fps = 0.0
        if not camera:
            file_fps = float(cap.get(cv2.CAP_PROP_FPS))
            if not file_fps > 0:
                raise _fail(f"{source} reports fps {file_fps}; cannot time it", 1)
        store = EventStore(db)
        stack.callback(store.close)
        # Pipeline time must never go backwards: monotonic for a camera, frame index / fps for a
        # file. The store gets wall-clock time (a file's time 0 is when this run started).
        wall_offset = time.time() - time.monotonic() if camera else time.time()
        pipeline = Pipeline(pose_backend, cfg, store=store, wall_offset=wall_offset)
        stack.callback(pipeline.close)
        fps = 0.0
        last_t: float | None = None
        last_now: float | None = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.monotonic()
            nudged = False
            if camera:
                t = now
                if last_t is not None and t <= last_t:
                    t, nudged = last_t + 1e-6, True  # the pipeline needs strictly increasing time
            else:
                t = idx / file_fps
            if last_now is not None and not nudged and now > last_now:
                inst = 1.0 / (now - last_now)
                fps = inst if fps == 0.0 else (1.0 - FPS_EMA_ALPHA) * fps + FPS_EMA_ALPHA * inst
            last_t, last_now = t, now
            result = pipeline.step(t, frame, idx, fps)
            idx += 1
            shown = overlay.draw(frame.copy(), result, cfg, fps)  # frame stays raw for the pilot save
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                _save_pilot(frame, shown, result, cfg, wall_offset)
            elif key in (ord("q"), 27):
                break
        if idx == 0:
            raise _fail(f"source {source!r} opened but yielded no frames", 1)


@app.command()
def replay(
    keypoints: Path = typer.Option(..., help="Keypoints file (version 1)."),
    station: Path = typer.Option(..., help="Station TOML file."),
    db: Path | None = typer.Option(None, help="SQLite event store (optional)."),
    out: Path | None = typer.Option(None, help="JSONL with one line per frame."),
    events_out: Path | None = typer.Option(None, help="JSON list of closed events."),
    render_dir: Path | None = typer.Option(None, help="Write overlay PNGs on a black canvas here."),
    render_every: int = typer.Option(15, min=1, help="Render every N-th frame."),
    wall_start: float | None = typer.Option(
        None, help="Epoch seconds of the first frame in --db (default: the session ends now)."
    ),
) -> None:
    """Run the pipeline over a keypoints file, time = frame index / fps."""
    if wall_start is not None and not math.isfinite(wall_start):
        raise _fail(f"--wall-start must be finite, got {wall_start}", 2)
    import cv2
    import numpy as np

    from . import overlay
    from .backends.replay import ReplayBackend
    from .pipeline import Pipeline
    from .store import EventStore

    cfg = _load_station(station)
    try:
        pose_backend = ReplayBackend(keypoints)
    except (OSError, ValueError, KeyError) as exc:
        raise _fail(f"cannot read keypoints file {keypoints}: {exc}", 1) from exc
    raw = pose_backend.raw

    store = EventStore(db) if db is not None else None
    fh = None
    events: list[dict] = []
    max_reba: int | None = None
    try:
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            fh = out.open("w", encoding="utf-8")
        if render_dir is not None:
            render_dir.mkdir(parents=True, exist_ok=True)
        # Stored events and status get wall-clock time; without --wall-start the session ends now.
        if wall_start is not None:
            wall_offset = wall_start
        elif db is not None:
            wall_offset = time.time() - (raw.n - 1) / raw.fps
        else:
            wall_offset = 0.0
        pipeline = Pipeline(pose_backend, cfg, store=store, wall_offset=wall_offset)
        for idx in range(raw.n):
            r = pipeline.step(idx / raw.fps, None, idx, raw.fps)
            reba, rula = r.reba, r.rula
            if r.closed_event is not None:
                events.append(_event_json(r.closed_event))
            if reba is not None:
                max_reba = reba.total if max_reba is None else max(max_reba, reba.total)
            if fh is not None:
                line = {
                    "t": r.t,
                    "reba": reba.total if reba is not None else None,
                    "band": reba.band.value if reba is not None else None,
                    "rula": rula.total if rula is not None else None,
                    "level": rula.level.value if rula is not None else None,
                    "event_active": r.event_active,
                    "partial": reba.partial if reba is not None else None,
                }
                fh.write(json.dumps(line) + "\n")
            if render_dir is not None and idx % render_every == 0:
                canvas = np.zeros((raw.height, raw.width, 3), np.uint8)
                overlay.draw(canvas, r, cfg, raw.fps)
                png = render_dir / f"frame_{idx:05d}.png"
                if not cv2.imwrite(str(png), canvas):
                    raise OSError(f"could not write {png}")
        flushed = pipeline.close()
        if flushed is not None:
            events.append(_event_json(flushed))
    finally:
        if fh is not None:
            fh.close()
        if store is not None:
            store.close()
        pose_backend.close()

    if events_out is not None:
        events_out.parent.mkdir(parents=True, exist_ok=True)
        events_out.write_text(json.dumps(events, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"frames={raw.n} events={len(events)} max_reba={max_reba if max_reba is not None else 'none'}")


@app.command()
def extract(
    video: Path = typer.Argument(..., help="Video clip."),
    out: Path = typer.Option(..., help="Keypoints file to write."),
    backend: str = typer.Option("ultralytics", help="ultralytics | hailo"),
    device: str | None = typer.Option(None, help="Torch device for ultralytics: mps | cuda | cpu."),
) -> None:
    """Run a pose backend over every frame of a video and save a keypoints file."""
    from .capture import extract_keypoints
    from .detections import save_keypoints

    if not video.exists():
        raise _fail(f"video not found: {video}", 1)
    _check_out_writable(out)  # before the backend loads and inference runs
    pose_backend = _make_live_backend(backend, device)
    try:
        raw = extract_keypoints(video, pose_backend)
    except (OSError, ValueError) as exc:
        raise _fail(f"extract failed: {exc}", 1) from exc
    finally:
        pose_backend.close()
    try:
        save_keypoints(out, raw)
    except OSError as exc:
        raise _fail(f"cannot write --out {out}: {exc}", 1) from exc
    typer.echo(f"frames={raw.n} fps={raw.fps:.3f} size={raw.width}x{raw.height} -> {out}")


@app.command()
def web(
    db: Path = typer.Option(Path("linesafe.db"), help="SQLite event store."),
    host: str = typer.Option("0.0.0.0", help="Interface to listen on; 0.0.0.0 serves the whole LAN."),
    port: int = typer.Option(8080, min=1, max=65535, help="TCP port."),
) -> None:
    """Serve the dashboard (station status, today's events, weekly summary) to phones on the LAN."""
    if not db.is_file():  # never create an empty store silently (a typo would show a blank dashboard)
        raise _fail(f"event store {db} does not exist; start `linesafe run --db {db}` first or fix the path", 2)
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError as exc:
        raise _fail(f"install the web extra: uv sync --extra web ({exc})", 1) from exc
    import sqlite3

    from .web import create_app

    try:
        dashboard = create_app(db)
    except (sqlite3.Error, RuntimeError) as exc:
        raise _fail(f"cannot open event store {db}: {exc}", 1) from exc
    uvicorn.run(dashboard, host=host, port=port)
