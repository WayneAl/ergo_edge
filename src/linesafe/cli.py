"""LineSafe command line.

``run`` scores a live camera and shows the overlay window, ``replay`` runs the same
pipeline from a keypoints file (no camera, no torch), ``extract`` turns a video into a
keypoints file. Heavy imports happen inside the commands; OpenCV windows are opened only
by ``run``.

Exit codes: 1 on backend or input-file errors, 2 on a bad station file or bad options.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)

LIVE_BACKENDS = ("ultralytics", "hailo")
FPS_EMA_ALPHA = 0.1
WINDOW = "LineSafe"
PILOT_DIR = Path("pilot")


def _fail(message: str, code: int) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


def _load_station(path: Path):
    from .config import load_station

    try:
        return load_station(path)
    except (OSError, ValueError) as exc:  # tomllib.TOMLDecodeError is a ValueError
        raise _fail(f"bad station file {path}: {exc}", 2) from exc


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


def _save_pilot(cv2, frame, result, cfg, wall_offset: float) -> None:
    """Key ``s``: the drawn frame as PNG plus its angles and scores as JSON."""
    now = datetime.now()
    stem = f"{cfg.station_id}_{now:%Y%m%d-%H%M%S}-{now.microsecond // 1000:03d}"
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    png = PILOT_DIR / f"{stem}.png"
    if not cv2.imwrite(str(png), frame):
        raise OSError(f"could not write {png}")
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
    (PILOT_DIR / f"{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    typer.echo(f"saved {png}")


def _open_writer(cv2, path: Path, cap, frame):
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not fps > 0:
        raise _fail(f"capture reports fps {fps}; cannot record {path} without it", 1)
    h, w = frame.shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise _fail(f"cannot open {path} for writing", 1)
    return writer


@app.command()
def run(
    source: str = typer.Option("0", help="Camera index or video path."),
    station: Path = typer.Option(..., help="Station TOML file."),
    backend: str = typer.Option("ultralytics", help="ultralytics | hailo"),
    device: str | None = typer.Option(None, help="Torch device for ultralytics: mps | cuda | cpu."),
    db: Path = typer.Option(Path("linesafe.db"), help="SQLite event store."),
    record: Path | None = typer.Option(None, help="Also write the raw frames to this mp4."),
) -> None:
    """Score a camera live with the overlay window. Keys: s saves a pilot frame, q or ESC quits."""
    import cv2

    from . import overlay
    from .pipeline import Pipeline
    from .store import EventStore

    cfg = _load_station(station)
    pose_backend = _make_live_backend(backend, device)
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    store = writer = pipeline = None
    try:
        if not cap.isOpened():
            raise _fail(f"cannot open source {source!r}", 1)
        store = EventStore(db)
        # Pipeline time is monotonic (it must never go backwards); the store gets wall-clock time.
        wall_offset = time.time() - time.monotonic()
        pipeline = Pipeline(pose_backend, cfg, store=store, wall_offset=wall_offset)
        fps = 0.0
        last_t: float | None = None
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = time.monotonic()
            if last_t is not None:
                if t <= last_t:
                    t = last_t + 1e-6  # the pipeline needs strictly increasing time
                inst = 1.0 / (t - last_t)
                fps = inst if fps == 0.0 else (1.0 - FPS_EMA_ALPHA) * fps + FPS_EMA_ALPHA * inst
            last_t = t
            if record is not None:
                if writer is None:
                    writer = _open_writer(cv2, record, cap, frame)
                writer.write(frame)  # raw frame, before the overlay draws on it
            result = pipeline.step(t, frame, idx, fps)
            overlay.draw(frame, result, cfg, fps)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                _save_pilot(cv2, frame, result, cfg, wall_offset)
            elif key in (ord("q"), 27):
                break
            idx += 1
    finally:
        if pipeline is not None:
            pipeline.close()
        pose_backend.close()
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        if store is not None:
            store.close()


@app.command()
def replay(
    keypoints: Path = typer.Option(..., help="Keypoints file (version 1)."),
    station: Path = typer.Option(..., help="Station TOML file."),
    db: Path | None = typer.Option(None, help="SQLite event store (optional)."),
    out: Path | None = typer.Option(None, help="JSONL with one line per frame."),
    events_out: Path | None = typer.Option(None, help="JSON list of closed events."),
    render_dir: Path | None = typer.Option(None, help="Write overlay PNGs on a black canvas here."),
    render_every: int = typer.Option(15, min=1, help="Render every N-th frame."),
) -> None:
    """Run the pipeline over a keypoints file, time = frame index / fps."""
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
        pipeline = Pipeline(pose_backend, cfg, store=store)
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

    pose_backend = _make_live_backend(backend, device)
    try:
        raw = extract_keypoints(video, pose_backend)
    except (OSError, ValueError) as exc:
        raise _fail(f"extract failed: {exc}", 1) from exc
    finally:
        pose_backend.close()
    save_keypoints(out, raw)
    typer.echo(f"frames={raw.n} fps={raw.fps:.3f} size={raw.width}x{raw.height} -> {out}")
