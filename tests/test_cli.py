import json
import math
import sys
import types

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner
from linesafe import cli
from linesafe.cli import app
from linesafe.config import StationConfig
from linesafe.detections import save_keypoints
from linesafe.geometry import compute_angles
from linesafe.pipeline import FrameResult
from linesafe.activity import ActivityFlags
from linesafe.pose import PoseFrame
from linesafe.reba import score_reba
from tests.synth import session, side_pose

runner = CliRunner()
UPRIGHT = {"trunk_deg": 0.0}
BEND = {"trunk_deg": 64.0, "arm_deg": 95.0, "elbow_deg": 90.0, "knee_deg": 40.0}
STATION = '[station]\nid = "S1"\nreba_load = 2\nreba_coupling = 1\nreba_wrist = 2\n'


def write_inputs(tmp_path, station_text=STATION, segments=((3, UPRIGHT), (6, BEND), (5, UPRIGHT))):
    raw = session(list(segments))
    kp = tmp_path / "synth.keypoints.json"
    save_keypoints(kp, raw)
    st = tmp_path / "station.toml"
    st.write_text(station_text, encoding="utf-8")
    return raw, kp, st


def test_replay_writes_jsonl_events_renders_and_summary(tmp_path):
    raw, kp, st = write_inputs(tmp_path)
    out, ev, rd = tmp_path / "frames.jsonl", tmp_path / "events.json", tmp_path / "render"
    res = runner.invoke(app, ["replay", "--keypoints", str(kp), "--station", str(st), "--db", str(tmp_path / "e.db"),
                              "--out", str(out), "--events-out", str(ev), "--render-dir", str(rd),
                              "--render-every", "60"])
    assert res.exit_code == 0, res.output
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == raw.n
    assert set(lines[0]) == {"t", "reba", "band", "rula", "level", "event_active", "partial"}
    assert lines[0]["t"] == 0.0 and lines[0]["reba"] == 3 and lines[0]["band"] == "low"
    assert any(line["event_active"] for line in lines)
    events = json.loads(ev.read_text(encoding="utf-8"))
    assert isinstance(events, list) and len(events) == 1
    assert events[0]["station"] == "S1" and events[0]["peak"] >= 8 and 3.0 <= events[0]["t_start"] <= 4.0
    max_reba = max(line["reba"] for line in lines if line["reba"] is not None)
    assert f"frames={raw.n} events=1 max_reba={max_reba}" in res.output
    pngs = sorted(p.name for p in rd.glob("*.png"))
    assert len(pngs) == math.ceil(raw.n / 60) and pngs[0] == "frame_00000.png" and "frame_00060.png" in pngs


def test_replay_bad_station_exits_2(tmp_path):
    _, kp, st = write_inputs(tmp_path, station_text='[station]\nid = "S1"\nreba_load = 7\n', segments=((1, UPRIGHT),))
    res = runner.invoke(app, ["replay", "--keypoints", str(kp), "--station", str(st)])
    assert res.exit_code == 2 and "reba_load" in res.output
    res = runner.invoke(app, ["replay", "--keypoints", str(kp), "--station", str(tmp_path / "missing.toml")])
    assert res.exit_code == 2 and "missing.toml" in res.output


# --- fix round 1 ------------------------------------------------------------------


def synthetic_frames(n, dt, t0=1000.0, size=(120, 160)):
    for i in range(n):
        frame = np.zeros((*size, 3), np.uint8)
        frame[:, : (i + 1) * 8] = (40, 160, 220)   # a bar that grows frame by frame
        yield t0 + i * dt, frame


def test_write_timed_video_uses_the_measured_rate_and_writes_timestamps(tmp_path):
    out = tmp_path / "session.mp4"
    n, fps = cli.write_timed_video(synthetic_frames(20, 0.1), out)
    assert n == 20 and fps == pytest.approx(10.0, abs=0.1)
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0, abs=0.1)
        count = 0
        while cap.read()[0]:
            count += 1
    finally:
        cap.release()
    assert count == 20
    stamps = json.loads((tmp_path / "session.mp4.timestamps.json").read_text(encoding="utf-8"))
    assert len(stamps) == 20 and stamps[0] == 0.0 and stamps[-1] == pytest.approx(1.9)
    assert all(b > a for a, b in zip(stamps, stamps[1:]))
    assert sorted(p.name for p in tmp_path.iterdir()) == ["session.mp4", "session.mp4.timestamps.json"]


def test_write_timed_video_fails_loud(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        cli.write_timed_video(iter([]), tmp_path / "a.mp4")
    with pytest.raises(ValueError, match="2 frames"):
        cli.write_timed_video(synthetic_frames(1, 0.1), tmp_path / "b.mp4")
    frames = list(synthetic_frames(3, 0.1))
    frames[2] = (frames[1][0], frames[2][1])
    with pytest.raises(ValueError, match="increase"):
        cli.write_timed_video(iter(frames), tmp_path / "c.mp4")
    assert list(tmp_path.iterdir()) == []   # nothing half-written is left behind


def test_record_validates_options_before_opening_a_camera():
    res = runner.invoke(app, ["record", "--source", "clip.mp4", "--out", "x.mp4", "--seconds", "5"])
    assert res.exit_code == 2 and "camera index" in res.output
    res = runner.invoke(app, ["record", "--out", "x.mp4", "--seconds", "0"])
    assert res.exit_code == 2 and "--seconds" in res.output


def test_run_has_no_record_option():
    res = runner.invoke(app, ["run", "--help"])
    assert res.exit_code == 0 and "--record" not in res.output


def pilot_result():
    pose = PoseFrame(12.5, *side_pose(64, 95, 90, 40), bbox=(0.0, 0.0, 1280.0, 720.0))
    angles = compute_angles(pose)
    cfg = StationConfig(station_id="S 1/a")
    flags = ActivityFlags(static=False, repeated=False, rapid=False, rula_muscle_use=False)
    return cfg, FrameResult(t=12.5, pose=pose, view=angles.view, wrong_view=False, angles=angles,
                            reba=score_reba(angles, cfg), rula=None, activity=flags, event_active=False,
                            closed_event=None)


def test_save_pilot_writes_raw_drawn_and_json(tmp_path):
    cfg, result = pilot_result()
    raw = np.random.default_rng(0).integers(0, 256, (72, 128, 3), dtype=np.uint8)
    drawn = raw.copy()
    drawn[10:20, 10:20] = 255
    stem = cli._save_pilot(raw, drawn, result, cfg, 1000.0, tmp_path / "pilot")
    assert stem is not None and stem.parent == tmp_path / "pilot" and stem.name.startswith("S_1_a_")
    raw_png, drawn_png, js = (stem.parent / f"{stem.name}{sfx}" for sfx in ("_raw.png", ".png", ".json"))
    assert np.array_equal(cv2.imread(str(raw_png)), raw)
    assert np.array_equal(cv2.imread(str(drawn_png)), drawn)
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["t"] == 1012.5 and payload["angles"]["trunk_flex"] == 64.0 and payload["rula"] is None
    assert payload["reba"]["total"] == result.reba.total


def test_save_pilot_failure_is_reported_not_raised(tmp_path, capsys):
    cfg, result = pilot_result()
    (tmp_path / "file").write_text("x", encoding="utf-8")
    frame = np.zeros((8, 8, 3), np.uint8)
    assert cli._save_pilot(frame, frame, result, cfg, 0.0, tmp_path / "file" / "pilot") is None
    assert "pilot save failed" in capsys.readouterr().err


def test_extract_checks_the_video_before_building_a_backend(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(cli, "_make_live_backend", lambda *a: built.append(a))
    res = runner.invoke(app, ["extract", str(tmp_path / "missing.mp4"), "--out", str(tmp_path / "k.json")])
    assert res.exit_code == 1 and "video not found" in res.output and built == []


def test_web_refuses_a_missing_database(tmp_path, monkeypatch):
    served = []
    monkeypatch.setitem(sys.modules, "uvicorn", types.SimpleNamespace(run=lambda *a, **k: served.append(a)))
    db = tmp_path / "missing.db"
    res = runner.invoke(app, ["web", "--db", str(db)])
    assert res.exit_code == 2 and "missing.db" in res.output and "does not exist" in res.output
    assert not db.exists() and served == []   # no empty store is created, nothing is served
