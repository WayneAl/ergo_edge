import json
import math

from typer.testing import CliRunner
from linesafe.cli import app
from linesafe.detections import save_keypoints
from tests.synth import session

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
