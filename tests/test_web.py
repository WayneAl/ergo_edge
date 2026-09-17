import sys
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from linesafe.cli import app as cli_app
from linesafe.events import Event
from linesafe.reba import Band
from linesafe.store import EventStore
from linesafe.web import BAND_RGB, create_app

DAY = 86400.0


def ev(t0, peak, band, drivers, duration=5.0):
    return Event(station="S1", t_start=t0, t_end=t0 + duration, peak=peak, band=band, drivers=drivers,
                 duration_s=duration)


@pytest.fixture
def seeded(tmp_path):
    """Two events (the older one very high) and one station status, times in wall-clock seconds."""
    p = tmp_path / "e.db"
    now = time.time()
    s = EventStore(p)
    s.add_event(ev(now - 3600, 11, Band.VERY_HIGH, ("trunk flexion 72°", "upper arm right 95°"), 12.5))
    s.add_event(ev(now - 60, 9, Band.HIGH, ("trunk flexion 64°", "activity +1"), 4.0))
    s.set_status("S1", now - 1, 9, "high", 6, ("trunk flexion 64°", "activity +1"), False, 27.5)
    s.close()
    return p, now


def test_status_returns_one_row(seeded):
    p, _ = seeded
    res = TestClient(create_app(p)).get("/api/status")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 1 and rows[0]["station"] == "S1" and rows[0]["reba_total"] == 9
    assert rows[0]["drivers"] == ["trunk flexion 64°", "activity +1"]


def test_events_limit_one_is_newest(seeded):
    p, now = seeded
    res = TestClient(create_app(p)).get("/api/events?limit=1")
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 1 and rows[0]["peak"] == 9 and rows[0]["t_start"] == pytest.approx(now - 60)


def test_events_since_and_bad_query(seeded):
    p, now = seeded
    client = TestClient(create_app(p))
    rows = client.get(f"/api/events?since={now - 600}").json()
    assert [r["peak"] for r in rows] == [9]
    assert client.get("/api/events?limit=0").status_code == 422
    res = client.get("/api/events?since=nan")
    assert res.status_code == 400 and "since" in res.json()["detail"]


def test_summary_counts_both_events(seeded):
    p, _ = seeded
    res = TestClient(create_app(p)).get("/api/summary")
    assert res.status_code == 200
    assert res.json() == {
        "days": 7,
        "events": 2,
        "by_band": {"high": 1, "very high": 1},
        "top_drivers": [["trunk flexion", 2], ["activity", 1], ["upper arm right", 1]],
        "total_high_seconds": 16.5,
    }


def test_summary_window_is_days_before_now(seeded):
    p, now = seeded
    s = EventStore(p)
    s.add_event(ev(now - 8 * DAY, 10, Band.HIGH, ("trunk twist",), 30.0))
    s.close()
    client = TestClient(create_app(p))
    assert client.get("/api/summary?days=7").json()["events"] == 2
    week_and_a_bit = client.get("/api/summary?days=9").json()
    assert week_and_a_bit["days"] == 9 and week_and_a_bit["events"] == 3
    assert week_and_a_bit["by_band"] == {"high": 2, "very high": 1}
    assert week_and_a_bit["total_high_seconds"] == 46.5
    assert client.get("/api/summary?days=0").status_code == 422


def test_top_drivers_at_most_five_most_frequent_first(tmp_path):
    p = tmp_path / "e.db"
    now = time.time()
    s = EventStore(p)
    kinds = ["trunk flexion 70°", "neck flexion 30°", "upper arm right 95°", "knee flexion 40°", "load (station)",
             "trunk twist", "coupling (station)"]
    for i in range(len(kinds)):
        s.add_event(ev(now - 100 * (i + 1), 9, Band.HIGH, tuple(kinds[: i + 1])))
    s.close()
    top = TestClient(create_app(p)).get("/api/summary").json()["top_drivers"]
    assert top == [["trunk flexion", 7], ["neck flexion", 6], ["upper arm right", 5], ["knee flexion", 4],
                   ["load (station)", 3]]


def test_index_is_self_contained_html(seeded):
    p, _ = seeded
    with TestClient(create_app(p)) as client:  # runs the lifespan (store closed on exit)
        res = client.get("/")
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/html")
    page = res.text
    assert "<title>LineSafe</title>" in page and "LineSafe" in page
    assert "http://" not in page and "https://" not in page
    assert "This week" in page
    assert "Plain-language weekly summary: generated on the UGen300 in Stage II." in page
    for route in ("/api/status", "/api/events", "/api/summary"):
        assert route in page


def test_band_colours_match_the_overlay(seeded):
    from linesafe.overlay import BAND_BGR

    p, _ = seeded
    page = TestClient(create_app(p)).get("/").text
    for band, (b, g, r) in BAND_BGR.items():
        assert BAND_RGB[band] == f"#{r:02x}{g:02x}{b:02x}"
        assert BAND_RGB[band] in page


def test_cli_web_runs_uvicorn(tmp_path, monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))
    res = CliRunner().invoke(cli_app, ["web", "--db", str(tmp_path / "e.db"), "--host", "127.0.0.1",
                                       "--port", "8765"])
    assert res.exit_code == 0, res.output
    assert len(calls) == 1 and calls[0][1] == {"host": "127.0.0.1", "port": 8765}
    assert {r.path for r in calls[0][0].routes} >= {"/", "/api/status", "/api/events", "/api/summary"}


def test_cli_web_without_the_extra_exits_1(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)  # makes `import uvicorn` raise ImportError
    res = CliRunner().invoke(cli_app, ["web", "--db", str(tmp_path / "e.db")])
    assert res.exit_code == 1 and "install the web extra: uv sync --extra web" in res.output
