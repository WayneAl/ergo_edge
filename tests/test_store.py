import sqlite3

import pytest

from linesafe.events import Event
from linesafe.reba import Band
from linesafe.store import EventStore

def ev(t0, peak=9):
    return Event(station="S1", t_start=t0, t_end=t0 + 5, peak=peak, band=Band.HIGH, drivers=("trunk flexion 64°",), duration_s=5.0)

def test_events_newest_first_and_since(tmp_path):
    s = EventStore(tmp_path / "a.db"); s.add_event(ev(100)); s.add_event(ev(200, 10))
    rows = s.events()
    assert [r["t_start"] for r in rows] == [200, 100] and rows[0]["drivers"] == ["trunk flexion 64°"]
    assert [r["t_start"] for r in s.events(since=150)] == [200]

def test_status_upsert_and_persistence(tmp_path):
    p = tmp_path / "a.db"; s = EventStore(p)
    s.set_status("S1", 1.0, 5, "medium", 4, ("x",), False, 27.5)
    s.set_status("S1", 2.0, 9, "high", 6, ("y",), True, 26.0)
    s.close()
    rows = EventStore(p).status()
    assert len(rows) == 1 and rows[0]["reba_total"] == 9 and rows[0]["partial"] in (1, True)

def test_reader_sees_writer(tmp_path):
    p = tmp_path / "a.db"; w = EventStore(p); r = EventStore(p)
    w.add_event(ev(1))
    assert len(r.events()) == 1

class _NoWalConnection(sqlite3.Connection):
    """A filesystem where SQLite cannot switch to WAL: the pragma just reports the current mode."""
    def execute(self, sql, *args):
        if sql.upper().replace(" ", "") == "PRAGMAJOURNAL_MODE=WAL":
            sql = "PRAGMA journal_mode"
        return super().execute(sql, *args)

def test_file_database_without_wal_raises(tmp_path, monkeypatch):
    EventStore(":memory:").close()                            # in-memory databases have no WAL; allowed
    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *a, **k: real_connect(*a, factory=_NoWalConnection, **k))
    with pytest.raises(RuntimeError, match="WAL"):
        EventStore(tmp_path / "a.db")
