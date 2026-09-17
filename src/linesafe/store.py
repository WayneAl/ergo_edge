"""SQLite store for closed High-risk events and the latest per-station status.

One file, WAL journal so a dashboard process can read while the pipeline writes.
Every write commits immediately. The connection may be shared across threads
(``check_same_thread=False``); a lock serialises its use. ``drivers`` is stored as
a JSON list and returned as a list.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
from pathlib import Path

from .events import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    station TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    peak INTEGER NOT NULL,
    band TEXT NOT NULL,
    drivers TEXT NOT NULL,
    duration_s REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS status (
    station TEXT PRIMARY KEY,
    t REAL NOT NULL,
    reba_total INTEGER,
    band TEXT,
    rula_total INTEGER,
    drivers TEXT NOT NULL,
    partial INTEGER NOT NULL,
    fps REAL NOT NULL
);
"""


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["drivers"] = json.loads(d["drivers"])
    return d


class EventStore:
    def __init__(self, path: Path | str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def add_event(self, e: Event) -> int:
        """Insert a closed event; returns its row id."""
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (station, t_start, t_end, peak, band, drivers, duration_s)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    e.station,
                    _finite("Event.t_start", e.t_start),
                    _finite("Event.t_end", e.t_end),
                    e.peak,
                    e.band.value,
                    json.dumps(list(e.drivers), ensure_ascii=False),
                    _finite("Event.duration_s", e.duration_s),
                ),
            )
            return int(cur.lastrowid)

    def set_status(
        self,
        station: str,
        t: float,
        reba_total: int | None,
        band: str | None,
        rula_total: int | None,
        drivers: tuple[str, ...],
        partial: bool,
        fps: float,
    ) -> None:
        """Insert or replace the latest status of ``station``."""
        if not isinstance(station, str) or not station.strip():
            raise ValueError(f"station must be a non-blank str, got {station!r}")
        if not isinstance(partial, bool):
            raise ValueError(f"partial must be a bool, got {partial!r}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO status (station, t, reba_total, band, rula_total, drivers, partial, fps)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(station) DO UPDATE SET t = excluded.t, reba_total = excluded.reba_total,"
                " band = excluded.band, rula_total = excluded.rula_total, drivers = excluded.drivers,"
                " partial = excluded.partial, fps = excluded.fps",
                (
                    station,
                    _finite("t", t),
                    reba_total,
                    band,
                    rula_total,
                    json.dumps(list(drivers), ensure_ascii=False),
                    int(partial),
                    _finite("fps", fps),
                ),
            )

    def events(self, limit: int = 100, since: float | None = None) -> list[dict]:
        """Newest first (by ``t_start``); ``since`` keeps events with ``t_start >= since``."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"limit must be an int >= 1, got {limit!r}")
        sql = "SELECT * FROM events"
        args: list[float | int] = []
        if since is not None:
            sql += " WHERE t_start >= ?"
            args.append(_finite("since", since))
        sql += " ORDER BY t_start DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row(r) for r in rows]

    def status(self) -> list[dict]:
        """Latest status of every station, by station id."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM status ORDER BY station").fetchall()
        return [_row(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
