"""LAN dashboard: JSON routes over the :class:`~linesafe.store.EventStore` and one page.

:func:`create_app` builds the FastAPI app; fastapi is imported inside it, so the rest of
the package never needs the ``web`` extra. Routes:

* ``GET /api/status`` - latest status of every station (``EventStore.status``).
* ``GET /api/events?limit=50&since=<float>`` - events newest first (``EventStore.events``).
* ``GET /api/summary?days=7`` - counts over the events with ``t_start >= time.time() - days * 86400``,
  ``days`` 1..366.
* ``GET /`` - one self-contained HTML page (inline CSS and JS, no external URL) that polls
  status and events every 2 s and the summary every 30 s.

Store times are wall-clock seconds.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from .reba import Band
from .rula import rula_level
from .store import EventStore

if TYPE_CHECKING:
    from fastapi import FastAPI

# The overlay's BAND_BGR as CSS RGB hex; tests/test_web.py keeps the two in step.
BAND_RGB: dict[Band, str] = {
    Band.NEGLIGIBLE: "#2e8b57",
    Band.LOW: "#2e8b57",
    Band.MEDIUM: "#d9930d",
    Band.HIGH: "#e4692f",
    Band.VERY_HIGH: "#c8371b",
}
TOP_DRIVERS = 5
DAY_S = 86400
MAX_SUMMARY_DAYS = 366
_ALL_ROWS = 2**63 - 1  # SQLite's largest LIMIT: every matching row
_MAGNITUDE = re.compile(r"\s+[+-]?\d+(?:\.\d+)?°?$")


def driver_kind(driver: str) -> str:
    """A driver without its magnitude, e.g. ``"trunk flexion 64°"`` -> ``"trunk flexion"``.

    ``"activity +2"`` -> ``"activity"``. Weekly counts group by kind; the angle differs per event.
    """
    return _MAGNITUDE.sub("", driver)


def summarize(events: list[dict], days: int) -> dict:
    """Weekly-summary counts over store event rows.

    ``by_band`` always has ``high`` and ``very high`` and counts any other band an event has;
    ``top_drivers`` counts each driver kind (see :func:`driver_kind`) once per event, most
    frequent first, ties by name; ``total_high_seconds`` sums ``duration_s``, rounded to 0.1 s.
    """
    by_band = {Band.HIGH.value: 0, Band.VERY_HIGH.value: 0}
    kinds: Counter[str] = Counter()
    for e in events:
        by_band[e["band"]] = by_band.get(e["band"], 0) + 1
        kinds.update({driver_kind(d) for d in e["drivers"]})
    top = sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_DRIVERS]
    return {
        "days": days,
        "events": len(events),
        "by_band": by_band,
        "top_drivers": [[kind, n] for kind, n in top],
        "total_high_seconds": round(float(sum(e["duration_s"] for e in events)), 1),
    }


def create_app(db_path: Path) -> FastAPI:
    """The dashboard app over the store at ``db_path`` (opened now, closed on shutdown)."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse

    store = EventStore(db_path)
    page = render_page()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            store.close()

    # No /docs, /redoc or /openapi.json: the docs pages load a CDN and break on an offline LAN.
    app = FastAPI(title="LineSafe", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/api/status")
    def status() -> list[dict]:
        return store.status()

    @app.get("/api/events")
    def events(limit: int = Query(50, ge=1, le=_ALL_ROWS), since: float | None = None) -> list[dict]:
        if since is not None and not math.isfinite(since):
            raise HTTPException(status_code=400, detail=f"since must be a finite number, got {since!r}")
        return store.events(limit, since)

    @app.get("/api/summary")
    def summary(days: int = Query(7, ge=1, le=MAX_SUMMARY_DAYS)) -> dict:
        return summarize(store.events(_ALL_ROWS, time.time() - days * DAY_S), days)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    return app


def render_page() -> str:
    """The dashboard page with the band colours and RULA levels filled in."""
    rules = []
    for band, rgb in BAND_RGB.items():
        text = "#141414" if band is Band.MEDIUM else "#ffffff"  # dark on amber, white otherwise, as the overlay
        rules.append(f".band-{band.value.replace(' ', '-')} {{ --band: {rgb}; --on-band: {text}; }}")
    band_css = "\n".join(rules)
    levels = json.dumps({str(total): rula_level(total).value for total in range(1, 8)})
    return _PAGE.replace("__BAND_CSS__", band_css).replace("__RULA_LEVELS__", levels)


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>LineSafe</title>
<link rel="icon" href="data:,">
<style>
:root {
  --bg: #f3f2ee; --panel: #ffffff; --ink: #1b1b19; --muted: #66665f; --line: #dcdbd4;
  --live: #2e8b57; --warn: #b3261e; --none: #8b8b84;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #111110; --panel: #1c1c1a; --ink: #ecebe6; --muted: #a3a29b; --line: #34342f; --warn: #ff8a80; }
}
.band-none { --band: var(--none); --on-band: #ffffff; }
__BAND_CSS__
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); line-height: 1.4; -webkit-text-size-adjust: 100%; }
header { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; justify-content: space-between;
  gap: 12px; padding: 12px 16px; background: var(--bg); border-bottom: 1px solid var(--line); }
h1 { margin: 0; font-size: 20px; font-weight: 800; letter-spacing: .01em; }
.conn { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--muted); }
.conn::before { content: ""; width: 9px; height: 9px; border-radius: 50%; background: var(--none); }
.conn.live::before { background: var(--live); }
.conn.down { color: var(--warn); font-weight: 600; }
.conn.down::before { background: var(--warn); }
main { max-width: 980px; margin: 0 auto; padding: 16px; display: grid; gap: 28px; }
h2 { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; margin: 0 0 10px;
  font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); }
h2 span { font-weight: 500; text-transform: none; letter-spacing: 0; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; }
.empty { margin: 0; padding: 14px 16px; color: var(--muted); }
.note { margin: 8px 0 0; font-size: 13px; color: var(--muted); }
.stations { display: grid; gap: 12px; }
@media (min-width: 640px) { .stations { grid-template-columns: repeat(auto-fill, minmax(290px, 1fr)); } }
.card { padding: 14px 16px 16px; border-top: 6px solid var(--band); }
.card.stale { opacity: .6; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.station { font-size: 17px; font-weight: 700; }
.ago { font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; }
.stale .ago { color: var(--warn); font-weight: 600; }
.score { display: flex; align-items: center; gap: 16px; margin: 10px 0 8px; }
.reba { font-size: 72px; font-weight: 800; line-height: .9; font-variant-numeric: tabular-nums; }
.reba small { display: block; margin-bottom: 4px; font-size: 12px; font-weight: 700; letter-spacing: .1em;
  color: var(--muted); }
.chip { padding: 6px 12px; border-radius: 8px; background: var(--band); color: var(--on-band);
  font-size: 20px; font-weight: 800; text-transform: uppercase; letter-spacing: .03em; }
.rula { font-weight: 600; }
.drivers { margin: 6px 0 0; padding: 0; list-style: none; color: var(--muted); }
.partial { margin-top: 8px; font-size: 13px; font-weight: 600; color: var(--warn); }
.events { margin: 0; padding: 0; list-style: none; }
.event { display: grid; grid-template-columns: auto 1fr auto; grid-template-areas: "time band dur" "drv drv drv";
  gap: 2px 12px; padding: 10px 16px; border-top: 1px solid var(--line); }
.event:first-child { border-top: 0; }
.time { grid-area: time; font-variant-numeric: tabular-nums; }
.band { grid-area: band; display: flex; align-items: center; gap: 6px; }
.band b { font-variant-numeric: tabular-nums; }
.dot { width: 10px; height: 10px; border-radius: 50%; background: var(--band); }
.dur { grid-area: dur; font-variant-numeric: tabular-nums; }
.drv { grid-area: drv; font-size: 14px; color: var(--muted); }
.week { padding: 16px; }
.stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
@media (min-width: 640px) { .stats { grid-template-columns: repeat(4, 1fr); } }
.stat { padding: 8px 12px; border-left: 4px solid var(--band, var(--line)); }
.stat b { display: block; font-size: 26px; font-weight: 800; font-variant-numeric: tabular-nums; }
.stat span { font-size: 13px; color: var(--muted); }
h3 { margin: 16px 0 6px; font-size: 15px; }
.top { margin: 0; padding: 0; list-style: none; }
.top li { display: flex; justify-content: space-between; gap: 12px; padding: 6px 0; border-top: 1px solid var(--line); }
.top li:first-child { border-top: 0; }
.count { color: var(--muted); font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
<header>
  <h1>LineSafe</h1>
  <div id="conn" class="conn" role="status">connecting</div>
</header>
<main>
  <section>
    <h2>Stations</h2>
    <div id="stations" class="stations"><p class="empty panel">Loading.</p></div>
  </section>
  <section>
    <h2>Today <span id="events-note"></span></h2>
    <ol id="events" class="events panel"><li class="empty">Loading.</li></ol>
  </section>
  <section>
    <h2>This week <span id="w-days"></span></h2>
    <div class="week panel">
      <div class="stats">
        <div class="stat"><b id="w-events">-</b><span>High-risk events</span></div>
        <div class="stat"><b id="w-seconds">-</b><span>time at High or above</span></div>
        <div id="w-bands" style="display: contents"></div>
      </div>
      <h3>Top drivers</h3>
      <ol id="w-top" class="top"></ol>
      <p class="note">Plain-language weekly summary: generated on the UGen300 in Stage II.</p>
    </div>
  </section>
</main>
<script>
"use strict";
const POLL_MS = 2000;
const SUMMARY_POLL_MS = 30000;
const STALE_S = 10;
const TIMEOUT_MS = 5000;
const EVENTS_LIMIT = 50; // shown; one more is requested to know whether there are more
let lastSummaryMs = null; // Date.now() of the last rendered summary
const RULA_LEVELS = __RULA_LEVELS__;
let clockOffset = 0; // server clock minus this device's clock, in s; 0 unless they differ by more than 2 s

const $ = (id) => document.getElementById(id);
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}
const bandClass = (band) => "band-" + (band ? band.replace(/ /g, "-") : "none");
const serverNow = () => Date.now() / 1000 + clockOffset;

function duration(seconds) {
  const s = Math.round(seconds);
  if (s < 60) return s + " s";
  const m = Math.floor(s / 60);
  if (m < 60) return m + " min " + String(s % 60).padStart(2, "0") + " s";
  return Math.floor(m / 60) + " h " + String(m % 60).padStart(2, "0") + " min";
}

function clockTime(t) {
  const d = new Date(t * 1000);
  return [d.getHours(), d.getMinutes(), d.getSeconds()].map((n) => String(n).padStart(2, "0")).join(":");
}

function sinceText(t) {
  const s = Math.max(0, Math.round(serverNow() - t));
  if (s < 120) return "updated " + s + " s ago";
  if (s < 7200) return "updated " + Math.floor(s / 60) + " min ago";
  return "updated " + Math.floor(s / 3600) + " h ago";
}

function tickAgo() {
  for (const node of document.querySelectorAll(".ago")) {
    const t = Number(node.dataset.t);
    node.textContent = sinceText(t);
    node.closest(".card").classList.toggle("stale", serverNow() - t > STALE_S);
  }
}

function startOfToday() {
  const d = new Date(serverNow() * 1000);
  d.setHours(0, 0, 0, 0);
  return d.getTime() / 1000;
}

async function getJSON(path) {
  const options = { cache: "no-store" };
  let timer = null;
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
    options.signal = AbortSignal.timeout(TIMEOUT_MS);
  } else if (typeof AbortController !== "undefined") {
    const c = new AbortController();
    timer = setTimeout(() => c.abort(), TIMEOUT_MS);
    options.signal = c.signal;
  }
  try {
    const res = await fetch(path, options);
    if (!res.ok) throw new Error(path + " returned " + res.status);
    const date = Date.parse(res.headers.get("date") || "");
    if (!Number.isNaN(date)) {
      const offset = (date - Date.now()) / 1000;
      clockOffset = Math.abs(offset) > 2 ? offset : 0;
    }
    return await res.json(); // inside try: the timeout also covers reading the body
  } finally {
    if (timer !== null) clearTimeout(timer);
  }
}

function renderStations(rows) {
  const box = $("stations");
  box.replaceChildren();
  if (!rows.length) {
    box.append(el("p", "empty panel", "No station has reported yet."));
    return;
  }
  for (const r of rows) {
    const card = el("article", "panel card " + bandClass(r.band));
    const head = el("div", "card-head");
    const ago = el("span", "ago");
    ago.dataset.t = r.t;
    head.append(el("span", "station", r.station), ago);
    const score = el("div", "score");
    const reba = el("div", "reba");
    reba.append(el("small", "", "REBA"), document.createTextNode(r.reba_total === null ? "--" : String(r.reba_total)));
    score.append(reba, el("span", "chip", r.band === null ? "no score" : r.band));
    const rula = r.rula_total === null ? "RULA --" : "RULA " + r.rula_total + " \\u00b7 " + RULA_LEVELS[r.rula_total];
    const drivers = el("ul", "drivers");
    for (const d of r.drivers.slice(0, 2)) drivers.append(el("li", "", d));
    card.append(head, score, el("div", "rula", rula), drivers);
    if (r.partial) card.append(el("div", "partial", "Partial score: some joints not visible"));
    box.append(card);
  }
  tickAgo();
}

function renderEvents(rows) {
  const list = $("events");
  list.replaceChildren();
  $("events-note").textContent = rows.length > EVENTS_LIMIT ? "latest " + EVENTS_LIMIT + " shown" : "";
  if (!rows.length) {
    list.append(el("li", "empty", "No High-risk events today."));
    return;
  }
  for (const e of rows.slice(0, EVENTS_LIMIT)) {
    const li = el("li", "event " + bandClass(e.band));
    const when = clockTime(e.t_start);
    const band = el("span", "band");
    band.append(el("i", "dot"), el("b", "", "peak " + e.peak), document.createTextNode(e.band));
    const drivers = e.drivers.length ? e.drivers.join(", ") : "no drivers";
    li.append(el("span", "time", when), band, el("span", "dur", duration(e.duration_s)),
      el("span", "drv", e.station + " \\u00b7 " + drivers));
    list.append(li);
  }
}

function renderSummary(s) {
  $("w-days").textContent = "last " + s.days + " days";
  $("w-events").textContent = String(s.events);
  $("w-seconds").textContent = duration(s.total_high_seconds);
  const bands = $("w-bands");
  bands.replaceChildren();
  for (const [band, n] of Object.entries(s.by_band)) {
    const stat = el("div", "stat " + bandClass(band));
    stat.append(el("b", "", String(n)), el("span", "", band));
    bands.append(stat);
  }
  const top = $("w-top");
  top.replaceChildren();
  if (!s.top_drivers.length) top.append(el("li", "count", "No events in this window."));
  for (const [kind, n] of s.top_drivers) {
    const li = el("li");
    li.append(el("span", "", kind), el("span", "count", n + (n === 1 ? " event" : " events")));
    top.append(li);
  }
}

function setConn(state, text) {
  const conn = $("conn");
  conn.className = "conn " + state;
  conn.textContent = text;
}

async function poll() {
  try {
    // status and events every poll; the summary every SUMMARY_POLL_MS, retried on the next poll if it failed
    const summaryDue = lastSummaryMs === null || Date.now() - lastSummaryMs >= SUMMARY_POLL_MS;
    const [status, events, summary] = await Promise.all([
      getJSON("/api/status"),
      getJSON("/api/events?limit=" + (EVENTS_LIMIT + 1) + "&since=" + startOfToday()),
      summaryDue ? getJSON("/api/summary?days=7") : null,
    ]);
    renderStations(status);
    renderEvents(events);
    if (summary !== null) {
      renderSummary(summary);
      lastSummaryMs = Date.now();
    }
    setConn("live", "live");
  } catch (err) {
    setConn("down", "connection lost, retrying");
  } finally {
    setTimeout(poll, POLL_MS);
  }
}

poll();
setInterval(tickAgo, 1000);
</script>
</body>
</html>
"""
