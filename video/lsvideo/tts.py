from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import edge_tts

from . import render
from .model import Shot, VideoError, Voice, WordTiming


# Streaming loop copied from yt_audiobook src/yt_audiobook/providers/tts.py (EdgeTTSProvider._run).
async def _stream(text: str, voice: str) -> tuple[bytes, list[WordTiming]]:
    comm = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    audio = bytearray()
    words: list[WordTiming] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"] / 10_000_000
            words.append(WordTiming(chunk["text"], start, start + chunk["duration"] / 10_000_000))
    return bytes(audio), words


def voice_for(shot: Shot, voice: str, cache_dir: Path) -> Voice | None:
    """Narration audio + word timings for one shot, from the cache when the voice and text are unchanged."""
    if not shot.narration:
        return None
    key = hashlib.sha1(f"{voice}\n{shot.narration}".encode()).hexdigest()[:16]
    mp3, meta = cache_dir / f"{key}.mp3", cache_dir / f"{key}.json"
    if meta.exists() and mp3.exists():
        try:
            data = json.loads(meta.read_text())
            words = tuple(WordTiming(str(t), float(s), float(e)) for t, s, e in data["words"])
            dur_s = float(data["dur_s"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise VideoError(f"shot {shot.id}: cached narration {meta} is unreadable ({e}); "
                             f"delete it to re-synthesize") from e
        print(f"tts {shot.id}: cached")
        return Voice(shot.id, mp3, words, dur_s)

    try:
        audio, word_list = asyncio.run(_stream(shot.narration, voice))
    except Exception as exc:
        raise VideoError(f"shot {shot.id}: Edge TTS failed and no cached audio: {exc}") from exc
    if not word_list:
        raise VideoError(f"shot {shot.id}: Edge TTS returned no WordBoundary events")
    cache_dir.mkdir(parents=True, exist_ok=True)
    mp3.write_bytes(audio)
    dur_s = render.probe_duration(mp3)
    # The json is written last, so a json without its mp3 never exists.
    meta.write_text(json.dumps({"voice": voice, "text": shot.narration, "dur_s": dur_s,
                                "words": [[w.text, w.start, w.end] for w in word_list]}, indent=1))
    print(f"tts {shot.id}: synthesized {dur_s:.2f}s")
    return Voice(shot.id, mp3, tuple(word_list), dur_s)
