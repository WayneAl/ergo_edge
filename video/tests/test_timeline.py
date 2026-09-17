from pathlib import Path

import pytest

from lsvideo.model import Clip, Inset, Shot, Still, VideoError, Voice
from lsvideo.timeline import place

P = Path("x")


def still(id, narration="Hi.", lead=0.5, hold=1.0):
    return Shot(id, Still(1, P), narration, lead, hold)


def clip(id, length, narration="Hi.", lead=0.5):
    return Shot(id, Clip(P, 10.0, 10.0 + length), narration, lead, 0.0)


def voice(id, dur):
    return Voice(id, P, (), dur)


def test_still_duration_is_lead_plus_voice_plus_hold_rounded_up_to_a_frame():
    [p] = place([still("a")], {"a": voice("a", 4.128)}, 180)
    assert p.t0 == 0.0
    assert p.dur == pytest.approx(169 / 30)   # 5.628 s -> 168.84 frames -> 169


def test_clip_duration_is_the_clip_length():
    [p] = place([clip("a", 55.0)], {"a": voice("a", 20.0)}, 180)
    assert p.dur == pytest.approx(55.0)


def test_t0_accumulates_rounded_durations():
    placed = place([still("a"), clip("b", 3.0)], {"a": voice("a", 4.128), "b": voice("b", 1.0)}, 180)
    assert placed[1].t0 == pytest.approx(169 / 30)


def test_narration_running_past_the_clip_end_names_the_shot():
    with pytest.raises(VideoError, match="shot lift: narration 9.80s starting at 0.50s runs past the clip end at 10.00s"):
        place([clip("lift", 10.0)], {"lift": voice("lift", 9.8)}, 180)


def test_pip_shorter_than_main_names_the_shot():
    shot = Shot("lift", Inset(Clip(P, 0.0, 10.0), Clip(P, 0.0, 9.0)), "", 0.5, 0.0)
    with pytest.raises(VideoError, match="shot lift: pip clip 9.00s is shorter than main clip 10.00s"):
        place([shot], {}, 180)


def test_total_over_the_limit_lists_every_shot():
    with pytest.raises(VideoError, match=r"total 12.00s exceeds 10s: a=6.00s, b=6.00s"):
        place([clip("a", 6.0, ""), clip("b", 6.0, "")], {}, 10)


def test_silent_shot_needs_no_voice():
    [p] = place([still("a", narration="")], {}, 180)
    assert p.dur == pytest.approx(1.5)


def test_narrated_shot_without_a_voice_is_rejected():
    with pytest.raises(VideoError, match="shot a: no voice synthesized"):
        place([still("a")], {}, 180)
