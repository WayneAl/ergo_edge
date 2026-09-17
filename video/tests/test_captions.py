import json
from pathlib import Path

import pytest

from lsvideo.captions import align, cues, format_ts_ass, group_lines, to_ass
from lsvideo.model import Cue, Placed, Shot, Still, Voice, WordTiming

FIXTURE = json.loads((Path(__file__).parent / "fixtures/edge_words.json").read_text())
TEXT = FIXTURE["text"]
WORDS = tuple(WordTiming(w["text"], w["start"], w["end"]) for w in FIXTURE["words"])


def texts(rows, text):
    return [text[r[0].start : r[-1].end] for r in rows]


def test_align_recovers_trailing_punctuation_from_the_script():
    by_text = {}
    for a in align(WORDS, TEXT):
        by_text.setdefault(a.word.text, a)
    assert by_text["keypoints"].punct == "."
    assert by_text["angles"].punct == ","
    assert by_text["UGen300"].punct == ""


def test_align_matches_the_slash_token_inside_reba_rula():
    aligned = align(WORDS, TEXT)
    i = [a.word.text for a in aligned].index("/")
    assert TEXT[aligned[i - 1].start : aligned[i + 1].end] == "REBA/RULA"


def test_align_skips_a_token_missing_from_the_script():
    words = WORDS[:8] + (WordTiming("seventeen", 2.0, 2.4),) + WORDS[8:]
    aligned = align(words, TEXT)
    assert len(aligned) == len(WORDS)
    assert [a.word.text for a in aligned][8:10] == ["17", "body"]


def test_lines_wrap_at_max_chars_and_break_after_a_sentence():
    rows = texts(group_lines(align(WORDS, TEXT), TEXT, 42), TEXT)
    assert rows[0] == "A camera feeds a pose model that finds 17"
    assert rows[1] == "body keypoints."
    assert rows[2].startswith("Plain geometry")


def test_every_line_is_a_substring_of_the_script_within_max_chars():
    for row in texts(group_lines(align(WORDS, TEXT), TEXT, 42), TEXT):
        assert row in TEXT
        assert len(row) <= 42


def test_cue_times_are_offset_by_shot_t0_plus_lead():
    shot = Shot("a", Still(1, Path("x")), TEXT, 0.5, 1.0)
    out = cues([Placed(shot, 10.0, 25.0)], {"a": Voice("a", Path("x"), WORDS, 19.0)})
    assert out[0].t0 == pytest.approx(10.5 + WORDS[0].start)
    assert out[0].text == "A camera feeds a pose model that finds 17\nbody keypoints."


def two_sentence_voice(gap):
    words = (WordTiming("One", 0.0, 0.3), WordTiming("two", 0.35, 0.6),
             WordTiming("Three", 0.6 + gap, 0.9 + gap), WordTiming("four", 0.95 + gap, 1.2 + gap))
    shot = Shot("a", Still(1, Path("x")), "One two. Three four.", 0.0, 0.0)
    return [Placed(shot, 0.0, 5.0)], {"a": Voice("a", Path("x"), words, 2.0)}


def test_a_short_pause_between_cues_is_bridged():
    placed, voices = two_sentence_voice(0.3)
    first, second = cues(placed, voices, lines_per_cue=1)
    assert first.t1 == pytest.approx(second.t0)


def test_a_long_pause_between_cues_is_kept():
    placed, voices = two_sentence_voice(0.8)
    first, _ = cues(placed, voices, lines_per_cue=1)
    assert first.t1 == pytest.approx(0.6)


def test_to_ass_formats_timestamps_and_escapes_newlines():
    ass = to_ass([Cue(1.0, 62.5, "Line {one}\nLine two")])
    assert "PlayResX: 1920" in ass
    assert "Style: Default,IBM Plex Sans,54," in ass
    assert "Dialogue: 0,0:00:01.00,0:01:02.50,Default,,0,0,0,,Line (one)\\NLine two" in ass
    assert format_ts_ass(3725.456) == "1:02:05.46"
