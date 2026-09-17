import json
import re
from pathlib import Path

import pytest

from lsvideo.captions import align, cues, format_ts_ass, layout, to_ass
from lsvideo.model import Cue, Placed, Shot, Still, Voice, WordTiming

FIXTURE = json.loads((Path(__file__).parent / "fixtures/edge_words.json").read_text())
TEXT = FIXTURE["text"]
WORDS = tuple(WordTiming(w["text"], w["start"], w["end"]) for w in FIXTURE["words"])


def line_texts(cue, text):
    return [text[line[0].start : line[-1].end] for line in cue]


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


def test_no_cue_crosses_a_sentence_end():
    for cue in layout(align(WORDS, TEXT), TEXT):
        joined = " ".join(line_texts(cue, TEXT))
        ends = [m.start() for m in re.finditer(r"[!?…]|\.(?= |$)", joined)]   # "2.5" is not a sentence end
        assert ends in ([], [len(joined) - 1]), joined


def test_every_cue_has_at_most_two_lines_within_max_chars():
    for cue in layout(align(WORDS, TEXT), TEXT):
        assert 1 <= len(cue) <= 2
        for line in line_texts(cue, TEXT):
            assert line in TEXT
            assert len(line) <= 42


def test_cue_times_are_offset_by_shot_t0_plus_lead():
    shot = Shot("a", Still(1, Path("x")), TEXT, 0.5, 1.0)
    out = cues([Placed(shot, 10.0, 25.0)], {"a": Voice("a", Path("x"), WORDS, 19.0)})
    assert out[0].t0 == pytest.approx(10.5 + WORDS[0].start)
    assert out[0].text == "A camera feeds a pose model\nthat finds 17 body keypoints."


def two_sentence_voice(gap):
    words = (WordTiming("One", 0.0, 0.3), WordTiming("two", 0.35, 0.6),
             WordTiming("Three", 0.6 + gap, 0.9 + gap), WordTiming("four", 0.95 + gap, 1.2 + gap))
    shot = Shot("a", Still(1, Path("x")), "One two. Three four.", 0.0, 0.0)
    return [Placed(shot, 0.0, 5.0)], {"a": Voice("a", Path("x"), words, 2.0)}


def test_a_short_pause_between_cues_is_bridged():
    placed, voices = two_sentence_voice(0.3)
    first, second = cues(placed, voices)
    assert first.t1 == pytest.approx(second.t0)


def test_a_long_pause_between_cues_is_kept():
    placed, voices = two_sentence_voice(0.8)
    first, _ = cues(placed, voices)
    assert first.t1 == pytest.approx(0.6)


# Real narration from video/shots.toml, with Edge-shaped tokens (words only; Edge drops the punctuation).
PROBLEM = ("A safety manager watches one station, scores each posture by eye on a paper REBA or RULA sheet, "
           "and types it into a monthly report.")
CLOSE = "LineSafe. Continuous ergonomic risk scoring at the workstation, offline, on an ASUS UGen300."


def edge_shaped_cues(text):
    tokens = re.findall(r"\w+(?:[.']\w+)*|/", text)
    words = tuple(WordTiming(w, k * 0.4, k * 0.4 + 0.3) for k, w in enumerate(tokens))
    shot = Shot("a", Still(1, Path("x")), text, 0.0, 0.0)
    return cues([Placed(shot, 0.0, 60.0)], {"a": Voice("a", Path("x"), words, words[-1].end)})


def test_a_long_sentence_has_no_orphan_line():
    two_line_cues = [c.text.split("\n") for c in edge_shaped_cues(PROBLEM) if "\n" in c.text]
    assert two_line_cues
    for lines in two_line_cues:
        assert all(len(line) >= 12 for line in lines), lines


def test_a_short_sentence_is_a_single_one_line_cue():
    assert edge_shaped_cues(CLOSE)[0].text == "LineSafe."


def test_to_ass_formats_timestamps_and_escapes_newlines():
    ass = to_ass([Cue(1.0, 62.5, "Line {one}\nLine two")])
    assert "PlayResX: 1920" in ass
    assert "Style: Default,IBM Plex Sans,54," in ass
    assert "Dialogue: 0,0:00:01.00,0:01:02.50,Default,,0,0,0,,Line (one)\\NLine two" in ass
    assert format_ts_ass(3725.456) == "1:02:05.46"
