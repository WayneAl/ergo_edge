from pathlib import Path

import pytest

from lsvideo.model import Clip, Inset, Still, VideoError
from lsvideo.shots import load

HEAD = 'voice = "en-US-AndrewNeural"\n'


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "shots.toml"
    p.write_text(HEAD + body)
    return p


def test_slide_shot_resolves_png_under_build_slides(tmp_path):
    s = load(write(tmp_path, '[[shot]]\nid = "problem"\nslide = 2\nnarration = "Hi."\n'), tmp_path)
    shot = s.shots[0]
    assert shot.visual == Still(page=2, png=tmp_path / "video/build/slides/02.png")
    assert (shot.lead_s, shot.hold_s) == (0.5, 1.0)
    assert s.max_total_s == 180


def test_clip_with_pip_becomes_inset(tmp_path):
    body = ('[[shot]]\nid = "lift"\nclip = "video/footage/a.mp4"\nin_s = 1.0\nout_s = 9.0\ncrop = [0, 50, 1280, 720]\n'
            'pip = "video/footage/b.mp4"\npip_in_s = 2.0\npip_out_s = 12.0\n')
    shot = load(write(tmp_path, body), tmp_path).shots[0]
    assert shot.visual == Inset(
        main=Clip(tmp_path / "video/footage/a.mp4", 1.0, 9.0, (0, 50, 1280, 720)),
        pip=Clip(tmp_path / "video/footage/b.mp4", 2.0, 12.0, None),
    )
    assert shot.hold_s == 0.0


def test_shot_with_both_slide_and_clip_is_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\nclip = "a.mp4"\nin_s = 0\nout_s = 1\n'
    with pytest.raises(VideoError, match="shot x: set exactly one of slide or clip"):
        load(write(tmp_path, body), tmp_path)


def test_hold_s_on_clip_is_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nclip = "a.mp4"\nin_s = 0\nout_s = 1\nhold_s = 2\n'
    with pytest.raises(VideoError, match="shot x: hold_s applies to slide shots only"):
        load(write(tmp_path, body), tmp_path)


def test_duplicate_ids_are_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\n[[shot]]\nid = "x"\nslide = 2\n'
    with pytest.raises(VideoError, match="x"):
        load(write(tmp_path, body), tmp_path)


def test_in_s_must_be_before_out_s(tmp_path):
    body = '[[shot]]\nid = "x"\nclip = "a.mp4"\nin_s = 5\nout_s = 5\n'
    with pytest.raises(VideoError, match=r"shot x: need 0 <= in_s < out_s"):
        load(write(tmp_path, body), tmp_path)


def test_unknown_key_is_rejected(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\nnaration = "typo"\n'
    with pytest.raises(VideoError, match="shot x: unknown keys"):
        load(write(tmp_path, body), tmp_path)


def test_narration_whitespace_is_collapsed(tmp_path):
    body = '[[shot]]\nid = "x"\nslide = 1\nnarration = """\nOne  two.\n   Three.\n"""\n'
    assert load(write(tmp_path, body), tmp_path).shots[0].narration == "One two. Three."
