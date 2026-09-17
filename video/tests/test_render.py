import pytest

from lsvideo.model import VideoError
from lsvideo.render import check_font

OK = "[Parsed_subtitles_0 @ 0x141806a10] fontselect: (IBM Plex Sans, 400, 0) -> IBMPlexSans-Regular, 0, IBMPlexSans-Regular"
FALLBACK = ("[Parsed_subtitles_0 @ 0x13f6151c0] fontselect: (IBM Plex Sans, 400, 0) -> "
            "/System/Library/Fonts/Helvetica.ttc, -1, Helvetica")


def test_font_resolved_to_plex_passes():
    check_font("noise\n" + OK + "\nmore noise")


def test_font_fallback_to_helvetica_fails():
    with pytest.raises(VideoError, match="libass fell back to /System/Library/Fonts/Helvetica.ttc for IBM Plex Sans"):
        check_font(OK + "\n" + FALLBACK)


def test_missing_fontselect_line_fails():
    with pytest.raises(VideoError, match="no fontselect line for IBM Plex Sans"):
        check_font("frame=  100 fps=30")
