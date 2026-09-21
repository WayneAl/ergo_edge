from dataclasses import replace

import numpy as np
from linesafe.activity import ActivityFlags
from linesafe.config import StationConfig
from linesafe.geometry import View, compute_angles
from linesafe.overlay import BAND_BGR, draw
from linesafe.pipeline import FrameResult
from linesafe.pose import PoseFrame
from linesafe.reba import Band, score_reba
from linesafe.rula import score_rula
from tests.synth import front_pose, side_pose

CFG = StationConfig(station_id="S1", reba_load=2, reba_coupling=1, reba_wrist=2)
FLAGS = ActivityFlags(static=False, repeated=False, rapid=False, rula_muscle_use=False)
RED, YELLOW = (0, 0, 255), (0, 255, 255)


def result(pose=None, view=None, wrong_view=False, angles=None, reba=None, rula=None, event_active=False):
    return FrameResult(t=1.0, pose=pose, view=view, wrong_view=wrong_view, angles=angles, reba=reba, rula=rula,
                       activity=FLAGS, event_active=event_active, closed_event=None)


def blank():
    return np.zeros((720, 1280, 3), np.uint8)


def has_colour(img, bgr):
    return bool(np.all(img == np.array(bgr, np.uint8), axis=2).any())


def check_drawn(frame, out):
    assert out is frame and out.shape == (720, 1280, 3) and out.dtype == np.uint8 and out.any()


def test_high_partial_with_event_active():
    pose = PoseFrame(1.0, *side_pose(64, 95, 90, 40), bbox=(0.0, 0.0, 1280.0, 720.0))
    angles = compute_angles(pose)
    reba = replace(score_reba(angles, CFG), partial=True, missing=("neck",))
    assert reba.band is Band.HIGH
    frame = blank()
    out = draw(frame, result(pose, View.SIDE, False, angles, reba, score_rula(angles, CFG), True), CFG, 30.0)
    check_drawn(frame, out)
    for edge in (out[:12, :], out[-12:, :], out[:, :12], out[:, -12:]):   # 12 px event border
        assert (edge == RED).all()
    assert has_colour(out[:300, :600], BAND_BGR[Band.HIGH])   # REBA chip in the top-left panel
    assert has_colour(out, YELLOW)   # PARTIAL line


def test_no_person():
    frame = blank()
    out = draw(frame, result(), CFG, 30.0)
    check_drawn(frame, out)
    assert not has_colour(out, RED)   # no border, no banner, FPS not red
    assert not has_colour(out[:300, :600], BAND_BGR[Band.HIGH])   # no score chip


def test_wrong_view_banner():
    pose = PoseFrame(1.0, *front_pose(), bbox=(0.0, 0.0, 1280.0, 720.0))
    angles = compute_angles(pose, view_ok=False)
    frame = blank()
    out = draw(frame, result(pose, View.FRONT, True, angles), CFG, 30.0)
    check_drawn(frame, out)
    assert has_colour(out[:, :40], RED) and has_colour(out[:, -40:], RED)   # full-width red banner


def test_fps_below_15_is_red():
    top_right = (slice(0, 90), slice(1280 - 320, 1280))
    assert has_colour(draw(blank(), result(), CFG, 10.0)[top_right], RED)
    assert not has_colour(draw(blank(), result(), CFG, 15.0)[top_right], RED)
    assert not has_colour(draw(blank(), result(), CFG, 0.0)[top_right], RED)


def test_no_person_text_is_drawn_in_the_centre():
    centre = (slice(330, 390), slice(440, 840))
    nobody = draw(blank(), result(), CFG, 30.0)
    pose = PoseFrame(1.0, *side_pose(0, x0=150.0, y_hip=450.0, torso=120.0), bbox=(0.0, 0.0, 1280.0, 720.0))
    someone = draw(blank(), result(pose, View.SIDE), CFG, 30.0)
    assert not someone[centre].any()   # the person is drawn away from the centre
    assert (nobody[centre] != someone[centre]).any()


def test_panel_fits_narrow_and_small_frames():
    long_driver = "upper arm left 95° with a driver text far longer than any frame is wide " * 3
    for h, w in ((1280, 720), (240, 320)):
        pose = PoseFrame(1.0, *side_pose(64, 95, 90, 40, x0=0.4 * w, y_hip=h / 2, torso=h / 8), bbox=(0.0, 0.0, w, h))
        angles = compute_angles(pose)
        reba = replace(score_reba(angles, CFG), drivers=(long_driver, long_driver), partial=True,
                       missing=tuple(f"segment {i}" for i in range(40)))
        frame = np.full((h, w, 3), 128, np.uint8)
        out = draw(frame, result(pose, View.SIDE, False, angles, reba, score_rula(angles, CFG)), CFG, 30.0)
        assert out.shape == (h, w, 3)
        assert (out[: h - h // 8, -2:] == 128).all(), (h, w)   # nothing from the panel runs off the right edge
