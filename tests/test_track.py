import numpy as np
import pytest
from linesafe.detections import Detection
from linesafe.track import Tracker, TrackerParams
from linesafe import keypoints as K

def det(x1, y1, x2, y2, score=0.9, shoulder_conf=0.9, conf=0.9, dx=0.0):
    k = np.zeros((17, 2), np.float32)
    k[:, 0] = (x1 + x2) / 2 + dx; k[:, 1] = np.linspace(y1, y2, 17)
    c = np.full(17, conf, np.float32); c[K.L_SHOULDER] = c[K.R_SHOULDER] = shoulder_conf
    return Detection(kpts=k, conf=c, bbox=(x1, y1, x2, y2), score=score)

def test_seeds_largest_with_confident_shoulders():
    tr = Tracker()
    big_bad = det(0, 0, 400, 800, shoulder_conf=0.1); small_ok = det(500, 0, 700, 400)
    p = tr.update(0.0, [big_bad, small_ok])
    assert p.bbox == small_ok.bbox and tr.locked

def test_roi_excludes_outside_centre():
    tr = Tracker(roi=(0, 0, 450, 1000))
    p = tr.update(0.0, [det(500, 0, 900, 800), det(0, 0, 200, 400)])
    assert p.bbox == (0.0, 0.0, 200.0, 400.0)

def test_follows_target_not_bigger_bystander():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    p = tr.update(1 / 30, [det(900, 0, 1500, 900), det(5, 0, 205, 400)])
    assert p.bbox == (5.0, 0.0, 205.0, 400.0)

def test_short_miss_keeps_lock():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    assert tr.update(0.2, []) is None and tr.locked
    p = tr.update(0.3, [det(900, 0, 1500, 900), det(2, 0, 202, 400)])
    assert p.bbox == (2.0, 0.0, 202.0, 400.0)

def test_long_miss_reseeds_on_largest():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    assert tr.update(0.3, []) is None
    p = tr.update(0.9, [det(900, 0, 1500, 900), det(1600, 0, 1700, 200)])   # nothing overlaps the old bbox
    assert p.bbox == (900.0, 0.0, 1500.0, 900.0)

def test_first_frame_passes_through_and_constant_stays():
    tr = Tracker(); d = det(0, 0, 200, 400)
    p0 = tr.update(0.0, [d]); p1 = tr.update(1 / 30, [d])
    np.testing.assert_allclose(p0.kpts, d.kpts); np.testing.assert_allclose(p1.kpts, d.kpts, atol=1e-4)

def test_low_conf_keypoint_held_then_dropped():
    tr = Tracker(); d = det(0, 0, 200, 400)
    tr.update(0.0, [d])
    low = det(0, 0, 200, 400); low.conf[K.L_ELBOW] = 0.0   # Detection stores float32 arrays
    p = tr.update(0.05, [low])
    assert p.conf[K.L_ELBOW] == pytest.approx(0.9, abs=1e-6)
    np.testing.assert_allclose(p.kpts[K.L_ELBOW], d.kpts[K.L_ELBOW], atol=1e-4)
    p = tr.update(0.2, [low])
    assert p.conf[K.L_ELBOW] == 0.0

def test_non_increasing_time_raises():
    tr = Tracker(); d = det(0, 0, 200, 400)
    tr.update(1.0, [d])
    with pytest.raises(ValueError):
        tr.update(1.0, [d])
