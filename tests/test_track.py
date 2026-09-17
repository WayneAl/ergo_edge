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

# --- fix round 1 ---------------------------------------------------------------

def test_update_time_must_strictly_increase_across_all_calls():
    tr = Tracker()
    tr.update(1.0, [])
    with pytest.raises(ValueError, match="Tracker.update t"):
        tr.update(1.0, [])
    tr = Tracker()
    tr.update(1.0, [det(0, 0, 200, 400)])
    with pytest.raises(ValueError, match="Tracker.update t"):
        tr.update(0.9, [])

def test_inverted_bbox_raises_zero_size_allowed():
    tr = Tracker()
    with pytest.raises(ValueError, match="Detection.bbox"):
        tr.update(0.0, [det(0, 0, 200, 400), det(200, 0, 0, 400)])
    with pytest.raises(ValueError, match="Detection.bbox"):
        tr.update(0.0, [det(0, 400, 200, 0)])
    p = tr.update(0.0, [det(100, 0, 100, 400)])   # zero width is allowed
    assert p.bbox == (100.0, 0.0, 100.0, 400.0)

def test_failed_update_leaves_no_trace():
    d0 = det(0, 0, 200, 400); d1 = det(2, 0, 202, 400, dx=3.0); d2 = det(4, 0, 204, 400, dx=6.0)
    bad_kpt = det(2, 0, 202, 400, dx=3.0); bad_kpt.kpts[K.L_WRIST, 0] = np.nan
    tr, ref = Tracker(), Tracker()
    tr.update(0.0, [d0]); ref.update(0.0, [d0])
    with pytest.raises(ValueError, match="Detection.kpts"):
        tr.update(1 / 30, [bad_kpt])
    bad_conf = det(2, 0, 202, 400, dx=3.0); bad_conf.conf[K.L_WRIST] = np.inf
    with pytest.raises(ValueError, match="Detection.conf"):
        tr.update(1 / 30, [bad_conf])
    with pytest.raises(ValueError, match="Detection.bbox"):
        tr.update(1 / 30, [d1, det(50, 0, 10, 400)])
    with pytest.raises(ValueError, match="Tracker.update t"):
        tr.update(0.0, [d1])
    for t, d in ((1 / 30, d1), (2 / 30, d2)):
        p, q = tr.update(t, [d]), ref.update(t, [d])
        np.testing.assert_array_equal(p.kpts, q.kpts)
        np.testing.assert_array_equal(p.conf, q.conf)
        assert p.bbox == q.bbox and p.t == q.t

def test_non_finite_raw_point_that_would_be_output_raises():
    tr = Tracker()
    d = det(0, 0, 200, 400); d.conf[K.L_ANKLE] = 0.0; d.kpts[K.L_ANKLE, 1] = np.nan
    with pytest.raises(ValueError, match="Detection.kpts"):
        tr.update(0.0, [d])
    assert not tr.locked

def test_reseed_after_loss_applies_roi_and_shoulder_gates():
    tr = Tracker(roi=(0, 0, 450, 1000))
    tr.update(0.0, [det(0, 0, 200, 400)])
    dets = [
        det(500, 0, 1100, 900),                       # big, centre outside the roi
        det(0, 450, 440, 1000, shoulder_conf=0.29),   # big, inside, shoulders just below conf_min
        det(300, 600, 400, 800),                      # small, inside, confident
    ]
    p = tr.update(0.6, dets)
    assert p.bbox == (300.0, 600.0, 400.0, 800.0) and tr.locked

def test_reseed_does_not_leak_held_keypoints_from_old_target():
    tr = Tracker(TrackerParams(lost_s=0.0, hold_s=0.5))
    a = det(0, 0, 200, 400)
    tr.update(0.0, [a])
    a_low = det(0, 0, 200, 400); a_low.conf[K.L_ELBOW] = 0.0
    p = tr.update(0.05, [a_low])
    assert p.conf[K.L_ELBOW] == pytest.approx(0.9, abs=1e-6)   # held from A
    b = det(900, 0, 1500, 900); b.conf[K.L_ELBOW] = 0.0
    p = tr.update(0.1, [b])
    assert p.bbox == b.bbox
    assert p.conf[K.L_ELBOW] == 0.0
    np.testing.assert_array_equal(p.kpts, b.kpts)   # confident points pass through, elbow is B's raw point
    ok = np.arange(17) != K.L_ELBOW
    np.testing.assert_array_equal(p.conf[ok], b.conf[ok])

def test_overlapping_bigger_bystander_with_lower_iou_does_not_steal_lock():
    tr = Tracker(); tr.update(0.0, [det(0, 0, 200, 400)])
    bystander = det(0, 0, 300, 500)    # IoU 80000/150000 = 0.533 >= iou_min, area 150000
    target = det(10, 0, 210, 400)      # IoU 76000/84000 = 0.905, area 80000
    p = tr.update(1 / 30, [bystander, target])
    assert p.bbox == target.bbox

# --- final review ----------------------------------------------------------------

def test_track_id_counts_seeds_not_matches():
    tr = Tracker()
    assert tr.track_id == 0
    tr.update(0.0, [])
    bad = det(0, 0, 200, 400); bad.conf[K.L_ANKLE] = 0.0; bad.kpts[K.L_ANKLE, 1] = np.nan
    with pytest.raises(ValueError, match="Detection.kpts"):
        tr.update(0.05, [bad])
    assert tr.track_id == 0                                   # nothing seeded yet, a rejected seed leaves no trace
    tr.update(0.1, [det(0, 0, 200, 400)])
    assert tr.track_id == 1                                   # first lock
    tr.update(0.2, [det(2, 0, 202, 400)])
    assert tr.track_id == 1                                   # match
    assert tr.update(0.5, []) is None and tr.track_id == 1    # short miss (0.3 s)
    tr.update(0.6, [det(4, 0, 204, 400)])
    assert tr.track_id == 1                                   # match after a short miss
    p = tr.update(1.2, [det(900, 0, 1500, 900)])              # 0.6 s since the last match: lost, re-seed
    assert p.bbox == (900.0, 0.0, 1500.0, 900.0) and tr.track_id == 2
    assert tr.update(2.0, []) is None and not tr.locked and tr.track_id == 2   # lock dropped, nobody seeded
    tr.update(2.1, [det(0, 0, 200, 400)])
    assert tr.track_id == 3
