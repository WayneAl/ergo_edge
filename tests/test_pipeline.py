import pytest
from linesafe import keypoints as K
from linesafe.backends.replay import ReplayBackend
from linesafe.config import StationConfig
from linesafe.detections import RawDetections, save_keypoints
from linesafe.geometry import LEFT, RIGHT, Measured, Missing, View, compute_angles
from linesafe.pipeline import Pipeline, PipelineParams
from linesafe.pose import PoseFrame
from linesafe.reba import Band
from linesafe.store import EventStore
from tests.synth import session, side_pose

FPS = 30.0
UPRIGHT = {"trunk_deg": 0.0}
BEND = {"trunk_deg": 64.0, "arm_deg": 95.0, "elbow_deg": 90.0, "knee_deg": 40.0}
BEND_CFG = StationConfig(station_id="S1", reba_load=2, reba_coupling=1, reba_wrist=2)


def deg(a):
    assert isinstance(a, Measured), a
    return a.deg


def replay(tmp_path, raw: RawDetections) -> ReplayBackend:
    path = tmp_path / "synth.keypoints.json"
    save_keypoints(path, raw)
    return ReplayBackend(path)


def run_all(pipe: Pipeline, backend: ReplayBackend):
    results = [pipe.step(i / backend.raw.fps, None, i) for i in range(backend.raw.n)]
    closed = [r.closed_event for r in results if r.closed_event is not None]
    flushed = pipe.close()
    return results, closed + ([flushed] if flushed is not None else [])


@pytest.mark.parametrize("trunk,arm,elbow,knee,facing", [
    (64, 95, 90, 40, 1), (64, 95, 90, 40, -1), (-25, -40, 10, 70, 1), (100, 200, 45, 120, -1), (30, 250, 150, 10, 1),
])
def test_synth_pose_roundtrip(trunk, arm, elbow, knee, facing):
    kpts, conf = side_pose(trunk, arm, elbow, knee, facing=facing)
    a = compute_angles(PoseFrame(0.0, kpts, conf, (0.0, 0.0, 1280.0, 720.0)))
    assert a.view is View.SIDE and a.facing == facing
    assert a.trunk_twisted is False and a.legs_bilateral is True
    assert deg(a.trunk_flex) == pytest.approx(trunk, abs=1)
    assert deg(a.neck_flex) == pytest.approx(0, abs=1)
    for side in (LEFT, RIGHT):
        assert deg(a.upper_arm[side]) == pytest.approx(arm, abs=1)
        assert deg(a.lower_arm[side]) == pytest.approx(elbow, abs=1)
        assert deg(a.knee[side]) == pytest.approx(knee, abs=1)


def test_replay_bending_session_makes_one_high_event(tmp_path):
    backend = replay(tmp_path, session([(3, UPRIGHT), (6, BEND), (5, UPRIGHT)]))
    store = EventStore(tmp_path / "events.db")
    results, closed = run_all(Pipeline(backend, BEND_CFG, store=store), backend)
    assert len(closed) == 1
    e = closed[0]
    assert e.peak >= 8 and 3.0 <= e.t_start <= 4.0
    rows = store.events()
    assert len(rows) == 1 and rows[0]["t_start"] == pytest.approx(e.t_start) and rows[0]["peak"] == e.peak
    # Mid-bend, after the rapid flag has expired: trunk 4, neck 1, legs 2 -> A 5 + load 2 = 7;
    # upper arm 4, lower arm 1, wrist 2 -> B 5 + coupling 1 = 6; Table C (7, 6) = 9.
    mid = results[int(8.0 * FPS)].reba
    assert mid is not None and mid.total == 9 and mid.band is Band.HIGH and mid.activity == 0


def test_front_view_sets_wrong_view_after_two_seconds(tmp_path):
    backend = replay(tmp_path, session([(3, {"front": True})]))
    results, _ = run_all(Pipeline(backend, StationConfig(station_id="S1")), backend)
    at = lambda s: results[round(s * FPS)]
    assert at(1.9).view is View.FRONT and at(1.9).wrong_view is False
    assert at(2.1).view is View.FRONT and at(2.1).wrong_view is True
    assert isinstance(at(2.1).angles.trunk_flex, Missing)
    assert all(r.reba is None and r.rula is None for r in results)


def test_front_view_is_unscored_before_the_banner(tmp_path):
    raw = session([(3, {"front": True})])
    for dets in raw.frames:
        dets[0].kpts[K.NOSE, 0] += 6.0   # nose off the ear midpoint: facing resolves even from the front
    backend = replay(tmp_path, raw)
    results, _ = run_all(Pipeline(backend, StationConfig(station_id="S1")), backend)
    r = results[round(0.5 * FPS)]
    assert r.view is View.FRONT and r.angles.facing == 1 and r.wrong_view is False
    assert r.reba is None and r.rula is None and isinstance(r.angles.trunk_flex, Missing)
    assert all(x.reba is None and x.rula is None for x in results)


def test_dropped_detection_does_not_reset_wrong_view(tmp_path):
    raw = session([(3, {"front": True})])
    for i in range(45, raw.n, 45):   # one dropped detection every 1.5 s
        raw.frames[i] = []
    backend = replay(tmp_path, raw)
    results, _ = run_all(Pipeline(backend, StationConfig(station_id="S1")), backend)
    assert results[45].pose is None and results[45].wrong_view is False
    assert results[round(1.9 * FPS)].wrong_view is False
    assert results[round(2.1 * FPS)].wrong_view is True


def test_step_rejects_bad_fps_before_any_state_change(tmp_path):
    backend = replay(tmp_path, session([(1, UPRIGHT)]))
    pipe = Pipeline(backend, StationConfig(station_id="S1"))
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError, match="fps"):
            pipe.step(0.0, None, 0, fps=bad)
    assert pipe.step(0.0, None, 0, fps=30.0).reba is not None   # t = 0.0 is still accepted


def test_no_person_gives_no_score(tmp_path):
    raw = session([(1, UPRIGHT)])
    raw.frames[10:20] = [[] for _ in range(10)]   # 10 frames with nobody detected
    backend = replay(tmp_path, raw)
    results, _ = run_all(Pipeline(backend, StationConfig(station_id="S1")), backend)
    for r in results[10:20]:
        assert r.pose is None and r.view is None and r.wrong_view is False
        assert r.angles is None and r.reba is None and r.rula is None
    assert results[5].reba is not None


def test_store_writes_are_shifted_by_wall_offset(tmp_path):
    backend = replay(tmp_path, session([(3, UPRIGHT), (6, BEND), (5, UPRIGHT)]))
    store = EventStore(tmp_path / "events.db")
    offset = 1.7e9
    params = PipelineParams(status_every_s=0.5)
    results, closed = run_all(Pipeline(backend, BEND_CFG, params, store=store, wall_offset=offset), backend)
    assert len(closed) == 1
    row = store.events()[0]
    assert row["t_start"] == pytest.approx(closed[0].t_start + offset)
    assert row["t_end"] == pytest.approx(closed[0].t_end + offset)
    status = store.status()
    assert len(status) == 1 and status[0]["station"] == "S1"
    # Status is written at most every 0.5 s: the last write is within 0.5 s of the last frame.
    assert results[-1].t + offset - 0.5 < status[0]["t"] <= results[-1].t + offset + 1e-6


def test_new_person_starts_a_fresh_activity_history(tmp_path):
    a = session([(50, {"trunk_deg": 30.0})])                     # person A, static, hip at x 640
    b = session([(15, {"trunk_deg": 35.0, "x0": 200.0})])        # person B, static, far from A: IoU 0
    gap = [[] for _ in range(round(0.6 * FPS))]                   # > lost_s: the tracker drops A, then seeds B
    raw = RawDetections(source="synth", fps=FPS, width=a.width, height=a.height, backend="synth",
                        frames=a.frames + gap + b.frames)
    backend = replay(tmp_path, raw)
    results, _ = run_all(Pipeline(backend, StationConfig(station_id="S1")), backend)
    b0 = a.n + len(gap)
    r = results[b0 + round(10 * FPS)]                             # B's t + 10 s = 60.6 s since A appeared
    assert r.pose is not None and r.pose.bbox[2] < 400            # the tracked person is B
    # 30° and 35° stay within the 10° band and the 0.6 s gap is under max_gap_s: one history would be static
    assert r.activity.static is False
