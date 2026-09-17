import sys
from pathlib import Path

import numpy as np
import pytest

from linesafe.backends.base import make_backend
from linesafe.backends.hailo_backend import HailoBackend, HailoUnavailable
from linesafe.backends.replay import ReplayBackend
from linesafe.detections import Detection, RawDetections, save_keypoints
from linesafe.keypoints import N_KPTS


def _three_frame_keypoints(tmp_path: Path) -> tuple[Path, RawDetections]:
    """Write a synthetic 3-frame keypoints file (2, 1 and 0 people).

    Returns the file and the in-memory detections that were written to it.
    """
    rng = np.random.default_rng(0)

    def det() -> Detection:
        return Detection(
            kpts=rng.uniform(0.0, 640.0, (N_KPTS, 2)),
            conf=rng.uniform(0.0, 1.0, N_KPTS),
            bbox=(100.0, 50.0, 300.0, 450.0),
            score=float(rng.uniform(0.0, 1.0)),
        )

    raw = RawDetections(
        source="synthetic",
        fps=30.0,
        width=640,
        height=480,
        backend="test",
        frames=[[det(), det()], [det()], []],
    )
    path = tmp_path / "synthetic.keypoints.json"
    save_keypoints(path, raw)
    return path, raw


def test_replay_backend_returns_file_detections(tmp_path):
    path, raw = _three_frame_keypoints(tmp_path)
    backend = make_backend("replay", keypoints_path=path)

    assert backend.name == "replay"
    assert isinstance(backend, ReplayBackend)

    dets = backend.infer(None, 0)
    assert len(dets) == len(raw.frames[0]) == 2
    assert dets[0].kpts.shape == (N_KPTS, 2)
    assert dets[0].conf.shape == (N_KPTS,)
    assert 0.0 <= dets[0].score <= 1.0
    for written, replayed in zip(raw.frames[0], dets):
        # save_keypoints rounds pixels to 2 decimal places
        assert np.allclose(replayed.kpts, written.kpts, atol=1e-2)

    # frame 2 was written with nobody in it
    assert backend.infer(None, 2) == []


def test_replay_backend_index_error_past_end(tmp_path):
    path, _ = _three_frame_keypoints(tmp_path)
    backend = ReplayBackend(path)

    assert backend.raw.n == 3
    with pytest.raises(IndexError):
        backend.infer(None, backend.raw.n)


def test_make_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend 'nope'"):
        make_backend("nope")


def test_hailo_backend_unavailable(monkeypatch):
    # `None` in sys.modules makes `import hailo_platform` raise ImportError,
    # so the test is deterministic on a machine that has pyHailoRT too.
    monkeypatch.setitem(sys.modules, "hailo_platform", None)

    with pytest.raises(HailoUnavailable, match="pyHailoRT"):
        HailoBackend()

    with pytest.raises(HailoUnavailable):
        make_backend("hailo")
