import json

import numpy as np
import pytest

from linesafe.detections import (
    Detection,
    RawDetections,
    load_keypoints,
    save_keypoints,
)
from linesafe.keypoints import N_KPTS


def _three_frame_raw() -> RawDetections:
    """Synthetic 3-frame detections (2, 3 and 0 people), unrounded values."""
    rng = np.random.default_rng(0)

    def det() -> Detection:
        return Detection(
            kpts=rng.uniform(0.0, 900.0, (N_KPTS, 2)),
            conf=rng.uniform(0.0, 1.0, N_KPTS),
            bbox=tuple(float(v) for v in rng.uniform(0.0, 700.0, 4)),
            score=float(rng.uniform(0.0, 1.0)),
        )

    return RawDetections(
        source="synthetic",
        fps=29.97002997,
        width=902,
        height=720,
        backend="test",
        frames=[[det(), det()], [det(), det(), det()], []],
    )


def test_detection_validates_shape():
    with pytest.raises(ValueError):
        Detection(np.zeros((16, 2)), np.zeros(17), (0, 0, 1, 1), 1.0)


def test_detection_validates_conf_and_bbox():
    with pytest.raises(ValueError, match="Detection.conf"):
        Detection(np.zeros((17, 2)), np.zeros(16), (0, 0, 1, 1), 1.0)
    with pytest.raises(ValueError, match="bbox"):
        Detection(np.zeros((17, 2)), np.zeros(17), (0, 0, 1), 1.0)

    # float64 in, float32 stored
    det = Detection(np.zeros((17, 2)), np.zeros(17), (0, 0, 1, 1), 1.0)
    assert det.kpts.dtype == np.float32
    assert det.conf.dtype == np.float32


def test_load_keypoints_validates_header(tmp_path):
    payload = {
        "version": 1,
        "source": "synthetic",
        "fps": 0,
        "width": 100,
        "height": 100,
        "backend": "test",
        "frames": [[]],
    }
    bad_fps = tmp_path / "zero_fps.keypoints.json"
    bad_fps.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fps"):
        load_keypoints(bad_fps)

    missing = tmp_path / "no_width.keypoints.json"
    missing.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "width"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="width"):
        load_keypoints(missing)


def test_raw_detections_validates_dimensions():
    with pytest.raises(ValueError, match="height"):
        RawDetections(
            source="s", fps=30.0, width=100, height=0, backend="test", frames=[]
        )


def test_keypoints_roundtrip(tmp_path):
    raw = _three_frame_raw()
    assert raw.n == 3

    out = tmp_path / "rt.keypoints.json"
    save_keypoints(out, raw)
    back = load_keypoints(out)

    assert back.n == raw.n
    assert back.source == raw.source
    assert back.backend == raw.backend
    assert back.width == raw.width and back.height == raw.height
    assert back.fps == pytest.approx(raw.fps, abs=1e-2)
    assert back.frames[0][0].kpts.shape == (17, 2)
    assert back.frames[0][0].conf.shape == (17,)
    # frames[i] holds EVERY detection in frame i, including a frame with none.
    assert [len(f) for f in back.frames] == [len(f) for f in raw.frames]
    assert [len(f) for f in back.frames] == [2, 3, 0]
    for a_dets, b_dets in zip(raw.frames, back.frames):
        for a, b in zip(a_dets, b_dets):
            assert np.allclose(a.kpts, b.kpts, atol=1e-2)
            assert np.allclose(a.conf, b.conf, atol=1e-2)
            assert np.allclose(a.bbox, b.bbox, atol=1e-2)
            assert b.score == pytest.approx(a.score, abs=1e-2)
