from linesafe.keypoints import (
    L_EYE,
    L_WRIST,
    N_KPTS,
    R_EYE,
    R_WRIST,
    SKELETON,
)


def test_skeleton_indices_in_range():
    assert len(SKELETON) == 19
    for a, b in SKELETON:
        assert 0 <= a < N_KPTS
        assert 0 <= b < N_KPTS
        assert a != b
    # COCO's eye-to-eye edge, not golf's wrist-to-wrist "shaft line".
    assert (L_EYE, R_EYE) in SKELETON
    assert (L_WRIST, R_WRIST) not in SKELETON
