"""REBA and RULA lookup tables.

Values are from the published worksheets:
  REBA: Hignett & McAtamney, Applied Ergonomics 31 (2000) 201-205 (Tables A, B, C).
  RULA: McAtamney & Corlett, Applied Ergonomics 24(2) (1993) 91-99 (Tables A, B, C).

Every lookup takes 1-based integer scores exactly as written on the worksheet and
raises ``ValueError`` naming the argument when a score is not an int or is out of
range. Nothing is clamped, except RULA Table C, whose worksheet rows read "8+" and
columns "7+".
"""

from __future__ import annotations

# [neck][trunk][legs]
REBA_A = (
    ((1, 2, 3, 4), (2, 3, 4, 5), (2, 4, 5, 6), (3, 5, 6, 7), (4, 6, 7, 8)),
    ((1, 2, 3, 4), (3, 4, 5, 6), (4, 5, 6, 7), (5, 6, 7, 8), (6, 7, 8, 9)),
    ((3, 3, 5, 6), (4, 5, 6, 7), (5, 6, 7, 8), (6, 7, 8, 9), (7, 8, 9, 9)),
)

# [upper_arm][lower_arm][wrist]
REBA_B = (
    ((1, 2, 2), (1, 2, 3)),
    ((1, 2, 3), (2, 3, 4)),
    ((3, 4, 5), (4, 5, 5)),
    ((4, 5, 5), (5, 6, 7)),
    ((6, 7, 8), (7, 8, 8)),
    ((7, 8, 8), (8, 9, 9)),
)

# [score_a][score_b]
REBA_C = (
    (1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7),
    (1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8),
    (2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8),
    (3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9),
    (4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9),
    (6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10),
    (7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11),
    (8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11),
    (9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12),
    (10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12),
    (11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12),
    (12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12),
)

# [upper_arm][lower_arm] -> (w1t1, w1t2, w2t1, w2t2, w3t1, w3t2, w4t1, w4t2)
RULA_A = (
    ((1, 2, 2, 2, 2, 3, 3, 3), (2, 2, 2, 2, 3, 3, 3, 3), (2, 3, 3, 3, 3, 3, 4, 4)),
    ((2, 3, 3, 3, 3, 4, 4, 4), (3, 3, 3, 3, 3, 4, 4, 4), (3, 4, 4, 4, 4, 4, 5, 5)),
    ((3, 3, 4, 4, 4, 4, 5, 5), (3, 4, 4, 4, 4, 4, 5, 5), (4, 4, 4, 4, 4, 5, 5, 5)),
    ((4, 4, 4, 4, 4, 5, 5, 5), (4, 4, 4, 4, 4, 5, 5, 5), (4, 4, 4, 5, 5, 5, 6, 6)),
    ((5, 5, 5, 5, 5, 6, 6, 7), (5, 6, 6, 6, 6, 7, 7, 7), (6, 6, 6, 7, 7, 7, 7, 8)),
    ((7, 7, 7, 7, 7, 8, 8, 9), (8, 8, 8, 8, 8, 9, 9, 9), (9, 9, 9, 9, 9, 9, 9, 9)),
)

# [neck] -> (t1l1, t1l2, t2l1, t2l2, ..., t6l1, t6l2)
RULA_B = (
    (1, 3, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7),
    (2, 3, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7),
    (3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7),
    (5, 5, 5, 6, 6, 7, 7, 7, 7, 7, 8, 8),
    (7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8),
    (8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9),
)

# [wrist_arm 1..8+][neck_trunk_leg 1..7+]
RULA_C = (
    (1, 2, 3, 3, 4, 5, 5),
    (2, 2, 3, 4, 4, 5, 5),
    (3, 3, 3, 4, 4, 5, 6),
    (3, 3, 3, 4, 5, 6, 6),
    (4, 4, 4, 5, 6, 7, 7),
    (4, 4, 5, 6, 6, 7, 7),
    (5, 5, 6, 6, 7, 7, 7),
    (5, 5, 6, 7, 7, 7, 7),
)


def _check(name: str, value: int, low: int, high: int | None) -> int:
    """Return ``value - 1`` (a 0-based index) after validating the 1-based score."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < low or (high is not None and value > high):
        bound = f"{low}..{high}" if high is not None else f">= {low}"
        raise ValueError(f"{name} must be {bound}, got {value}")
    return value - 1


def reba_table_a(neck: int, trunk: int, legs: int) -> int:
    n = _check("neck", neck, 1, 3)
    t = _check("trunk", trunk, 1, 5)
    g = _check("legs", legs, 1, 4)
    return REBA_A[n][t][g]


def reba_table_b(upper_arm: int, lower_arm: int, wrist: int) -> int:
    u = _check("upper_arm", upper_arm, 1, 6)
    lo = _check("lower_arm", lower_arm, 1, 2)
    w = _check("wrist", wrist, 1, 3)
    return REBA_B[u][lo][w]


def reba_table_c(score_a: int, score_b: int) -> int:
    a = _check("score_a", score_a, 1, 12)
    b = _check("score_b", score_b, 1, 12)
    return REBA_C[a][b]


def rula_table_a(upper_arm: int, lower_arm: int, wrist: int, wrist_twist: int) -> int:
    u = _check("upper_arm", upper_arm, 1, 6)
    lo = _check("lower_arm", lower_arm, 1, 3)
    w = _check("wrist", wrist, 1, 4)
    tw = _check("wrist_twist", wrist_twist, 1, 2)
    return RULA_A[u][lo][w * 2 + tw]


def rula_table_b(neck: int, trunk: int, legs: int) -> int:
    n = _check("neck", neck, 1, 6)
    t = _check("trunk", trunk, 1, 6)
    g = _check("legs", legs, 1, 2)
    return RULA_B[n][t * 2 + g]


def rula_table_c(wrist_arm: int, neck_trunk_leg: int) -> int:
    """RULA Table C; scores above the last row ("8+") or column ("7+") use that row or column."""
    a = _check("wrist_arm", wrist_arm, 1, None)
    b = _check("neck_trunk_leg", neck_trunk_leg, 1, None)
    return RULA_C[min(a, 7)][min(b, 6)]
