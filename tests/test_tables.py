import itertools
import pytest
from linesafe import tables as T
from tests.fixtures import worksheet_tables as W

def test_reba_table_a_every_cell():
    for n, t, g in itertools.product(range(1, 4), range(1, 6), range(1, 5)):
        assert T.reba_table_a(n, t, g) == W.REBA_A[n][t - 1][g - 1], (n, t, g)

def test_reba_table_b_every_cell():
    for u, l, w in itertools.product(range(1, 7), range(1, 3), range(1, 4)):
        assert T.reba_table_b(u, l, w) == W.REBA_B[u][l - 1][w - 1], (u, l, w)

def test_reba_table_c_every_cell():
    for a, b in itertools.product(range(1, 13), range(1, 13)):
        assert T.reba_table_c(a, b) == W.REBA_C[a - 1][b - 1], (a, b)

def test_rula_table_a_every_cell():
    for u, l, w, tw in itertools.product(range(1, 7), range(1, 4), range(1, 5), range(1, 3)):
        assert T.rula_table_a(u, l, w, tw) == W.RULA_A[u][l - 1][(w - 1) * 2 + (tw - 1)], (u, l, w, tw)

def test_rula_table_b_every_cell():
    for n, t, g in itertools.product(range(1, 7), range(1, 7), range(1, 3)):
        assert T.rula_table_b(n, t, g) == W.RULA_B[n][(t - 1) * 2 + (g - 1)], (n, t, g)

def test_rula_table_c_every_cell_and_clamps():
    for a, b in itertools.product(range(1, 9), range(1, 8)):
        assert T.rula_table_c(a, b) == W.RULA_C[a - 1][b - 1], (a, b)
    assert T.rula_table_c(11, 9) == W.RULA_C[7][6]
    assert T.rula_table_c(9, 1) == W.RULA_C[7][0]
    assert T.rula_table_c(1, 8) == W.RULA_C[0][6]

# Each function's arguments in order with their valid 1-based upper bound; None = clamps (RULA Table C).
RANGES = [
    (T.reba_table_a, [("neck", 3), ("trunk", 5), ("legs", 4)]),
    (T.reba_table_b, [("upper_arm", 6), ("lower_arm", 2), ("wrist", 3)]),
    (T.reba_table_c, [("score_a", 12), ("score_b", 12)]),
    (T.rula_table_a, [("upper_arm", 6), ("lower_arm", 3), ("wrist", 4), ("wrist_twist", 2)]),
    (T.rula_table_b, [("neck", 6), ("trunk", 6), ("legs", 2)]),
    (T.rula_table_c, [("wrist_arm", None), ("neck_trunk_leg", None)]),
]

def _out_of_range_cases():
    for fn, params in RANGES:
        for i, (name, high) in enumerate(params):
            for bad in (0,) if high is None else (0, high + 1):
                args = [1] * len(params)
                args[i] = bad
                yield pytest.param(fn, tuple(args), name, id=f"{fn.__name__}-{name}={bad}")

@pytest.mark.parametrize("fn, args, name", list(_out_of_range_cases()))
def test_out_of_range_raises_naming_argument(fn, args, name):
    with pytest.raises(ValueError, match=rf"^{name} "):
        fn(*args)

@pytest.mark.parametrize("fn, args, name", [
    (T.reba_table_a, (1.0, 1, 1), "neck"),
    (T.rula_table_a, (1, 1, 1, 2.0), "wrist_twist"),
])
def test_non_int_raises(fn, args, name):
    with pytest.raises(ValueError, match=rf"^{name} "):
        fn(*args)
