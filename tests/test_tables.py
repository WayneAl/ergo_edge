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

@pytest.mark.parametrize("call, name", [
    (lambda: T.reba_table_a(0, 1, 1), "neck"), (lambda: T.reba_table_a(1, 6, 1), "trunk"),
    (lambda: T.reba_table_a(1, 1, 5), "legs"), (lambda: T.reba_table_b(7, 1, 1), "upper_arm"),
    (lambda: T.reba_table_b(1, 3, 1), "lower_arm"), (lambda: T.reba_table_b(1, 1, 4), "wrist"),
    (lambda: T.reba_table_c(13, 1), "score_a"), (lambda: T.reba_table_c(1, 0), "score_b"),
    (lambda: T.rula_table_a(1, 1, 5, 1), "wrist"), (lambda: T.rula_table_a(1, 1, 1, 3), "wrist_twist"),
    (lambda: T.rula_table_b(7, 1, 1), "neck"), (lambda: T.rula_table_c(0, 1), "wrist_arm"),
])
def test_out_of_range_raises_naming_argument(call, name):
    with pytest.raises(ValueError, match=name):
        call()

def test_non_int_raises():
    with pytest.raises(ValueError):
        T.reba_table_a(1.0, 1, 1)
