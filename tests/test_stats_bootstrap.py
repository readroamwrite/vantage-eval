import math

import numpy as np
import pytest

from vantage.stats import bootstrap_ci, pair_by_case, paired_bootstrap_diff, proportion_ci


def test_bootstrap_ci_brackets_mean_and_is_reproducible():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    est = bootstrap_ci(values)
    assert est.point == pytest.approx(0.7)
    assert est.lo <= 0.7 <= est.hi
    assert est.n == 10 and est.level == 0.95
    assert est == bootstrap_ci(values)
    assert str(est).startswith("0.700 [")


def test_bootstrap_ci_edge_cases():
    empty = bootstrap_ci([])
    assert empty.n == 0 and math.isnan(empty.point) and str(empty) == "n/a"
    single = bootstrap_ci([0.4])
    assert (single.point, single.lo, single.hi) == (0.4, 0.4, 0.4)
    assert bootstrap_ci([1, 1, 1]).lo == 1.0


def test_bootstrap_ci_custom_stat():
    est = bootstrap_ci([1, 2, 3, 4, 100], stat=np.median)
    assert est.point == 3


def test_paired_diff_sign_and_zero_exclusion():
    a = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    b = [1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
    est = paired_bootstrap_diff(a, b)
    assert est.point == pytest.approx(0.6)
    assert est.excludes_zero
    same = paired_bootstrap_diff(a, a)
    assert same.point == 0 and not same.excludes_zero
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1], [1, 2])


def test_pair_by_case_aligns_and_drops_unmatched():
    rows_a = [
        {"case_id": "x", "repeat_idx": 0, "value": 1},
        {"case_id": "y", "repeat_idx": 0, "value": 0},
    ]
    rows_b = [
        {"case_id": "y", "repeat_idx": 0, "value": 1},
        {"case_id": "z", "repeat_idx": 0, "value": 1},
    ]
    va, vb, ids = pair_by_case(rows_a, rows_b)
    assert ids == [("y", 0)] and va == [0.0] and vb == [1.0]


def test_proportion_ci_is_wilson_and_never_degenerate():
    none = proportion_ci([0.0] * 42)
    assert none.point == 0.0 and none.lo == 0.0 and none.hi == pytest.approx(0.0838, abs=1e-3)
    every = proportion_ci([1.0] * 42)
    assert every.hi == 1.0 and every.lo == pytest.approx(0.9162, abs=1e-3)
    some = proportion_ci([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    assert some.point == pytest.approx(0.7) and some.lo < 0.7 < some.hi and some.n == 10
    assert some.lo == pytest.approx(0.3968, abs=1e-3) and some.hi == pytest.approx(0.8922, abs=1e-3)
    assert str(proportion_ci([])) == "n/a"
