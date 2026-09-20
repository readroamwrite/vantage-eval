import math

import pytest

from vantage.stats import (
    accuracy,
    auroc,
    calibration,
    cohens_kappa,
    confusion,
    kappa_ci,
    position_bias,
    precision_recall,
    self_consistency,
)


def test_accuracy_and_confusion():
    pred = ["pass", "pass", "fail", "fail"]
    gold = ["pass", "fail", "fail", "fail"]
    assert accuracy(pred, gold).point == 0.75
    assert confusion(pred, gold, ["pass", "fail"]) == {
        "pass": {"pass": 1, "fail": 0},
        "fail": {"pass": 1, "fail": 2},
    }
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_cohens_kappa_known_values():
    # Textbook example: observed 0.7, expected 0.5 -> kappa 0.4
    a = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    assert cohens_kappa(a, b) == pytest.approx(0.4)
    assert cohens_kappa(["a", "b"], ["a", "b"]) == 1.0
    assert math.isnan(cohens_kappa(["a", "a"], ["a", "a"]))
    est = kappa_ci(a, b, n_boot=200)
    assert est.lo <= est.point <= est.hi and est.n == 50


def test_precision_recall():
    pred = ["fail", "fail", "pass", "pass"]
    gold = ["fail", "pass", "fail", "pass"]
    pr = precision_recall(pred, gold, positive="fail")
    assert pr["precision"].point == 0.5 and pr["recall"].point == 0.5 and pr["f1"].point == 0.5


def test_position_bias_counts_first_slot_preference():
    normal = ["A", "A", "B", "A", "B"]
    swapped = ["A", "B", "B", "B", "A"]
    result = position_bias(normal, swapped)
    assert result["n"] == 5 and result["n_inconsistent"] == 3
    assert result["consistency"].point == pytest.approx(0.4)
    # inconsistent pairs: (A,B) first-slot, (A,B) first-slot, (B,A) second-slot
    assert result["first_position_rate"].point == pytest.approx(2 / 3)


def test_self_consistency():
    result = self_consistency([["p", "p", "p"], ["p", "f", "p"], ["f", "p", "p", "f"], []])
    assert result["n_items"] == 3 and result["samples_per_item"] == 4
    assert result["majority_agreement"].point == pytest.approx((1 + 2 / 3 + 0.5) / 3)
    assert result["unanimity"].point == pytest.approx(1 / 3)


def test_calibration_and_auroc():
    conf = [0.9, 0.8, 0.7, 0.2, 0.1, 0.3]
    correct = [True, True, False, False, False, True]
    result = calibration(conf, correct, n_bins=2)
    assert result["n"] == 6 and result["auroc"] == pytest.approx(8 / 9)
    assert 0 <= result["ece"] <= 1 and 0 <= result["brier"] <= 1
    assert len(result["bins"]) == 2 and result["bins"][1][2] == 3
    assert auroc([1, 1, 1], [1, 0, 1]) == 0.5
    assert math.isnan(auroc([1, 2], [1, 1]))
    empty = calibration([], [], n_bins=3)
    assert empty["n"] == 0 and math.isnan(empty["ece"])
