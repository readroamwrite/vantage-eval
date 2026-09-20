import math

import pytest

from vantage.stats.diagnostics import (
    build_item_matrix,
    dead_items,
    difficulty_signature,
    failure_taxonomy,
    item_discrimination,
    paraphrase_consistency,
    saturation,
    spearman,
)
from vantage.taxonomy import (
    register_threat,
    structural_coverage,
    tags_by_prefix,
    taxonomy_gaps,
    threats,
)


def test_saturation_flags():
    assert saturation([1.0] * 30)["saturated"] is True
    assert saturation([0.0] * 30)["floored"] is True
    mid = saturation([1, 0] * 15)
    assert not mid["saturated"] and not mid["floored"] and mid["headroom"] == pytest.approx(0.5)


def test_item_matrix_dead_items_and_discrimination():
    rows = {
        "m1": [
            {"case_id": "a", "value": 1},
            {"case_id": "b", "value": 0},
            {"case_id": "c", "value": 1},
            {"case_id": "d", "value": 1},
        ],
        "m2": [
            {"case_id": "a", "value": 1},
            {"case_id": "b", "value": 0},
            {"case_id": "c", "value": 0},
            {"case_id": "d", "value": 0},
        ],
        "m3": [
            {"case_id": "a", "value": 1},
            {"case_id": "b", "value": 0},
            {"case_id": "c", "value": 1},
            {"case_id": "d", "value": 0},
        ],
    }
    matrix = build_item_matrix(rows)
    assert dead_items(matrix) == {"all_pass": ["a"], "all_fail": ["b"]}
    disc = item_discrimination(matrix)
    assert disc["a"]["variance"] == 0 and math.isnan(disc["a"]["item_total_corr"])
    assert disc["c"]["variance"] > 0
    averaged = build_item_matrix(
        {"m": [{"case_id": "a", "value": 1}, {"case_id": "a", "value": 0}]}
    )
    assert averaged["a"]["m"] == 0.5


def test_paraphrase_consistency():
    result = paraphrase_consistency({"p1": [1, 1, 1], "p2": [1, 0, 1], "p3": [0, 0]})
    assert result["n_bases"] == 3 and result["inconsistent"] == ["p2"]
    assert result["agreement"].point == pytest.approx((1 + 2 / 3 + 1) / 3)


def test_failure_taxonomy_adjusts_for_non_capability_failures():
    labels = ["correct"] * 6 + ["wrong_answer"] * 2 + ["refusal"] * 2
    result = failure_taxonomy(labels)
    assert result["raw_accuracy"].point == pytest.approx(0.6)
    assert result["adjusted_accuracy"].point == pytest.approx(0.75)
    assert result["n_excluded"] == 2 and result["classes"]["refusal"]["n"] == 2


def test_spearman_and_difficulty_signature():
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(spearman([1, 1], [1, 2]))
    honest = {
        1: [1] * 10,
        2: [1] * 9 + [0],
        3: [1] * 6 + [0] * 4,
        4: [1] * 3 + [0] * 7,
        5: [0] * 10,
    }
    sig = difficulty_signature(honest, n_boot=50)
    assert (
        sig["spearman"] == pytest.approx(1.0)
        and sig["monotone"]
        and sig["easy_failure_rate"].point == pytest.approx(0.05)
    )
    sandbag = {
        1: [0] * 5 + [1] * 5,
        2: [0] * 5 + [1] * 5,
        3: [0] * 5 + [1] * 5,
        4: [1] * 5 + [0] * 5,
        5: [0] * 10,
    }
    sig2 = difficulty_signature(sandbag, n_boot=50)
    assert sig2["easy_failure_rate"].point == pytest.approx(0.5) and sig2["spearman"] < 0.99


def test_taxonomy_helpers():
    assert tags_by_prefix(["tier:2", "cap:x", "tier:1", "plain"]) == {
        "cap": ["cap:x"],
        "tag": ["plain"],
        "tier": ["tier:1", "tier:2"],
    }
    assert "threat:grader_tampering" in threats()
    gaps = taxonomy_gaps(["threat:grader_tampering", "cap:x"])
    assert (
        "threat:grader_tampering" not in gaps
        and "threat:fake_success" in gaps
        and "threat:none" not in gaps
    )

    class M:
        def __init__(self, name, view):
            self.name, self.view = name, view

    cov = structural_coverage(
        [M("out", "output")], ["threat:grader_tampering", "threat:fake_success"]
    )
    assert cov["threat:grader_tampering"]["structurally_detectable"] is False
    assert cov["threat:fake_success"]["covered_by"] == ["out"]
    cov = structural_coverage([M("full", "trajectory")], ["threat:grader_tampering"])
    assert cov["threat:grader_tampering"]["structurally_detectable"] is True
    register_threat("threat:custom", "output", "x")
    assert "threat:custom" in threats()
    with pytest.raises(ValueError):
        register_threat("custom", "output", "x")
