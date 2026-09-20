"""Judge reliability statistics: agreement, bias, consistency and calibration."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np

from vantage.stats.bootstrap import Estimate, bootstrap_ci


def accuracy(pred: Sequence[str], gold: Sequence[str], **kwargs: Any) -> Estimate:
    """Fraction of items where ``pred`` equals ``gold``, with a bootstrap CI."""
    _check_lengths(pred, gold)
    return bootstrap_ci([float(p == g) for p, g in zip(pred, gold, strict=True)], **kwargs)


def confusion(
    pred: Sequence[str], gold: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Counts of ``gold -> pred`` for every label pair."""
    _check_lengths(pred, gold)
    table = {g: {p: 0 for p in labels} for g in labels}
    for p, g in zip(pred, gold, strict=True):
        table.setdefault(g, {}).setdefault(p, 0)
        table[g][p] += 1
    return table


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two label sequences.

    Returns ``nan`` when chance agreement is perfect (all labels identical).
    """
    _check_lengths(a, b)
    n = len(a)
    if n == 0:
        return math.nan
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b.get(label, 0) for label in counts_a) / (n * n)
    if math.isclose(expected, 1.0):
        return math.nan
    return (observed - expected) / (1 - expected)


def kappa_ci(a: Sequence[str], b: Sequence[str], *, n_boot: int = 2000, seed: int = 0) -> Estimate:
    """Bootstrap interval for Cohen's kappa."""
    _check_lengths(a, b)
    n = len(a)
    point = cohens_kappa(a, b)
    if n < 2:
        return Estimate(point, point, point, n)
    rng = np.random.default_rng(seed)
    arr_a, arr_b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k = cohens_kappa(list(arr_a[idx]), list(arr_b[idx]))
        if not math.isnan(k):
            boots.append(k)
    if not boots:
        return Estimate(point, point, point, n)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), n)


def precision_recall(
    pred: Sequence[str], gold: Sequence[str], positive: str
) -> dict[str, Estimate]:
    """Precision, recall and F1 for one label, each with a bootstrap CI."""
    _check_lengths(pred, gold)
    pairs = list(zip(pred, gold, strict=True))
    predicted_pos = [float(g == positive) for p, g in pairs if p == positive]
    actual_pos = [float(p == positive) for p, g in pairs if g == positive]
    precision, recall = bootstrap_ci(predicted_pos), bootstrap_ci(actual_pos)
    return {
        "precision": precision,
        "recall": recall,
        "f1": _f1_bootstrap(pairs, positive),
    }


def _f1_bootstrap(
    pairs: list[tuple[str, str]], positive: str, *, n_boot: int = 2000, seed: int = 0
) -> Estimate:
    """F1 with a percentile interval from resampling the (pred, gold) pairs together.

    Precision and recall share the same true positives, so their intervals
    cannot be combined into an F1 interval; the pairs have to be resampled
    jointly.
    """
    n = len(pairs)
    pred = np.array([p == positive for p, _ in pairs], dtype=bool)
    gold = np.array([g == positive for _, g in pairs], dtype=bool)
    point = _f1(*_precision_recall_point(pred, gold))
    if n < 2:
        return Estimate(point, point, point, n)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    tp = (pred[idx] & gold[idx]).sum(axis=1)
    pp, ap = pred[idx].sum(axis=1), gold[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boots = np.where(pp + ap > 0, 2 * tp / np.maximum(pp + ap, 1), 0.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), n)


def _precision_recall_point(pred: np.ndarray, gold: np.ndarray) -> tuple[float, float]:
    tp = float((pred & gold).sum())
    precision = tp / pred.sum() if pred.sum() else math.nan
    recall = tp / gold.sum() if gold.sum() else math.nan
    return precision, recall


def _f1(p: float, r: float) -> float:
    if math.isnan(p) or math.isnan(r) or p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def position_bias(labels_normal: Sequence[str], labels_swapped: Sequence[str]) -> dict[str, Any]:
    """Measure how much presentation order changes pairwise verdicts.

    Both inputs are un-swapped labels (``"A"`` always names the same
    response) from the same pairs judged in both orders.

    Returns:
        ``consistency``: fraction of pairs with the same choice in both orders.
        ``first_position_rate``: among inconsistent pairs, the fraction where
        the judge picked whichever response was shown first (as opposed to
        second). ``n_inconsistent`` and ``n`` give the counts.
    """
    _check_lengths(labels_normal, labels_swapped)
    consistent = [float(x == y) for x, y in zip(labels_normal, labels_swapped, strict=True)]
    # In the normal order A was first; in the swapped order B was first. A judge
    # that always picks the first slot answers A then B.
    first_slot = [
        float(x == "A" and y == "B")
        for x, y in zip(labels_normal, labels_swapped, strict=True)
        if x != y
    ]
    return {
        "n": len(consistent),
        "consistency": bootstrap_ci(consistent),
        "n_inconsistent": len(first_slot),
        "first_position_rate": bootstrap_ci(first_slot),
    }


def self_consistency(labels_per_item: Sequence[Sequence[str]]) -> dict[str, Any]:
    """Agreement among repeated judgements of the same items.

    Returns:
        ``majority_agreement``: mean fraction of samples agreeing with each
        item's majority label. ``unanimity``: fraction of items where every
        sample agreed. ``n_items`` and ``samples_per_item``.
    """
    majority: list[float] = []
    unanimous: list[float] = []
    for labels in labels_per_item:
        if not labels:
            continue
        top = Counter(labels).most_common(1)[0][1]
        majority.append(top / len(labels))
        unanimous.append(float(top == len(labels)))
    return {
        "n_items": len(majority),
        "samples_per_item": max((len(x) for x in labels_per_item), default=0),
        "majority_agreement": bootstrap_ci(majority),
        "unanimity": bootstrap_ci(unanimous),
    }


def calibration(
    conf: Sequence[float], correct: Sequence[bool], *, n_bins: int = 5
) -> dict[str, Any]:
    """Calibration of confidence against correctness.

    Returns:
        ``ece`` (expected calibration error), ``brier`` score, ``auroc`` of
        confidence as a predictor of correctness, and ``bins`` as a list of
        ``(lo, hi, n, mean_confidence, accuracy)`` tuples.
    """
    _check_lengths(conf, correct)
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = c.size
    if n == 0:
        return {"n": 0, "ece": math.nan, "brier": math.nan, "auroc": math.nan, "bins": []}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[tuple[float, float, int, float, float]] = []
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (c >= lo) & (c < hi) if i < n_bins - 1 else (c >= lo) & (c <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append((float(lo), float(hi), 0, math.nan, math.nan))
            continue
        mean_conf, acc = float(c[mask].mean()), float(y[mask].mean())
        ece += (count / n) * abs(acc - mean_conf)
        bins.append((float(lo), float(hi), count, mean_conf, acc))
    return {
        "n": int(n),
        "ece": float(ece),
        "brier": float(np.mean((c - y) ** 2)),
        "auroc": auroc(c, y),
        "bins": bins,
    }


def auroc(scores: Sequence[float], positives: Sequence[float]) -> float:
    """Area under the ROC curve via the rank statistic; ties count half.

    Returns ``nan`` when either class is empty.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(positives, dtype=float) > 0.5
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    sorted_scores = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _check_lengths(a: Sequence[Any], b: Sequence[Any]) -> None:
    if len(a) != len(b):
        raise ValueError(f"sequences differ in length: {len(a)} vs {len(b)}")


def adjust_for_judge(
    judge_values: Sequence[float],
    *,
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    n_boot: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Correct a judge-measured pass rate for the judge's own error rates.

    Uses the Rogan-Gladen estimator: with sensitivity ``se`` (the judge says
    pass when the answer truly passes) and specificity ``sp`` (the judge
    says fail when it truly fails), the true pass rate is
    ``(observed + sp - 1) / (se + sp - 1)``. Uncertainty is propagated by
    resampling both the judged items and the confusion matrix the
    sensitivity and specificity came from.

    Args:
        judge_values: ``1`` where the judge said pass, ``0`` otherwise.
        tp: Truly-pass items the judge called pass.
        fn: Truly-pass items the judge called fail.
        fp: Truly-fail items the judge called pass.
        tn: Truly-fail items the judge called fail.
        n_boot: Number of resamples.
        seed: Random seed.

    Returns:
        The corrected pass rate, clipped to ``[0, 1]``, with a 95% interval.
        Empty input or a judge no better than chance gives NaN bounds.
    """
    obs = np.asarray(judge_values, dtype=float)
    n_pos, n_neg = tp + fn, fp + tn
    if obs.size == 0 or n_pos == 0 or n_neg == 0:
        return Estimate(math.nan, math.nan, math.nan, int(obs.size))
    se, sp = tp / n_pos, tn / n_neg
    if se + sp - 1 <= 0:
        return Estimate(math.nan, math.nan, math.nan, int(obs.size))
    point = _rogan_gladen(float(obs.mean()), se, sp)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        o = float(obs[rng.integers(0, obs.size, size=obs.size)].mean())
        se_b = rng.binomial(n_pos, se) / n_pos
        sp_b = rng.binomial(n_neg, sp) / n_neg
        if se_b + sp_b - 1 <= 0:
            continue
        boots.append(_rogan_gladen(o, se_b, sp_b))
    if not boots:
        return Estimate(point, point, point, int(obs.size))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Estimate(point, float(lo), float(hi), int(obs.size))


def _rogan_gladen(observed: float, sensitivity: float, specificity: float) -> float:
    value = (observed + specificity - 1) / (sensitivity + specificity - 1)
    return max(0.0, min(1.0, value))
