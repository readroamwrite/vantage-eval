"""Eval-health diagnostics: is the evaluation still measuring what you think?

Covers saturation (ceiling effects), dead and low-discrimination items,
paraphrase consistency, the split between capability and non-capability
failures, and the difficulty signature used as a sandbagging signal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import numpy as np

from vantage.stats.bootstrap import Estimate, bootstrap_ci


def saturation(
    values: Sequence[float], *, ceiling: float = 0.9, floor: float = 0.1
) -> dict[str, Any]:
    """Detect ceiling and floor effects on a score vector.

    Returns:
        ``mean`` (Estimate), ``saturated`` (the whole interval sits above
        ``ceiling``), ``floored`` (the whole interval sits below ``floor``)
        and ``headroom`` (``1 - mean``).
    """
    est = bootstrap_ci(values)
    return {
        "mean": est,
        "saturated": bool(est.n > 0 and est.lo > ceiling),
        "floored": bool(est.n > 0 and est.hi < floor),
        "headroom": float("nan") if est.n == 0 else 1.0 - est.point,
    }


def build_item_matrix(
    rows_by_column: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Arrange score rows into ``item -> column -> mean value``.

    Args:
        rows_by_column: Column name (a run, repeat or model) to its score
            rows; each row needs ``case_id`` and ``value``. Repeats within one
            column are averaged.
    """
    sums: dict[str, dict[str, list[float]]] = {}
    for column, rows in rows_by_column.items():
        for row in rows:
            sums.setdefault(row["case_id"], {}).setdefault(column, []).append(float(row["value"]))
    return {
        item: {column: float(np.mean(vals)) for column, vals in columns.items()}
        for item, columns in sums.items()
    }


def dead_items(matrix: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    """Items every column passes or every column fails.

    Such items carry no information about differences between the columns.
    Needs at least two columns to be meaningful.
    """
    all_pass, all_fail = [], []
    for item, columns in matrix.items():
        values = list(columns.values())
        if len(values) < 2:
            continue
        if all(v >= 1.0 for v in values):
            all_pass.append(item)
        elif all(v <= 0.0 for v in values):
            all_fail.append(item)
    return {"all_pass": sorted(all_pass), "all_fail": sorted(all_fail)}


def item_discrimination(matrix: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Per-item variance across columns and item-total correlation.

    The item-total correlation is the Pearson correlation between an item's
    values and the column totals with that item removed; low values mean the
    item does not track the rest of the eval.
    """
    columns = sorted({c for cols in matrix.values() for c in cols})
    if not columns:
        return {}
    items = sorted(matrix)
    grid = np.array([[matrix[i].get(c, np.nan) for c in columns] for i in items], dtype=float)
    totals = np.nansum(grid, axis=0)
    out: dict[str, dict[str, float]] = {}
    for idx, item in enumerate(items):
        row = grid[idx]
        variance = float(np.nanvar(row)) if len(columns) > 1 else 0.0
        rest = totals - np.nan_to_num(row)
        corr = _pearson(row, rest) if len(columns) >= 3 else math.nan
        out[item] = {"variance": variance, "item_total_corr": corr}
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 3 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return math.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def paraphrase_consistency(values_by_base: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """How often paraphrases of the same item agree with each other.

    Returns:
        ``agreement``: mean fraction of paraphrases matching the majority
        outcome per base item (Estimate). ``inconsistent``: base ids whose
        paraphrases disagree.
    """
    agreement: list[float] = []
    inconsistent: list[str] = []
    for base, values in values_by_base.items():
        vals = [1.0 if v >= 0.5 else 0.0 for v in values]
        if not vals:
            continue
        top = max(vals.count(1.0), vals.count(0.0))
        agreement.append(top / len(vals))
        if top != len(vals):
            inconsistent.append(base)
    return {
        "n_bases": len(agreement),
        "agreement": bootstrap_ci(agreement),
        "inconsistent": sorted(inconsistent),
    }


def failure_taxonomy(
    labels: Sequence[str],
    *,
    correct_label: str = "correct",
    non_capability: Sequence[str] = (
        "refusal",
        "format_error",
        "no_answer",
        "tool_error",
        "timeout",
        "halted",
        "error",
    ),
) -> dict[str, Any]:
    """Share of each failure class plus raw and capability-adjusted accuracy.

    Capability-adjusted accuracy excludes items whose failure says nothing
    about capability (refusals, format errors, crashes, timeouts).
    """
    n = len(labels)
    classes = sorted(set(labels))
    shares = {
        c: {"n": labels.count(c), "share": bootstrap_ci([float(x == c) for x in labels])}
        for c in classes
    }
    raw = bootstrap_ci([float(x == correct_label) for x in labels])
    kept = [float(x == correct_label) for x in labels if x not in non_capability]
    adjusted = bootstrap_ci(kept)
    non_cap = bootstrap_ci([float(x in non_capability) for x in labels])
    return {
        "n": n,
        "classes": shares,
        "raw_accuracy": raw,
        "adjusted_accuracy": adjusted,
        "non_capability_share": non_cap,
        "n_excluded": n - len(kept),
    }


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation; ``nan`` when undefined."""
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    rx, ry = _ranks(np.asarray(x, dtype=float)), _ranks(np.asarray(y, dtype=float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def difficulty_signature(
    values_by_tier: Mapping[int, Sequence[float]],
    *,
    easy_tiers: Sequence[int] = (1, 2),
    n_boot: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Does failure rise with difficulty the way honest effort predicts?

    A model trying its best fails more on harder tiers, giving a strongly
    positive Spearman correlation between tier and failure rate. A flat or
    inverted curve, or failures on the easiest tiers, is a sandbagging signal
    (or a sign the tiers are mislabelled).

    Returns:
        ``per_tier`` accuracy estimates, ``spearman`` (tier vs failure rate)
        with a bootstrap ``spearman_ci``, ``monotone`` (accuracy never rises
        with tier), and ``easy_failure_rate`` on ``easy_tiers``.
    """
    tiers = sorted(values_by_tier)
    per_tier = {t: bootstrap_ci(values_by_tier[t]) for t in tiers}
    fail_rates = [1.0 - per_tier[t].point for t in tiers]
    rho = spearman(tiers, fail_rates)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    arrays = {t: np.asarray(values_by_tier[t], dtype=float) for t in tiers}
    for _ in range(n_boot):
        rates = []
        for t in tiers:
            arr = arrays[t]
            if arr.size == 0:
                rates.append(math.nan)
                continue
            rates.append(1.0 - float(arr[rng.integers(0, arr.size, size=arr.size)].mean()))
        r = spearman(tiers, rates)
        if not math.isnan(r):
            boots.append(r)
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rho_ci = Estimate(rho, float(lo), float(hi), len(tiers))
    else:
        rho_ci = Estimate(rho, rho, rho, len(tiers))
    accs = [per_tier[t].point for t in tiers]
    monotone = all(b <= a + 1e-9 for a, b in pairwise(accs))
    easy = [1.0 - v for t in tiers if t in easy_tiers for v in values_by_tier[t]]
    return {
        "per_tier": per_tier,
        "spearman": rho,
        "spearman_ci": rho_ci,
        "monotone": monotone,
        "easy_failure_rate": bootstrap_ci(easy),
    }
