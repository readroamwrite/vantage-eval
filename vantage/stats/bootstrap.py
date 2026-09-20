"""Bootstrap confidence intervals and paired comparisons."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Stat = Callable[[np.ndarray], float]


@dataclass(slots=True, frozen=True)
class Estimate:
    """A point estimate with a confidence interval.

    Attributes:
        point: The statistic on the observed data.
        lo: Lower confidence bound.
        hi: Upper confidence bound.
        n: Number of observations.
        level: Confidence level, for example ``0.95``.
    """

    point: float
    lo: float
    hi: float
    n: int
    level: float = 0.95

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a"
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "n": self.n, "level": self.level}

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely above or below zero."""
        return self.lo > 0 or self.hi < 0


def _empty(level: float) -> Estimate:
    return Estimate(math.nan, math.nan, math.nan, 0, level)


def bootstrap_ci(
    values: Sequence[float],
    stat: Stat = np.mean,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap interval for ``stat`` over ``values``.

    Args:
        values: Observations.
        stat: Statistic to bootstrap; defaults to the mean.
        n_boot: Number of resamples.
        alpha: One minus the confidence level.
        seed: Random seed, so reports are reproducible.

    Returns:
        The estimate. Empty input gives NaN bounds with ``n=0``; a single
        observation gives a degenerate interval.
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 0:
        return _empty(1 - alpha)
    point = float(stat(arr))
    if n == 1:
        return Estimate(point, point, point, 1, 1 - alpha)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    boots = samples.mean(axis=1) if stat is np.mean else np.apply_along_axis(stat, 1, samples)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Estimate(point, float(lo), float(hi), n, 1 - alpha)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Bootstrap interval for ``mean(b - a)`` over paired observations.

    Args:
        a: Baseline values.
        b: Comparison values, aligned with ``a`` item by item.
        n_boot: Number of resamples.
        alpha: One minus the confidence level.
        seed: Random seed.

    Returns:
        The estimated difference; positive means ``b`` is higher.

    Raises:
        ValueError: If the sequences differ in length.
    """
    if len(a) != len(b):
        raise ValueError(f"paired inputs differ in length: {len(a)} vs {len(b)}")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)


def pair_by_case(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "value",
    keys: tuple[str, ...] = ("case_id", "repeat_idx"),
) -> tuple[list[float], list[float], list[tuple[Any, ...]]]:
    """Align two sets of score rows on their case (and repeat) ids.

    Rows present on only one side are dropped, so the result is a strictly
    paired sample.

    Args:
        rows_a: Score rows from run A.
        rows_b: Score rows from run B.
        value_key: Which field holds the value.
        keys: Fields that identify a pair.

    Returns:
        ``(values_a, values_b, ids)`` in a stable order.
    """

    def index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], float]:
        return {tuple(r[k] for k in keys): float(r[value_key]) for r in rows}

    index_a, index_b = index(rows_a), index(rows_b)
    ids = [k for k in index_a if k in index_b]
    return [index_a[k] for k in ids], [index_b[k] for k in ids], ids
