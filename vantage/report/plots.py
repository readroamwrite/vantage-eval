"""Matplotlib figures for experiment reports. Matplotlib is optional."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _plt() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - exercised only without the extra
        return None
    return plt


def reliability_diagram(
    bins: list[tuple[float, float, int, float, float]], path: Path, title: str
) -> bool:
    """Plot judge confidence against accuracy per bin.

    Returns:
        Whether the figure was written (false when matplotlib is missing).
    """
    plt = _plt()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999", label="perfect calibration")
    xs = [b[3] for b in bins if b[2] > 0]
    ys = [b[4] for b in bins if b[2] > 0]
    sizes = [20 + 3 * b[2] for b in bins if b[2] > 0]
    ax.scatter(xs, ys, s=sizes, color="#2b6cb0", zorder=3, label="bins (size = n)")
    ax.set_xlabel("mean stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def grouped_bars(
    groups: list[str],
    series: dict[str, list[float]],
    path: Path,
    *,
    title: str,
    ylabel: str,
    errors: dict[str, list[tuple[float, float]]] | None = None,
) -> bool:
    """Grouped bar chart with optional asymmetric error bars.

    Args:
        groups: X-axis category names.
        series: Series name to one value per group.
        path: Output file.
        title: Figure title.
        ylabel: Y-axis label.
        errors: Series name to ``(lo, hi)`` absolute bounds per group.
    """
    plt = _plt()
    if plt is None:
        return False
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(groups) + 2), 4))
    width = 0.8 / max(1, len(series))
    x = np.arange(len(groups))
    for i, (name, values) in enumerate(series.items()):
        offsets = x + (i - (len(series) - 1) / 2) * width
        yerr = None
        if errors and name in errors:
            lo = [max(0.0, v - b[0]) for v, b in zip(values, errors[name], strict=True)]
            hi = [max(0.0, b[1] - v) for v, b in zip(values, errors[name], strict=True)]
            yerr = [lo, hi]
        ax.bar(offsets, values, width, label=name, yerr=yerr, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
