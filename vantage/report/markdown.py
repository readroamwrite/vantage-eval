"""Markdown renderers for runs and comparisons."""

from __future__ import annotations

from typing import Any

from vantage.stats import Estimate, bootstrap_ci
from vantage.store import Store


def summarize_run(store: Store, run_id: int) -> dict[str, Any]:
    """Collect per-scorer estimates and status counts for a run.

    Returns:
        ``{"run": row, "metrics": {scorer: Estimate}, "statuses": {status: n}}``.
    """
    run = store.get_run(run_id)
    rows = store.scores_for_run(run_id)
    by_scorer: dict[str, list[float]] = {}
    for row in rows:
        by_scorer.setdefault(row["scorer"], []).append(float(row["value"]))
    metrics = {name: bootstrap_ci(values) for name, values in by_scorer.items()}
    return {"run": run, "metrics": metrics, "statuses": store.status_counts(run_id)}


def metrics_table(metrics: dict[str, Estimate]) -> str:
    """Render ``{scorer: Estimate}`` as a markdown table."""
    lines = ["| scorer | mean | 95% CI | n |", "|---|---|---|---|"]
    for name, est in metrics.items():
        lines.append(f"| {name} | {est.point:.3f} | [{est.lo:.3f}, {est.hi:.3f}] | {est.n} |")
    return "\n".join(lines)


def run_summary_markdown(store: Store, run_id: int) -> str:
    """Render one run as a markdown section."""
    summary = summarize_run(store, run_id)
    run = summary["run"]
    header = [
        f"## Run {run['id']}: {run['name']}",
        "",
        f"- target: `{run['target_id']}`",
        f"- dataset: `{run['dataset_name']}` ({run['dataset_hash']})",
        f"- condition: {run['condition'] or '-'}",
        f"- status: {run['status']}, trajectories: {run['n_trajectories']}",
        f"- outcomes: {', '.join(f'{k}={v}' for k, v in sorted(summary['statuses'].items())) or '-'}",
        "",
    ]
    return "\n".join(header) + metrics_table(summary["metrics"]) + "\n"


def _est(est: Estimate | None) -> str:
    return "-" if est is None else str(est)


def _est_dict(est: dict[str, Any] | None) -> str:
    """Format an ``Estimate.to_dict()`` payload as ``point [lo, hi]``."""
    if not est:
        return "-"
    return f"{est.get('point', float('nan')):.3f} [{est.get('lo', float('nan')):.3f}, {est.get('hi', float('nan')):.3f}]"


def compare_markdown(result: dict[str, Any]) -> str:
    """Render ``analysis.compare_runs`` output."""
    a, b = result["run_a"], result["run_b"]
    lines = [
        f"## Compare run {a['id']} ({a['name']}) vs run {b['id']} ({b['name']})",
        "",
        "Paired on shared cases; diff is B minus A with a 95% bootstrap CI.",
        "",
        "| scorer | mean A | mean B | diff (B - A) | pairs | flipped up | flipped down |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, block in result["scorers"].items():
        lines.append(
            f"| {name} | {block['mean_a'].point:.3f} | {block['mean_b'].point:.3f} | {block['diff']} | "
            f"{block['n_pairs']} | {len(block['flipped_up'])} | {len(block['flipped_down'])} |"
        )
    return "\n".join(lines) + "\n"


def diagnose_markdown(result: dict[str, Any]) -> str:
    """Render ``analysis.diagnose_run`` output as an eval-health report."""
    run = result["run"]
    if result.get("scorer") is None:
        return f"## Diagnose run {run['id']}: no scores stored\n"
    sat = result["saturation"]
    lines = [
        f"## Eval health: run {run['id']} ({run['name']}), scorer `{result['scorer']}`",
        "",
    ]
    warnings = result.get("warnings", [])
    lines += ["### Checks", ""] + ([f"- {w}" for w in warnings] or ["- all checks passed"]) + [""]
    lines += [
        f"- mean: {sat['mean']}  (n={result['n']})",
        f"- saturated (CI above 0.9): {sat['saturated']}; floored (CI below 0.1): {sat['floored']}; headroom: {sat['headroom']:.3f}",
    ]
    if result["dead_items"] is not None:
        dead = result["dead_items"]
        lines.append(
            f"- dead items across {result['n_columns']} columns: {len(dead['all_pass'])} always pass, {len(dead['all_fail'])} always fail"
        )
    if result["low_discrimination"]:
        lines.append(
            f"- low-discrimination items (item-total corr < 0.1): {len(result['low_discrimination'])}"
        )
    if result["paraphrase"] is not None:
        para = result["paraphrase"]
        lines.append(
            f"- paraphrase agreement: {para['agreement']} over {para['n_bases']} base items; inconsistent: {len(para['inconsistent'])}"
        )
    if result["difficulty"] is not None:
        diff = result["difficulty"]
        flag = (
            ""
            if diff["spearman"] == diff["spearman"] and diff["spearman"] > 0.5
            else "  <-- weak or inverted difficulty signature"
        )
        lines.append(
            f"- difficulty signature: Spearman(tier, failure rate) = {diff['spearman_ci']}{flag}"
        )
        lines.append(f"- easy-tier failure rate: {diff['easy_failure_rate']}")
    if result["failures"] is not None:
        f = result["failures"]
        lines.append(
            f"- raw accuracy {f['raw_accuracy']} vs capability-adjusted {f['adjusted_accuracy']} "
            f"({f['n_excluded']} non-capability failures excluded; share {f['non_capability_share']})"
        )
    if result["coverage_gaps"]:
        lines.append(f"- threat-model nodes with no items: {', '.join(result['coverage_gaps'])}")
    lines.append("")
    if result["difficulty"] is not None:
        lines += [
            "### Accuracy by tier",
            "",
            metrics_table({f"tier {t}": e for t, e in result["difficulty"]["per_tier"].items()}),
            "",
        ]
    if result["failures"] is not None:
        lines += ["### Failure classes", "", "| class | n | share |", "|---|---|---|"]
        for label, block in result["failures"]["classes"].items():
            lines.append(f"| {label} | {block['n']} | {block['share']} |")
        lines.append("")
    for prefix, tags in result["by_tag"].items():
        if prefix in ("tier", "base", "para"):
            continue
        lines += [f"### By `{prefix}` tag", "", metrics_table(tags), ""]
    return "\n".join(lines)


def coverage_markdown(result: dict[str, Any]) -> str:
    """Render ``analysis.coverage_report`` output."""
    run = result["run"]
    lines = [f"## Coverage: run {run['id']} ({run['name']})", ""]
    for prefix, tags in result["counts"].items():
        lines += [f"### `{prefix}` tags", "", "| tag | items | pass rate |", "|---|---|---|"]
        rates = result["pass_rates"].get(prefix, {})
        for tag, n in tags.items():
            lines.append(f"| {tag} | {n} | {_est(rates.get(tag))} |")
        lines.append("")
    if result["structural"]:
        lines += [
            f"### Structural monitor coverage (monitors: {', '.join(result['monitors']) or 'none configured'})",
            "",
            "| threat | evidence | detectable by configured monitors | covered by |",
            "|---|---|---|---|",
        ]
        for tag, block in result["structural"].items():
            lines.append(
                f"| {tag} | {block['evidence']} | {'yes' if block['structurally_detectable'] else 'NO'} | {', '.join(block['covered_by']) or '-'} |"
            )
        lines.append("")
    measured = result.get("measured")
    if measured and measured["monitors"]:
        lines += [
            "### Measured detection (from the monitoring experiment)",
            "",
            "| monitor | view | TPR | FPR | AUROC |",
            "|---|---|---|---|---|",
        ]
        for name, block in measured["monitors"].items():
            auc = block.get("auroc")
            auc_text = f"{auc:.3f}" if isinstance(auc, int | float) else "-"
            lines.append(
                f"| {name} | {block.get('view', '-')} | {_est_dict(block.get('tpr'))} | "
                f"{_est_dict(block.get('fpr'))} | {auc_text} |"
            )
        lines.append("")
    if result["gaps"]:
        lines.append(f"Threat-model nodes with no items in this run: {', '.join(result['gaps'])}")
    return "\n".join(lines) + "\n"
