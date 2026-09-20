"""Experiment 1: how reliable is an LLM judge?

Measures a judge model against rule-checkable gold labels on arithmetic word
problems: accuracy and Cohen's kappa overall and per planted variant, position
bias on pairwise comparisons judged in both orders, self-consistency under
repeated sampling, and calibration of stated confidence. Every number carries
a bootstrap confidence interval.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from vantage.cases import Dataset
from vantage.config import make_client, make_target, parse_model_spec
from vantage.experiments import datagen
from vantage.judge import Judge
from vantage.report.markdown import metrics_table
from vantage.report.plots import grouped_bars, reliability_diagram
from vantage.runner import Runner
from vantage.scorers import get_scorer
from vantage.stats import (
    Estimate,
    accuracy,
    bootstrap_ci,
    calibration,
    confusion,
    kappa_ci,
    position_bias,
    precision_recall,
    self_consistency,
)
from vantage.store import Store
from vantage.targets import ScriptedTarget

NAME = "judge_reliability"
Progress = Callable[[str], None]


@dataclass(slots=True)
class JudgeExperimentConfig:
    """Knobs for the judge reliability experiment."""

    target: str = "ollama:qwen2.5:3b"
    judge: str = "ollama:qwen2.5:7b"
    rubric: str = "data/rubrics/correct.md"
    n_problems: int = 120
    n_items: int = 200
    n_pairs: int = 80
    n_consistency: int = 40
    n_samples: int = 5
    seed: int = 0
    concurrency: int = 2
    out_dir: Path = field(default_factory=lambda: Path("results"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the experiments table."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def _log(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


async def _collect_real_items(
    store: Store,
    cfg: JudgeExperimentConfig,
    problems: list[datagen.Problem],
    progress: Progress | None,
) -> list:
    """Run the target on the problems and label its answers with the numeric rule."""
    dataset = Dataset("judge_problems", [p.to_case() for p in problems])
    target = make_target(cfg.target, store, max_tokens=200)
    _log(progress, f"collecting {len(problems)} real responses from {cfg.target}")
    result = await Runner(store, concurrency=cfg.concurrency).run(
        target, dataset, [get_scorer("numeric")], name=f"{NAME}:responses@{cfg.target}"
    )
    responses = {
        r.case.id: r.trajectory.answer for r in result.results if r.trajectory.status == "ok"
    }
    correct = {
        r.case.id: bool(r.score("numeric") and r.score("numeric").passed) for r in result.results
    }
    return datagen.real_items(problems, responses, correct)


async def _grade_items(
    store: Store,
    cfg: JudgeExperimentConfig,
    items: list,
    judge_scorer: Any,
    progress: Progress | None,
) -> Any:
    dataset = Dataset("judge_items", items)
    target = ScriptedTarget({c.id: c.meta["response"] for c in items}, target_id="candidates")
    _log(progress, f"grading {len(items)} items with {cfg.judge}")
    return await Runner(store, concurrency=cfg.concurrency).run(
        target, dataset, [get_scorer("gold"), judge_scorer], name=f"{NAME}:grading@{cfg.judge}"
    )


def _agreement_block(result: Any) -> dict[str, Any]:
    pred, gold, variants, conf, parse_ok = [], [], [], [], []
    for r in result.results:
        judge, truth = r.score("judge:correct"), r.score("gold")
        if judge is None or truth is None:
            continue
        parse_ok.append(float(judge.label != "unparsed"))
        pred.append(judge.label)
        gold.append(truth.label)
        variants.append(r.case.meta["variant"])
        conf.append(judge.confidence)
    per_variant: dict[str, Estimate] = {}
    for variant in sorted(set(variants)):
        idx = [i for i, v in enumerate(variants) if v == variant]
        per_variant[variant] = accuracy([pred[i] for i in idx], [gold[i] for i in idx])
    usable = [i for i, c in enumerate(conf) if c is not None]
    cal = calibration([conf[i] for i in usable], [pred[i] == gold[i] for i in usable])
    return {
        "n": len(pred),
        "accuracy": accuracy(pred, gold),
        "kappa": kappa_ci(pred, gold, n_boot=500),
        "parse_rate": bootstrap_ci(parse_ok),
        "fail_detection": precision_recall(pred, gold, positive="fail"),
        "confusion": confusion(pred, gold, ["pass", "fail", "unparsed"]),
        "per_variant": per_variant,
        "calibration": cal,
    }


async def _pairwise_block(
    judge: Judge, pairs: list[dict[str, Any]], progress: Progress | None
) -> dict[str, Any]:
    from vantage.trajectory import Trajectory

    _log(progress, f"pairwise comparisons: {len(pairs)} pairs x 2 orders")
    normal, swapped, correct_normal, correct_swapped, long_wrong = [], [], [], [], []
    for pair in pairs:
        case = pair["problem"].to_case()
        a = Trajectory.single_turn(case.id, "cand", case.prompt_text, pair["correct"])
        b = Trajectory.single_turn(case.id, "cand", case.prompt_text, pair["incorrect"])
        v1 = await judge.compare(a, b, case)
        v2 = await judge.compare(a, b, case, swap=True)
        normal.append(v1.label)
        swapped.append(v2.label)
        correct_normal.append(float(v1.label == "A"))
        correct_swapped.append(float(v2.label == "A"))
        long_wrong.append(pair["long_wrong"])
    bias = position_bias(normal, swapped)
    long_idx = [i for i, flag in enumerate(long_wrong) if flag]
    short_idx = [i for i, flag in enumerate(long_wrong) if not flag]
    picked_long = [float(normal[i] == "B") for i in long_idx] + [
        float(swapped[i] == "B") for i in long_idx
    ]
    return {
        "n_pairs": len(pairs),
        "consistency": bias["consistency"],
        "n_inconsistent": bias["n_inconsistent"],
        "first_position_rate": bias["first_position_rate"],
        "accuracy_correct_first": bootstrap_ci(correct_normal),
        "accuracy_correct_second": bootstrap_ci(correct_swapped),
        "accuracy_when_wrong_is_long": bootstrap_ci(
            [correct_normal[i] for i in long_idx] + [correct_swapped[i] for i in long_idx]
        ),
        "accuracy_when_wrong_is_short": bootstrap_ci(
            [correct_normal[i] for i in short_idx] + [correct_swapped[i] for i in short_idx]
        ),
        "picked_longer_wrong_rate": bootstrap_ci(picked_long),
    }


async def _consistency_block(
    judge: Judge, items: list, n_samples: int, progress: Progress | None
) -> dict[str, Any]:
    from vantage.trajectory import Trajectory

    _log(progress, f"self-consistency: {len(items)} items x {n_samples} samples")
    labels_per_item: list[list[str]] = []
    majority_correct: list[float] = []
    for case in items:
        traj = Trajectory.single_turn(case.id, "cand", case.prompt_text, case.meta["response"])
        verdicts = await judge.grade_n(traj, case, n_samples)
        labels = [v.label for v in verdicts]
        labels_per_item.append(labels)
        top = max(set(labels), key=labels.count)
        majority_correct.append(float(top == case.meta["gold"]))
    stats = self_consistency(labels_per_item)
    stats["majority_vote_accuracy"] = bootstrap_ci(majority_correct)
    return stats


def _fmt(est: Estimate) -> str:
    return str(est)


def render_report(
    cfg: JudgeExperimentConfig, results: dict[str, Any], figures: dict[str, str]
) -> str:
    """Render the experiment's markdown report."""
    agree, pair, cons = results["agreement"], results["pairwise"], results["consistency"]
    cal = agree["calibration"]
    lines = [
        "# Experiment 1: judge reliability",
        "",
        f"Judge `{cfg.judge}` graded candidate answers to arithmetic word problems against the rubric "
        f"`{cfg.rubric}`. Gold labels come from a numeric rule (real responses from `{cfg.target}`) "
        "or from construction (planted responses). All intervals are 95% bootstrap.",
        "",
        "## Agreement with gold labels",
        "",
        metrics_table(
            {
                "accuracy": agree["accuracy"],
                "cohen_kappa": agree["kappa"],
                "parse_rate": agree["parse_rate"],
                "fail_precision": agree["fail_detection"]["precision"],
                "fail_recall": agree["fail_detection"]["recall"],
                "fail_f1": agree["fail_detection"]["f1"],
            }
        ),
        "",
        "Confusion (rows = gold, columns = judge):",
        "",
        "| gold \\ judge | pass | fail | unparsed |",
        "|---|---|---|---|",
    ]
    for gold in ("pass", "fail"):
        row = agree["confusion"].get(gold, {})
        lines.append(
            f"| {gold} | {row.get('pass', 0)} | {row.get('fail', 0)} | {row.get('unparsed', 0)} |"
        )
    lines += ["", "### Accuracy per response variant", "", metrics_table(agree["per_variant"])]
    if "per_variant" in figures:
        lines += ["", f"![per-variant accuracy]({figures['per_variant']})"]
    lines += [
        "",
        "## Position bias (pairwise, both orders)",
        "",
        metrics_table(
            {
                "consistency_across_orders": pair["consistency"],
                "first_position_rate_when_inconsistent": pair["first_position_rate"],
                "accuracy_correct_shown_first": pair["accuracy_correct_first"],
                "accuracy_correct_shown_second": pair["accuracy_correct_second"],
                "accuracy_when_wrong_is_long": pair["accuracy_when_wrong_is_long"],
                "accuracy_when_wrong_is_short": pair["accuracy_when_wrong_is_short"],
                "picked_longer_wrong_rate": pair["picked_longer_wrong_rate"],
            }
        ),
        f"\n{pair['n_pairs']} pairs; {pair['n_inconsistent']} changed verdict when the order was swapped.",
        "",
        "## Self-consistency (repeated sampling at temperature 0.7)",
        "",
        metrics_table(
            {
                "majority_agreement": cons["majority_agreement"],
                "unanimity": cons["unanimity"],
                "majority_vote_accuracy": cons["majority_vote_accuracy"],
            }
        ),
        f"\n{cons['n_items']} items x {cons['samples_per_item']} samples.",
        "",
        "## Calibration of stated confidence",
        "",
        f"- ECE: {cal['ece']:.3f}",
        f"- Brier: {cal['brier']:.3f}",
        f"- AUROC of confidence as a correctness predictor: {cal['auroc']:.3f}",
        "",
        "| bin | n | mean confidence | accuracy |",
        "|---|---|---|---|",
    ]
    for lo, hi, n, mc, acc in cal["bins"]:
        lines.append(
            f"| [{lo:.1f}, {hi:.1f}] | {n} | {'-' if n == 0 else f'{mc:.2f}'} | {'-' if n == 0 else f'{acc:.2f}'} |"
        )
    if "reliability" in figures:
        lines += ["", f"![reliability diagram]({figures['reliability']})"]
    return "\n".join(lines) + "\n"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Estimate):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


async def run_experiment(
    store: Store, cfg: JudgeExperimentConfig, progress: Progress | None = None
) -> dict[str, Any]:
    """Run the whole experiment, store its numbers and write the report.

    Returns:
        The results dictionary, JSON-serialisable.
    """
    problems = datagen.generate_problems(cfg.n_problems, cfg.seed)
    real = await _collect_real_items(store, cfg, problems, progress)
    planted = datagen.planted_items(problems, cfg.seed)
    items = datagen.balanced_sample(real, cfg.n_items // 2, cfg.seed) + datagen.balanced_sample(
        planted, cfg.n_items - cfg.n_items // 2, cfg.seed
    )

    judge_client = make_client(cfg.judge, store)
    judge_model = parse_model_spec(cfg.judge)[1]
    judge = Judge(judge_client, judge_model, cfg.rubric, max_tokens=200)
    judge_scorer = get_scorer(
        f"judge:{cfg.rubric}", client=judge_client, model=judge_model, max_tokens=200
    )

    graded = await _grade_items(store, cfg, items, judge_scorer, progress)
    agreement = _agreement_block(graded)
    pairwise = await _pairwise_block(
        judge, datagen.pairwise_pairs(problems, cfg.seed, cfg.n_pairs), progress
    )
    consistency_items = datagen.balanced_sample(planted, cfg.n_consistency, cfg.seed + 1)
    consistency = await _consistency_block(judge, consistency_items, cfg.n_samples, progress)

    results = {
        "agreement": agreement,
        "pairwise": pairwise,
        "consistency": consistency,
        "n_real_items": len(real),
        "n_planted_items": len(planted),
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    if reliability_diagram(
        agreement["calibration"]["bins"],
        cfg.out_dir / "judge_reliability_calibration.png",
        f"{judge_model} calibration",
    ):
        figures["reliability"] = "judge_reliability_calibration.png"
    variants = list(agreement["per_variant"])
    if grouped_bars(
        variants,
        {"accuracy": [agreement["per_variant"][v].point for v in variants]},
        cfg.out_dir / "judge_reliability_variants.png",
        title="judge accuracy per response variant",
        ylabel="accuracy vs gold",
        errors={
            "accuracy": [
                (agreement["per_variant"][v].lo, agreement["per_variant"][v].hi) for v in variants
            ]
        },
    ):
        figures["per_variant"] = "judge_reliability_variants.png"
    report = render_report(cfg, results, figures)
    (cfg.out_dir / "judge_reliability.md").write_text(report, encoding="utf-8")
    store.save_experiment(
        NAME,
        params=cfg.to_dict(),
        results=_jsonable(results),
        run_ids=[graded.run_id],
    )
    _log(progress, f"wrote {cfg.out_dir / 'judge_reliability.md'}")
    return results


def main(cfg: JudgeExperimentConfig, db: Path, progress: Progress | None = print) -> dict[str, Any]:
    """Synchronous entry point used by the CLI."""
    with Store(db) as store:
        return asyncio.run(run_experiment(store, cfg, progress))
