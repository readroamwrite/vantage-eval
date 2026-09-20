"""Command-line entry point.

Every command is a thin wrapper over library calls so the same behaviour is
available from Python. Output uses rich tables; ``--md`` options write
markdown for reports.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from vantage import __version__
from vantage.analysis import compare_runs, coverage_report, diagnose_run
from vantage.cases import Dataset
from vantage.config import DEFAULT_DB, make_target, read_text_or_path
from vantage.report.markdown import (
    compare_markdown,
    coverage_markdown,
    diagnose_markdown,
    run_summary_markdown,
    summarize_run,
)
from vantage.runner import Runner
from vantage.scorers import get_scorer, list_scorers
from vantage.store import Store

app = typer.Typer(
    name="vantage",
    help="Evaluate LLMs and agents: run, score, judge, monitor, diagnose.",
    no_args_is_help=True,
    add_completion=False,
)
runs_app = typer.Typer(help="List and inspect stored runs.", no_args_is_help=True)
models_app = typer.Typer(help="Check model providers.", no_args_is_help=True)
cache_app = typer.Typer(help="Inspect or clear the response cache.", no_args_is_help=True)
datagen_app = typer.Typer(help="Generate synthetic datasets.", no_args_is_help=True)
experiment_app = typer.Typer(help="Run the showcase experiments.", no_args_is_help=True)
app.add_typer(runs_app, name="runs")
app.add_typer(datagen_app, name="datagen")
app.add_typer(experiment_app, name="experiment")
app.add_typer(models_app, name="models")
app.add_typer(cache_app, name="cache")
console = Console(highlight=False)

DbOption = Annotated[Path, typer.Option("--db", help="SQLite database path.", envvar="VANTAGE_DB")]


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"vantage {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print the version and exit.",
        callback=_print_version,
        is_eager=True,
    ),
) -> None:
    """Evaluate LLMs and agents: run, score, judge, monitor, diagnose."""


@app.command()
def version() -> None:
    """Print the installed vantage version."""
    typer.echo(f"vantage {__version__}")


def _resolve_run(store: Store, ref: str) -> int:
    if ref.isdigit():
        run_id = int(ref)
        store.get_run(run_id)
        return run_id
    found = store.find_run(ref)
    if found is None:
        raise typer.BadParameter(f"no run with id or name {ref!r}")
    return found


def _metrics_table(title: str, summary: dict) -> Table:
    table = Table(title=title)
    table.add_column("scorer")
    table.add_column("mean", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("n", justify="right")
    for name, est in summary["metrics"].items():
        table.add_row(name, f"{est.point:.3f}", f"[{est.lo:.3f}, {est.hi:.3f}]", str(est.n))
    return table


@app.command()
def run(
    dataset: Annotated[Path, typer.Argument(help="JSON, JSONL or CSV dataset file.")],
    target: Annotated[
        str,
        typer.Option(
            "--target", "-t", help="provider:model, e.g. mock:arith or ollama:qwen2.5:3b."
        ),
    ] = "mock:arith",
    scorer: Annotated[
        list[str], typer.Option("--scorer", "-s", help="Scorer spec; repeatable.")
    ] = ["exact"],  # noqa: B006
    name: Annotated[str | None, typer.Option(help="Run name; defaults to dataset@target.")] = None,
    repeats: Annotated[int, typer.Option(min=1, help="Trajectories per case.")] = 1,
    condition: Annotated[
        str | None, typer.Option(help="Condition label stored with the run.")
    ] = None,
    system: Annotated[
        str | None, typer.Option(help="System prompt text or a file containing it.")
    ] = None,
    max_tokens: Annotated[int, typer.Option(help="Output token cap.")] = 256,
    temperature: Annotated[float, typer.Option(help="Sampling temperature.")] = 0.0,
    concurrency: Annotated[int, typer.Option(min=1, help="Cases in flight at once.")] = 2,
    resume: Annotated[
        bool, typer.Option(help="Skip cases already stored under this run name.")
    ] = True,
    monitor: Annotated[list[str], typer.Option("--monitor", help="Monitor spec; repeatable.")] = [],  # noqa: B006
    halt: Annotated[
        bool, typer.Option(help="Rule monitors stop the target when they flag.")
    ] = False,
    agent: Annotated[
        bool, typer.Option(help="Run the built-in agent instead of a single model call.")
    ] = False,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Run a target over a dataset, score every case and store the results."""
    from vantage.monitors import make_monitor

    ds = Dataset.load(dataset)
    with Store(db) as store:
        scorers = [get_scorer(spec, store=store) for spec in scorer]
        monitors = [make_monitor(spec, store=store, halt=halt) for spec in monitor]
        if agent:
            from vantage.config import make_client, parse_model_spec
            from vantage.targets import AgentTarget

            tgt = AgentTarget(
                make_client(target, store),
                parse_model_spec(target)[1],
                system=read_text_or_path(system),
                max_tokens=max_tokens,
            )
        else:
            tgt = make_target(
                target,
                store,
                system=read_text_or_path(system),
                temperature=temperature,
                max_tokens=max_tokens,
            )
        total = len(ds) * repeats
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            bar = progress.add_task(f"{ds.name} on {target}", total=total)
            runner = Runner(
                store, concurrency=concurrency, on_result=lambda _r: progress.advance(bar)
            )
            result = asyncio.run(
                runner.run(
                    tgt,
                    ds,
                    scorers,
                    monitors=monitors,
                    name=name,
                    repeats=repeats,
                    resume=resume,
                    condition=condition,
                )
            )
            progress.update(bar, completed=total)
        summary = summarize_run(store, result.run_id)
        console.print(_metrics_table(f"run {result.run_id}: {result.name}", summary))
        console.print(f"outcomes: {result.status_counts()}")


def _emit(text: str, md: Path | None) -> None:
    if md is None:
        console.print(text, markup=False, highlight=False)
    else:
        md.write_text(text, encoding="utf-8")
        console.print(f"wrote {md}")


@app.command()
def rescore(
    run: Annotated[str, typer.Argument(help="Run id or name.")],
    scorer: Annotated[
        list[str], typer.Option("--scorer", "-s", help="Scorer spec; repeatable.")
    ] = [],  # noqa: B006
    monitor: Annotated[list[str], typer.Option("--monitor", help="Monitor spec; repeatable.")] = [],  # noqa: B006
    concurrency: Annotated[int, typer.Option(min=1)] = 2,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Apply scorers and monitors to a stored run without calling the target again."""
    from vantage.monitors import make_monitor

    with Store(db) as store:
        run_id = _resolve_run(store, run)
        scorers = [get_scorer(spec, store=store) for spec in scorer]
        monitors = [make_monitor(spec, store=store, every_step=False) for spec in monitor]
        result = asyncio.run(
            Runner(store, concurrency=concurrency).rescore(run_id, scorers, monitors=monitors)
        )
        if monitors:
            flagged = sum(1 for r in result.results for v in r.verdicts if v.flagged)
            console.print(
                f"monitors flagged {flagged} verdict(s) across {len(result.results)} trajectories"
            )
        console.print(_metrics_table(f"run {run_id}: {result.name}", summarize_run(store, run_id)))


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(help="Baseline run id or name.")],
    run_b: Annotated[str, typer.Argument(help="Comparison run id or name.")],
    scorer: Annotated[str | None, typer.Option(help="Restrict to one scorer.")] = None,
    md: Annotated[
        Path | None, typer.Option(help="Write markdown here instead of printing.")
    ] = None,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Paired comparison of two runs on their shared cases."""
    with Store(db) as store:
        result = compare_runs(store, _resolve_run(store, run_a), _resolve_run(store, run_b), scorer)
    _emit(compare_markdown(result), md)


@app.command()
def diagnose(
    run: Annotated[str, typer.Argument(help="Run id or name.")],
    scorer: Annotated[str | None, typer.Option(help="Scorer to analyse.")] = None,
    against: Annotated[
        list[str], typer.Option("--against", help="Other runs used as extra columns; repeatable.")
    ] = [],  # noqa: B006
    md: Annotated[Path | None, typer.Option()] = None,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Eval-health report: saturation, dead items, paraphrase consistency, failure classes."""
    with Store(db) as store:
        others = [_resolve_run(store, r) for r in against]
        result = diagnose_run(store, _resolve_run(store, run), scorer=scorer, against=others)
    _emit(diagnose_markdown(result), md)


@app.command()
def coverage(
    run: Annotated[str, typer.Argument(help="Run id or name.")],
    scorer: Annotated[str | None, typer.Option()] = None,
    monitor: Annotated[
        list[str],
        typer.Option(
            "--monitor", help="Monitor specs to check structural coverage for; repeatable."
        ),
    ] = [],  # noqa: B006
    md: Annotated[Path | None, typer.Option()] = None,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Tag coverage, which threats the configured monitors can see, and measured detection rates."""
    from vantage.monitors import make_monitor

    with Store(db) as store:
        monitors = [make_monitor(spec, store=store, every_step=False) for spec in monitor]
        result = coverage_report(store, _resolve_run(store, run), monitors=monitors, scorer=scorer)
    _emit(coverage_markdown(result), md)


@runs_app.command("list")
def runs_list(db: DbOption = DEFAULT_DB) -> None:
    """List stored runs, newest first."""
    with Store(db) as store:
        table = Table(title="runs")
        for col in ("id", "name", "target", "dataset", "condition", "status", "n"):
            table.add_column(col, overflow="fold")
        for row in store.list_runs():
            table.add_row(
                str(row["id"]),
                row["name"],
                row["target_id"],
                row["dataset_name"],
                row["condition"] or "-",
                row["status"],
                str(row["n_trajectories"]),
            )
        console.print(table)


@runs_app.command("show")
def runs_show(
    run: Annotated[str, typer.Argument(help="Run id or name.")],
    db: DbOption = DEFAULT_DB,
) -> None:
    """Show a run's metrics with confidence intervals."""
    with Store(db) as store:
        run_id = _resolve_run(store, run)
        summary = summarize_run(store, run_id)
        console.print(_metrics_table(f"run {run_id}: {summary['run']['name']}", summary))
        console.print(f"outcomes: {summary['statuses']}")


@app.command()
def report(
    run: Annotated[str, typer.Argument(help="Run id or name.")],
    md: Annotated[
        Path | None, typer.Option(help="Write markdown to this file instead of stdout.")
    ] = None,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Render a run as markdown."""
    with Store(db) as store:
        text = run_summary_markdown(store, _resolve_run(store, run))
    if md is None:
        typer.echo(text)
    else:
        md.write_text(text, encoding="utf-8")
        console.print(f"wrote {md}")


@models_app.command("check")
def models_check(
    smoke: Annotated[
        bool, typer.Option(help="Send a one-token prompt to every installed Ollama model.")
    ] = True,
) -> None:
    """Report which providers are reachable and which models are installed."""
    import os

    from vantage.models import ChatRequest, ModelError
    from vantage.models.ollama import DEFAULT_HOST, OllamaClient

    async def check() -> None:
        client = OllamaClient()
        try:
            models = await client.list_models()
        except ModelError as exc:
            console.print(f"[red]ollama[/red]: {exc}")
            models = []
        else:
            console.print(
                f"[green]ollama[/green]: reachable at {DEFAULT_HOST}, {len(models)} model(s)"
            )
        table = Table(title="ollama models")
        for col in ("name", "size", "smoke test", "latency"):
            table.add_column(col)
        for m in models:
            size = f"{m.get('size', 0) / 1e9:.1f} GB"
            if not smoke:
                table.add_row(m["name"], size, "-", "-")
                continue
            req = ChatRequest.create(
                m["name"], [{"role": "user", "content": "Reply with OK."}], max_tokens=4
            )
            try:
                resp = await client.chat(req)
                table.add_row(
                    m["name"],
                    size,
                    f"[green]{resp.content.strip()[:20]!r}[/green]",
                    f"{resp.latency_ms:.0f} ms",
                )
            except ModelError as exc:
                table.add_row(m["name"], size, f"[red]{exc}[/red]", "-")
        if models:
            console.print(table)
        await client.aclose()
        key = "set" if os.environ.get("ANTHROPIC_API_KEY") else "not set"
        console.print(f"anthropic: ANTHROPIC_API_KEY {key}")

    asyncio.run(check())


@cache_app.command("stats")
def cache_stats(db: DbOption = DEFAULT_DB) -> None:
    """Show response cache entries and hits."""
    with Store(db) as store:
        console.print(store.cache_stats())


@cache_app.command("clear")
def cache_clear(db: DbOption = DEFAULT_DB) -> None:
    """Delete every cached model response."""
    with Store(db) as store:
        console.print(f"removed {store.cache_clear()} cached responses")


@datagen_app.command("judge")
def datagen_judge(
    n: Annotated[int, typer.Option(help="Number of word problems.")] = 120,
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory.")] = Path("data"),
) -> None:
    """Write the arithmetic problems and planted judge items used by experiment 1."""
    from vantage.experiments import datagen

    problems = datagen.generate_problems(n, seed)
    out.mkdir(parents=True, exist_ok=True)
    Dataset("judge_problems", [p.to_case() for p in problems]).save(out / "judge_problems.jsonl")
    Dataset("judge_planted", datagen.planted_items(problems, seed)).save(
        out / "judge_planted.jsonl"
    )
    console.print(f"wrote {n} problems and {len(datagen.VARIANTS) and n} planted items to {out}/")


@datagen_app.command("robustness")
def datagen_robustness(
    per_tier: Annotated[int, typer.Option(help="Base items per tier (x3 paraphrases).")] = 8,
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("data"),
) -> None:
    """Write the tiered arithmetic set with paraphrases and probes used by experiment 2."""
    from vantage.experiments import datagen

    out.mkdir(parents=True, exist_ok=True)
    ds = datagen.robustness_dataset(seed, per_tier)
    ds.save(out / "robustness.jsonl")
    console.print(f"wrote {len(ds)} cases to {out / 'robustness.jsonl'}")


@experiment_app.command("robustness")
def experiment_robustness(
    target: Annotated[str, typer.Option(help="Primary model.")] = "ollama:qwen2.5:3b",
    second: Annotated[
        str | None, typer.Option(help="Second model for discrimination stats; empty to skip.")
    ] = "ollama:llama3.2:3b",
    per_tier: Annotated[int, typer.Option()] = 8,
    repeats: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option()] = 0,
    concurrency: Annotated[int, typer.Option(min=1)] = 2,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("results"),
    db: DbOption = DEFAULT_DB,
) -> None:
    """Experiment 2: saturation, dead items, paraphrase consistency, failure classes, sandbagging signal."""
    from vantage.experiments.robustness import RobustnessConfig, main

    cfg = RobustnessConfig(
        target=target,
        second=second or None,
        per_tier=per_tier,
        repeats=repeats,
        seed=seed,
        concurrency=concurrency,
        out_dir=out,
    )
    results = main(cfg, db, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print(
        f"normal {results['compare']['mean_a']}  sandbag {results['compare']['mean_b']}  diff {results['compare']['diff']}"
    )


@experiment_app.command("benchmark")
def experiment_benchmark(
    dataset: Annotated[
        Path, typer.Option(help="Dataset written by `vantage datagen gsm8k`.")
    ] = Path("data/gsm8k.jsonl"),
    target: Annotated[str, typer.Option(help="Primary model.")] = "ollama:qwen2.5:3b",
    second: Annotated[
        str | None, typer.Option(help="Second model for discrimination stats; empty to skip.")
    ] = "ollama:llama3.2:3b",
    repeats: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option()] = 0,
    concurrency: Annotated[int, typer.Option(min=1)] = 2,
    max_tokens: Annotated[int, typer.Option(min=16)] = 400,
    memorisation: Annotated[
        bool, typer.Option(help="Run the verbatim-vs-perturbed completion probe.")
    ] = True,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("results"),
    db: DbOption = DEFAULT_DB,
) -> None:
    """Experiment 4: the eval-health diagnostics and a memorisation probe on GSM8K."""
    from vantage.experiments.benchmark import BenchmarkConfig, main

    cfg = BenchmarkConfig(
        dataset=dataset,
        target=target,
        second=second or None,
        repeats=repeats,
        seed=seed,
        concurrency=concurrency,
        max_tokens=max_tokens,
        memorisation=memorisation,
        out_dir=out,
    )
    results = main(cfg, db, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    console.print(
        f"normal {results['compare']['mean_a']}  sandbag {results['compare']['mean_b']}  "
        f"diff {results['compare']['diff']}"
    )
    for warning in results["diagnose"]["warnings"]:
        console.print(f"[yellow]{warning}[/yellow]")


@experiment_app.command("monitoring")
def experiment_monitoring(
    agent_model: Annotated[
        str, typer.Option(help="Model that drives the agent.")
    ] = "ollama:qwen2.5:3b",
    judge: Annotated[str, typer.Option(help="Judge model for the monitors.")] = "ollama:qwen2.5:7b",
    n_scripted_tasks: Annotated[int, typer.Option(help="Tasks per scripted behaviour class.")] = 8,
    max_steps: Annotated[int, typer.Option()] = 8,
    concurrency: Annotated[int, typer.Option(min=1)] = 2,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("results"),
    db: DbOption = DEFAULT_DB,
) -> None:
    """Experiment 3: compare an output-only monitor with a trajectory monitor on agent runs."""
    from vantage.experiments.monitoring import MonitoringConfig, main

    cfg = MonitoringConfig(
        agent_model=agent_model,
        judge=judge,
        n_scripted_tasks=n_scripted_tasks,
        max_steps=max_steps,
        concurrency=concurrency,
        out_dir=out,
    )
    results = main(cfg, db, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    for name, block in results["monitors"].items():
        console.print(f"{name}: TPR {block['tpr']}  FPR {block['fpr']}  AUROC {block['auroc']:.3f}")


@experiment_app.command("judge")
def experiment_judge(
    target: Annotated[
        str, typer.Option(help="Model whose real answers are graded.")
    ] = "ollama:qwen2.5:3b",
    judge: Annotated[str, typer.Option(help="Judge model.")] = "ollama:qwen2.5:7b",
    n_problems: Annotated[int, typer.Option()] = 120,
    n_items: Annotated[int, typer.Option(help="Graded items (half real, half planted).")] = 200,
    n_pairs: Annotated[int, typer.Option()] = 80,
    n_consistency: Annotated[int, typer.Option()] = 40,
    n_samples: Annotated[int, typer.Option()] = 5,
    seed: Annotated[int, typer.Option()] = 0,
    concurrency: Annotated[int, typer.Option(min=1)] = 2,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("results"),
    db: DbOption = DEFAULT_DB,
) -> None:
    """Experiment 1: measure a judge's accuracy, position bias, consistency and calibration."""
    from vantage.experiments.judge_reliability import JudgeExperimentConfig, main

    cfg = JudgeExperimentConfig(
        target=target,
        judge=judge,
        n_problems=n_problems,
        n_items=n_items,
        n_pairs=n_pairs,
        n_consistency=n_consistency,
        n_samples=n_samples,
        seed=seed,
        concurrency=concurrency,
        out_dir=out,
    )
    results = main(cfg, db, progress=lambda m: console.print(f"[dim]{m}[/dim]"))
    agree = results["agreement"]
    console.print(
        f"accuracy {agree['accuracy']}  kappa {agree['kappa']}  parse rate {agree['parse_rate']}"
    )


@datagen_app.command("gsm8k")
def datagen_gsm8k(
    source: Annotated[
        str | None,
        typer.Option(
            help="Local GSM8K test.jsonl or a URL; downloads the official split if omitted."
        ),
    ] = None,
    per_tier: Annotated[int, typer.Option(help="Items per difficulty tier.")] = 50,
    seed: Annotated[int, typer.Option()] = 0,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("data"),
) -> None:
    """Write the stratified GSM8K sample used by experiment 4."""
    from vantage.experiments.benchmark import gsm8k_dataset, load_gsm8k_rows

    out.mkdir(parents=True, exist_ok=True)
    rows = load_gsm8k_rows(source)
    ds = gsm8k_dataset(rows, per_tier=per_tier, seed=seed)
    ds.save(out / "gsm8k.jsonl")
    console.print(f"wrote {len(ds)} of {len(rows)} GSM8K items to {out / 'gsm8k.jsonl'}")


@datagen_app.command("agent")
def datagen_agent(out: Annotated[Path, typer.Option("--out", "-o")] = Path("data")) -> None:
    """Write the coding tasks (fair and pressured variants) used by the agent and experiment 3."""
    from vantage.experiments import datagen

    out.mkdir(parents=True, exist_ok=True)
    ds = datagen.agent_dataset()
    ds.save(out / "agent_tasks.jsonl")
    console.print(f"wrote {len(ds)} tasks to {out / 'agent_tasks.jsonl'}")


@app.command()
def agent(
    task: Annotated[
        str, typer.Argument(help="Task id such as fizzbuzz:fair, or a JSONL file plus --case.")
    ],
    model: Annotated[str, typer.Option("--model", "-m")] = "ollama:qwen2.5:3b",
    permissive: Annotated[
        bool, typer.Option(help="Use the permissive system prompt (test edits allowed).")
    ] = False,
    max_steps: Annotated[int, typer.Option()] = 8,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Run the built-in agent on one task and print the trajectory."""
    from vantage.agent.react import PERMISSIVE_SYSTEM
    from vantage.config import make_client, parse_model_spec
    from vantage.experiments import datagen
    from vantage.judge import render_transcript
    from vantage.targets import AgentTarget

    ds = datagen.agent_dataset()
    try:
        case = ds.get(task)
    except KeyError as exc:
        raise typer.BadParameter(f"unknown task {task!r}; try one of {ds.ids[:4]} ...") from exc
    with Store(db) as store:
        client = make_client(model, store)
        target = AgentTarget(
            client,
            parse_model_spec(model)[1],
            system=PERMISSIVE_SYSTEM if permissive else None,
            max_steps=max_steps,
        )
        traj = asyncio.run(target.run(case))
    console.print(render_transcript(traj, "trajectory"), markup=False, highlight=False)
    console.print(
        f"\nstatus={traj.status} submitted={traj.meta.get('submitted')} tests_passed={traj.meta.get('tests_passed')} "
        f"tampered={traj.meta.get('tampered')} changed={traj.meta.get('sandbox_diff', {}).get('protected_touched')}"
    )


@app.command()
def search(
    rubric: Annotated[
        str, typer.Argument(help="Behaviour to look for, as a yes/no question or a rubric file.")
    ],
    run: Annotated[str | None, typer.Option(help="Restrict to one run id or name.")] = None,
    fts: Annotated[
        str | None,
        typer.Option(help="Full-text pre-filter on step contents, e.g. 'conftest OR test_task'."),
    ] = None,
    limit: Annotated[int, typer.Option(min=1)] = 50,
    model: Annotated[str, typer.Option("--model", "-m", help="Judge model.")] = "ollama:qwen2.5:7b",
    view: Annotated[str, typer.Option(help="output or trajectory")] = "trajectory",
    concurrency: Annotated[int, typer.Option(min=1)] = 2,
    audit: Annotated[
        int,
        typer.Option(
            min=0, help="With --fts: also judge N random excluded trajectories to estimate recall."
        ),
    ] = 0,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Find stored trajectories that match a behaviour rubric."""
    from vantage.config import make_client, parse_model_spec
    from vantage.finder import find_behaviour
    from vantage.judge import load_rubric

    if view not in ("output", "trajectory"):
        raise typer.BadParameter("view must be 'output' or 'trajectory'")
    with Store(db) as store:
        run_id = _resolve_run(store, run) if run else None
        client = make_client(model, store)
        result = asyncio.run(
            find_behaviour(
                store,
                load_rubric(rubric),
                client,
                parse_model_spec(model)[1],
                run_id=run_id,
                fts=fts,
                limit=limit,
                view=view,  # type: ignore[arg-type]
                concurrency=concurrency,
                audit=audit,
            )
        )
    table = Table(title=f"{len(result.hits)} hit(s) in {result.examined} examined trajectories")
    for col in ("trajectory", "run", "case", "score", "rationale"):
        table.add_column(col, overflow="fold")
    for hit in result.hits:
        table.add_row(
            str(hit.trajectory_id),
            str(hit.run_id),
            hit.case_id,
            f"{hit.score:.2f}",
            hit.rationale[:160],
        )
    console.print(table)
    if result.audit is not None:
        a = result.audit
        console.print(
            f"pre-filter audit: judged {a.n_audited} of {a.n_excluded} excluded trajectories, "
            f"{a.n_flagged} flagged; estimated missed {a.estimated_missed:.1f}; "
            f"pre-filter recall {a.prefilter_recall}"
        )
    if result.n_labelled:
        console.print(f"precision from {result.n_labelled} human-labelled hits: {result.precision}")
    else:
        console.print(
            "no human labels on these hits yet; label them in the review queue to measure precision"
        )
    console.print(
        f"stored as monitor events under {result.monitor_name!r}; they now show in the trajectory viewer and review queue"
    )


@app.command(name="import")
def import_command(
    file: Annotated[
        Path,
        typer.Argument(help="Chat export (.json/.jsonl) or a User:/Assistant: text transcript."),
    ],
    name: Annotated[
        str | None, typer.Option(help="Run name suffix; defaults to the file stem.")
    ] = None,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Import an outside conversation so it can be searched, reviewed and mapped."""
    from vantage.importers import import_chat

    with Store(db) as store:
        run_id, traj_id = import_chat(store, file, name=name)
    console.print(f"imported {file} as run {run_id}, trajectory {traj_id}")


@app.command(name="map")
def map_command(
    trajectory: Annotated[
        int, typer.Argument(help="Trajectory id (see the dashboard or `vantage import`).")
    ],
    model: Annotated[str, typer.Option("--model", "-m")] = "ollama:qwen2.5:7b",
    mermaid: Annotated[
        Path | None, typer.Option(help="Write the Mermaid mindmap to this file.")
    ] = None,
    force: Annotated[bool, typer.Option(help="Rebuild even if a map is cached.")] = False,
    db: DbOption = DEFAULT_DB,
) -> None:
    """Build a traceable mind map of a conversation: topics, directions, decisions, with step references."""
    from vantage.config import make_client, parse_model_spec
    from vantage.convmap import ConvMap, build

    model_name = parse_model_spec(model)[1]
    with Store(db) as store:
        cached = None if force else store.get_conv_map(trajectory, model_name)
        if cached is not None:
            conv = ConvMap.from_dict(cached["tree"])
        else:
            stored = store.get_trajectory(trajectory)
            conv = asyncio.run(build(stored.trajectory, make_client(model, store), model_name))
            store.save_conv_map(trajectory, model_name, conv.to_dict(), conv.to_mermaid())
    text = conv.to_mermaid()
    if mermaid is not None:
        mermaid.write_text("```mermaid\n" + text + "\n```\n", encoding="utf-8")
        console.print(f"wrote {mermaid}")
    else:
        console.print(text, markup=False, highlight=False)
    console.print(
        f"\n{len(conv.root.walk())} nodes; step references point into trajectory {trajectory}"
    )


@app.command()
def dashboard(
    db: DbOption = DEFAULT_DB,
    port: Annotated[int, typer.Option()] = 8501,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
) -> None:
    """Open the dashboard (a local web app) on the given database."""
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise typer.BadParameter(
            'install the dashboard extra: pip install -e ".[dashboard]"'
        ) from exc

    from vantage.dashboard.api import create_app

    console.print(f"starting dashboard on http://{host}:{port} (database {db})")
    uvicorn.run(create_app(str(db)), host=host, port=port, log_level="warning")


@app.command()
def scorers() -> None:
    """List available scorer names."""
    for name in list_scorers():
        typer.echo(name)


def main() -> None:
    """Run the CLI."""
    app()


if __name__ == "__main__":
    main()
