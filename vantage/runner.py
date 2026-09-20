"""The runner: executes a target over a dataset, scores and stores everything.

Runs are resumable (cases already stored are skipped), concurrent up to a
limit, and retried on provider errors. Monitors are wired into the target's
``on_step`` hook so they can halt a run in progress. ``Runner.rescore`` applies
new scorers or monitors to stored trajectories without calling the target,
which turns the store into a re-analysable corpus.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vantage.cases import Case, Dataset
from vantage.monitors import Monitor
from vantage.results import Score, Verdict
from vantage.scorers import Scorer
from vantage.stats import Estimate, bootstrap_ci
from vantage.store import Store, StoredTrajectory
from vantage.targets import Target
from vantage.trajectory import Trajectory

log = logging.getLogger(__name__)

MONITOR_ERROR_PREFIX = "monitor error: "

ResultHook = Callable[["CaseResult"], None]


@dataclass(slots=True)
class CaseResult:
    """Everything produced for one case in one run.

    Attributes:
        case: The case.
        repeat_idx: Which repeat this is.
        trajectory_id: Store id of the trajectory.
        trajectory: The trajectory.
        scores: Scores from every scorer, in scorer order.
        verdicts: Monitor verdicts, step-level first then final.
    """

    case: Case
    repeat_idx: int
    trajectory_id: int
    trajectory: Trajectory
    scores: list[Score] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)

    def score(self, name: str) -> Score | None:
        """Return the score named ``name``, if present."""
        return next((s for s in self.scores if s.name == name), None)


@dataclass(slots=True)
class RunResult:
    """A completed (or resumed) run.

    Attributes:
        run_id: Store id.
        name: Run name.
        results: One entry per case and repeat.
    """

    run_id: int
    name: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def scorer_names(self) -> list[str]:
        """Scorer names present in the results, in first-seen order."""
        names: list[str] = []
        for r in self.results:
            for s in r.scores:
                if s.name not in names:
                    names.append(s.name)
        return names

    def values(self, scorer: str) -> list[float]:
        """Values of ``scorer`` across results that have it."""
        return [s.value for r in self.results for s in r.scores if s.name == scorer]

    def metric(self, scorer: str, **kwargs: Any) -> Estimate:
        """Mean of ``scorer`` with a bootstrap confidence interval."""
        return bootstrap_ci(self.values(scorer), **kwargs)

    def summary(self) -> dict[str, Estimate]:
        """Metric per scorer."""
        return {name: self.metric(name) for name in self.scorer_names}

    def status_counts(self) -> dict[str, int]:
        """How many trajectories ended in each status."""
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.trajectory.status] = counts.get(r.trajectory.status, 0) + 1
        return counts


class Runner:
    """Execute targets over datasets and persist the results.

    Args:
        store: Where everything is written.
        concurrency: Maximum cases in flight at once.
        retries: Extra attempts when the target returns an error trajectory.
        timeout_s: Per-case wall-clock limit.
        on_result: Called after each case completes, for progress display.
    """

    def __init__(
        self,
        store: Store,
        *,
        concurrency: int = 2,
        retries: int = 2,
        timeout_s: float = 120.0,
        on_result: ResultHook | None = None,
    ) -> None:
        self.store = store
        self.concurrency = max(1, concurrency)
        self.retries = max(0, retries)
        self.timeout_s = timeout_s
        self.on_result = on_result

    async def run(
        self,
        target: Target,
        dataset: Dataset,
        scorers: Sequence[Scorer] = (),
        *,
        monitors: Sequence[Monitor] = (),
        name: str | None = None,
        repeats: int = 1,
        resume: bool = True,
        condition: str | None = None,
        notes: str | None = None,
    ) -> RunResult:
        """Run ``target`` on every case of ``dataset`` and score the results.

        Args:
            target: The thing under test.
            dataset: The cases.
            scorers: Applied to every finished trajectory.
            monitors: Observe each step and the final trajectory; may halt.
            name: Run name; defaults to ``"<dataset>@<target id>"``. A run with
                the same name is resumed when ``resume`` is true, otherwise it
                is deleted and started afresh.
            repeats: How many trajectories to collect per case.
            resume: Skip cases already stored under this run name.
            condition: Experimental condition label stored with the run.
            notes: Free text stored with the run.

        Returns:
            The run result, including previously stored cases when resuming.
        """
        name = name or f"{dataset.name}@{target.id}"
        run_id = self._prepare_run(name, target, dataset, scorers, resume, condition, notes)
        done = self.store.completed_case_ids(run_id)
        pending = [(c, r) for c in dataset for r in range(repeats) if (c.id, r) not in done]
        log.info("run %s: %d pending, %d already stored", name, len(pending), len(done))

        results: list[CaseResult] = self._load_existing(run_id) if done else []
        sem = asyncio.Semaphore(self.concurrency)

        async def one(case: Case, repeat_idx: int) -> None:
            async with sem:
                result = await self._run_case(
                    run_id, target, case, repeat_idx, scorers, monitors, condition
                )
            results.append(result)
            if self.on_result is not None:
                self.on_result(result)

        try:
            async with asyncio.TaskGroup() as group:
                for case, repeat_idx in pending:
                    group.create_task(one(case, repeat_idx))
        except BaseException:
            self.store.set_run_status(run_id, "failed")
            raise
        self.store.set_run_status(run_id, "done")
        order = {case_id: i for i, case_id in enumerate(dataset.ids)}
        results.sort(key=lambda r: (order[r.case.id], r.repeat_idx))
        return RunResult(run_id=run_id, name=name, results=results)

    def _prepare_run(
        self,
        name: str,
        target: Target,
        dataset: Dataset,
        scorers: Sequence[Scorer],
        resume: bool,
        condition: str | None,
        notes: str | None,
    ) -> int:
        existing = self.store.find_run(name)
        if existing is not None:
            if resume:
                self.store.set_run_status(existing, "running")
                return existing
            self.store.delete_run(existing)
        return self.store.create_run(
            name,
            target.id,
            dataset,
            target_cfg=target.config(),
            scorer_names=[s.name for s in scorers],
            condition=condition,
            notes=notes,
        )

    async def _run_case(
        self,
        run_id: int,
        target: Target,
        case: Case,
        repeat_idx: int,
        scorers: Sequence[Scorer],
        monitors: Sequence[Monitor],
        condition: str | None,
    ) -> CaseResult:
        step_verdicts: list[Verdict] = []
        step_monitors = [m for m in monitors if m.every_step]

        async def on_step(partial: Trajectory) -> Verdict | None:
            halting: Verdict | None = None
            for monitor in step_monitors:
                verdict = await monitor.observe(partial, case, final=False)
                verdict.step_index = partial.n_steps - 1
                if verdict.flagged:
                    step_verdicts.append(verdict)
                if verdict.halt and halting is None:
                    halting = verdict
            return halting

        traj = await self._run_with_retries(target, case, on_step if step_monitors else None)
        traj.meta.setdefault("repeat_idx", repeat_idx)
        if condition is not None:
            traj.meta.setdefault("condition", condition)

        scores = await self._apply_scorers(traj, case, scorers)
        final_verdicts = await self._apply_monitors(traj, case, monitors)

        traj_id = self.store.save_trajectory(run_id, case, traj, repeat_idx=repeat_idx)
        self.store.save_scores(traj_id, scores)
        self.store.save_verdicts(traj_id, step_verdicts + final_verdicts)
        return CaseResult(case, repeat_idx, traj_id, traj, scores, step_verdicts + final_verdicts)

    async def _run_with_retries(self, target: Target, case: Case, on_step: Any) -> Trajectory:
        traj: Trajectory | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                traj = await asyncio.wait_for(target.run(case, on_step), timeout=self.timeout_s)
            except TimeoutError:
                traj = Trajectory(
                    case.id, target.id, status="timeout", error=f"exceeded {self.timeout_s}s"
                )
            except Exception as exc:
                log.exception("target %s failed on case %s", target.id, case.id)
                traj = Trajectory(
                    case.id, target.id, status="error", error=f"{type(exc).__name__}: {exc}"
                )
            traj.meta["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
            traj.meta["attempts"] = attempt + 1
            if traj.status != "error":
                break
        assert traj is not None
        return traj

    @staticmethod
    async def _apply_scorers(
        traj: Trajectory, case: Case, scorers: Sequence[Scorer]
    ) -> list[Score]:
        scores: list[Score] = []
        for scorer in scorers:
            try:
                scores.append(await scorer.score(traj, case))
            except Exception as exc:
                log.exception("scorer %s failed on case %s", scorer.name, case.id)
                scores.append(
                    Score(scorer.name, 0.0, passed=None, label="scorer_error", rationale=str(exc))
                )
        return scores

    @staticmethod
    async def _apply_monitors(
        traj: Trajectory, case: Case, monitors: Sequence[Monitor]
    ) -> list[Verdict]:
        return [await Runner._observe_with_retry(monitor, traj, case) for monitor in monitors]

    @staticmethod
    async def _observe_with_retry(monitor: Monitor, traj: Trajectory, case: Case) -> Verdict:
        """Run a monitor on a finished trajectory, retrying once on failure.

        A monitor that still fails yields a verdict with ``score=None`` and a
        reason starting with ``MONITOR_ERROR_PREFIX``; analyses must treat
        such verdicts as missing, never as "not flagged".
        """
        last_error: Exception | None = None
        for _ in range(2):
            try:
                return await monitor.observe(traj, case, final=True)
            except Exception as exc:
                last_error = exc
                log.warning("monitor %s failed on case %s: %s", monitor.name, case.id, exc)
        return Verdict(
            monitor.name,
            False,
            score=None,
            reason=f"{MONITOR_ERROR_PREFIX}{type(last_error).__name__}: {last_error}",
            view=monitor.view,
        )

    def _load_existing(self, run_id: int) -> list[CaseResult]:
        score_rows = self.store.scores_for_run(run_id)
        verdict_rows = self.store.verdicts_for_run(run_id)
        results: list[CaseResult] = []
        for stored in self.store.trajectories(run_id):
            scores = [
                Score(
                    r["scorer"],
                    r["value"],
                    r["passed"],
                    r["label"],
                    r["confidence"],
                    r["rationale"],
                    r["meta"],
                )
                for r in score_rows
                if r["trajectory_id"] == stored.id
            ]
            verdicts = [
                Verdict(
                    r["monitor"],
                    r["flagged"],
                    r["halt"],
                    r["score"],
                    r["reason"],
                    r["step_idx"],
                    r["view"],
                )
                for r in verdict_rows
                if r["trajectory_id"] == stored.id
            ]
            results.append(
                CaseResult(
                    stored.case, stored.repeat_idx, stored.id, stored.trajectory, scores, verdicts
                )
            )
        return results

    async def rescore(
        self,
        run_id: int,
        scorers: Sequence[Scorer] = (),
        *,
        monitors: Sequence[Monitor] = (),
    ) -> RunResult:
        """Apply scorers and monitors to a stored run without re-running the target.

        Existing scores with the same scorer name and existing events from the
        same monitors are replaced.

        Args:
            run_id: The run to re-analyse.
            scorers: Scorers to apply.
            monitors: Monitors to apply on the finished trajectories.

        Returns:
            The run result with the new scores and verdicts included.
        """
        run = self.store.get_run(run_id)
        sem = asyncio.Semaphore(self.concurrency)

        async def one(stored: StoredTrajectory) -> None:
            async with sem:
                scores = await self._apply_scorers(stored.trajectory, stored.case, scorers)
                verdicts = await self._apply_monitors(stored.trajectory, stored.case, monitors)
            self.store.save_scores(stored.id, scores)
            for monitor in monitors:
                self.store.clear_verdicts(stored.id, monitor.name)
            self.store.save_verdicts(stored.id, verdicts)

        async with asyncio.TaskGroup() as group:
            for stored in self.store.trajectories(run_id):
                group.create_task(one(stored))
        return RunResult(run_id=run_id, name=run["name"], results=self._load_existing(run_id))
