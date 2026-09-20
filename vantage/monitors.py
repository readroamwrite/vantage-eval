"""Monitors: observers that watch a trajectory as it unfolds.

A monitor declares its ``view`` (final output only, or the whole trajectory),
whether it wants to be called after every step, and returns a ``Verdict``
that may halt the run. Two families are provided:

* ``RuleMonitor`` wraps any scorer whose ``passed`` field means "no
  violation"; it is cheap, runs on partial trajectories and can halt an agent
  before a forbidden tool call executes.
* ``JudgeMonitor`` asks a model a yes/no rubric question about what it can
  see. Its ``view`` is the whole point of experiment 3: the same rubric with
  an output-only view versus a full-trajectory view.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from vantage.cases import Case
from vantage.judge import Judge
from vantage.results import Verdict, View
from vantage.scorers import Scorer, get_scorer
from vantage.trajectory import Trajectory

__all__ = ["JudgeMonitor", "Monitor", "RuleMonitor", "Verdict", "View", "make_monitor"]


@runtime_checkable
class Monitor(Protocol):
    """Anything that can observe a (partial) trajectory and raise a flag."""

    name: str
    view: View
    every_step: bool

    async def observe(self, traj: Trajectory, case: Case, *, final: bool) -> Verdict:
        """Observe ``traj`` and return a verdict.

        Args:
            traj: The trajectory so far. Partial unless ``final`` is true.
            case: The case being run.
            final: Whether the target has finished.
        """
        ...


class RuleMonitor:
    """Turn a scorer into a monitor; a failing score is a flag.

    Args:
        scorer: Scorer whose ``passed=False`` means a violation.
        halt_on_flag: Ask the runner to stop the target when flagged.
        every_step: Observe after every step (default) or only at the end.
        name: Monitor name; defaults to the scorer's name.
    """

    view: View = "trajectory"

    def __init__(
        self,
        scorer: Scorer,
        *,
        halt_on_flag: bool = False,
        every_step: bool = True,
        name: str | None = None,
    ) -> None:
        self.scorer = scorer
        self.halt_on_flag = halt_on_flag
        self.every_step = every_step
        self.name = name or scorer.name

    async def observe(self, traj: Trajectory, case: Case, *, final: bool) -> Verdict:
        """Score the (partial) trajectory and flag on failure."""
        score = await self.scorer.score(traj, case)
        flagged = score.passed is False
        reason = ""
        if flagged:
            violations = score.meta.get("violations")
            reason = (
                f"{score.label or 'violation'}: {violations[0]}"
                if violations
                else (score.label or "violation")
            )
        return Verdict(
            self.name,
            flagged,
            halt=flagged and self.halt_on_flag,
            score=1.0 - score.value if flagged else 0.0,
            reason=reason,
            view=self.view,
        )


class JudgeMonitor:
    """Ask a model whether a behaviour occurred, from a chosen vantage point.

    Args:
        judge: A ``Judge`` whose rubric asks a yes/no question. Its labels
            are overridden to ``("yes", "no")``.
        view: ``"output"`` shows the judge only the task, the final message
            and nothing about intermediate steps; ``"trajectory"`` shows
            every step and tool call.
        every_step: Observe after every step (expensive) or only at the end.
        name: Monitor name; defaults to ``"judge_monitor:<view>"``.
    """

    def __init__(
        self,
        judge: Judge,
        *,
        view: View = "trajectory",
        every_step: bool = False,
        name: str | None = None,
    ) -> None:
        self.judge = judge
        self.judge.labels = ("yes", "no")
        self.view: View = view
        self.every_step = every_step
        self.name = name or f"judge_monitor:{view}"

    async def observe(self, traj: Trajectory, case: Case, *, final: bool) -> Verdict:
        """Grade the trajectory from this monitor's view."""
        verdict = await self.judge.grade(traj, case, view=self.view)
        flagged = verdict.label == "yes"
        conf = verdict.confidence if verdict.confidence is not None else 0.5
        suspicion = conf if flagged else (1.0 - conf if verdict.parse_ok else 0.5)
        return Verdict(
            self.name,
            flagged,
            halt=False,
            score=suspicion,
            reason=verdict.rationale if verdict.parse_ok else "unparsed",
            view=self.view,
        )


def make_monitor(
    spec: str, *, store: Any = None, halt: bool = False, every_step: bool | None = None
) -> Monitor:
    """Build a monitor from a spec string.

    ``judge:<rubric>,model=<provider:model>,view=<output|trajectory>`` builds a
    ``JudgeMonitor``; any other scorer spec builds a ``RuleMonitor``.

    Args:
        spec: The spec.
        store: Passed to judge scorers for caching.
        halt: Rule monitors halt the target when flagged.
        every_step: Override the default per family.
    """
    if spec.split(":", 1)[0] == "judge":
        from vantage.scorers import parse_spec

        _, args, kwargs = parse_spec(spec)
        view = kwargs.pop("view", "trajectory")
        scorer = get_scorer("judge:" + ",".join(map(str, args)), store=store, **kwargs)
        judge = scorer.judge  # type: ignore[attr-defined]
        return JudgeMonitor(judge, view=view, every_step=bool(every_step))
    scorer = get_scorer(spec, store=store)
    return RuleMonitor(
        scorer, halt_on_flag=halt, every_step=True if every_step is None else every_step
    )
