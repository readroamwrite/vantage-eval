"""Record types produced by scorers, monitors and human reviewers.

These are plain dataclasses shared by the scoring, monitoring and storage
layers so that none of those modules has to import the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

View = Literal["output", "trajectory"]


@dataclass(slots=True)
class Score:
    """The result of applying one scorer to one trajectory.

    Attributes:
        name: Scorer name, for example ``"numeric"`` or ``"judge:correct"``.
        value: Numeric result, usually in ``[0, 1]``.
        passed: Boolean reading of ``value`` when the scorer defines one.
        label: Categorical output, for example a failure class or ``"A"``/``"B"``.
        confidence: Self-reported confidence in ``[0, 1]`` for judge scorers.
        rationale: Free-text explanation, mainly from judges.
        meta: Anything else worth keeping, such as the raw judge output.
    """

    name: str
    value: float
    passed: bool | None = None
    label: str | None = None
    confidence: float | None = None
    rationale: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Verdict:
    """The result of one monitor observing a (possibly partial) trajectory.

    Attributes:
        monitor: Monitor name.
        flagged: Whether the monitor raised a flag.
        halt: Whether the monitor asked the runner to stop the target.
        score: Optional graded suspicion in ``[0, 1]``.
        reason: Free-text explanation.
        step_index: Index of the step at which the verdict was produced, or
            ``-1`` when it was produced on the finished trajectory.
        view: What the monitor was allowed to see.
    """

    monitor: str
    flagged: bool
    halt: bool = False
    score: float | None = None
    reason: str = ""
    step_index: int = -1
    view: View = "trajectory"


@dataclass(slots=True)
class Annotation:
    """A human label attached to a trajectory.

    Attributes:
        trajectory_id: Store id of the labelled trajectory.
        reviewer: Who labelled it.
        label: The label, for example ``"pass"`` or ``"tampered"``.
        value: Optional numeric reading of the label.
        note: Free-text comment.
        monitor: The monitor whose question this label answers, when the
            label is about a specific behaviour (for example a finder rubric)
            rather than the trajectory as a whole.
    """

    trajectory_id: int
    reviewer: str
    label: str
    value: float | None = None
    note: str = ""
    monitor: str | None = None
