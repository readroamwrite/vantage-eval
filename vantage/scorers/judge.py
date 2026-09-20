"""Scorer that wraps a ``Judge`` so rubric grading plugs into runs like any rule."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vantage.cases import Case
from vantage.judge import UNPARSED, Judge
from vantage.models import ModelClient
from vantage.results import Score, View
from vantage.scorers import register
from vantage.trajectory import Trajectory


def rubric_stem(rubric: str) -> str:
    """Short stable name for a rubric: its file stem, or a hash of inline text.

    Two different inline rubrics must not share a scorer name, because scores
    are unique per ``(trajectory, scorer)`` and the second would overwrite
    the first.
    """
    path = Path(rubric)
    if path.suffix in (".md", ".txt") and path.is_file():
        return path.stem
    return hashlib.sha256(rubric.strip().encode("utf-8")).hexdigest()[:8]


@register("judge")
class JudgeScorer:
    """Grade with a rubric and turn the verdict into a ``Score``.

    Spec form: ``judge:<rubric path or text>,model=<provider:model>,view=output``.

    Args:
        rubric: Rubric text or file path.
        model: ``provider:model`` spec used when ``client`` is not given.
        client: Model client to use; overrides ``model`` lookup.
        view: What the judge sees.
        labels: Allowed labels.
        pass_labels: Labels that count as value ``1``.
        name: Score name; defaults to ``"judge:<rubric stem>"``.
        **judge_kwargs: Passed through to ``Judge``.
    """

    def __init__(
        self,
        rubric: str,
        *,
        model: str | None = None,
        client: ModelClient | None = None,
        view: View = "output",
        labels: Sequence[str] = ("pass", "fail"),
        pass_labels: Sequence[str] = ("pass",),
        name: str | None = None,
        store: Any = None,
        **judge_kwargs: Any,
    ) -> None:
        if client is None:
            if model is None:
                raise ValueError("judge scorer needs either client= or model=provider:model")
            from vantage.config import make_client, parse_model_spec

            client = make_client(model, store)
            model_name = parse_model_spec(model)[1]
        else:
            model_name = model or getattr(client, "model", "unknown")
        self.name = name or f"judge:{rubric_stem(rubric)}"
        self.view: View = view
        self.pass_labels = set(pass_labels)
        self.judge = Judge(
            client, model_name, rubric, labels=labels, name=self.name, **judge_kwargs
        )

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Grade ``traj`` and map the label to a numeric score."""
        verdict = await self.judge.grade(traj, case, view=self.view)
        if not verdict.parse_ok:
            return Score(
                self.name,
                0.0,
                passed=None,
                label=UNPARSED,
                rationale=verdict.rationale,
                meta={"parse_ok": False, "raw": verdict.raw, "judge": self.judge.config_hash},
            )
        passed = verdict.label in self.pass_labels
        return Score(
            self.name,
            1.0 if passed else 0.0,
            passed=passed,
            label=verdict.label,
            confidence=verdict.confidence,
            rationale=verdict.rationale,
            meta={
                "parse_ok": True,
                "view": self.view,
                "attempts": verdict.meta.get("attempts", 1),
                "judge": self.judge.config_hash,
            },
        )
