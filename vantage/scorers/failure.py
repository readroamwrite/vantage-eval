"""Failure classification: separate capability failures from everything else.

A wrong answer says something about capability. A refusal, a formatting
error, a timeout or a tool crash usually does not, yet all of them count as
failures in a plain accuracy number. This scorer labels each trajectory with
a failure class so that reports can show both raw accuracy and
capability-adjusted accuracy, and so that non-capability failure rates are
visible instead of silently lowering the score.
"""

from __future__ import annotations

import re
from typing import Any

from vantage.cases import Case
from vantage.results import Score
from vantage.scorers import Scorer, get_scorer, register
from vantage.scorers.rules import parse_json
from vantage.trajectory import Trajectory

CAPABILITY_FAILURES = frozenset({"wrong_answer"})
NON_CAPABILITY_FAILURES = frozenset(
    {"refusal", "format_error", "no_answer", "tool_error", "timeout", "halted", "error"}
)
ALL_LABELS = ("correct", *sorted(CAPABILITY_FAILURES), *sorted(NON_CAPABILITY_FAILURES))

_REFUSAL_PATTERNS = [
    r"\bI(?:'m| am) (?:sorry|afraid)\b[^.!?\n]{0,40}\bI (?:can(?:'t|not)|won't|am unable)\b",
    r"\bI (?:can(?:'t|not)|won't|will not|am unable to|am not able to) (?:help|assist|provide|answer|comply|do that)\b",
    r"\bnot able to help\b",
    r"\bcannot assist\b",
    r"\bas an AI\b[^.!?\n]{0,80}\bI (?:can(?:'t|not)|am unable)\b",
    r"\bI must decline\b",
    r"\bI(?:'m| am) not (?:comfortable|willing)\b",
]
_REFUSAL = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE | re.DOTALL)


def looks_like_refusal(text: str) -> bool:
    """Return whether ``text`` reads as a refusal to answer.

    Deliberately pattern-based so it is cheap and auditable; ambiguous cases
    can be sent to a judge or a human via the review queue.
    """
    head = text.strip()[:400]
    return bool(_REFUSAL.search(head))


@register("failure")
class FailureClassifier:
    """Label each trajectory with a failure class.

    Args:
        correctness: Spec of the scorer that decides correct vs wrong, for
            example ``"numeric"`` or ``"exact"``.
        expect_json: Treat un-parseable JSON output as a format error. Also
            triggered per case by the ``format:json`` tag.
        name: Score name.
    """

    def __init__(
        self, correctness: str = "numeric", expect_json: bool = False, name: str = "failure"
    ) -> None:
        self.correctness: Scorer = (
            get_scorer(correctness) if isinstance(correctness, str) else correctness
        )
        self.expect_json = expect_json
        self.name = name

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Classify ``traj``; ``value`` is ``1`` only for ``correct``."""
        label, detail = await self._classify(traj, case)
        return Score(
            self.name,
            1.0 if label == "correct" else 0.0,
            passed=label == "correct",
            label=label,
            meta={
                "capability_failure": label in CAPABILITY_FAILURES,
                "non_capability_failure": label in NON_CAPABILITY_FAILURES,
                **detail,
            },
        )

    async def _classify(self, traj: Trajectory, case: Case) -> tuple[str, dict[str, Any]]:
        if traj.status in ("timeout", "error", "halted"):
            return traj.status, {"error": traj.error}
        if traj.status == "format_error":
            return "format_error", {"error": traj.error}
        if any(s.role == "tool" and s.meta.get("error") for s in traj.steps):
            return "tool_error", {}
        answer = traj.answer
        if not answer.strip():
            return "no_answer", {}
        if looks_like_refusal(answer):
            return "refusal", {}
        wants_json = self.expect_json or case.has_tag("format:json")
        if wants_json and parse_json(answer) is None:
            return "format_error", {"reason": "expected JSON"}
        result = await self.correctness.score(traj, case)
        if result.passed:
            return "correct", {"correctness": result.name}
        if result.label == "no_number":
            return "no_answer", {"correctness": result.name}
        return "wrong_answer", {"correctness": result.name}
