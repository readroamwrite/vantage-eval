"""Deterministic rule-based scorers.

These are cheap, reproducible and need no model: exact match, substring,
regex, JSON validity, numeric tolerance, multiple-choice letter and length
limits. Each returns ``value`` in ``{0, 1}`` with ``passed`` set, except where
noted.
"""

from __future__ import annotations

import json
import re
from typing import Any

from vantage.cases import Case
from vantage.results import Score
from vantage.scorers import register
from vantage.trajectory import Trajectory

_WHITESPACE = re.compile(r"\s+")
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FINAL_CUE = re.compile(r"final answer\s*(?:is|:|=)?[\s*$]*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
_ANSWER_CUE = re.compile(
    r"(?:answer|result|total)\s*(?:is|:|=)[\s*$]*(-?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE
)
_CHOICE = re.compile(r"(?<![A-Za-z])(?:\(([A-Ea-e])\)|([A-E]))(?![A-Za-z])")
_JSON_BLOCK = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def normalize_text(text: str) -> str:
    """Lower-case, trim and collapse whitespace."""
    return _WHITESPACE.sub(" ", text.strip()).casefold()


def extract_number(text: str) -> float | None:
    """Pull the answer number out of free text.

    Takes, in order of preference, the last number after ``"final answer"``,
    the last number after a weaker cue such as ``"answer is"`` or ``"total:"``,
    and otherwise the last number in the text. Later cues win because a
    solution often says "the total is 5 per box" on the way to its answer.
    Thousands separators are removed.

    Returns:
        The number, or ``None`` if the text contains none.
    """
    for cue in (_FINAL_CUE, _ANSWER_CUE):
        found = cue.findall(text)
        if found:
            return float(found[-1].replace(",", ""))
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def extract_choice(text: str) -> str | None:
    """First standalone option letter A-E in ``text``, upper-cased.

    Lower-case letters only count inside parentheses, so the article in
    "a tie between B and C" is not read as option A.
    """
    match = _CHOICE.search(text)
    if match is None:
        return None
    return (match.group(1) or match.group(2)).upper()


def _bool_score(name: str, passed: bool, **meta: Any) -> Score:
    return Score(name=name, value=1.0 if passed else 0.0, passed=passed, meta=meta)


@register("exact")
class ExactMatch:
    """Pass when the answer equals ``case.expected`` after normalisation.

    Args:
        normalize: Ignore case and surrounding/repeated whitespace.
    """

    name = "exact"

    def __init__(self, normalize: bool = True) -> None:
        self.normalize = normalize

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Compare the answer text with the expected text."""
        got, want = traj.answer, str(case.expected)
        if self.normalize:
            got, want = normalize_text(got), normalize_text(want)
        return _bool_score(self.name, got == want)


@register("contains")
class Contains:
    """Pass when the answer contains a needle.

    Args:
        needle: Text to look for. Defaults to ``case.expected``.
        case_sensitive: Match case exactly.
    """

    name = "contains"

    def __init__(self, needle: str | None = None, case_sensitive: bool = False) -> None:
        self.needle = needle
        self.case_sensitive = case_sensitive

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Check for the needle in the answer."""
        needle = self.needle if self.needle is not None else str(case.expected)
        haystack = traj.answer
        if not self.case_sensitive:
            needle, haystack = needle.casefold(), haystack.casefold()
        return _bool_score(self.name, needle in haystack, needle=needle)


@register("regex")
class Regex:
    """Pass when a regular expression matches the answer.

    Args:
        pattern: The pattern. Defaults to ``case.expected``.
        full: Require the whole answer to match rather than any part.
    """

    name = "regex"

    def __init__(self, pattern: str | None = None, full: bool = False) -> None:
        self.pattern = pattern
        self.full = full

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Apply the pattern with ``re.search`` or ``re.fullmatch``."""
        pattern = self.pattern if self.pattern is not None else str(case.expected)
        compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        matched = (
            compiled.fullmatch(traj.answer.strip()) if self.full else compiled.search(traj.answer)
        )
        return _bool_score(self.name, matched is not None, pattern=pattern)


@register("json_valid")
class JsonValid:
    """Pass when the answer is (or contains) valid JSON.

    Args:
        lenient: Also accept JSON embedded in surrounding prose or code fences.
        require_keys: Keys the top-level object must contain.
    """

    name = "json_valid"

    def __init__(self, lenient: bool = True, require_keys: list[str] | None = None) -> None:
        self.lenient = lenient
        self.require_keys = list(require_keys or [])

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Parse the answer as JSON and check required keys."""
        parsed = parse_json(traj.answer, lenient=self.lenient)
        if parsed is None:
            return _bool_score(self.name, False, reason="not valid JSON")
        missing = [k for k in self.require_keys if not (isinstance(parsed, dict) and k in parsed)]
        return _bool_score(self.name, not missing, missing_keys=missing)


def parse_json(text: str, *, lenient: bool = True) -> Any | None:
    """Parse ``text`` as JSON, optionally extracting an embedded object.

    Returns:
        The parsed value, or ``None`` when nothing parses.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if not lenient:
            return None
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(fenced)
    except json.JSONDecodeError:
        pass
    block = _JSON_BLOCK.search(text)
    if block is None:
        return None
    try:
        return json.loads(block.group(1))
    except json.JSONDecodeError:
        return None


@register("numeric")
class Numeric:
    """Pass when the answer's number is within a tolerance of ``case.expected``.

    Args:
        tol: Absolute tolerance, or relative when ``relative`` is set.
        relative: Interpret ``tol`` as a fraction of the expected value.
    """

    name = "numeric"

    def __init__(self, tol: float = 1e-6, relative: bool = False) -> None:
        self.tol = float(tol)
        self.relative = relative

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Extract a number from the answer and compare it."""
        try:
            want = float(case.expected)
        except (TypeError, ValueError):
            return _bool_score(self.name, False, reason="expected is not numeric")
        got = extract_number(traj.answer)
        if got is None:
            return Score(self.name, 0.0, passed=False, label="no_number", meta={"expected": want})
        limit = self.tol * abs(want) if self.relative else self.tol
        passed = abs(got - want) <= limit
        return Score(
            self.name, 1.0 if passed else 0.0, passed=passed, meta={"got": got, "expected": want}
        )


def lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def overlap_f1(answer: str, reference: str, *, truncate: bool = True) -> float:
    """ROUGE-L style F-measure between two texts over case-folded words.

    Args:
        answer: The model's text.
        reference: The text it should match.
        truncate: Cut ``answer`` to the reference's word count first, so that
            a model which keeps writing past the reference is not penalised.
    """
    want = normalize_text(reference).split()
    got = normalize_text(answer).split()
    if truncate:
        got = got[: len(want)]
    if not want or not got:
        return 0.0
    lcs = lcs_length(got, want)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(got), lcs / len(want)
    return 2 * precision * recall / (precision + recall)


@register("overlap")
class Overlap:
    """Word-level ROUGE-L F-measure between the answer and ``case.expected``.

    Unlike the other rule scorers, ``value`` is continuous in ``[0, 1]``;
    ``passed`` is ``value >= threshold``. The memorisation probe in the
    benchmark experiment uses it to measure how much of a withheld
    continuation the model reproduces.

    Args:
        threshold: Pass mark on the F-measure.
        truncate: Compare only the first ``len(expected)`` words of the answer.
    """

    name = "overlap"

    def __init__(self, threshold: float = 0.5, truncate: bool = True) -> None:
        self.threshold = float(threshold)
        self.truncate = truncate

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Score the answer's overlap with the expected text."""
        if case.expected is None:
            return _bool_score(self.name, False, reason="no expected text")
        value = overlap_f1(traj.answer, str(case.expected), truncate=self.truncate)
        return Score(self.name, value, passed=value >= self.threshold, meta={"f1": value})


@register("choice")
class Choice:
    """Pass when the first answer letter (A-E) equals ``case.expected``."""

    name = "choice"

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Find the first standalone option letter in the answer."""
        got = extract_choice(traj.answer)
        want = str(case.expected).strip().upper()
        return Score(
            self.name,
            1.0 if got == want else 0.0,
            passed=got == want,
            label=got,
            meta={"expected": want},
        )


@register("max_length")
class MaxLength:
    """Pass when the answer is at most ``limit`` characters (or words).

    Args:
        limit: The maximum.
        unit: ``"chars"`` or ``"words"``.
    """

    name = "max_length"

    def __init__(self, limit: int = 500, unit: str = "chars") -> None:
        if unit not in ("chars", "words"):
            raise ValueError("unit must be 'chars' or 'words'")
        self.limit = int(limit)
        self.unit = unit

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Measure the answer and compare with the limit."""
        length = len(traj.answer) if self.unit == "chars" else len(traj.answer.split())
        return _bool_score(self.name, length <= self.limit, length=length, limit=self.limit)


@register("gold")
class GoldLabel:
    """Emit a known label stored on the case, for judge-vs-gold studies.

    Args:
        key: ``case.meta`` field holding the label.
        pass_labels: Labels that count as value ``1``.
    """

    name = "gold"

    def __init__(self, key: str = "gold", pass_labels: list[str] | None = None) -> None:
        self.key = key
        self.pass_labels = set(pass_labels or ["pass"])

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Read the gold label from the case."""
        label = case.meta.get(self.key)
        if label is None:
            return Score(self.name, 0.0, passed=None, label="missing")
        passed = str(label) in self.pass_labels
        return Score(self.name, 1.0 if passed else 0.0, passed=passed, label=str(label))
