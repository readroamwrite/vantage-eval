"""LLM-as-judge: grade trajectories against a rubric with a model.

The judge is a measured instrument, not an oracle. Every verdict records
whether the model's JSON parsed, its self-reported confidence and its
rationale, so that a judge's reliability (agreement with gold labels,
position bias, self-consistency, calibration) can be estimated with
``vantage.stats.agreement`` before its scores are trusted.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vantage.cases import Case
from vantage.models import ChatRequest, ModelClient
from vantage.results import View
from vantage.scorers.rules import parse_json
from vantage.trajectory import Trajectory

UNPARSED = "unparsed"
_TOOL_OUTPUT_LIMIT = 1500


@dataclass(slots=True)
class JudgeVerdict:
    """One judge decision.

    Attributes:
        label: One of the judge's labels, or ``"unparsed"`` when the model's
            output could not be read even after a repair attempt.
        confidence: Self-reported confidence in ``[0, 1]``, if given.
        rationale: The model's explanation.
        raw: The model's raw text.
        parse_ok: Whether ``label`` came from well-formed output.
        meta: Extra details such as the presented order in pairwise mode.
    """

    label: str
    confidence: float | None
    rationale: str
    raw: str
    parse_ok: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "raw": self.raw,
            "parse_ok": self.parse_ok,
            "meta": dict(self.meta),
        }


def load_rubric(text_or_path: str) -> str:
    """Return rubric text, reading it from disk when ``text_or_path`` is a file."""
    path = Path(text_or_path)
    if path.suffix in (".md", ".txt") and path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return text_or_path.strip()


def render_transcript(traj: Trajectory, view: View = "trajectory") -> str:
    """Render a trajectory as text for the judge prompt.

    Args:
        traj: The trajectory.
        view: ``"output"`` shows only what ``Trajectory.output_only`` keeps;
            ``"trajectory"`` shows every step, tool call and (truncated) tool
            result.
    """
    shown = traj.output_only() if view == "output" else traj
    lines: list[str] = []
    for idx, step in enumerate(shown.steps):
        if step.role == "system":
            continue
        header = f"[{idx}] {step.role.upper()}"
        if step.role == "tool":
            header += f" RESULT ({step.tool_call_id or '?'})"
        content = step.content.strip()
        if step.role == "tool" and len(content) > _TOOL_OUTPUT_LIMIT:
            content = (
                content[:_TOOL_OUTPUT_LIMIT]
                + f"\n... [{len(step.content) - _TOOL_OUTPUT_LIMIT} more chars]"
            )
        lines.append(f"{header}:\n{content}" if content else f"{header}:")
        for call in step.tool_calls:
            lines.append(
                f"    TOOL CALL {call.name}({json.dumps(call.args, ensure_ascii=False)}) id={call.id}"
            )
    if traj.status != "ok" and view == "trajectory":
        lines.append(
            f"[run ended with status {traj.status}{': ' + traj.error if traj.error else ''}]"
        )
    return "\n\n".join(lines)


def parse_verdict(text: str, labels: Sequence[str]) -> JudgeVerdict:
    """Read a JSON verdict out of model text.

    Accepts ``verdict``, ``label`` or ``answer`` for the label field,
    confidence on a 0-1 or 0-100 scale, and case-insensitive labels.
    """
    data = parse_json(text)
    if not isinstance(data, dict):
        return JudgeVerdict(UNPARSED, None, "", text, False)
    raw_label = data.get("verdict", data.get("label", data.get("answer")))
    label = _match_label(raw_label, labels)
    if label is None:
        return JudgeVerdict(UNPARSED, None, str(data.get("rationale", "")), text, False)
    return JudgeVerdict(
        label,
        _normalize_confidence(data.get("confidence")),
        str(data.get("rationale", data.get("reasoning", ""))),
        text,
        True,
    )


_FIRST_WORD = re.compile(r"[^\W_]+")


def _match_label(value: Any, labels: Sequence[str]) -> str | None:
    """Map a raw verdict value onto one of ``labels``.

    Accepts the label itself (any case) or the label as the first word of a
    longer value such as ``"yes, because ..."`` or ``"(B)"``. A value that
    merely starts with a label's letters, like ``"Both"`` for ``B``, is not a
    match; the caller treats that as unparsed.
    """
    if value is None:
        return None
    text = str(value).strip().casefold()
    by_fold = {label.casefold(): label for label in labels}
    if text in by_fold:
        return by_fold[text]
    first = _FIRST_WORD.search(text)
    if first is not None and first.group(0) in by_fold:
        return by_fold[first.group(0)]
    return None


def _normalize_confidence(value: Any) -> float | None:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf > 1.0:
        conf = conf / 100.0
    return max(0.0, min(1.0, conf))


class Judge:
    """Grade trajectories with a model against a rubric.

    Args:
        client: Model client.
        model: Model name.
        rubric: Rubric text or path to a ``.md``/``.txt`` file.
        labels: Allowed verdict labels, in preference order.
        temperature: Sampling temperature for ``grade``; ``grade_n`` overrides it.
        max_tokens: Output cap.
        use_reference: Show ``case.expected`` to the judge when available.
        cot: Ask for a reasoning field before the verdict.
        name: Short name used in scores and reports.
    """

    def __init__(
        self,
        client: ModelClient,
        model: str,
        rubric: str,
        *,
        labels: Sequence[str] = ("pass", "fail"),
        temperature: float = 0.0,
        max_tokens: int = 300,
        use_reference: bool = True,
        cot: bool = False,
        name: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.rubric = load_rubric(rubric)
        self.labels = tuple(labels)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_reference = use_reference
        self.cot = cot
        self.name = name or "judge"

    def config(self) -> dict[str, Any]:
        """Settings that define this judge, for provenance.

        Includes the rendered system prompt, so a change to the prompt
        template changes the hash even when the rubric does not.
        """
        return {
            "model": self.model,
            "labels": list(self.labels),
            "temperature": self.temperature,
            "use_reference": self.use_reference,
            "cot": self.cot,
            "rubric": self.rubric,
            "system_prompt": self._system_prompt(self.labels),
        }

    @property
    def config_hash(self) -> str:
        """Short hash of :meth:`config`, stored with every score this judge produces."""
        blob = json.dumps(self.config(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def _system_prompt(self, labels: Sequence[str]) -> str:
        fields = (
            '{"reasoning": "<brief step-by-step check>", "verdict": "<label>", "confidence": <0-100>}'
            if self.cot
            else '{"verdict": "<label>", "confidence": <0-100>, "rationale": "<one sentence>"}'
        )
        return (
            "You are a careful, strict evaluator. Apply the rubric exactly as written. "
            f"The verdict must be one of: {', '.join(labels)}. "
            "Confidence is how sure you are that your verdict is right, from 0 to 100. "
            f"Reply with a single JSON object and nothing else: {fields}"
        )

    def _grade_prompt(self, traj: Trajectory, case: Case, view: View) -> str:
        parts = [f"## Task given to the model\n{case.prompt_text}"]
        if self.use_reference and case.expected is not None:
            parts.append(f"## Reference answer\n{case.expected}")
        title = "Model response" if view == "output" else "Full transcript of the model's run"
        parts.append(f"## {title}\n{render_transcript(traj, view)}")
        parts.append(f"## Rubric\n{self.rubric}")
        return "\n\n".join(parts)

    async def _ask(
        self,
        prompt: str,
        labels: Sequence[str],
        *,
        temperature: float,
        client: ModelClient | None = None,
        seed: int | None = 0,
    ) -> JudgeVerdict:
        client = client or self.client
        messages = [
            {"role": "system", "content": self._system_prompt(labels)},
            {"role": "user", "content": prompt},
        ]
        request = ChatRequest.create(
            self.model,
            messages,
            temperature=temperature,
            max_tokens=self.max_tokens,
            json_mode=True,
            seed=seed,
        )
        response = await client.chat(request)
        verdict = parse_verdict(response.content, labels)
        verdict.meta["attempts"] = 1
        if verdict.parse_ok:
            return verdict
        repair = [
            *messages,
            {"role": "assistant", "content": response.content},
            {
                "role": "user",
                "content": (
                    "That was not a valid JSON object with a verdict from the allowed labels "
                    f"({', '.join(labels)}). Reply again with only the JSON object."
                ),
            },
        ]
        request = ChatRequest.create(
            self.model,
            repair,
            temperature=temperature,
            max_tokens=self.max_tokens,
            json_mode=True,
            seed=seed,
        )
        response = await client.chat(request)
        verdict = parse_verdict(response.content, labels)
        verdict.meta["attempts"] = 2
        return verdict

    async def grade(self, traj: Trajectory, case: Case, *, view: View = "output") -> JudgeVerdict:
        """Grade one trajectory.

        Args:
            traj: The trajectory to grade.
            case: Its case, for the task text and reference answer.
            view: What the judge is allowed to see.
        """
        verdict = await self._ask(
            self._grade_prompt(traj, case, view), self.labels, temperature=self.temperature
        )
        verdict.meta["view"] = view
        return verdict

    async def grade_n(
        self,
        traj: Trajectory,
        case: Case,
        n: int,
        *,
        temperature: float = 0.7,
        view: View = "output",
    ) -> list[JudgeVerdict]:
        """Grade the same trajectory ``n`` times with sampling, for self-consistency.

        Each sample uses its own sampling seed (``1..n``) so providers that
        honour seeds produce genuinely different draws, and when the client is
        a ``CachedClient`` each sample also has its own cache key.
        """
        prompt = self._grade_prompt(traj, case, view)
        verdicts: list[JudgeVerdict] = []
        for i in range(n):
            client = (
                self.client.with_sample(i) if hasattr(self.client, "with_sample") else self.client
            )
            verdict = await self._ask(
                prompt, self.labels, temperature=temperature, client=client, seed=i + 1
            )
            verdict.meta.update({"view": view, "sample_idx": i})
            verdicts.append(verdict)
        return verdicts

    async def compare(
        self, a: Trajectory, b: Trajectory, case: Case, *, swap: bool = False, view: View = "output"
    ) -> JudgeVerdict:
        """Ask which of two responses better satisfies the rubric.

        Args:
            a: First response.
            b: Second response.
            case: The shared case.
            swap: Present ``b`` first. The returned label is always un-swapped,
                so ``"A"`` means ``a`` regardless of presentation order.
            view: What the judge sees of each trajectory.

        Returns:
            A verdict whose label is ``"A"`` or ``"B"`` (or ``"unparsed"``), with
            ``meta["presented_first"]`` recording the order shown.
        """
        first, second = (b, a) if swap else (a, b)
        parts = [f"## Task given to the model\n{case.prompt_text}"]
        if self.use_reference and case.expected is not None:
            parts.append(f"## Reference answer\n{case.expected}")
        parts.append(f"## Response A\n{render_transcript(first, view)}")
        parts.append(f"## Response B\n{render_transcript(second, view)}")
        parts.append(
            f"## Rubric\n{self.rubric}\n\nWhich response better satisfies the rubric? "
            "Answer A or B. If both are equally good or equally bad, pick the one you would show a user."
        )
        verdict = await self._ask("\n\n".join(parts), ("A", "B"), temperature=self.temperature)
        verdict.meta.update(
            {"presented_first": "b" if swap else "a", "swapped": swap, "view": view}
        )
        if verdict.parse_ok and swap:
            verdict.label = "B" if verdict.label == "A" else "A"
        return verdict
