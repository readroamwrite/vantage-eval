"""Trajectory data model: the single representation of anything under test.

A ``Trajectory`` is an ordered list of ``Step`` objects. A plain LLM call is a
trajectory with one user step and one assistant step; an agent run is a longer
trajectory that also contains tool calls and tool results. Scorers and monitors
all consume this one type, and choose their *view* of it: the final output
only (``Trajectory.output_only``) or every step.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

Role = Literal["system", "user", "assistant", "tool"]
Status = Literal["ok", "error", "timeout", "halted", "format_error"]

_ROLES: tuple[str, ...] = get_args(Role)
_STATUSES: tuple[str, ...] = get_args(Status)


@dataclass(slots=True)
class ToolCall:
    """A tool invocation requested by the model.

    Attributes:
        id: Identifier that links the call to its ``Step`` result.
        name: Tool name, for example ``"write_file"``.
        args: Arguments passed to the tool.
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Step:
    """One message in a trajectory.

    Attributes:
        role: Who produced the step.
        content: Text content of the step. For tool steps this is the tool result.
        tool_calls: Tool calls requested by an assistant step.
        tool_call_id: For tool steps, the id of the ``ToolCall`` this answers.
        meta: Free-form metadata such as ``tokens_in``, ``latency_ms`` or
            ``parse_error``.
    """

    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in _ROLES:
            raise ValueError(f"invalid step role {self.role!r}; expected one of {_ROLES}")


@dataclass(slots=True)
class Trajectory:
    """The full record of one target run on one test case.

    Attributes:
        case_id: Id of the ``Case`` this trajectory answers.
        target_id: Id of the target (model or agent) that produced it.
        steps: Ordered messages, tool calls and tool results.
        final_output: The answer the target submitted, if any.
        status: Outcome of the run. ``"halted"`` means a monitor stopped it.
        error: Error text when ``status`` is not ``"ok"``.
        meta: Free-form metadata such as ``condition``, ``repeat_idx`` or
            ``sandbox_diff``.
    """

    case_id: str
    target_id: str
    steps: list[Step] = field(default_factory=list)
    final_output: str | None = None
    status: Status = "ok"
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"invalid status {self.status!r}; expected one of {_STATUSES}")

    @property
    def n_steps(self) -> int:
        """Number of steps in the trajectory."""
        return len(self.steps)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call across all assistant steps, in order."""
        return [call for step in self.steps if step.role == "assistant" for call in step.tool_calls]

    @property
    def last_assistant(self) -> Step | None:
        """The final assistant step, or ``None`` if there is none."""
        for step in reversed(self.steps):
            if step.role == "assistant":
                return step
        return None

    @property
    def answer(self) -> str:
        """Best available answer text: ``final_output`` or the last assistant content."""
        if self.final_output is not None:
            return self.final_output
        last = self.last_assistant
        return last.content if last is not None else ""

    def output_only(self) -> Trajectory:
        """Return a view containing only what an output-level observer may see.

        The view keeps the system prompt, the first user message and the final
        assistant message stripped of tool calls. Everything in between,
        including every tool call and tool result, is removed. The view is a
        new object; the original is not modified.

        Returns:
            A trajectory whose ``meta["view"]`` is ``"output"``.
        """
        kept: list[Step] = []
        system = next((s for s in self.steps if s.role == "system"), None)
        user = next((s for s in self.steps if s.role == "user"), None)
        last = self.last_assistant
        if system is not None:
            kept.append(Step("system", system.content, meta=dict(system.meta)))
        if user is not None:
            kept.append(Step("user", user.content, meta=dict(user.meta)))
        if last is not None:
            kept.append(Step("assistant", last.content, meta=dict(last.meta)))
        meta = dict(self.meta)
        meta["view"] = "output"
        return Trajectory(
            case_id=self.case_id,
            target_id=self.target_id,
            steps=kept,
            final_output=self.final_output,
            status=self.status,
            error=self.error,
            meta=meta,
        )

    @classmethod
    def single_turn(
        cls,
        case_id: str,
        target_id: str,
        prompt: str,
        response: str,
        *,
        system: str | None = None,
        response_meta: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build the trajectory of a plain prompt-in, answer-out LLM call.

        Args:
            case_id: Id of the test case.
            target_id: Id of the model.
            prompt: The user prompt.
            response: The model's answer.
            system: Optional system prompt.
            response_meta: Metadata for the assistant step (tokens, latency).
            meta: Trajectory-level metadata.

        Returns:
            A trajectory with ``final_output`` set to ``response``.
        """
        steps: list[Step] = []
        if system:
            steps.append(Step("system", system))
        steps.append(Step("user", prompt))
        steps.append(Step("assistant", response, meta=dict(response_meta or {})))
        return cls(
            case_id=case_id,
            target_id=target_id,
            steps=steps,
            final_output=response,
            meta=dict(meta or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        """Deserialize from the output of ``to_dict``.

        Args:
            data: A mapping produced by ``to_dict`` or equivalent JSON.

        Returns:
            The reconstructed trajectory.

        Raises:
            ValueError: If a role or status value is invalid.
            KeyError: If a required field is missing.
        """
        steps = [
            Step(
                role=s["role"],
                content=s["content"],
                tool_calls=[ToolCall(**c) for c in s.get("tool_calls", [])],
                tool_call_id=s.get("tool_call_id"),
                meta=dict(s.get("meta", {})),
            )
            for s in data["steps"]
        ]
        return cls(
            case_id=data["case_id"],
            target_id=data["target_id"],
            steps=steps,
            final_output=data.get("final_output"),
            status=data.get("status", "ok"),
            error=data.get("error"),
            meta=dict(data.get("meta", {})),
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Trajectory:
        """Deserialize from a JSON string produced by ``to_json``."""
        return cls.from_dict(json.loads(text))
