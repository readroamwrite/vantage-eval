"""Targets: the things under evaluation.

A target turns a ``Case`` into a ``Trajectory``. ``LLMTarget`` wraps a single
model call; agent targets (added in ``vantage.agent``) run multi-step loops.
Targets accept an optional ``on_step`` hook that monitors use to observe the
partial trajectory after every step and, if they choose, halt the run.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from vantage.cases import Case
from vantage.models import ChatRequest, ModelClient
from vantage.results import Verdict
from vantage.trajectory import Step, Trajectory

StepHook = Callable[[Trajectory], Awaitable[Verdict | None]]


@runtime_checkable
class Target(Protocol):
    """Anything that can be evaluated."""

    id: str

    async def run(self, case: Case, on_step: StepHook | None = None) -> Trajectory:
        """Produce a trajectory for ``case``.

        Args:
            case: The item to answer.
            on_step: Called with the partial trajectory after each step. If it
                returns a halting ``Verdict`` the target must stop and mark the
                trajectory ``"halted"``.
        """
        ...

    def config(self) -> dict[str, Any]:
        """Return provenance information to store with each run."""
        ...


class LLMTarget:
    """A single-call language model target.

    Args:
        client: Model client to call.
        model: Model name passed to the client.
        system: Default system prompt. A case's own ``system`` takes priority.
        temperature: Sampling temperature.
        max_tokens: Output token cap.
        json_mode: Ask the provider for JSON output.
        target_id: Override the generated id ``"llm:<model>"``.
    """

    def __init__(
        self,
        client: ModelClient,
        model: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        json_mode: bool = False,
        target_id: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_mode = json_mode
        self.id = target_id or f"llm:{model}"

    def config(self) -> dict[str, Any]:
        """Return the settings that define this target."""
        return {
            "kind": "llm",
            "model": self.model,
            "system": self.system,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "json_mode": self.json_mode,
        }

    def _messages(self, case: Case) -> list[dict[str, str]]:
        messages = case.messages()
        if case.system is None and self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        return messages

    async def run(self, case: Case, on_step: StepHook | None = None) -> Trajectory:
        """Call the model once and wrap the exchange as a trajectory."""
        messages = self._messages(case)
        steps = [Step(m["role"], m["content"]) for m in messages]  # type: ignore[arg-type]
        traj = Trajectory(case_id=case.id, target_id=self.id, steps=steps)
        request = ChatRequest.create(
            self.model,
            messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            json_mode=self.json_mode,
        )
        try:
            response = await self.client.chat(request)
        except Exception as exc:
            traj.status = "error"
            traj.error = f"{type(exc).__name__}: {exc}"
            return traj
        traj.steps.append(Step("assistant", response.content, meta=response.step_meta()))
        traj.final_output = response.content
        if on_step is not None:
            verdict = await on_step(traj)
            if verdict is not None and verdict.halt:
                traj.status = "halted"
                traj.meta["halted_at_step"] = traj.n_steps - 1
                traj.meta["halted_by"] = verdict.monitor
        return traj


class ScriptedTarget:
    """A target that replays pre-written responses or trajectories.

    Used for judge studies (grading known-good and known-bad responses), for
    monitor studies (replaying trajectories with known behaviours) and in
    tests. Each step of a replayed trajectory is offered to ``on_step`` in
    turn, so monitors can halt a scripted run exactly as they would a live one.

    Args:
        scripts: Map from case id to either a response string or a full
            ``Trajectory``. Strings become single-turn trajectories.
        target_id: Id recorded on produced trajectories.
        default: Response for cases without a script; ``None`` raises.
    """

    def __init__(
        self,
        scripts: dict[str, str | Trajectory],
        *,
        target_id: str = "scripted",
        default: str | None = None,
    ) -> None:
        self.scripts = dict(scripts)
        self.id = target_id
        self.default = default

    def config(self) -> dict[str, Any]:
        """Return the settings that define this target."""
        return {"kind": "scripted", "n_scripts": len(self.scripts)}

    async def run(self, case: Case, on_step: StepHook | None = None) -> Trajectory:
        """Replay the script for ``case``.

        Raises:
            KeyError: If the case has no script and no default was given.
        """
        script = self.scripts.get(case.id, self.default)
        if script is None:
            raise KeyError(f"no script for case {case.id!r}")
        if isinstance(script, str):
            full = Trajectory.single_turn(
                case.id, self.id, case.prompt_text, script, system=case.system
            )
        else:
            full = Trajectory.from_dict(script.to_dict())
            full.case_id, full.target_id = case.id, self.id
        if on_step is None:
            return full
        partial = Trajectory(case.id, self.id, meta=dict(full.meta))
        for index, step in enumerate(full.steps):
            partial.steps.append(step)
            verdict = await on_step(partial)
            if verdict is not None and verdict.halt:
                partial.status = "halted"
                partial.meta["halted_at_step"] = index
                partial.meta["halted_by"] = verdict.monitor
                partial.final_output = None
                return partial
        full.meta.update(partial.meta)
        return full


class AgentTarget:
    """The built-in tool-using agent as a target.

    Each case must carry ``meta["files"]`` (relative path to content) that
    seeds the sandbox. See ``vantage.agent`` for the loop and the tools.

    Args:
        client: Model client.
        model: Model name.
        system: Behavioural system prompt; ``None`` uses the honest default.
        max_steps: Tool calls allowed per run.
        max_tokens: Output cap per turn.
        sandbox_timeout_s: Wall-clock limit for each test run.
        target_id: Override the generated id ``"agent:<model>"``.
    """

    def __init__(
        self,
        client: ModelClient,
        model: str,
        *,
        system: str | None = None,
        max_steps: int = 8,
        max_tokens: int = 400,
        sandbox_timeout_s: float = 20.0,
        target_id: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system = system
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.sandbox_timeout_s = sandbox_timeout_s
        self.id = target_id or f"agent:{model}"

    def config(self) -> dict[str, Any]:
        """Return the settings that define this target."""
        return {
            "kind": "agent",
            "model": self.model,
            "system": self.system,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
        }

    async def run(self, case: Case, on_step: StepHook | None = None) -> Trajectory:
        """Run the agent on ``case`` in a fresh sandbox."""
        from vantage.agent.react import run_agent
        from vantage.agent.sandbox import Sandbox

        files = case.meta.get("files")
        if not isinstance(files, dict):
            raise ValueError(f"case {case.id!r} has no meta['files'] to seed the sandbox")
        with Sandbox(files, timeout_s=self.sandbox_timeout_s) as sandbox:
            return await run_agent(
                self.client,
                self.model,
                case,
                sandbox,
                system=self.system,
                max_steps=self.max_steps,
                max_tokens=self.max_tokens,
                on_step=on_step,
                target_id=self.id,
                hidden_test=case.meta.get("hidden_test"),
            )
