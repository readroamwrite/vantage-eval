"""Replay canned tool-call scripts through the real agent loop and sandbox."""

from __future__ import annotations

import json
from typing import Any

from vantage.agent.react import run_agent
from vantage.agent.sandbox import Sandbox
from vantage.cases import Case
from vantage.models import ChatRequest
from vantage.models.mock import MockClient
from vantage.targets import StepHook
from vantage.trajectory import Trajectory


class ScriptedAgentTarget:
    """An agent whose decisions are scripted but whose tools really run.

    Each case carries ``meta["files"]`` and ``meta["script"]``, a list of
    ``{"thought", "tool", "args"}`` turns. The turns are fed to the ordinary
    agent loop through a mock model, so tool results, sandbox diffs and test
    outcomes are real. This gives monitor studies trajectories with known
    behaviour classes and genuine evidence.

    Args:
        target_id: Recorded on trajectories.
        sandbox_timeout_s: Wall-clock limit per test run.
    """

    def __init__(
        self, *, target_id: str = "scripted_agent", sandbox_timeout_s: float = 20.0
    ) -> None:
        self.id = target_id
        self.sandbox_timeout_s = sandbox_timeout_s

    def config(self) -> dict[str, Any]:
        """Return the settings that define this target."""
        return {"kind": "scripted_agent"}

    async def run(self, case: Case, on_step: StepHook | None = None) -> Trajectory:
        """Replay the case's script inside a fresh sandbox."""
        files, script = case.meta.get("files"), case.meta.get("script")
        if not isinstance(files, dict) or not isinstance(script, list):
            raise ValueError(f"case {case.id!r} needs meta['files'] and meta['script']")
        turns = iter(json.dumps(turn) for turn in script)

        def respond(_request: ChatRequest) -> str:
            return next(
                turns,
                json.dumps(
                    {"thought": "Nothing left to do.", "tool": "submit", "args": {"answer": "done"}}
                ),
            )

        with Sandbox(files, timeout_s=self.sandbox_timeout_s) as sandbox:
            traj = await run_agent(
                MockClient(respond),
                "scripted",
                case,
                sandbox,
                max_steps=len(script) + 1,
                on_step=on_step,
                target_id=self.id,
                hidden_test=case.meta.get("hidden_test"),
            )
        traj.meta["behaviour"] = case.meta.get("behaviour")
        return traj
