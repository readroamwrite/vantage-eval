"""Rebuild an agent's sandbox from a stored trajectory to recover ground truth.

Agent trajectories record every ``write_file`` call with its full content, so
the final state of the sandbox can be reconstructed long after the temporary
directory is gone. That makes it possible to apply checks that did not exist
when a run was recorded, such as the held-out tests, without calling any
model again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vantage.agent.sandbox import IMPL_FILE, Sandbox, run_hidden_tests
from vantage.cases import Case
from vantage.trajectory import Trajectory

REPLAY_KEY = "replay"


@dataclass(slots=True)
class ReplayResult:
    """What the harness observes after replaying a trajectory's file writes.

    Attributes:
        files: Final sandbox contents, path to text.
        protected_touched: Changed paths that match the current protected patterns.
        final_tests_passed: Whether the visible suite passes in the rebuilt sandbox.
        hidden_tests_passed: Whether the held-out suite passes against the
            final implementation, or ``None`` when the case has none.
        skipped_calls: ``write_file`` calls that never executed (halted or errored).
    """

    files: dict[str, str]
    protected_touched: list[str]
    final_tests_passed: bool
    hidden_tests_passed: bool | None
    skipped_calls: int


def executed_writes(traj: Trajectory) -> tuple[list[tuple[str, str]], int]:
    """Return the ``(path, content)`` of every ``write_file`` call that actually ran.

    A call counts as executed when a non-error tool step answers it. A call
    without a tool step (the run was halted before it ran) or with an error
    step (bad path) is skipped and counted.
    """
    answered = {s.tool_call_id for s in traj.steps if s.role == "tool" and not s.meta.get("error")}
    writes: list[tuple[str, str]] = []
    skipped = 0
    for call in traj.tool_calls:
        if call.name != "write_file":
            continue
        if call.id not in answered or "path" not in call.args:
            skipped += 1
            continue
        content = call.args.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        writes.append((str(call.args["path"]), content))
    return writes, skipped


def hidden_test_for(case: Case) -> str | None:
    """The case's held-out suite, looked up by task name for cases stored before it existed."""
    hidden = case.meta.get("hidden_test")
    if isinstance(hidden, str):
        return hidden
    task = case.meta.get("task")
    if not task:
        return None
    from vantage.experiments import datagen

    for spec in datagen.AGENT_TASKS:
        if spec["name"] == task:
            return datagen.hidden_test_file(spec)
    return None


async def replay(traj: Trajectory, case: Case, *, timeout_s: float = 20.0) -> ReplayResult | None:
    """Rebuild the sandbox from ``case.meta["files"]`` plus the executed writes and test it.

    Returns ``None`` when the case carries no sandbox files, which means the
    trajectory did not come from the agent.
    """
    files = case.meta.get("files")
    if not isinstance(files, dict):
        return None
    writes, skipped = executed_writes(traj)
    with Sandbox(files, timeout_s=timeout_s) as sandbox:
        for path, content in writes:
            try:
                sandbox.write_file(path, content)
            except ValueError:
                skipped += 1
        final = await sandbox.run_pytest()
        protected = sandbox.diff().protected_touched(sandbox.protected)
        final_files = {rel: sandbox.read_file(rel) for rel in sandbox.list_files()}
        hidden_test = hidden_test_for(case)
        hidden: bool | None = None
        if hidden_test is not None:
            impl = final_files.get(IMPL_FILE)
            if impl is None:
                hidden = False
            else:
                hidden = (await run_hidden_tests(impl, hidden_test, timeout_s=timeout_s)).ok
    return ReplayResult(final_files, protected, final.ok, hidden, skipped)


def needs_replay(traj: Trajectory) -> bool:
    """Whether this is a finished agent trajectory recorded without the harness checks."""
    return "sandbox_diff" in traj.meta and "final_tests_passed" not in traj.meta


async def ensure_ground_truth(traj: Trajectory, case: Case, *, timeout_s: float = 20.0) -> bool:
    """Make the harness-side ground truth available in ``traj.meta``.

    Trajectories recorded by the current agent loop already carry
    ``final_tests_passed`` and ``hidden_tests_passed``. Older ones are
    replayed once; the result is memoised under ``meta["replay"]`` so that
    several scorers on the same trajectory share a single replay.

    Returns:
        Whether ground truth is available after the call.
    """
    if not needs_replay(traj):
        return "final_tests_passed" in traj.meta
    if REPLAY_KEY not in traj.meta:
        result = await replay(traj, case, timeout_s=timeout_s)
        if result is None:
            traj.meta[REPLAY_KEY] = None
            return False
        traj.meta[REPLAY_KEY] = {
            "protected_touched": result.protected_touched,
            "skipped_calls": result.skipped_calls,
            "files": sorted(result.files),
        }
        traj.meta["final_tests_passed"] = result.final_tests_passed
        traj.meta["hidden_tests_passed"] = result.hidden_tests_passed
    return traj.meta[REPLAY_KEY] is not None


def replayed_protected(traj: Trajectory) -> list[str]:
    """Protected paths found by a replay, if one happened."""
    info: dict[str, Any] | None = traj.meta.get(REPLAY_KEY)
    return list(info.get("protected_touched", [])) if info else []
