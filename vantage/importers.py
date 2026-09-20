"""Import outside conversations as trajectories.

Supported inputs:

* a JSON list of ``{"role": ..., "content": ...}`` messages, or an object
  with a ``messages`` list (the shape most chat exports use); content may be
  a string or a list of text blocks;
* a plain-text transcript with ``User:`` / ``Assistant:`` (or ``Human:`` /
  ``AI:``) line prefixes.

Imported chats are stored under a run named ``imported/<name>`` so that
every analysis, the finder and the conversation map treat them like any
other trajectory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vantage.cases import Case, Dataset
from vantage.store import Store
from vantage.trajectory import Step, Trajectory

_ROLE_ALIASES = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "ai": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}
_TEXT_PREFIX = re.compile(r"^(user|human|assistant|ai|system)\s*:\s*", re.IGNORECASE)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def parse_messages(data: Any) -> list[Step]:
    """Turn exported JSON into steps.

    Raises:
        ValueError: If the data has no recognisable messages.
    """
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        data = data["messages"]
    if not isinstance(data, list) or not data:
        raise ValueError("expected a non-empty list of messages")
    steps: list[Step] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"message is not an object: {item!r}")
        role = _ROLE_ALIASES.get(str(item.get("role", "")).lower())
        if role is None:
            raise ValueError(f"unknown role {item.get('role')!r}")
        steps.append(Step(role, _content_text(item.get("content", ""))))  # type: ignore[arg-type]
    return steps


def parse_text_transcript(text: str) -> list[Step]:
    """Turn a ``User:``/``Assistant:`` transcript into steps.

    Raises:
        ValueError: If no prefixed lines are found.
    """
    steps: list[Step] = []
    role: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if role is not None:
            steps.append(Step(role, "\n".join(buffer).strip()))  # type: ignore[arg-type]

    for line in text.splitlines():
        match = _TEXT_PREFIX.match(line)
        if match:
            flush()
            role = _ROLE_ALIASES[match.group(1).lower()]
            buffer = [line[match.end() :]]
        else:
            buffer.append(line)
    flush()
    if not steps:
        raise ValueError("no 'User:' / 'Assistant:' prefixed lines found")
    return steps


def load_chat(path: str | Path) -> Trajectory:
    """Read a chat file into a trajectory.

    Args:
        path: ``.json``/``.jsonl`` for message exports, anything else is
            treated as a text transcript.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".json", ".jsonl"):
        if path.suffix.lower() == ".jsonl":
            data: Any = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            data = json.loads(raw)
        steps = parse_messages(data)
    else:
        steps = parse_text_transcript(raw)
    last = next((s for s in reversed(steps) if s.role == "assistant"), None)
    return Trajectory(
        case_id=path.stem,
        target_id="imported",
        steps=steps,
        final_output=last.content if last else None,
        meta={"source": str(path), "imported": True},
    )


def import_chat(store: Store, path: str | Path, *, name: str | None = None) -> tuple[int, int]:
    """Store a chat file as a one-case run.

    Returns:
        ``(run_id, trajectory_id)``.
    """
    traj = load_chat(path)
    name = name or Path(path).stem
    first_user = next((s.content for s in traj.steps if s.role == "user"), "")
    case = Case(traj.case_id, first_user, tags=["source:imported"], meta={"source": str(path)})
    run_name = f"imported/{name}"
    run_id = store.find_run(run_name)
    if run_id is None:
        run_id = store.create_run(
            run_name, "imported", Dataset(run_name, [case]), notes=f"imported from {path}"
        )
    traj_id = store.save_trajectory(run_id, case, traj)
    store.set_run_status(run_id, "done")
    return run_id, traj_id
