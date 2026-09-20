"""Conversation maps: a traceable mind map of what a conversation was about.

A model reads the transcript and returns a tree of topics, the directions
each went in, questions raised, decisions made and outcomes. Every node
keeps the indices of the steps it came from, so a reader can always trace a
summary back to the exact messages. Long conversations are mapped in chunks
and merged, preserving the step references. Maps render to Mermaid (which
GitHub displays natively) and to Graphviz for the dashboard.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from vantage.models import ChatRequest, ModelClient
from vantage.scorers.rules import parse_json
from vantage.trajectory import Trajectory

KINDS = ("topic", "direction", "question", "decision", "outcome")
_MERMAID_UNSAFE = re.compile(r"[()\[\]{}\"'`|<>]")


@dataclass(slots=True)
class Node:
    """One node of a conversation map."""

    title: str
    kind: str = "topic"
    summary: str = ""
    step_refs: list[int] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize recursively."""
        return {
            "title": self.title,
            "kind": self.kind,
            "summary": self.summary,
            "step_refs": list(self.step_refs),
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Deserialize recursively, tolerating the model's loose field names."""
        refs = data.get("step_refs", data.get("steps", []))
        if isinstance(refs, int):
            refs = [refs]
        return cls(
            title=str(data.get("title", data.get("name", "untitled")))[:120],
            kind=str(data.get("kind", "topic")).lower(),
            summary=str(data.get("summary", "")),
            step_refs=[
                int(r)
                for r in refs
                if isinstance(r, int | float | str) and str(r).lstrip("-").isdigit()
            ],
            children=[cls.from_dict(c) for c in data.get("children", []) if isinstance(c, dict)],
        )

    def walk(self) -> list[Node]:
        """This node and every descendant, depth first."""
        out = [self]
        for child in self.children:
            out.extend(child.walk())
        return out


@dataclass(slots=True)
class ConvMap:
    """A conversation map with provenance."""

    root: Node
    model: str
    n_steps: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize."""
        return {"root": self.root.to_dict(), "model": self.model, "n_steps": self.n_steps}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConvMap:
        """Deserialize."""
        return cls(
            Node.from_dict(data["root"]), str(data.get("model", "")), int(data.get("n_steps", 0))
        )

    def to_mermaid(self) -> str:
        """Render as a Mermaid ``mindmap`` block."""
        lines = ["mindmap", f"  root(({_safe(self.root.title)}))"]

        def emit(node: Node, depth: int) -> None:
            for child in node.children:
                refs = f" [{','.join(map(str, child.step_refs))}]" if child.step_refs else ""
                lines.append("  " * (depth + 1) + f"{_safe(child.title)}{refs}")
                emit(child, depth + 1)

        emit(self.root, 1)
        return "\n".join(lines)

    def to_dot(self) -> str:
        """Render as Graphviz DOT for ``st.graphviz_chart``."""
        colors = {
            "topic": "#dbeafe",
            "direction": "#e0f2fe",
            "question": "#fef9c3",
            "decision": "#dcfce7",
            "outcome": "#fce7f3",
        }
        lines = [
            "digraph map {",
            "  rankdir=LR;",
            '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        ]
        counter = 0

        def emit(node: Node, parent: str | None) -> None:
            nonlocal counter
            ident = f"n{counter}"
            counter += 1
            refs = f"\\n[steps {', '.join(map(str, node.step_refs))}]" if node.step_refs else ""
            label = _dot_escape(f"{node.title}{refs}")
            lines.append(
                f'  {ident} [label="{label}", fillcolor="{colors.get(node.kind, "#eeeeee")}"];'
            )
            if parent is not None:
                lines.append(f"  {parent} -> {ident};")
            for child in node.children:
                emit(child, ident)

        emit(self.root, None)
        lines.append("}")
        return "\n".join(lines)


def _safe(text: str) -> str:
    return _MERMAID_UNSAFE.sub("", text).strip() or "untitled"


def _dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def validate(node: Node, n_steps: int) -> Node:
    """Drop invalid step references and unknown kinds, in place."""
    for item in node.walk():
        item.step_refs = sorted({r for r in item.step_refs if 0 <= r < n_steps})
        if item.kind not in KINDS:
            item.kind = "topic"
    return node


def render_for_map(traj: Trajectory, start: int = 0, end: int | None = None) -> str:
    """Transcript text with ``[index]`` markers the model must cite."""
    end = traj.n_steps if end is None else end
    lines = []
    for idx in range(start, end):
        step = traj.steps[idx]
        if step.role == "system":
            continue
        content = step.content.strip()
        if len(content) > 1500:
            content = content[:1500] + " ..."
        calls = "; ".join(
            f"{c.name}({json.dumps(c.args, ensure_ascii=False)[:200]})" for c in step.tool_calls
        )
        body = content if content else ""
        if calls:
            body = (body + "\n" if body else "") + f"tool calls: {calls}"
        lines.append(f"[{idx}] {step.role}: {body}")
    return "\n\n".join(lines)


_SYSTEM = (
    "You map conversations. Read the transcript, whose messages are numbered [index], and return a JSON "
    "tree describing what it was about. Root: the overall subject. Children: the distinct topics, each "
    "with the directions it went in, questions raised, decisions made and outcomes reached. Every node "
    "has: title (short), kind (topic|direction|question|decision|outcome), summary (one sentence), "
    "steps (list of the message indices it is based on), children (list). Cite indices faithfully; a "
    "reader must be able to jump from any node to the exact messages. Reply with a single JSON object only."
)

_MERGE_SYSTEM = (
    "You merge partial conversation maps into one. You receive several JSON trees mapping consecutive "
    "parts of one conversation. Return one JSON tree with the same node format (title, kind, summary, "
    "steps, children), merging duplicate topics and keeping every steps index from the inputs. Reply "
    "with a single JSON object only."
)


async def _ask_tree(
    client: ModelClient, model: str, system: str, user: str, *, max_tokens: int
) -> dict[str, Any] | None:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for attempt in range(2):
        response = await client.chat(
            ChatRequest.create(model, messages, max_tokens=max_tokens, json_mode=True)
        )
        data = parse_json(response.content)
        if isinstance(data, dict):
            return data
        if attempt == 0:
            messages = [
                *messages,
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": "That was not a JSON object. Reply with only the JSON tree.",
                },
            ]
    return None


async def build(
    traj: Trajectory,
    client: ModelClient,
    model: str,
    *,
    chunk_steps: int = 40,
    max_tokens: int = 1500,
) -> ConvMap:
    """Map a trajectory, chunking and merging when it is long.

    Args:
        traj: The conversation.
        client: Model client.
        model: Model name.
        chunk_steps: Steps per chunk; conversations longer than this are
            mapped in pieces and merged.
        max_tokens: Output cap per model call.

    Returns:
        The validated map.

    Raises:
        ValueError: If the model never returned a usable tree.
    """
    n = traj.n_steps
    if n <= chunk_steps:
        data = await _ask_tree(client, model, _SYSTEM, render_for_map(traj), max_tokens=max_tokens)
        if data is None:
            raise ValueError("the model did not return a JSON map")
        return ConvMap(validate(Node.from_dict(data), n), model, n)

    partials: list[dict[str, Any]] = []
    for start in range(0, n, chunk_steps):
        end = min(n, start + chunk_steps)
        data = await _ask_tree(
            client,
            model,
            _SYSTEM,
            f"(messages {start} to {end - 1} of {n})\n\n" + render_for_map(traj, start, end),
            max_tokens=max_tokens,
        )
        if data is not None:
            partials.append(validate(Node.from_dict(data), n).to_dict())
    if not partials:
        raise ValueError("the model did not return a JSON map for any chunk")
    merged = await _ask_tree(
        client,
        model,
        _MERGE_SYSTEM,
        json.dumps(partials, ensure_ascii=False),
        max_tokens=max_tokens,
    )
    if merged is None:
        root = Node("Conversation", children=[Node.from_dict(p) for p in partials])
    else:
        root = Node.from_dict(merged)
        cited = {r for node in root.walk() for r in node.step_refs}
        for partial in partials:
            for node in Node.from_dict(partial).walk():
                if node.step_refs and not set(node.step_refs) <= cited and node is not None:
                    root.children.append(Node(node.title, node.kind, node.summary, node.step_refs))
                    cited |= set(node.step_refs)
    return ConvMap(validate(root, n), model, n)


def steps_for(traj: Trajectory, node: Node) -> list[tuple[int, str, str]]:
    """The ``(index, role, content)`` triples a node cites."""
    return [
        (i, traj.steps[i].role, traj.steps[i].content)
        for i in node.step_refs
        if 0 <= i < traj.n_steps
    ]
