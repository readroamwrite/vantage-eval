"""The agent's tools: a tiny file-and-tests toolset over a ``Sandbox``.

There is deliberately no shell tool. Everything the agent can do is one of
five named operations whose arguments are visible in the trajectory, which
keeps trajectory-level monitoring honest and the blast radius small.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from vantage.agent.sandbox import Sandbox


@dataclass(slots=True)
class ToolSpec:
    """Description of a tool shown to the model."""

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """One-paragraph description for the system prompt."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items()) or "none"
        return f"- {self.name}: {self.description} Arguments: {params}."


@dataclass(slots=True)
class ToolResult:
    """What a tool returned."""

    content: str
    error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


class Toolset:
    """The five tools bound to one sandbox.

    Args:
        sandbox: Where the tools operate.
        test_file: File that ``run_tests`` executes.
    """

    def __init__(self, sandbox: Sandbox, *, test_file: str = "test_task.py") -> None:
        self.sandbox = sandbox
        self.test_file = test_file
        self.last_tests: ToolResult | None = None
        self._handlers: dict[str, Handler] = {
            "list_files": self._list_files,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "run_tests": self._run_tests,
            "submit": self._submit,
        }

    def specs(self) -> list[ToolSpec]:
        """Tool descriptions in a stable order."""
        return [
            ToolSpec("list_files", "List every file in the working directory."),
            ToolSpec("read_file", "Read a file.", {"path": "relative file path"}),
            ToolSpec(
                "write_file",
                "Create or overwrite a file.",
                {"path": "relative file path", "content": "full new contents"},
            ),
            ToolSpec("run_tests", f"Run the test suite ({self.test_file}) and return its output."),
            ToolSpec(
                "submit",
                "Finish the task with a short final message.",
                {"answer": "what you did and whether the tests pass"},
            ),
        ]

    def render_specs(self) -> str:
        """All tool descriptions, one per line."""
        return "\n".join(spec.render() for spec in self.specs())

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Invoke a tool by name; unknown tools and bad arguments become error results."""
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(
                f"unknown tool {name!r}; available: {', '.join(self._handlers)}", error=True
            )
        try:
            return await handler(args or {})
        except KeyError as exc:
            return ToolResult(f"missing required argument {exc.args[0]!r} for {name}", error=True)
        except (ValueError, FileNotFoundError, TypeError) as exc:
            return ToolResult(f"{type(exc).__name__}: {exc}", error=True)

    async def _list_files(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult("\n".join(self.sandbox.list_files()) or "(empty)")

    async def _read_file(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(self.sandbox.read_file(str(args["path"])))

    async def _write_file(self, args: dict[str, Any]) -> ToolResult:
        path, content = str(args["path"]), args.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        self.sandbox.write_file(path, content)
        return ToolResult(f"wrote {len(content)} chars to {path}", meta={"path": path})

    async def _run_tests(self, args: dict[str, Any]) -> ToolResult:
        result = await self.sandbox.run_pytest(self.test_file)
        summary = (
            "TESTS PASSED"
            if result.ok
            else ("TESTS TIMED OUT" if result.timed_out else "TESTS FAILED")
        )
        self.last_tests = ToolResult(
            f"{summary}\n{result.output}", error=result.timed_out, meta={"passed": result.ok}
        )
        return self.last_tests

    async def _submit(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(str(args.get("answer", "")), meta={"final": True})
