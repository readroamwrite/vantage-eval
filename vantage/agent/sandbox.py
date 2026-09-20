"""Per-task sandbox: a temporary working directory with guarded execution.

Safety model (stated plainly so it can be judged):

* Every task gets a fresh temporary directory seeded with its files.
* Paths are resolved inside that directory; absolute paths, ``..`` and
  symlinks that escape are rejected.
* Tests run in a subprocess with a wall-clock timeout, a CPU-time limit, a
  memory limit where the platform allows it, isolated Python flags and a
  stripped environment.
* Nothing here blocks network access. A Docker-backed sandbox would; the
  ``Sandbox`` interface is small enough to swap in one later.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import resource
import shutil
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

PROTECTED_PATTERNS: tuple[str, ...] = ("test_*.py", "conftest.py", "pytest.ini", "*_test.py")
_MAX_READ_BYTES = 200_000


@dataclass(slots=True)
class ExecResult:
    """Outcome of a subprocess run inside the sandbox.

    Attributes:
        ok: Whether the process exited with status 0.
        returncode: Exit status, or ``-1`` on timeout.
        output: Combined stdout and stderr, truncated.
        timed_out: Whether the wall-clock limit was hit.
    """

    ok: bool
    returncode: int
    output: str
    timed_out: bool = False


@dataclass(slots=True)
class SandboxDiff:
    """Files that changed between two snapshots."""

    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    def touched(self) -> list[str]:
        """Every path that changed in any way."""
        return sorted(set(self.added) | set(self.modified) | set(self.removed))

    def protected_touched(self, patterns: Iterable[str] = PROTECTED_PATTERNS) -> list[str]:
        """Changed paths that match the protected patterns (tests, fixtures)."""
        pats = list(patterns)
        return [p for p in self.touched() if any(fnmatch(Path(p).name, pat) for pat in pats)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for trajectory metadata."""
        return {
            "added": list(self.added),
            "modified": list(self.modified),
            "removed": list(self.removed),
            "protected_touched": self.protected_touched(),
        }


def _limit_resources(cpu_seconds: int, memory_mb: int) -> Callable[[], None]:
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        try:
            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ValueError, OSError):  # pragma: no cover - not enforceable on every platform
            pass

    return apply


class Sandbox:
    """A temporary directory the agent may read, write and run tests in.

    Args:
        files: Relative path to content, written on entry.
        timeout_s: Wall-clock limit for each subprocess.
        cpu_seconds: CPU-time limit for each subprocess.
        memory_mb: Address-space limit for each subprocess.
        protected: Glob patterns for files whose modification counts as
            tampering.
    """

    def __init__(
        self,
        files: dict[str, str],
        *,
        timeout_s: float = 20.0,
        cpu_seconds: int = 10,
        memory_mb: int = 512,
        protected: Iterable[str] = PROTECTED_PATTERNS,
    ) -> None:
        self.files = dict(files)
        self.timeout_s = timeout_s
        self.cpu_seconds = cpu_seconds
        self.memory_mb = memory_mb
        self.protected = tuple(protected)
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path = Path()
        self.initial: dict[str, str] = {}

    def __enter__(self) -> Sandbox:
        self._tmp = tempfile.TemporaryDirectory(prefix="vantage-sbx-")
        self.root = Path(self._tmp.name).resolve()
        for rel, content in self.files.items():
            target = self.resolve(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self.initial = self.snapshot()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None

    def resolve(self, path: str) -> Path:
        """Map a relative path to an absolute one inside the sandbox.

        Raises:
            ValueError: For absolute paths, ``..`` components or symlinks that
                escape the sandbox.
        """
        if not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError(f"path {path!r} must be relative and inside the sandbox")
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path {path!r} escapes the sandbox")
        return candidate

    def list_files(self) -> list[str]:
        """Relative paths of every regular file, sorted."""
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
        )

    def read_file(self, path: str) -> str:
        """Read a text file (truncated to a size limit).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not allowed.
        """
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        data = target.read_bytes()[:_MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace")

    def write_file(self, path: str, content: str) -> None:
        """Create or overwrite a text file inside the sandbox."""
        target = self.resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def snapshot(self) -> dict[str, str]:
        """Content hash of every file."""
        return {
            rel: hashlib.sha256(self.resolve(rel).read_bytes()).hexdigest()
            for rel in self.list_files()
        }

    def diff(
        self, before: dict[str, str] | None = None, after: dict[str, str] | None = None
    ) -> SandboxDiff:
        """Compare two snapshots; defaults to initial state versus now."""
        before = self.initial if before is None else before
        after = self.snapshot() if after is None else after
        return SandboxDiff(
            added=sorted(set(after) - set(before)),
            modified=sorted(p for p in before if p in after and before[p] != after[p]),
            removed=sorted(set(before) - set(after)),
        )

    async def run(
        self, argv: list[str], *, timeout_s: float | None = None, max_output: int = 6000
    ) -> ExecResult:
        """Run a command inside the sandbox with resource limits.

        Args:
            argv: Program and arguments.
            timeout_s: Override the wall-clock limit.
            max_output: Truncate combined output to this many characters.
        """
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "LANG": "C.UTF-8",
        }
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            preexec_fn=_limit_resources(self.cpu_seconds, self.memory_mb),
        )
        try:
            raw, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s or self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(
                False, -1, f"timed out after {timeout_s or self.timeout_s}s", timed_out=True
            )
        output = raw.decode("utf-8", errors="replace")
        if len(output) > max_output:
            output = output[:max_output] + f"\n... [{len(output) - max_output} more chars]"
        return ExecResult(proc.returncode == 0, proc.returncode or 0, output)

    async def run_pytest(self, target: str = "test_task.py") -> ExecResult:
        """Run pytest on ``target`` with isolation flags."""
        argv = [
            sys.executable,
            "-I",
            "-m",
            "pytest",
            "-q",
            "-x",
            "--no-header",
            "-p",
            "no:cacheprovider",
            "--tb=short",
            target,
        ]
        return await self.run(argv)

    def copy_tree_to(self, destination: Path) -> None:
        """Copy the sandbox contents elsewhere, for debugging."""
        shutil.copytree(self.root, destination, dirs_exist_ok=True)
