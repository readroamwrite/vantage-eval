"""Background jobs for the dashboard's model-calling pages.

The behaviour finder and the conversation map call a judge model and can run
for minutes. The API starts them in a thread, returns a job id, and the page
polls for progress. Each job opens its own ``Store`` so it never shares a
connection with request handlers.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from vantage.store import Store

JobBody = Callable[[Store, Callable[[int, int], None]], Coroutine[Any, Any, Any]]


@dataclass(slots=True)
class Job:
    """State of one background job.

    Attributes:
        id: Random identifier the client polls with.
        kind: ``"finder"`` or ``"convmap"``.
        status: ``"running"``, ``"done"`` or ``"error"``.
        done: Units of work completed so far.
        total: Units of work expected.
        result: JSON-safe result once finished.
        error: Error text when ``status`` is ``"error"``.
    """

    id: str
    kind: str
    status: str = "running"
    done: int = 0
    total: int = 0
    result: Any = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the API."""
        with self._lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "status": self.status,
                "done": self.done,
                "total": self.total,
                "result": self.result,
                "error": self.error,
            }


class JobRegistry:
    """In-memory table of jobs for one dashboard process."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, body: JobBody) -> Job:
        """Run ``body`` in a thread and return its job record immediately.

        Args:
            kind: Label shown to the client.
            body: Coroutine factory taking a fresh store and a progress callback.
        """
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._prune()
            self._jobs[job.id] = job

        def progress(done: int, total: int) -> None:
            with job._lock:
                job.done, job.total = done, total

        def run() -> None:
            try:
                with Store(self._db_path) as store:
                    result = asyncio.run(body(store, progress))
                with job._lock:
                    job.result, job.status = result, "done"
            except Exception as exc:
                with job._lock:
                    job.error = f"{exc}\n{traceback.format_exc(limit=3)}"
                    job.status = "error"

        threading.Thread(target=run, name=f"vantage-{kind}-{job.id}", daemon=True).start()
        return job

    def _prune(self, keep: int = 50) -> None:
        """Drop the oldest finished jobs so results do not accumulate forever."""
        finished = [jid for jid, job in self._jobs.items() if job.status != "running"]
        for jid in finished[: max(0, len(self._jobs) - keep)]:
            del self._jobs[jid]

    def get(self, job_id: str) -> Job | None:
        """Look up a job by id."""
        with self._lock:
            return self._jobs.get(job_id)
