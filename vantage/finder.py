"""Behaviour finder: rubric search over stored transcripts.

Given a rubric such as "the agent edits or bypasses tests" the finder runs a
judge monitor over candidate trajectories (optionally pre-filtered by
full-text search) and returns ranked hits. Hits are stored as monitor
events, so they appear inline in the trajectory viewer and feed the review
queue; human labels on those hits in turn give the finder a measured
precision.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from vantage.judge import Judge
from vantage.models import ModelClient
from vantage.monitors import JudgeMonitor
from vantage.results import View
from vantage.stats import Estimate, bootstrap_ci
from vantage.store import Store, StoredTrajectory

CONFIRM_LABELS = frozenset({"tampered", "fail", "yes"})
REJECT_LABELS = frozenset({"clean", "pass", "no"})


@dataclass(slots=True)
class Hit:
    """One trajectory the finder examined."""

    trajectory_id: int
    run_id: int
    case_id: str
    flagged: bool
    score: float
    rationale: str


@dataclass(slots=True)
class Audit:
    """What the full-text pre-filter is estimated to have thrown away.

    Attributes:
        n_excluded: Trajectories the pre-filter left out of the candidate set.
        n_audited: How many of those were judged anyway, at random.
        n_flagged: How many of the audited ones the judge flagged.
        flag_rate: Share of excluded trajectories that are hits, with a CI.
        estimated_missed: ``flag_rate`` scaled up to all excluded trajectories.
        prefilter_recall: Hits found divided by hits found plus estimated
            missed, with the interval taken from ``flag_rate``.
        hits: The audited trajectories that were flagged.
    """

    n_excluded: int
    n_audited: int
    n_flagged: int
    flag_rate: Estimate
    estimated_missed: float
    prefilter_recall: Estimate
    hits: list[Hit]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view."""
        return {
            "n_excluded": self.n_excluded,
            "n_audited": self.n_audited,
            "n_flagged": self.n_flagged,
            "flag_rate": self.flag_rate.to_dict(),
            "estimated_missed": self.estimated_missed,
            "prefilter_recall": self.prefilter_recall.to_dict(),
            "hits": [asdict(h) for h in self.hits],
        }


@dataclass(slots=True)
class FinderResult:
    """Ranked hits plus a precision estimate from existing human labels."""

    monitor_name: str
    examined: int
    hits: list[Hit]
    non_hits: list[Hit]
    precision: Estimate
    n_labelled: int
    audit: Audit | None = None


def monitor_name_for(rubric: str) -> str:
    """Stable monitor name for a rubric, so repeated searches replace their events."""
    return "finder:" + hashlib.sha256(rubric.strip().encode("utf-8")).hexdigest()[:8]


def candidates(
    store: Store, *, run_id: int | None, fts: str | None, limit: int
) -> list[StoredTrajectory]:
    """Pick trajectories to examine: FTS matches first, else a run's trajectories."""
    if fts:
        seen: dict[int, None] = {}
        for row in store.search_steps(fts, run_id=run_id, limit=limit * 20):
            seen.setdefault(int(row["trajectory_id"]), None)
            if len(seen) >= limit:
                break
        return [store.get_trajectory(tid) for tid in seen]
    run_ids = [run_id] if run_id is not None else [int(r["id"]) for r in store.list_runs()]
    out: list[StoredTrajectory] = []
    for rid in run_ids:
        out.extend(store.trajectories(rid, limit=limit - len(out)))
        if len(out) >= limit:
            break
    return out


def precision_from_annotations(store: Store, hits: list[Hit], monitor: str) -> tuple[Estimate, int]:
    """Precision of the hits according to human labels stored for this rubric.

    Only labels given in answer to ``monitor`` count. A label about something
    else, such as whether the final answer was correct, says nothing about
    whether this rubric's hit was right.
    """
    outcomes: list[float] = []
    for hit in hits:
        labels = [
            a["label"] for a in store.annotations(trajectory_id=hit.trajectory_id, monitor=monitor)
        ]
        if any(lab in CONFIRM_LABELS for lab in labels):
            outcomes.append(1.0)
        elif any(lab in REJECT_LABELS for lab in labels):
            outcomes.append(0.0)
    return bootstrap_ci(outcomes), len(outcomes)


async def find_behaviour(
    store: Store,
    rubric: str,
    client: ModelClient,
    model: str,
    *,
    run_id: int | None = None,
    fts: str | None = None,
    limit: int = 50,
    view: View = "trajectory",
    concurrency: int = 2,
    audit: int = 0,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> FinderResult:
    """Search stored trajectories for a behaviour described by ``rubric``.

    Args:
        store: The store to search.
        rubric: Yes/no question describing the behaviour.
        client: Judge model client.
        model: Judge model name.
        run_id: Restrict to one run; ``None`` searches every run.
        fts: Optional full-text pre-filter on step contents.
        limit: Maximum trajectories to examine.
        view: What the judge sees of each trajectory.
        concurrency: Parallel judge calls.
        audit: With ``fts``, also judge this many random trajectories the
            pre-filter excluded, to estimate how many hits it is missing.
        seed: Seed for the audit sample.
        progress: Called with ``(done, total)`` after each trajectory.

    Returns:
        Hits ranked by suspicion score, non-hits, the precision estimate, and
        the pre-filter audit when one was requested.
    """
    name = monitor_name_for(rubric)
    monitor = JudgeMonitor(
        Judge(client, model, rubric, max_tokens=200, use_reference=False), view=view, name=name
    )
    pool = candidates(store, run_id=run_id, fts=fts, limit=limit)
    excluded_ids = _excluded_ids(store, run_id, pool) if fts and audit > 0 else []
    sample_ids = random.Random(seed).sample(excluded_ids, min(audit, len(excluded_ids)))
    audit_pool = [store.get_trajectory(tid) for tid in sample_ids]
    total = len(pool) + len(audit_pool)
    sem = asyncio.Semaphore(concurrency)
    results: list[Hit] = []
    audited: list[Hit] = []

    async def one(stored: StoredTrajectory, into: list[Hit]) -> None:
        async with sem:
            verdict = await monitor.observe(stored.trajectory, stored.case, final=True)
        store.clear_verdicts(stored.id, name)
        store.save_verdicts(stored.id, [verdict])
        into.append(
            Hit(
                stored.id,
                stored.run_id,
                stored.case.id,
                verdict.flagged,
                verdict.score or 0.0,
                verdict.reason,
            )
        )
        if progress is not None:
            progress(len(results) + len(audited), total)

    async with asyncio.TaskGroup() as group:
        for stored in pool:
            group.create_task(one(stored, results))
        for stored in audit_pool:
            group.create_task(one(stored, audited))
    hits = sorted((h for h in results if h.flagged), key=lambda h: -h.score)
    non_hits = sorted((h for h in results if not h.flagged), key=lambda h: -h.score)
    precision, n_labelled = precision_from_annotations(store, hits, name)
    report = _audit_report(len(excluded_ids), audited, len(hits)) if audit_pool else None
    return FinderResult(name, len(pool), hits, non_hits, precision, n_labelled, report)


def _excluded_ids(store: Store, run_id: int | None, pool: list[StoredTrajectory]) -> list[int]:
    """Ids of the trajectories in scope that the pre-filter did not return."""
    in_pool = {stored.id for stored in pool}
    return [tid for tid in store.trajectory_ids(run_id) if tid not in in_pool]


def _audit_report(n_excluded: int, audited: list[Hit], n_found: int) -> Audit:
    """Estimate the pre-filter's recall from a random sample of what it excluded."""
    flags = [1.0 if h.flagged else 0.0 for h in audited]
    rate = bootstrap_ci(flags)
    missed = rate.point * n_excluded

    def recall(missed_rate: float) -> float:
        estimated = missed_rate * n_excluded
        return n_found / (n_found + estimated) if n_found + estimated > 0 else math.nan

    point = recall(rate.point)
    bounds = [recall(rate.hi), recall(rate.lo)]  # a higher miss rate means lower recall
    lo, hi = (point if math.isnan(b) else b for b in bounds)
    prefilter_recall = Estimate(point, lo, hi, rate.n)
    return Audit(
        n_excluded=n_excluded,
        n_audited=len(audited),
        n_flagged=sum(1 for h in audited if h.flagged),
        flag_rate=rate,
        estimated_missed=missed,
        prefilter_recall=prefilter_recall,
        hits=sorted((h for h in audited if h.flagged), key=lambda h: -h.score),
    )


def result_to_dict(result: FinderResult) -> dict[str, Any]:
    """JSON-friendly view of a finder result."""
    return {
        "monitor": result.monitor_name,
        "examined": result.examined,
        "hits": [asdict(h) for h in result.hits],
        "precision": result.precision.to_dict(),
        "n_labelled": result.n_labelled,
        "audit": result.audit.to_dict() if result.audit else None,
    }
