"""Read-side helpers for the dashboard API.

Every function here takes a ``Store`` and returns plain JSON-compatible data,
so the same shapes can be served over HTTP and asserted in tests without a
running server.
"""

from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from vantage.analysis import compare_runs, diagnose_run, primary_scorer
from vantage.report.markdown import summarize_run
from vantage.stats import Estimate, cohens_kappa
from vantage.store import Store, StoredTrajectory

REVIEW_LABELS = ("pass", "fail", "tampered", "clean", "unsure", "yes", "no")
"""Labels a reviewer can attach to a trajectory. ``yes`` and ``no`` answer a
specific monitor's question and carry no correctness value."""

LABEL_VALUES: dict[str, float | None] = {
    "pass": 1.0,
    "clean": 1.0,
    "fail": 0.0,
    "tampered": 0.0,
    "unsure": None,
    "yes": None,
    "no": None,
}


def jsonable(value: Any) -> Any:
    """Convert estimates, dataclasses, tuples and NaN into JSON-safe data.

    Args:
        value: Any object produced by the analysis layer.

    Returns:
        The same structure with ``Estimate`` as dicts, dataclasses as dicts,
        tuple keys joined with ``:`` and non-finite floats as ``None``.
    """
    if isinstance(value, Estimate):
        return jsonable(value.to_dict())
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {_key(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        return jsonable(value.item())
    return value


def _key(key: Any) -> str:
    if isinstance(key, tuple):
        return ":".join(str(k) for k in key)
    return str(key)


def run_row(store: Store, run: dict[str, Any]) -> dict[str, Any]:
    """One run with its per-scorer estimates and status counts."""
    summary = summarize_run(store, int(run["id"]))
    return {
        "id": int(run["id"]),
        "name": run["name"],
        "target": run["target_id"],
        "dataset": run["dataset_name"],
        "condition": run["condition"] or "",
        "status": run["status"],
        "n": int(run["n_trajectories"]),
        "created_at": run.get("created_at"),
        "metrics": jsonable(summary["metrics"]),
        "statuses": summary["statuses"],
    }


def list_runs(store: Store) -> list[dict[str, Any]]:
    """Every run, newest first, with metrics."""
    return [run_row(store, run) for run in store.list_runs()]


def case_rows(store: Store, run_id: int) -> list[dict[str, Any]]:
    """One row per trajectory with scores keyed by scorer name."""
    flagged = {v["trajectory_id"] for v in store.verdicts_for_run(run_id) if v["flagged"]}
    rows: dict[int, dict[str, Any]] = {}
    for stored in store.trajectories(run_id):
        traj = stored.trajectory
        rows[stored.id] = {
            "trajectory_id": stored.id,
            "case_id": stored.case.id,
            "repeat": stored.repeat_idx,
            "status": traj.status,
            "steps": traj.n_steps,
            "tool_calls": len(traj.tool_calls),
            "flagged": stored.id in flagged,
            "tags": list(stored.case.tags),
            "answer": (traj.answer or "")[:160],
            "scores": {},
        }
    for row in store.scores_for_run(run_id):
        target = rows.get(int(row["trajectory_id"]))
        if target is None:
            continue
        target["scores"][row["scorer"]] = {
            "value": row["value"],
            "label": row["label"],
            "passed": row["passed"],
        }
    return list(rows.values())


def run_detail(store: Store, run_id: int) -> dict[str, Any]:
    """Run header, metrics, per-case table, scorer names and eval-health diagnosis."""
    run = store.get_run(run_id)
    scorer = primary_scorer(store, run_id)
    diagnosis = diagnose_run(store, run_id, scorer=scorer) if scorer else None
    names: list[str] = []
    for row in store.scores_for_run(run_id):
        if row["scorer"] not in names:
            names.append(row["scorer"])
    return {
        "run": run_row(store, run),
        "scorers": names,
        "primary_scorer": scorer,
        "cases": case_rows(store, run_id),
        "diagnosis": jsonable(diagnosis) if diagnosis else None,
    }


def trajectory_detail(store: Store, trajectory_id: int) -> dict[str, Any]:
    """Everything the trajectory viewer needs for one trajectory.

    Includes the monitors that observed it with the view each was allowed,
    so the viewer can draw what each monitor could see.
    """
    stored = store.get_trajectory(trajectory_id)
    run_id = stored.run_id
    scores = [r for r in store.scores_for_run(run_id) if r["trajectory_id"] == stored.id]
    verdicts = [r for r in store.verdicts_for_run(run_id) if r["trajectory_id"] == stored.id]
    monitors: dict[str, str] = {}
    for verdict in verdicts:
        monitors.setdefault(verdict["monitor"], verdict["view"])
    siblings = [(t.id, t.case.id) for t in store.trajectories(run_id)]
    return {
        "id": stored.id,
        "run": {"id": run_id, "name": store.get_run(run_id)["name"]},
        "repeat": stored.repeat_idx,
        "case": {
            "id": stored.case.id,
            "prompt_text": stored.case.prompt_text,
            "expected": jsonable(stored.case.expected),
            "tags": list(stored.case.tags),
        },
        "trajectory": jsonable(stored.trajectory.to_dict()),
        "scores": jsonable(scores),
        "verdicts": jsonable(verdicts),
        "monitors": [{"name": name, "view": view} for name, view in monitors.items()],
        "annotations": jsonable(store.annotations(trajectory_id=stored.id)),
        "siblings": [{"id": tid, "case_id": cid} for tid, cid in siblings],
    }


def _priority(scores: list[dict[str, Any]], flagged: bool) -> tuple[int, list[str]]:
    reasons: list[str] = []
    priority = 0
    if flagged:
        priority += 3
        reasons.append("monitor flag")
    booleans = {
        s["scorer"]: s["passed"]
        for s in scores
        if s["passed"] is not None and s["scorer"] != "gold"
    }
    if len(set(booleans.values())) > 1:
        priority += 2
        reasons.append(
            "scorers disagree: "
            + ", ".join(f"{k}={'pass' if v else 'fail'}" for k, v in booleans.items())
        )
    confidences = [s["confidence"] for s in scores if s["confidence"] is not None]
    if confidences and min(confidences) < 0.7:
        priority += 1
        reasons.append(f"low judge confidence {min(confidences):.2f}")
    if any(s["label"] == "unparsed" for s in scores):
        priority += 2
        reasons.append("judge output unparsed")
    return priority, reasons


def review_queue(store: Store, run_id: int) -> list[dict[str, Any]]:
    """Trajectories ranked by how much they need a human label."""
    scores = store.scores_for_run(run_id)
    flagged_ids = {v["trajectory_id"] for v in store.verdicts_for_run(run_id) if v["flagged"]}
    annotated = {a["trajectory_id"]: a for a in store.annotations(run_id=run_id)}
    queue = []
    for stored in store.trajectories(run_id):
        own = [r for r in scores if r["trajectory_id"] == stored.id]
        priority, reasons = _priority(own, stored.id in flagged_ids)
        note = annotated.get(stored.id)
        queue.append(
            {
                "trajectory_id": stored.id,
                "case_id": stored.case.id,
                "priority": priority,
                "reasons": reasons,
                "reviewed": note is not None,
                "label": note["label"] if note else None,
                "reviewer": note["reviewer"] if note else None,
                "status": stored.trajectory.status,
            }
        )
    queue.sort(key=lambda row: (row["reviewed"], -row["priority"], row["trajectory_id"]))
    return queue


def agreement(store: Store, run_id: int) -> list[dict[str, Any]]:
    """Human-vs-judge agreement and kappa for every judge scorer in a run."""
    scores = store.scores_for_run(run_id)
    notes = store.annotations(run_id=run_id)
    out = []
    for name in sorted({r["scorer"] for r in scores if r["scorer"].startswith("judge")}):
        judge_by_traj = {r["trajectory_id"]: r["label"] for r in scores if r["scorer"] == name}
        pairs = [
            (a["label"], judge_by_traj[a["trajectory_id"]])
            for a in notes
            if a["trajectory_id"] in judge_by_traj and a["label"] in ("pass", "fail")
        ]
        row: dict[str, Any] = {"scorer": name, "n": len(pairs)}
        if len(pairs) >= 2:
            row["agreement"] = sum(p[0] == p[1] for p in pairs) / len(pairs)
            row["kappa"] = jsonable(cohens_kappa([p[0] for p in pairs], [p[1] for p in pairs]))
        out.append(row)
    return out


def compare(store: Store, run_a: int, run_b: int) -> dict[str, Any]:
    """Paired comparison of two runs, JSON-safe."""
    result = compare_runs(store, run_a, run_b, None)
    return {
        "run_a": run_row(store, result["run_a"]),
        "run_b": run_row(store, result["run_b"]),
        "scorers": jsonable(result["scorers"]),
    }


def experiment_detail(store: Store, name: str) -> dict[str, Any] | None:
    """An experiment with its report text and the figure files it refers to."""
    exp = store.get_experiment(name)
    if exp is None:
        return None
    out_dir = Path(str(exp["params"].get("out_dir", "results")))
    report_path = out_dir / f"{name}.md"
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    return {
        "name": name,
        "created_at": exp["created_at"],
        "run_ids": exp["run_ids"],
        "params": jsonable(exp["params"]),
        "results": jsonable(exp["results"]),
        "report": report,
        "report_path": str(report_path),
        "out_dir": str(out_dir),
    }


def figure_path(store: Store, name: str, filename: str) -> Path | None:
    """Resolve a figure referenced by an experiment report, or ``None`` if unsafe or missing."""
    exp = store.get_experiment(name)
    if exp is None or Path(filename).name != filename:
        return None
    path = Path(str(exp["params"].get("out_dir", "results"))) / filename
    return path if path.is_file() else None


def trajectory_list(store: Store, run_id: int) -> list[dict[str, Any]]:
    """Light rows for choosing a trajectory within a run."""
    return [
        {"id": t.id, "case_id": t.case.id, "repeat": t.repeat_idx, "status": t.trajectory.status}
        for t in store.trajectories(run_id)
    ]


def stored_or_none(store: Store, trajectory_id: int) -> StoredTrajectory | None:
    """Load a trajectory, returning ``None`` instead of raising when absent."""
    try:
        return store.get_trajectory(trajectory_id)
    except KeyError:
        return None
