"""SQLite persistence for everything the framework produces.

The store holds runs, the cases they used, every trajectory with its steps,
scores, monitor events, human annotations, the model response cache,
experiment results and conversation maps. It is the single source of truth:
reports, the dashboard and re-scoring all read from here, so a run never has
to be repeated to be re-analysed.

Design notes:
    * Plain ``sqlite3`` with WAL mode, so the dashboard can read while a run
      writes.
    * One connection guarded by a re-entrant lock. Writes are short, so this
      is simpler and safer than a pool.
    * ``trajectories.traj_json`` is the canonical record; ``steps`` is a
      denormalised copy that feeds an FTS5 index for transcript search.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vantage.cases import Case, Dataset
from vantage.results import Annotation, Score, Verdict
from vantage.trajectory import Trajectory

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_cfg_json TEXT NOT NULL DEFAULT '{}',
    dataset_name TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    condition TEXT,
    scorers_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'running',
    notes TEXT
);
CREATE TABLE IF NOT EXISTS cases (
    id INTEGER PRIMARY KEY,
    dataset_hash TEXT NOT NULL,
    case_id TEXT NOT NULL,
    input_json TEXT NOT NULL,
    expected_json TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    system TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE (dataset_hash, case_id)
);
CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    repeat_idx INTEGER NOT NULL DEFAULT 0,
    target_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    final_output TEXT,
    n_steps INTEGER NOT NULL,
    n_tool_calls INTEGER NOT NULL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_ms REAL,
    traj_json TEXT NOT NULL,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (run_id, case_id, repeat_idx)
);
CREATE INDEX IF NOT EXISTS idx_trajectories_run ON trajectories(run_id);
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    tool_name TEXT,
    tool_args_json TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_steps_traj ON steps(trajectory_id);
CREATE VIRTUAL TABLE IF NOT EXISTS steps_fts USING fts5(
    content, tool_name, tool_args_json, content='steps', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS steps_ai AFTER INSERT ON steps BEGIN
    INSERT INTO steps_fts(rowid, content, tool_name, tool_args_json)
    VALUES (new.id, new.content, new.tool_name, new.tool_args_json);
END;
CREATE TRIGGER IF NOT EXISTS steps_ad AFTER DELETE ON steps BEGIN
    INSERT INTO steps_fts(steps_fts, rowid, content, tool_name, tool_args_json)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_args_json);
END;
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    scorer TEXT NOT NULL,
    value REAL NOT NULL,
    passed INTEGER,
    label TEXT,
    confidence REAL,
    rationale TEXT,
    meta_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE (trajectory_id, scorer)
);
CREATE INDEX IF NOT EXISTS idx_scores_traj ON scores(trajectory_id, scorer);
CREATE TABLE IF NOT EXISTS monitor_events (
    id INTEGER PRIMARY KEY,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    monitor TEXT NOT NULL,
    view TEXT NOT NULL,
    step_idx INTEGER NOT NULL,
    flagged INTEGER NOT NULL,
    halt INTEGER NOT NULL DEFAULT 0,
    score REAL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_monitor_events_traj ON monitor_events(trajectory_id);
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    reviewer TEXT NOT NULL,
    label TEXT NOT NULL,
    value REAL,
    note TEXT NOT NULL DEFAULT '',
    monitor TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    params_json TEXT NOT NULL DEFAULT '{}',
    results_json TEXT NOT NULL DEFAULT '{}',
    run_ids_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS conv_maps (
    id INTEGER PRIMARY KEY,
    trajectory_id INTEGER NOT NULL REFERENCES trajectories(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    tree_json TEXT NOT NULL,
    mermaid TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (trajectory_id, model)
);
"""


@dataclass(slots=True)
class StoredTrajectory:
    """A trajectory read back from the store together with its case.

    Attributes:
        id: Store id of the trajectory.
        run_id: Store id of the run it belongs to.
        repeat_idx: Which repeat of the case this is.
        case: The case the trajectory answers.
        trajectory: The trajectory itself.
    """

    id: int
    run_id: int
    repeat_idx: int
    case: Case
    trajectory: Trajectory


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in zip(row.keys(), tuple(row), strict=True):
        if key.endswith("_json") and isinstance(value, str):
            out[key[: -len("_json")]] = json.loads(value)
        else:
            out[key] = value
    return out


_FTS_OPERATORS = frozenset({"AND", "OR", "NOT"})


def fts_query(text: str) -> str:
    """Turn free text into a query FTS5 will accept.

    Every whitespace-separated word becomes a quoted term, so punctuation such
    as ``conftest.py`` or ``rm -rf`` cannot be read as syntax. The upper-case
    words ``AND``, ``OR`` and ``NOT`` are kept as operators when they sit
    between terms; anywhere else they are dropped.
    """
    out: list[str] = []
    for token in text.split():
        if token in _FTS_OPERATORS:
            if out and out[-1] not in _FTS_OPERATORS:
                out.append(token)
        else:
            out.append('"' + token.replace('"', '""') + '"')
    while out and out[-1] in _FTS_OPERATORS:
        out.pop()
    return " ".join(out) or '""'


class Store:
    """SQLite-backed repository for runs, trajectories, scores and more.

    Args:
        path: Database file. ``":memory:"`` gives a throwaway in-memory store.
    """

    def __init__(self, path: str | Path = "vantage.db") -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        self._migrate_fts()
        self._conn.executescript(SCHEMA)
        self._migrate_annotations()

    def _migrate_fts(self) -> None:
        """Rebuild the step index when an older database lacks the tool arguments column."""
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'steps_fts'"
        ).fetchone()
        if exists is None:
            return
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(steps_fts)")}
        if "tool_args_json" in columns:
            return
        self._conn.executescript(
            "DROP TRIGGER IF EXISTS steps_ai; DROP TRIGGER IF EXISTS steps_ad; DROP TABLE steps_fts;"
        )
        self._conn.executescript(SCHEMA)
        self._conn.execute("INSERT INTO steps_fts(steps_fts) VALUES ('rebuild')")

    def _migrate_annotations(self) -> None:
        """Add the ``monitor`` column to annotations written by older versions."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(annotations)")}
        if "monitor" not in columns:
            self._conn.execute("ALTER TABLE annotations ADD COLUMN monitor TEXT")

    def close(self) -> None:
        """Close the underlying connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [_row_to_dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ------------------------------------------------------------------ runs

    def create_run(
        self,
        name: str,
        target_id: str,
        dataset: Dataset,
        *,
        target_cfg: dict[str, Any] | None = None,
        scorer_names: Iterable[str] = (),
        condition: str | None = None,
        notes: str | None = None,
    ) -> int:
        """Create a run and register the dataset's cases.

        Args:
            name: Unique run name.
            target_id: Id of the target under test.
            dataset: Cases the run will evaluate; stored by content hash.
            target_cfg: Target configuration to keep for provenance.
            scorer_names: Names of the scorers the run applies.
            condition: Experimental condition label, if any.
            notes: Free text.

        Returns:
            The new run id.

        Raises:
            ValueError: If a run with ``name`` already exists.
        """
        dataset_hash = dataset.hash()
        with self._tx() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO runs (name, created_at, target_id, target_cfg_json, dataset_name,"
                    " dataset_hash, condition, scorers_json, notes)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        name,
                        _now(),
                        target_id,
                        _dumps(target_cfg or {}),
                        dataset.name,
                        dataset_hash,
                        condition,
                        _dumps(list(scorer_names)),
                        notes,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"run {name!r} already exists") from exc
            run_id = int(cur.lastrowid or 0)
            conn.executemany(
                "INSERT OR IGNORE INTO cases (dataset_hash, case_id, input_json, expected_json,"
                " tags_json, system, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        dataset_hash,
                        c.id,
                        _dumps(c.input),
                        _dumps(c.expected),
                        _dumps(c.tags),
                        c.system,
                        _dumps(c.meta),
                    )
                    for c in dataset
                ],
            )
        return run_id

    def find_run(self, name: str) -> int | None:
        """Return the id of the run named ``name``, or ``None``."""
        row = self._one("SELECT id FROM runs WHERE name = ?", (name,))
        return int(row["id"]) if row else None

    def get_run(self, run_id: int) -> dict[str, Any]:
        """Return one run's row, with ``n_trajectories`` added.

        Raises:
            KeyError: If the run does not exist.
        """
        row = self._one(
            "SELECT r.*, (SELECT COUNT(*) FROM trajectories t WHERE t.run_id = r.id)"
            " AS n_trajectories FROM runs r WHERE r.id = ?",
            (run_id,),
        )
        if row is None:
            raise KeyError(f"run {run_id} not found")
        return row

    def list_runs(self) -> list[dict[str, Any]]:
        """Return all runs, newest first, each with ``n_trajectories``."""
        return self._query(
            "SELECT r.*, (SELECT COUNT(*) FROM trajectories t WHERE t.run_id = r.id)"
            " AS n_trajectories FROM runs r ORDER BY r.id DESC"
        )

    def set_run_status(self, run_id: int, status: str) -> None:
        """Update a run's status (``running``, ``done`` or ``failed``)."""
        with self._tx() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))

    def delete_run(self, run_id: int) -> None:
        """Delete a run and everything attached to it."""
        with self._tx() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def completed_case_ids(self, run_id: int) -> set[tuple[str, int]]:
        """Return ``(case_id, repeat_idx)`` pairs already stored for a run."""
        rows = self._query(
            "SELECT case_id, repeat_idx FROM trajectories WHERE run_id = ?", (run_id,)
        )
        return {(r["case_id"], int(r["repeat_idx"])) for r in rows}

    # ------------------------------------------------------------ trajectories

    def save_trajectory(
        self, run_id: int, case: Case, traj: Trajectory, *, repeat_idx: int = 0
    ) -> int:
        """Persist a trajectory and its denormalised steps.

        Saving the same ``(run, case, repeat)`` again replaces the earlier
        record and everything attached to it.

        Returns:
            The trajectory's store id.
        """
        tokens_in = sum(int(s.meta.get("tokens_in", 0) or 0) for s in traj.steps)
        tokens_out = sum(int(s.meta.get("tokens_out", 0) or 0) for s in traj.steps)
        latency = sum(float(s.meta.get("latency_ms", 0) or 0) for s in traj.steps)
        with self._tx() as conn:
            conn.execute(
                "DELETE FROM trajectories WHERE run_id = ? AND case_id = ? AND repeat_idx = ?",
                (run_id, case.id, repeat_idx),
            )
            cur = conn.execute(
                "INSERT INTO trajectories (run_id, case_id, repeat_idx, target_id, status, error,"
                " final_output, n_steps, n_tool_calls, tokens_in, tokens_out, latency_ms,"
                " traj_json, meta_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    case.id,
                    repeat_idx,
                    traj.target_id,
                    traj.status,
                    traj.error,
                    traj.final_output,
                    traj.n_steps,
                    len(traj.tool_calls),
                    tokens_in,
                    tokens_out,
                    latency,
                    traj.to_json(),
                    _dumps(traj.meta),
                    _now(),
                ),
            )
            traj_id = int(cur.lastrowid or 0)
            conn.executemany(
                "INSERT INTO steps (trajectory_id, idx, role, content, tool_name, tool_args_json,"
                " meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        traj_id,
                        idx,
                        step.role,
                        step.content,
                        " ".join(c.name for c in step.tool_calls) if step.tool_calls else None,
                        _dumps([c.args for c in step.tool_calls]) if step.tool_calls else None,
                        _dumps(step.meta),
                    )
                    for idx, step in enumerate(traj.steps)
                ],
            )
        return traj_id

    def _cases_for(self, dataset_hash: str) -> dict[str, Case]:
        """All cases of one dataset, keyed by id, in a single query."""
        rows = self._query("SELECT * FROM cases WHERE dataset_hash = ?", (dataset_hash,))
        return {
            row["case_id"]: Case(
                id=row["case_id"],
                input=row["input"],
                expected=row["expected"],
                tags=row["tags"],
                system=row["system"],
                meta=row["meta"],
            )
            for row in rows
        }

    def trajectory_ids(self, run_id: int | None = None) -> list[int]:
        """Ids of stored trajectories, for one run or all runs, without loading them."""
        if run_id is None:
            rows = self._query("SELECT id FROM trajectories ORDER BY id")
        else:
            rows = self._query(
                "SELECT id FROM trajectories WHERE run_id = ? ORDER BY id", (run_id,)
            )
        return [int(row["id"]) for row in rows]

    def status_counts(self, run_id: int) -> dict[str, int]:
        """Number of trajectories per status for a run."""
        rows = self._query(
            "SELECT status, COUNT(*) AS n FROM trajectories WHERE run_id = ? GROUP BY status",
            (run_id,),
        )
        return {str(row["status"]): int(row["n"]) for row in rows}

    def _case_for(self, dataset_hash: str, case_id: str) -> Case:
        row = self._one(
            "SELECT * FROM cases WHERE dataset_hash = ? AND case_id = ?", (dataset_hash, case_id)
        )
        if row is None:
            raise KeyError(f"case {case_id!r} not found for dataset {dataset_hash}")
        return Case(
            id=row["case_id"],
            input=row["input"],
            expected=row["expected"],
            tags=row["tags"],
            system=row["system"],
            meta=row["meta"],
        )

    def get_trajectory(self, trajectory_id: int) -> StoredTrajectory:
        """Load one trajectory with its case.

        Raises:
            KeyError: If it does not exist.
        """
        row = self._one(
            "SELECT t.id, t.run_id, t.repeat_idx, t.case_id, t.traj_json, r.dataset_hash"
            " FROM trajectories t JOIN runs r ON r.id = t.run_id WHERE t.id = ?",
            (trajectory_id,),
        )
        if row is None:
            raise KeyError(f"trajectory {trajectory_id} not found")
        return StoredTrajectory(
            id=int(row["id"]),
            run_id=int(row["run_id"]),
            repeat_idx=int(row["repeat_idx"]),
            case=self._case_for(row["dataset_hash"], row["case_id"]),
            trajectory=Trajectory.from_dict(row["traj"]),
        )

    def trajectories(
        self, run_id: int, case_ids: Iterable[str] | None = None, *, limit: int | None = None
    ) -> list[StoredTrajectory]:
        """Load a run's trajectories, optionally restricted to some cases.

        Args:
            run_id: The run.
            case_ids: Keep only these cases.
            limit: Stop after this many rows, applied in SQL.
        """
        wanted = set(case_ids) if case_ids is not None else None
        sql = (
            "SELECT t.id, t.run_id, t.repeat_idx, t.case_id, t.traj_json, r.dataset_hash"
            " FROM trajectories t JOIN runs r ON r.id = t.run_id WHERE t.run_id = ?"
            " ORDER BY t.id"
        )
        params: tuple[Any, ...] = (run_id,)
        if limit is not None and wanted is None:
            sql += " LIMIT ?"
            params += (limit,)
        rows = self._query(sql, params)
        cases: dict[str, Case] = {}
        out: list[StoredTrajectory] = []
        for row in rows:
            if wanted is not None and row["case_id"] not in wanted:
                continue
            if not cases:
                cases = self._cases_for(row["dataset_hash"])
            out.append(
                StoredTrajectory(
                    id=int(row["id"]),
                    run_id=int(row["run_id"]),
                    repeat_idx=int(row["repeat_idx"]),
                    case=cases[row["case_id"]],
                    trajectory=Trajectory.from_dict(row["traj"]),
                )
            )
        return out

    # ------------------------------------------------------- scores/verdicts

    def save_scores(self, trajectory_id: int, scores: Iterable[Score]) -> None:
        """Persist scores for a trajectory, replacing any with the same scorer name."""
        now = _now()
        with self._tx() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO scores (trajectory_id, scorer, value, passed, label,"
                " confidence, rationale, meta_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        trajectory_id,
                        s.name,
                        float(s.value),
                        None if s.passed is None else int(s.passed),
                        s.label,
                        s.confidence,
                        s.rationale,
                        _dumps(s.meta),
                        now,
                    )
                    for s in scores
                ],
            )

    def scores_for_run(self, run_id: int, scorer: str | None = None) -> list[dict[str, Any]]:
        """Return score rows joined with case id, repeat, status and tags.

        Args:
            run_id: The run.
            scorer: Restrict to one scorer name when given.
        """
        sql = (
            "SELECT s.*, t.case_id, t.repeat_idx, t.status AS traj_status, c.tags_json"
            " FROM scores s JOIN trajectories t ON t.id = s.trajectory_id"
            " JOIN runs r ON r.id = t.run_id"
            " JOIN cases c ON c.dataset_hash = r.dataset_hash AND c.case_id = t.case_id"
            " WHERE t.run_id = ?"
        )
        params: tuple[Any, ...] = (run_id,)
        if scorer is not None:
            sql += " AND s.scorer = ?"
            params += (scorer,)
        sql += " ORDER BY t.id"
        rows = self._query(sql, params)
        for row in rows:
            row["passed"] = None if row["passed"] is None else bool(row["passed"])
        return rows

    def save_verdicts(self, trajectory_id: int, verdicts: Iterable[Verdict]) -> None:
        """Append monitor events for a trajectory."""
        with self._tx() as conn:
            conn.executemany(
                "INSERT INTO monitor_events (trajectory_id, monitor, view, step_idx, flagged,"
                " halt, score, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        trajectory_id,
                        v.monitor,
                        v.view,
                        v.step_index,
                        int(v.flagged),
                        int(v.halt),
                        v.score,
                        v.reason,
                    )
                    for v in verdicts
                ],
            )

    def clear_verdicts(self, trajectory_id: int, monitor: str) -> None:
        """Remove a monitor's events for a trajectory before re-scoring."""
        with self._tx() as conn:
            conn.execute(
                "DELETE FROM monitor_events WHERE trajectory_id = ? AND monitor = ?",
                (trajectory_id, monitor),
            )

    def verdicts_for_run(self, run_id: int, monitor: str | None = None) -> list[dict[str, Any]]:
        """Return monitor events for a run, joined with case id and repeat."""
        sql = (
            "SELECT m.*, t.case_id, t.repeat_idx FROM monitor_events m"
            " JOIN trajectories t ON t.id = m.trajectory_id WHERE t.run_id = ?"
        )
        params: tuple[Any, ...] = (run_id,)
        if monitor is not None:
            sql += " AND m.monitor = ?"
            params += (monitor,)
        sql += " ORDER BY m.id"
        rows = self._query(sql, params)
        for row in rows:
            row["flagged"] = bool(row["flagged"])
            row["halt"] = bool(row["halt"])
        return rows

    # ------------------------------------------------------------ annotations

    def save_annotation(self, annotation: Annotation) -> int:
        """Persist a human annotation and return its id."""
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO annotations (trajectory_id, reviewer, label, value, note, monitor,"
                " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    annotation.trajectory_id,
                    annotation.reviewer,
                    annotation.label,
                    annotation.value,
                    annotation.note,
                    annotation.monitor,
                    _now(),
                ),
            )
            return int(cur.lastrowid or 0)

    def delete_annotations(self, trajectory_id: int, *, reviewer: str, monitor: str | None) -> int:
        """Remove a reviewer's annotations on one trajectory; returns how many were removed."""
        with self._tx() as conn:
            cur = conn.execute(
                "DELETE FROM annotations WHERE trajectory_id = ? AND reviewer = ?"
                " AND ((monitor IS NULL AND ? IS NULL) OR monitor = ?)",
                (trajectory_id, reviewer, monitor, monitor),
            )
            return int(cur.rowcount)

    def annotations(
        self,
        *,
        run_id: int | None = None,
        trajectory_id: int | None = None,
        monitor: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return annotations for a run or a single trajectory.

        Args:
            run_id: Restrict to one run.
            trajectory_id: Restrict to one trajectory.
            monitor: Restrict to labels given in answer to one monitor's
                question (for example a finder rubric).
        """
        sql = (
            "SELECT a.*, t.case_id, t.run_id FROM annotations a"
            " JOIN trajectories t ON t.id = a.trajectory_id WHERE 1 = 1"
        )
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql += " AND t.run_id = ?"
            params += (run_id,)
        if trajectory_id is not None:
            sql += " AND a.trajectory_id = ?"
            params += (trajectory_id,)
        if monitor is not None:
            sql += " AND a.monitor = ?"
            params += (monitor,)
        return self._query(sql + " ORDER BY a.id", params)

    # ----------------------------------------------------------------- search

    def search_steps(
        self, query: str, *, run_id: int | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Full-text search over step contents.

        Args:
            query: An FTS5 match expression, for example ``"pytest OR conftest"``.
            run_id: Restrict to one run when given.
            limit: Maximum rows.

        Returns:
            Rows with ``trajectory_id``, ``run_id``, ``case_id``, ``idx``, ``role``
            and ``content``, best matches first.
        """
        sql = (
            "SELECT s.trajectory_id, t.run_id, t.case_id, s.idx, s.role, s.tool_name, s.content,"
            " s.tool_args_json FROM steps_fts f JOIN steps s ON s.id = f.rowid"
            " JOIN trajectories t ON t.id = s.trajectory_id WHERE steps_fts MATCH ?"
        )
        params: tuple[Any, ...] = (fts_query(query),)
        if run_id is not None:
            sql += " AND t.run_id = ?"
            params += (run_id,)
        sql += " ORDER BY rank LIMIT ?"
        params += (limit,)
        return self._query(sql, params)

    # ------------------------------------------------------------------ cache

    def cache_get(self, key: str) -> dict[str, Any] | None:
        """Return a cached response for ``key`` and count the hit."""
        with self._tx() as conn:
            row = conn.execute("SELECT response_json FROM cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            conn.execute("UPDATE cache SET hits = hits + 1 WHERE key = ?", (key,))
            return json.loads(row["response_json"])

    def cache_put(
        self, key: str, model: str, request: dict[str, Any], response: dict[str, Any]
    ) -> None:
        """Store a response under ``key``."""
        with self._tx() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, model, request_json, response_json, created_at, hits)"
                " VALUES (?, ?, ?, ?, ?, COALESCE((SELECT hits FROM cache WHERE key = ?), 0))",
                (key, model, _dumps(request), _dumps(response), _now(), key),
            )

    def cache_stats(self) -> dict[str, Any]:
        """Return entry count, total hits and per-model counts."""
        totals = self._one("SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits FROM cache")
        per_model = self._query("SELECT model, COUNT(*) AS entries FROM cache GROUP BY model")
        return {
            "entries": int(totals["entries"]) if totals else 0,
            "hits": int(totals["hits"]) if totals else 0,
            "models": {r["model"]: int(r["entries"]) for r in per_model},
        }

    def cache_clear(self) -> int:
        """Delete every cache entry and return how many were removed."""
        with self._tx() as conn:
            n = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            conn.execute("DELETE FROM cache")
        return int(n)

    # ------------------------------------------------------------ experiments

    def save_experiment(
        self,
        name: str,
        *,
        params: dict[str, Any],
        results: dict[str, Any],
        run_ids: Iterable[int] = (),
    ) -> None:
        """Create or replace an experiment's stored results."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO experiments (name, created_at, params_json, results_json, run_ids_json)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET created_at = excluded.created_at,"
                " params_json = excluded.params_json, results_json = excluded.results_json,"
                " run_ids_json = excluded.run_ids_json",
                (name, _now(), _dumps(params), _dumps(results), _dumps(list(run_ids))),
            )

    def get_experiment(self, name: str) -> dict[str, Any] | None:
        """Return an experiment row, or ``None``."""
        return self._one("SELECT * FROM experiments WHERE name = ?", (name,))

    def list_experiments(self) -> list[dict[str, Any]]:
        """Return all experiments, newest first."""
        return self._query("SELECT * FROM experiments ORDER BY id DESC")

    # -------------------------------------------------------------- conv maps

    def save_conv_map(
        self, trajectory_id: int, model: str, tree: dict[str, Any], mermaid: str
    ) -> None:
        """Create or replace the conversation map for a trajectory and model."""
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO conv_maps (trajectory_id, model, tree_json, mermaid, created_at)"
                " VALUES (?, ?, ?, ?, ?) ON CONFLICT(trajectory_id, model) DO UPDATE SET"
                " tree_json = excluded.tree_json, mermaid = excluded.mermaid,"
                " created_at = excluded.created_at",
                (trajectory_id, model, _dumps(tree), mermaid, _now()),
            )

    def get_conv_map(self, trajectory_id: int, model: str) -> dict[str, Any] | None:
        """Return the stored conversation map, or ``None``."""
        return self._one(
            "SELECT * FROM conv_maps WHERE trajectory_id = ? AND model = ?", (trajectory_id, model)
        )
