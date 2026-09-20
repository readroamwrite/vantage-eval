import pytest

from vantage.cases import Case, Dataset
from vantage.results import Annotation, Score, Verdict
from vantage.store import Store
from vantage.trajectory import Step, ToolCall, Trajectory


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "t.db") as s:
        yield s


@pytest.fixture
def dataset():
    return Dataset("d", [Case("a", "1+1?", "2", tags=["tier:1"]), Case("b", "2*3?", "6")])


def _traj(case_id: str, with_tool: bool = False) -> Trajectory:
    steps = [Step("user", "q")]
    if with_tool:
        steps.append(
            Step("assistant", "", tool_calls=[ToolCall("c1", "write_file", {"path": "test_x.py"})])
        )
        steps.append(Step("tool", "ok", tool_call_id="c1"))
    steps.append(
        Step(
            "assistant",
            "the answer is 42",
            meta={"tokens_in": 5, "tokens_out": 7, "latency_ms": 10},
        )
    )
    return Trajectory(case_id, "mock", steps=steps, final_output="42")


def test_create_run_registers_cases_and_rejects_duplicate_names(store, dataset):
    run_id = store.create_run("r1", "mock", dataset, scorer_names=["exact"])
    row = store.get_run(run_id)
    assert row["name"] == "r1"
    assert row["scorers"] == ["exact"]
    assert row["n_trajectories"] == 0
    assert store.find_run("r1") == run_id
    assert store.find_run("nope") is None
    with pytest.raises(ValueError):
        store.create_run("r1", "mock", dataset)


def test_save_and_load_trajectory_round_trip(store, dataset):
    run_id = store.create_run("r", "mock", dataset)
    traj = _traj("a", with_tool=True)
    tid = store.save_trajectory(run_id, dataset.get("a"), traj)
    stored = store.get_trajectory(tid)
    assert stored.trajectory == traj
    assert stored.case.id == "a"
    assert stored.case.tags == ["tier:1"]
    assert stored.case.expected == "2"
    assert store.get_run(run_id)["n_trajectories"] == 1


def test_resave_replaces_and_completed_ids_support_resume(store, dataset):
    run_id = store.create_run("r", "mock", dataset)
    store.save_trajectory(run_id, dataset.get("a"), _traj("a"))
    store.save_trajectory(run_id, dataset.get("a"), _traj("a"), repeat_idx=1)
    store.save_trajectory(run_id, dataset.get("a"), _traj("a"))  # replace repeat 0
    assert store.completed_case_ids(run_id) == {("a", 0), ("a", 1)}
    assert len(store.trajectories(run_id)) == 2
    assert [t.case.id for t in store.trajectories(run_id, case_ids=["b"])] == []


def test_scores_verdicts_and_annotations(store, dataset):
    run_id = store.create_run("r", "mock", dataset)
    tid = store.save_trajectory(run_id, dataset.get("a"), _traj("a"))
    store.save_scores(
        tid, [Score("exact", 1.0, passed=True), Score("judge", 0.5, label="pass", confidence=0.9)]
    )
    store.save_scores(tid, [Score("exact", 0.0, passed=False)])  # replaces
    rows = store.scores_for_run(run_id)
    by_name = {r["scorer"]: r for r in rows}
    assert by_name["exact"]["value"] == 0.0 and by_name["exact"]["passed"] is False
    assert by_name["judge"]["confidence"] == 0.9
    assert by_name["exact"]["tags"] == ["tier:1"]
    assert store.scores_for_run(run_id, scorer="judge")[0]["label"] == "pass"

    store.save_verdicts(tid, [Verdict("rule", True, halt=True, step_index=2, view="trajectory")])
    events = store.verdicts_for_run(run_id)
    assert (
        events[0]["flagged"] is True and events[0]["halt"] is True and events[0]["case_id"] == "a"
    )
    store.clear_verdicts(tid, "rule")
    assert store.verdicts_for_run(run_id) == []

    store.save_annotation(Annotation(tid, "vibha", "pass", value=1.0, note="fine"))
    notes = store.annotations(run_id=run_id)
    assert notes[0]["reviewer"] == "vibha" and notes[0]["label"] == "pass"


def test_full_text_search_finds_step_content(store, dataset):
    run_id = store.create_run("r", "mock", dataset)
    store.save_trajectory(run_id, dataset.get("a"), _traj("a", with_tool=True))
    hits = store.search_steps("answer")
    assert hits and hits[0]["case_id"] == "a" and hits[0]["role"] == "assistant"
    assert store.search_steps("answer", run_id=run_id + 1) == []
    assert store.search_steps("zebra") == []
    by_args = store.search_steps("test_x")
    assert by_args and by_args[0]["tool_name"] == "write_file"


def test_old_fts_index_is_rebuilt(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    with Store(path) as s:
        ds = Dataset("d", [Case("a", "q")])
        rid = s.create_run("r", "m", ds)
        s.save_trajectory(rid, ds.get("a"), _traj("a", with_tool=True))
    conn = sqlite3.connect(path)
    conn.executescript(
        "DROP TRIGGER steps_ai; DROP TRIGGER steps_ad; DROP TABLE steps_fts;"
        " CREATE VIRTUAL TABLE steps_fts USING fts5(content, tool_name, content='steps', content_rowid='id');"
    )
    conn.close()
    with Store(path) as s:
        assert s.search_steps("test_x")


def test_delete_run_cascades(store, dataset):
    run_id = store.create_run("r", "mock", dataset)
    tid = store.save_trajectory(run_id, dataset.get("a"), _traj("a"))
    store.save_scores(tid, [Score("exact", 1.0)])
    store.delete_run(run_id)
    with pytest.raises(KeyError):
        store.get_trajectory(tid)
    assert store.search_steps("answer") == []


def test_cache_round_trip_and_stats(store):
    assert store.cache_get("k") is None
    store.cache_put("k", "m", {"q": 1}, {"content": "x"})
    assert store.cache_get("k") == {"content": "x"}
    assert store.cache_get("k") == {"content": "x"}
    stats = store.cache_stats()
    assert stats == {"entries": 1, "hits": 2, "models": {"m": 1}}
    assert store.cache_clear() == 1
    assert store.cache_stats()["entries"] == 0


def test_experiments_and_conv_maps_upsert(store, dataset):
    store.save_experiment("e", params={"n": 1}, results={"acc": 0.5}, run_ids=[1])
    store.save_experiment("e", params={"n": 2}, results={"acc": 0.6})
    exp = store.get_experiment("e")
    assert exp["params"] == {"n": 2} and exp["results"] == {"acc": 0.6} and exp["run_ids"] == []
    assert len(store.list_experiments()) == 1

    run_id = store.create_run("r", "mock", dataset)
    tid = store.save_trajectory(run_id, dataset.get("a"), _traj("a"))
    store.save_conv_map(tid, "m", {"title": "t"}, "mindmap")
    store.save_conv_map(tid, "m", {"title": "t2"}, "mindmap2")
    assert store.get_conv_map(tid, "m")["tree"] == {"title": "t2"}
    assert store.get_conv_map(tid, "other") is None


def test_in_memory_store_works():
    with Store(":memory:") as s:
        assert s.list_runs() == []


def test_fts_query_is_safe_for_punctuation_and_operators(store, dataset):
    from vantage.store import fts_query

    assert fts_query("conftest.py") == '"conftest.py"'
    assert fts_query("rm -rf") == '"rm" "-rf"'
    assert fts_query("OR pytest OR conftest OR") == '"pytest" OR "conftest"'
    assert fts_query('say "hi"') == '"say" """hi"""'
    assert fts_query("") == '""'
    assert store.search_steps("conftest.py") == []  # no crash on punctuation


def test_annotations_can_be_scoped_to_a_monitor(store, dataset):
    from vantage.results import Annotation
    from vantage.trajectory import Trajectory

    run_id = store.create_run("r", "mock", dataset)
    tid = store.save_trajectory(
        run_id, dataset.get("a"), Trajectory.single_turn("a", "t", "q", "x")
    )
    store.save_annotation(Annotation(tid, "v", "fail"))
    store.save_annotation(Annotation(tid, "v", "yes", monitor="finder:abc"))
    assert len(store.annotations(trajectory_id=tid)) == 2
    assert [a["label"] for a in store.annotations(trajectory_id=tid, monitor="finder:abc")] == [
        "yes"
    ]


def test_old_annotations_table_gains_monitor_column(tmp_path):
    import sqlite3

    from vantage.store import Store

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.executescript(
        "CREATE TABLE annotations (id INTEGER PRIMARY KEY, trajectory_id INTEGER NOT NULL,"
        " reviewer TEXT NOT NULL, label TEXT NOT NULL, value REAL, note TEXT NOT NULL DEFAULT '',"
        " created_at TEXT NOT NULL);"
    )
    con.close()
    with Store(db) as store:
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(annotations)")}
    assert "monitor" in columns
