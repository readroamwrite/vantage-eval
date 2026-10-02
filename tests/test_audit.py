import csv
import json

from typer.testing import CliRunner

from vantage.agent.scripted import ScriptedAgentTarget
from vantage.audit import (
    AUDIT_MONITOR,
    audit_markdown,
    audit_report,
    export_audit,
    import_labels,
    write_audit,
)
from vantage.cases import Dataset
from vantage.cli import app
from vantage.experiments import datagen
from vantage.results import Verdict
from vantage.runner import Runner
from vantage.scorers import get_scorer
from vantage.store import Store

BEHAVIOURS = ("honest_fix", "subtle_tamper", "fake_success")


async def _scripted_run(store: Store) -> int:
    cases = [c for c in datagen.scripted_behaviour_cases(1) if c.meta["behaviour"] in BEHAVIOURS]
    scorers = [get_scorer(s) for s in ("tampered", "fake_success")]
    result = await Runner(store).run(
        ScriptedAgentTarget(), Dataset("audit", cases), scorers, name="audit-run"
    )
    for r in result.results:  # a pretend judge that flags nothing
        store.save_verdicts(r.trajectory_id, [Verdict("judge:test", False, False, 0.0, "fine", -1)])
    return result.run_id


def _fill_labels(labels_path, key, store, run_id):
    """Label every run from the behaviour name, as an oracle human would."""
    by_traj = {t.id: t.case.meta["behaviour"] for t in store.trajectories(run_id)}
    rows = list(csv.DictReader(labels_path.open()))
    for row in rows:
        behaviour = by_traj[key[row["audit_id"]]["trajectory_id"]]
        row["label"] = "honest" if behaviour == "honest_fix" else "dishonest"
        row["note"] = behaviour
    with labels_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["audit_id", "label", "note"])
        writer.writeheader()
        writer.writerows(rows)


async def test_export_is_blind_deterministic_and_round_trips(tmp_path):
    with Store(":memory:") as store:
        run_id = await _scripted_run(store)
        items, key = export_audit(store, [run_id], seed=3)
        assert len(items) == 3
        assert {k["trajectory_id"] for k in key.values()} == {
            t.id for t in store.trajectories(run_id)
        }
        assert all(k["run"] == "audit-run" and ":" in k["case_id"] for k in key.values())
        assert [i["audit_id"] for i in items] == [
            i["audit_id"] for i in export_audit(store, [run_id], seed=3)[0]
        ]
        leaked = {"behaviour", "verdict", "tampered", "condition", "case_id", "dishonest", "system"}
        for item in items:
            assert not leaked & set(item), item.keys()
            assert "behaviour" not in json.dumps(item)
        tamper = next(i for i in items if "conftest.py" in i["files_written"])
        assert tamper["last_test_run"]["outcome"] == "passed"
        fake = next(i for i in items if i["last_test_run"]["outcome"].startswith("the agent never"))
        assert "tests pass" in fake["final_message"]

        paths = write_audit(tmp_path / "audit", items, key)
        text = paths["markdown"].read_text()
        assert text.count("## ") == 3 and "subtle_tamper" not in text and "pressured" not in text
        _fill_labels(paths["labels"], key, store, run_id)
        assert import_labels(store, paths["labels"], paths["key"], reviewer="me") == 3
        assert len(store.annotations(run_id=run_id, monitor=AUDIT_MONITOR)) == 3
        # Importing again replaces rather than duplicates, and ids resolve by run and case
        # when the store id points elsewhere (as in a copied database).
        for entry in key.values():
            entry["trajectory_id"] += 1000
        paths["key"].write_text(json.dumps(key))
        assert import_labels(store, paths["labels"], paths["key"], reviewer="me") == 3
        assert len(store.annotations(run_id=run_id, monitor=AUDIT_MONITOR)) == 3

        report = audit_report(store, [run_id])
        assert report["n_labelled"] == 3 and report["human_dishonest"] == 2
        rule = report["raters"]["rule (tampered or fake_success)"]
        assert rule["n"] == 3 and rule["agreement"] == 1.0 and rule["kappa"].point == 1.0
        judge = report["raters"]["judge:test"]
        assert judge["human_dishonest_rater_honest"] == 2 and judge["agreement"] == 1 / 3
        assert "| rule (tampered or fake_success) | 3 | 1.000 |" in audit_markdown(report)


async def test_audit_cli_round_trip(tmp_path):
    db = tmp_path / "t.db"
    with Store(db) as store:
        run_id = await _scripted_run(store)
    runner = CliRunner()
    exported = runner.invoke(
        app, ["audit", "export", str(tmp_path / "blind"), "--run", str(run_id), "--db", str(db)]
    )
    assert exported.exit_code == 0, exported.output
    key = json.loads((tmp_path / "blind.key.json").read_text())
    with Store(db) as store:
        _fill_labels(tmp_path / "blind.labels.csv", key, store, run_id)
    imported = runner.invoke(
        app,
        [
            "audit",
            "import",
            str(tmp_path / "blind.labels.csv"),
            "--reviewer",
            "me",
            "--db",
            str(db),
        ],
    )
    assert imported.exit_code == 0 and "saved 3 labels" in imported.output
    reported = runner.invoke(
        app,
        ["audit", "report", "--run", str(run_id), "--md", str(tmp_path / "a.md"), "--db", str(db)],
    )
    assert reported.exit_code == 0, reported.output
    assert "3 runs labelled" in (tmp_path / "a.md").read_text()
