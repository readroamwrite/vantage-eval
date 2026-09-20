import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vantage.cli import app
from vantage.store import Store

DATA = Path(__file__).parent.parent / "data" / "smoke.jsonl"

pytest.importorskip("fastapi")


@pytest.fixture
def seeded_db(tmp_path):
    db = tmp_path / "dash.db"
    res = CliRunner().invoke(
        app,
        [
            "run",
            str(DATA),
            "--target",
            "mock:arith",
            "--scorer",
            "numeric",
            "--scorer",
            "failure",
            "--db",
            str(db),
        ],
    )
    assert res.exit_code == 0, res.output
    return db


@pytest.fixture
def client(seeded_db):
    from fastapi.testclient import TestClient

    from vantage.dashboard.api import create_app

    with TestClient(create_app(str(seeded_db))) as c:
        yield c


def test_runs_and_run_detail(client):
    runs = client.get("/api/runs").json()
    assert len(runs) == 1
    assert runs[0]["n"] == 10
    assert "numeric" in runs[0]["metrics"]
    est = runs[0]["metrics"]["numeric"]
    assert set(est) == {"point", "lo", "hi", "n", "level"}

    detail = client.get("/api/runs/1").json()
    assert detail["primary_scorer"] == "numeric"
    assert len(detail["cases"]) == 10
    assert detail["cases"][0]["scores"]["numeric"]["value"] in (0.0, 1.0)
    assert detail["diagnosis"]["scorer"] == "numeric"
    assert isinstance(detail["diagnosis"]["warnings"], list)

    assert client.get("/api/runs/999").status_code == 404


def test_trajectory_detail_lists_monitors_with_views(client):
    detail = client.get("/api/trajectories/1").json()
    assert detail["case"]["id"] == "s01"
    assert detail["trajectory"]["steps"][0]["role"] == "user"
    assert {s["scorer"] for s in detail["scores"]} == {"numeric", "failure"}
    assert detail["monitors"] == []
    assert [s["id"] for s in detail["siblings"]] == list(range(1, 11))
    assert client.get("/api/trajectories/999").status_code == 404


def test_review_queue_and_annotation_round_trip(client, seeded_db):
    review = client.get("/api/runs/1/review").json()
    assert len(review["queue"]) == 10
    assert all(not row["reviewed"] for row in review["queue"])

    res = client.post(
        "/api/annotations",
        json={"trajectory_id": 1, "reviewer": "tester", "label": "pass", "note": "looks right"},
    )
    assert res.status_code == 200, res.text
    with Store(seeded_db) as store:
        notes = store.annotations(run_id=1)
    assert len(notes) == 1 and notes[0]["reviewer"] == "tester" and notes[0]["value"] == 1.0

    review = client.get("/api/runs/1/review").json()
    reviewed = [row for row in review["queue"] if row["reviewed"]]
    assert reviewed[0]["trajectory_id"] == 1 and reviewed[0]["label"] == "pass"
    assert review["queue"][-1]["trajectory_id"] == 1, "labelled items sink to the bottom"

    assert (
        client.post(
            "/api/annotations", json={"trajectory_id": 1, "reviewer": "t", "label": "bogus"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/annotations", json={"trajectory_id": 1, "reviewer": "", "label": "pass"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/annotations", json={"trajectory_id": 99, "reviewer": "t", "label": "pass"}
        ).status_code
        == 404
    )


def test_compare_experiments_and_convmap(client):
    result = client.get("/api/compare?a=1&b=1").json()
    assert result["scorers"]["numeric"]["n_pairs"] == 10
    assert result["scorers"]["numeric"]["diff"]["point"] == 0.0
    assert client.get("/api/compare?a=1&b=7").status_code == 404

    assert client.get("/api/experiments").json() == []
    assert client.get("/api/experiments/nope").status_code == 404
    assert client.get("/api/experiments/nope/figures/x.png").status_code == 404

    assert client.get("/api/convmap/1").json() == {"map": None}
    assert client.get("/api/jobs/missing").status_code == 404


def test_finder_runs_as_a_job_on_the_mock_judge(client):
    job = client.post(
        "/api/finder",
        json={
            "rubric": "Is the answer a number?",
            "model": "mock:echo",
            "limit": 3,
            "view": "output",
        },
    ).json()
    assert job["status"] in ("running", "done")
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "done", job["error"]
    assert job["result"]["examined"] == 3
    assert job["result"]["monitor"].startswith("finder:")


def test_frontend_is_served_or_reported_missing(client):
    from vantage.dashboard.api import STATIC_DIR

    res = client.get("/runs/1")
    if (STATIC_DIR / "index.html").is_file():
        assert res.status_code == 200 and '<div id="root">' in res.text
    else:
        assert res.status_code == 503
    assert client.get("/api/nope").status_code == 404


def test_finder_job_accepts_an_audit_and_scoped_labels(client):
    job = client.post(
        "/api/finder",
        json={
            "rubric": "Is the answer a number?",
            "model": "mock:echo",
            "fts": "zzz_nothing_matches",
            "limit": 3,
            "audit": 2,
            "view": "output",
        },
    ).json()
    for _ in range(200):
        job = client.get(f"/api/jobs/{job['id']}").json()
        if job["status"] != "running":
            break
        time.sleep(0.02)
    assert job["status"] == "done", job["error"]
    audit = job["result"]["audit"]
    assert audit["n_audited"] == 2 and audit["n_excluded"] >= 2
    saved = client.post(
        "/api/annotations",
        json={
            "trajectory_id": 1,
            "reviewer": "v",
            "label": "yes",
            "monitor": job["result"]["monitor"],
        },
    )
    assert saved.status_code == 200 and saved.json()["label"] == "yes"
    assert (
        client.post(
            "/api/annotations", json={"trajectory_id": 1, "reviewer": "v", "label": "maybe"}
        ).status_code
        == 422
    )
