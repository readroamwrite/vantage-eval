import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vantage.analysis import diagnose_run, health_warnings, judge_corrected_rate
from vantage.cases import Case, Dataset
from vantage.cli import app
from vantage.models.mock import MockClient
from vantage.runner import Runner
from vantage.scorers import get_scorer
from vantage.stats import adjust_for_judge
from vantage.store import Store
from vantage.targets import LLMTarget

DATA = Path(__file__).parent.parent / "data" / "smoke.jsonl"


def test_rogan_gladen_point_and_interval():
    observed = [1] * 60 + [0] * 40  # judge says 60% pass
    est = adjust_for_judge(observed, tp=90, fn=10, fp=20, tn=80)  # se 0.9, sp 0.8
    assert est.point == pytest.approx(0.4 / 0.7, abs=1e-9)
    assert est.lo <= est.point <= est.hi and est.n == 100
    assert math.isnan(adjust_for_judge([], tp=1, fn=0, fp=0, tn=1).point)
    assert math.isnan(adjust_for_judge([1, 0], tp=5, fn=5, fp=5, tn=5).point)  # chance-level judge
    assert adjust_for_judge([1] * 10, tp=9, fn=1, fp=1, tn=9).point == 1.0  # clipped


def test_health_warnings_fire_on_bad_signals():
    from vantage.stats import Estimate

    diag = {
        "n": 20,
        "n_columns": 2,
        "saturation": {"saturated": True, "floored": False, "mean": Estimate(0.97, 0.93, 1.0, 20)},
        "dead_items": {"all_pass": ["a", "b", "c", "d", "e"], "all_fail": []},
        "low_discrimination": ["x", "y", "z"],
        "paraphrase": {"agreement": Estimate(0.6, 0.5, 0.7, 5)},
        "difficulty": {
            "spearman": 0.1,
            "spearman_ci": Estimate(0.1, -0.5, 0.6, 5),
            "easy_failure_rate": Estimate(0.3, 0.2, 0.4, 10),
        },
        "failures": {
            "non_capability_share": Estimate(0.2, 0.1, 0.3, 20),
            "raw_accuracy": Estimate(0.5, 0.4, 0.6, 20),
            "adjusted_accuracy": Estimate(0.6, 0.5, 0.7, 16),
        },
        "coverage_gaps": ["threat:x"],
        "judge_corrected": {
            "available": True,
            "corrected": Estimate(0.5, 0.4, 0.6, 20),
            "raw": Estimate(0.6, 0.5, 0.7, 20),
            "kappa": 0.7,
        },
    }
    names = [w.split(":")[0] for w in health_warnings(diag)]
    assert names == [
        "SATURATED",
        "DEAD_ITEMS",
        "LOW_DISCRIMINATION",
        "PARAPHRASE_SENSITIVE",
        "SANDBAG_ALARM",
        "NON_CAPABILITY_FAILURES",
        "COVERAGE_GAPS",
        "JUDGE_NOISE",
    ]
    clean = {
        **diag,
        "saturation": {"saturated": False, "floored": False, "mean": Estimate(0.5, 0.4, 0.6, 20)},
        "dead_items": None,
        "low_discrimination": [],
        "paraphrase": None,
        "difficulty": None,
        "failures": None,
        "coverage_gaps": [],
        "judge_corrected": None,
    }
    assert health_warnings(clean) == []


async def test_judge_corrected_rate_uses_stored_experiment():
    with Store(":memory:") as store:
        ds = Dataset("d", [Case(f"c{i}", "q", expected="x") for i in range(10)])
        target = LLMTarget(
            MockClient(lambda r: "x" if len(r.messages[-1]["content"]) % 2 else "y"), "m"
        )
        result = await Runner(store).run(
            target, ds, [get_scorer("exact", name="judge:fake")] if False else [get_scorer("exact")]
        )
        store.save_scores(result.results[0].trajectory_id, [])
        # rename: emulate a judge scorer column by re-saving scores under a judge name
        for r in result.results:
            score = r.score("exact")
            score.name = "judge:correct"
            store.save_scores(r.trajectory_id, [score])
        missing = judge_corrected_rate(store, result.run_id, "judge:correct")
        assert missing["available"] is False
        store.save_experiment(
            "judge_reliability",
            params={"judge": "j"},
            results={
                "agreement": {
                    "confusion": {
                        "pass": {"pass": 90, "fail": 10},
                        "fail": {"pass": 20, "fail": 80},
                    },
                    "kappa": {"point": 0.7},
                }
            },
        )
        found = judge_corrected_rate(store, result.run_id, "judge:correct")
        assert found["available"] and found["kappa"] == 0.7 and 0 <= found["corrected"].point <= 1
        diag = diagnose_run(store, result.run_id, scorer="judge:correct")
        assert any(w.startswith("JUDGE_NOISE") for w in diag["warnings"])


def test_coverage_cli_with_monitor_specs(tmp_path):
    db = tmp_path / "t.db"
    runner = CliRunner()
    assert (
        runner.invoke(app, ["run", str(DATA), "--scorer", "numeric", "--db", str(db)]).exit_code
        == 0
    )
    out = tmp_path / "cov.md"
    res = runner.invoke(
        app,
        [
            "coverage",
            "1",
            "--monitor",
            "forbidden:write_file,test_*.py",
            "--md",
            str(out),
            "--db",
            str(db),
        ],
    )
    assert res.exit_code == 0, res.output
    text = out.read_text()
    assert "Structural monitor coverage" in text and "forbidden:write_file:test_*.py" in text
    res = runner.invoke(app, ["diagnose", "1", "--db", str(db)])
    assert res.exit_code == 0 and "### Checks" in res.output
