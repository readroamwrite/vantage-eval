from pathlib import Path

from typer.testing import CliRunner

from vantage.analysis import compare_runs, coverage_report, diagnose_run
from vantage.cases import Case, Dataset
from vantage.cli import app
from vantage.models.mock import MockClient
from vantage.runner import Runner
from vantage.scorers import get_scorer
from vantage.store import Store
from vantage.targets import LLMTarget

DATA = Path(__file__).parent.parent / "data" / "smoke.jsonl"


def _tiered() -> Dataset:
    cases = []
    for tier in range(1, 4):
        for j in range(4):
            cases.append(
                Case(
                    f"t{tier}_{j}",
                    f"tier {tier} item {j}",
                    expected=1,
                    tags=[f"tier:{tier}", f"base:b{j}", "cap:x", "threat:none"],
                )
            )
    return Dataset("tiered", cases)


def _model(fail_from_tier: int):
    def respond(request):
        text = request.messages[-1]["content"]
        tier = int(text.split()[1])
        return "1" if tier < fail_from_tier else "0"

    return LLMTarget(MockClient(respond), f"m{fail_from_tier}")


async def test_compare_and_diagnose_on_stored_runs():
    with Store(":memory:") as store:
        runner = Runner(store)
        good = await runner.run(
            _model(3), _tiered(), [get_scorer("numeric"), get_scorer("failure")], repeats=2
        )
        weak = await runner.run(
            _model(2), _tiered(), [get_scorer("numeric"), get_scorer("failure")]
        )
        cmp_ = compare_runs(store, good.run_id, weak.run_id, "numeric")
        block = cmp_["scorers"]["numeric"]
        assert (
            block["n_pairs"] == 12 and block["diff"].point < 0 and len(block["flipped_down"]) == 4
        )

        diag = diagnose_run(store, good.run_id, against=[weak.run_id])
        assert diag["scorer"] == "numeric" and diag["n"] == 24
        assert (
            diag["difficulty"]["per_tier"][1].point == 1.0
            and diag["difficulty"]["per_tier"][3].point == 0.0
        )
        assert diag["dead_items"]["all_fail"] == ["t3_0", "t3_1", "t3_2", "t3_3"]
        assert diag["failures"]["raw_accuracy"].point > 0.6
        assert "threat:grader_tampering" in diag["coverage_gaps"]
        assert diag["paraphrase"]["n_bases"] == 4

        cov = coverage_report(store, good.run_id)
        assert (
            cov["counts"]["tier"]["tier:2"] == 8 and cov["pass_rates"]["cap"]["cap:x"].point > 0.6
        )
        assert cov["structural"]["threat:none"]["structurally_detectable"] is False


def test_cli_analysis_commands(tmp_path):
    db = tmp_path / "t.db"
    runner = CliRunner()
    for name, tgt in (("a", "mock:arith"), ("b", "mock:echo")):
        r = runner.invoke(
            app,
            [
                "run",
                str(DATA),
                "--target",
                tgt,
                "--scorer",
                "numeric",
                "--name",
                name,
                "--db",
                str(db),
            ],
        )
        assert r.exit_code == 0, r.output
    res = runner.invoke(app, ["rescore", "a", "--scorer", "failure", "--db", str(db)])
    assert res.exit_code == 0 and "failure" in res.output
    res = runner.invoke(app, ["compare", "a", "b", "--db", str(db)])
    assert res.exit_code == 0 and "diff (B - A)" in res.output
    res = runner.invoke(app, ["diagnose", "a", "--against", "b", "--db", str(db)])
    assert (
        res.exit_code == 0 and "Eval health" in res.output and "capability-adjusted" in res.output
    )
    out = tmp_path / "cov.md"
    res = runner.invoke(app, ["coverage", "a", "--md", str(out), "--db", str(db)])
    assert res.exit_code == 0 and "cap:arithmetic" in out.read_text()
