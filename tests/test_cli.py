from pathlib import Path

from typer.testing import CliRunner

from vantage.cli import app

DATA = Path(__file__).parent.parent / "data" / "smoke.jsonl"


def test_run_then_show_and_report(tmp_path):
    db = tmp_path / "t.db"
    runner = CliRunner()
    result = runner.invoke(
        app, ["run", str(DATA), "--target", "mock:arith", "--scorer", "numeric", "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert "numeric" in result.output

    listed = runner.invoke(app, ["runs", "list", "--db", str(db)])
    assert listed.exit_code == 0 and "smoke@mock:arith" in listed.output

    shown = runner.invoke(app, ["runs", "show", "1", "--db", str(db)])
    assert shown.exit_code == 0 and "numeric" in shown.output

    report = runner.invoke(app, ["report", "smoke@mock:arith", "--db", str(db)])
    assert report.exit_code == 0 and "| numeric |" in report.output

    again = runner.invoke(
        app, ["run", str(DATA), "--target", "mock:arith", "--scorer", "numeric", "--db", str(db)]
    )
    assert again.exit_code == 0

    missing = runner.invoke(app, ["runs", "show", "nope", "--db", str(db)])
    assert missing.exit_code != 0


def test_scorers_command_lists_names():
    result = CliRunner().invoke(app, ["scorers"])
    assert result.exit_code == 0 and "numeric" in result.output
