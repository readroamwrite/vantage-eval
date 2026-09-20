from typer.testing import CliRunner

import vantage
from vantage.cli import app


def test_version_string():
    assert vantage.__version__


def test_cli_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert vantage.__version__ in result.output
