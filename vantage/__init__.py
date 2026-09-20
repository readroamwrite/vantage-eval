"""vantage: a pluggable evaluation framework for LLMs and agents.

A single ``Trajectory`` type represents anything under test: a one-shot LLM
call is a two-step trajectory, an agent run is a many-step one. Scorers and
monitors choose their *vantage*: the final output only, or the whole
trajectory.

The names below are the public surface most integrations need; deeper
modules (``vantage.judge``, ``vantage.monitors``, ``vantage.stats``, ...) are
imported directly.
"""

from vantage.cases import Case, Dataset
from vantage.results import Annotation, Score, Verdict
from vantage.runner import CaseResult, Runner, RunResult
from vantage.store import Store
from vantage.targets import LLMTarget, ScriptedTarget, Target
from vantage.trajectory import Step, ToolCall, Trajectory

__version__ = "0.1.0"

__all__ = [
    "Annotation",
    "Case",
    "CaseResult",
    "Dataset",
    "LLMTarget",
    "RunResult",
    "Runner",
    "Score",
    "ScriptedTarget",
    "Step",
    "Store",
    "Target",
    "ToolCall",
    "Trajectory",
    "Verdict",
    "__version__",
]
