"""The code examples in docs/extending.md (formerly the README) must keep working."""

import asyncio

from vantage import Case, Dataset, Runner, Step, Store, Trajectory
from vantage.config import make_target, register_provider
from vantage.models.mock import MockClient
from vantage.monitors import RuleMonitor
from vantage.results import Score
from vantage.scorers import get_scorer, register


def test_public_imports_and_library_usage():
    async def main():
        with Store(":memory:") as store:
            target = make_target("mock:arith", store)
            result = await Runner(store).run(
                target, Dataset.load("data/smoke.jsonl"), [get_scorer("numeric")]
            )
            return result.metric("numeric")

    est = asyncio.run(main())
    assert est.n == 10 and 0 < est.point < 1
    assert Step("user", "x").role == "user"


async def test_custom_target_scorer_provider_and_monitor():
    class MyChatbot:
        id = "my-chatbot"

        def config(self):
            return {"kind": "test"}

        async def run(self, case: Case, on_step=None) -> Trajectory:
            return Trajectory.single_turn(
                case.id, self.id, case.prompt_text, "Source: docs. The answer is 4."
            )

    @register("mentions_source")
    class MentionsSource:
        name = "mentions_source"

        async def score(self, traj, case):
            ok = "source:" in traj.answer.lower()
            return Score(self.name, 1.0 if ok else 0.0, passed=ok)

    @register_provider("myprovider")
    def make_my_client(model: str):
        return MockClient(default=f"from {model}")

    with Store(":memory:") as store:
        result = await Runner(store).run(
            MyChatbot(), Dataset("d", [Case("c", "q")]), [get_scorer("mentions_source")]
        )
        assert result.values("mentions_source") == [1.0]
        traj = await make_target("myprovider:m1", store).run(Case("c", "q"))
        assert traj.final_output == "from m1"
    guard = RuleMonitor(get_scorer("forbidden:write_file,test_*.py"), halt_on_flag=True)
    assert guard.halt_on_flag and guard.view == "trajectory"
