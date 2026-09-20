from vantage.cases import Case, Dataset
from vantage.models.mock import MockClient
from vantage.results import Verdict
from vantage.runner import Runner
from vantage.scorers import get_scorer
from vantage.store import Store
from vantage.targets import LLMTarget


class FlagLongAnswers:
    """Test monitor: flags answers longer than 3 characters, halts if asked."""

    name = "long"
    view = "output"

    def __init__(self, every_step=False, halt=False):
        self.every_step = every_step
        self.halt_on_flag = halt

    async def observe(self, traj, case, *, final):
        flagged = len(traj.answer) > 3
        return Verdict(self.name, flagged, halt=flagged and self.halt_on_flag, view=self.view)


def _dataset():
    return Dataset("d", [Case("a", "2+2", "4"), Case("b", "3+3", "6"), Case("c", "name?", "x")])


def _target(**kw):
    return LLMTarget(MockClient(rules={"2+2": "4", "3+3": "7"}, default="long answer"), "m", **kw)


async def test_run_scores_stores_and_summarizes():
    with Store(":memory:") as store:
        seen = []
        runner = Runner(store, on_result=seen.append)
        result = await runner.run(_target(), _dataset(), [get_scorer("exact")])
        assert result.name == "d@llm:m"
        assert [r.case.id for r in result.results] == ["a", "b", "c"]
        assert result.values("exact") == [1.0, 0.0, 0.0]
        est = result.metric("exact")
        assert est.n == 3 and est.point < 0.4
        assert result.status_counts() == {"ok": 3}
        assert len(seen) == 3
        assert store.get_run(result.run_id)["status"] == "done"
        assert len(store.scores_for_run(result.run_id)) == 3


async def test_resume_skips_stored_cases_and_keeps_their_scores():
    with Store(":memory:") as store:
        runner = Runner(store)
        first = await runner.run(_target(), _dataset(), [get_scorer("exact")], name="r")
        client = MockClient(default="never called")
        second = await runner.run(
            LLMTarget(client, "m"), _dataset(), [get_scorer("exact")], name="r"
        )
        assert client.calls == []
        assert second.values("exact") == first.values("exact")
        assert [r.trajectory_id for r in second.results] == [r.trajectory_id for r in first.results]


async def test_no_resume_recreates_run():
    with Store(":memory:") as store:
        runner = Runner(store)
        await runner.run(_target(), _dataset(), name="r")
        client = MockClient(default="fresh")
        result = await runner.run(LLMTarget(client, "m"), _dataset(), name="r", resume=False)
        assert len(client.calls) == 3
        assert all(r.trajectory.final_output == "fresh" for r in result.results)


async def test_repeats_and_condition_are_recorded():
    with Store(":memory:") as store:
        result = await Runner(store).run(_target(), _dataset(), repeats=2, condition="sandbag")
        assert len(result.results) == 6
        assert {r.repeat_idx for r in result.results} == {0, 1}
        assert all(r.trajectory.meta["condition"] == "sandbag" for r in result.results)
        assert store.get_run(result.run_id)["condition"] == "sandbag"


async def test_error_trajectories_are_retried_then_recorded():
    with Store(":memory:") as store:
        target = LLMTarget(MockClient(fail_with=RuntimeError("down")), "m")
        result = await Runner(store, retries=1).run(target, _dataset(), [get_scorer("exact")])
        assert result.status_counts() == {"error": 3}
        assert all(r.trajectory.meta["attempts"] == 2 for r in result.results)
        assert result.values("exact") == [0.0, 0.0, 0.0]


async def test_timeout_is_recorded():
    import asyncio

    class Slow:
        id = "slow"

        def config(self):
            return {}

        async def run(self, case, on_step=None):
            await asyncio.sleep(1)

    with Store(":memory:") as store:
        result = await Runner(store, timeout_s=0.01, retries=0).run(Slow(), _dataset())
        assert result.status_counts() == {"timeout": 3}


async def test_monitors_flag_and_halt():
    with Store(":memory:") as store:
        monitors = [FlagLongAnswers(every_step=True, halt=True)]
        result = await Runner(store).run(_target(), _dataset(), monitors=monitors)
        by_case = {r.case.id: r for r in result.results}
        assert by_case["a"].trajectory.status == "ok"
        assert by_case["c"].trajectory.status == "halted"
        assert any(v.halt for v in by_case["c"].verdicts)
        events = store.verdicts_for_run(result.run_id)
        assert any(e["halt"] for e in events)


async def test_rescore_adds_scores_without_calling_target():
    with Store(":memory:") as store:
        runner = Runner(store)
        first = await runner.run(_target(), _dataset(), [get_scorer("exact")])
        again = await runner.rescore(
            first.run_id, [get_scorer("contains")], monitors=[FlagLongAnswers()]
        )
        assert set(again.scorer_names) == {"exact", "contains"}
        assert again.values("exact") == first.values("exact")
        assert sum(1 for r in again.results for v in r.verdicts if v.flagged) == 1
