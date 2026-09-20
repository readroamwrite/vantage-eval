from vantage.cases import Case
from vantage.models.mock import MockClient
from vantage.results import Verdict
from vantage.targets import LLMTarget, Target


async def test_llm_target_single_turn_trajectory():
    target = LLMTarget(MockClient(rules={"2+2": "4"}), "m", system="Be terse.")
    assert isinstance(target, Target)
    traj = await target.run(Case("c", "what is 2+2?"))
    assert traj.status == "ok"
    assert [s.role for s in traj.steps] == ["system", "user", "assistant"]
    assert traj.final_output == "4"
    assert traj.steps[-1].meta["model"] == "m"
    assert traj.steps[-1].meta["latency_ms"] == 1.0
    assert traj.target_id == "llm:m"
    assert target.config()["system"] == "Be terse."


async def test_case_system_prompt_wins_over_target_default():
    client = MockClient()
    target = LLMTarget(client, "m", system="target-sys")
    await target.run(Case("c", "q", system="case-sys"))
    assert client.calls[0].messages[0] == {"role": "system", "content": "case-sys"}


async def test_multi_turn_input_is_preserved():
    target = LLMTarget(MockClient(default="ok"), "m")
    traj = await target.run(
        Case(
            "c",
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        )
    )
    assert [s.role for s in traj.steps] == ["user", "assistant", "user", "assistant"]


async def test_provider_failure_becomes_error_trajectory():
    target = LLMTarget(MockClient(fail_with=RuntimeError("down")), "m")
    traj = await target.run(Case("c", "q"))
    assert traj.status == "error"
    assert "down" in (traj.error or "")
    assert traj.final_output is None


async def test_on_step_hook_can_halt():
    async def hook(traj):
        return Verdict("rule", flagged=True, halt=True)

    traj = await LLMTarget(MockClient(), "m").run(Case("c", "q"), on_step=hook)
    assert traj.status == "halted"
    assert traj.meta["halted_by"] == "rule"
