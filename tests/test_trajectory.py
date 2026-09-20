import pytest

from vantage.trajectory import Step, ToolCall, Trajectory


def _agent_traj() -> Trajectory:
    return Trajectory(
        case_id="t1",
        target_id="agent",
        steps=[
            Step("system", "be good"),
            Step("user", "fix the tests"),
            Step("assistant", "", tool_calls=[ToolCall("c1", "read_file", {"path": "task.py"})]),
            Step("tool", "def f(): ...", tool_call_id="c1"),
            Step("assistant", "done", meta={"tokens_out": 3}),
        ],
        final_output="done",
        meta={"condition": "honest"},
    )


def test_json_round_trip_preserves_everything():
    traj = _agent_traj()
    again = Trajectory.from_json(traj.to_json())
    assert again == traj


def test_single_turn_shape():
    traj = Trajectory.single_turn(
        "c", "m", "2+2?", "4", system="sys", response_meta={"latency_ms": 1}
    )
    assert [s.role for s in traj.steps] == ["system", "user", "assistant"]
    assert traj.final_output == "4"
    assert traj.answer == "4"
    assert traj.steps[-1].meta["latency_ms"] == 1


def test_tool_calls_and_counts():
    traj = _agent_traj()
    assert traj.n_steps == 5
    assert [c.name for c in traj.tool_calls] == ["read_file"]
    assert traj.last_assistant is not None and traj.last_assistant.content == "done"


def test_output_only_hides_tool_activity():
    view = _agent_traj().output_only()
    assert [s.role for s in view.steps] == ["system", "user", "assistant"]
    assert view.tool_calls == []
    assert view.meta["view"] == "output"
    assert view.meta["condition"] == "honest"
    assert view.final_output == "done"


def test_answer_falls_back_to_last_assistant():
    traj = Trajectory("c", "m", steps=[Step("user", "q"), Step("assistant", "a")])
    assert traj.answer == "a"
    assert Trajectory("c", "m").answer == ""


def test_invalid_role_and_status_rejected():
    with pytest.raises(ValueError):
        Step("robot", "x")
    with pytest.raises(ValueError):
        Trajectory("c", "m", status="weird")
