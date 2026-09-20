import json

from vantage.cases import Case
from vantage.judge import Judge, parse_verdict, render_transcript
from vantage.models.cache import CachedClient
from vantage.models.mock import MockClient
from vantage.scorers import get_scorer
from vantage.store import Store
from vantage.trajectory import Step, ToolCall, Trajectory


def _traj(answer="The answer is 4."):
    return Trajectory.single_turn("c", "m", "2+2?", answer)


def _json_judge(verdict="pass", confidence=90):
    return MockClient(
        default=json.dumps({"verdict": verdict, "confidence": confidence, "rationale": "ok"})
    )


def test_parse_verdict_variants():
    v = parse_verdict(
        '{"verdict": "PASS", "confidence": 80, "rationale": "fine"}', ("pass", "fail")
    )
    assert v.parse_ok and v.label == "pass" and v.confidence == 0.8 and v.rationale == "fine"
    v = parse_verdict('```json\n{"label": "fail", "confidence": 0.3}\n```', ("pass", "fail"))
    assert v.parse_ok and v.label == "fail" and v.confidence == 0.3
    v = parse_verdict('{"verdict": "maybe"}', ("pass", "fail"))
    assert not v.parse_ok and v.label == "unparsed"
    v = parse_verdict("no json here", ("pass", "fail"))
    assert not v.parse_ok and v.confidence is None
    v = parse_verdict('{"answer": "B", "confidence": "high"}', ("A", "B"))
    assert v.parse_ok and v.label == "B" and v.confidence is None


def test_render_transcript_views():
    traj = Trajectory(
        "c",
        "agent",
        steps=[
            Step("system", "sys"),
            Step("user", "fix"),
            Step("assistant", "", tool_calls=[ToolCall("1", "write_file", {"path": "test_x.py"})]),
            Step("tool", "x" * 2000, tool_call_id="1"),
            Step("assistant", "done"),
        ],
        final_output="done",
        status="halted",
        error="stopped",
    )
    full = render_transcript(traj, "trajectory")
    assert "TOOL CALL write_file" in full and "more chars]" in full and "status halted" in full
    out = render_transcript(traj, "output")
    assert "write_file" not in out and "done" in out and "sys" not in out


async def test_grade_prompt_contains_task_reference_and_rubric():
    client = _json_judge()
    judge = Judge(client, "m", "Only the final answer matters.")
    verdict = await judge.grade(_traj(), Case("c", "2+2?", expected=4))
    assert verdict.parse_ok and verdict.label == "pass" and verdict.confidence == 0.9
    prompt = client.calls[0].messages[-1]["content"]
    assert "2+2?" in prompt and "Reference answer\n4" in prompt and "final answer matters" in prompt
    assert client.calls[0].json_mode is True
    no_ref = Judge(MockClient(default='{"verdict":"pass"}'), "m", "r", use_reference=False)
    await no_ref.grade(_traj(), Case("c", "q", expected=4))
    assert "Reference answer" not in no_ref.client.calls[0].messages[-1]["content"]


async def test_repair_path_on_bad_json():
    replies = iter(["I think it's correct.", '{"verdict": "pass", "confidence": 60}'])
    client = MockClient(lambda r: next(replies))
    verdict = await Judge(client, "m", "r").grade(_traj(), Case("c", "q"))
    assert verdict.parse_ok and verdict.label == "pass" and verdict.meta["attempts"] == 2
    assert "not a valid JSON" in client.calls[1].messages[-1]["content"]

    always_bad = MockClient(default="nope")
    verdict = await Judge(always_bad, "m", "r").grade(_traj(), Case("c", "q"))
    assert not verdict.parse_ok and verdict.label == "unparsed" and len(always_bad.calls) == 2


async def test_compare_unswaps_labels():
    picks_first = MockClient(default='{"verdict": "A", "confidence": 70}')
    judge = Judge(picks_first, "m", "r")
    a, b = _traj("4"), _traj("5")
    normal = await judge.compare(a, b, Case("c", "q"))
    swapped = await judge.compare(a, b, Case("c", "q"), swap=True)
    assert normal.label == "A" and normal.meta["presented_first"] == "a"
    assert swapped.label == "B" and swapped.meta["swapped"] is True
    prompt = picks_first.calls[1].messages[-1]["content"]
    assert prompt.index("Response A\n") < prompt.index("5") < prompt.index("Response B\n")


async def test_grade_n_uses_distinct_cache_samples():
    inner = _json_judge()
    with Store(":memory:") as store:
        judge = Judge(CachedClient(inner, store), "m", "r")
        verdicts = await judge.grade_n(_traj(), Case("c", "q"), 3)
        assert [v.meta["sample_idx"] for v in verdicts] == [0, 1, 2]
        assert len(inner.calls) == 3
        assert all(c.temperature == 0.7 for c in inner.calls)
        assert [c.seed for c in inner.calls] == [1, 2, 3]  # distinct seeds, not identical draws
        again = await judge.grade_n(_traj(), Case("c", "q"), 3)
        assert len(inner.calls) == 3 and len(again) == 3


async def test_judge_scorer_maps_labels_to_scores():
    scorer = get_scorer("judge:data/rubrics/correct.md", client=_json_judge("fail", 55))
    assert scorer.name == "judge:correct"
    score = await scorer.score(_traj(), Case("c", "q", expected=4))
    assert (
        score.value == 0.0
        and score.passed is False
        and score.label == "fail"
        and score.confidence == 0.55
    )
    unparsed = get_scorer("judge:Is it right?", client=MockClient(default="???"))
    score = await unparsed.score(_traj(), Case("c", "q"))
    assert score.passed is None and score.label == "unparsed"


def test_parse_verdict_rejects_label_prefixes():
    from vantage.judge import parse_verdict

    assert parse_verdict('{"verdict": "Both are equally good"}', ["A", "B"]).label == "unparsed"
    assert parse_verdict('{"verdict": "Not sure"}', ["yes", "no"]).label == "unparsed"
    assert (
        parse_verdict('{"verdict": "yes, because it edited tests"}', ["yes", "no"]).label == "yes"
    )
    assert parse_verdict('{"verdict": "(B)"}', ["A", "B"]).label == "B"


def test_inline_rubrics_get_distinct_scorer_names(tmp_path):
    from vantage.scorers import get_scorer

    polite = get_scorer("judge:Is it polite?", store=None, model="mock:echo")
    correct = get_scorer("judge:Is it correct?", store=None, model="mock:echo")
    assert polite.name != correct.name and polite.name.startswith("judge:")
    rubric = tmp_path / "tone.md"
    rubric.write_text("Is the tone friendly?")
    assert get_scorer(f"judge:{rubric}", store=None, model="mock:echo").name == "judge:tone"


def test_halted_status_is_hidden_from_the_output_view():
    from vantage.judge import render_transcript
    from vantage.trajectory import Trajectory

    traj = Trajectory.single_turn("c", "m", "q", "partial")
    traj.status, traj.error = "halted", "monitor"
    assert "halted" in render_transcript(traj, "trajectory")
    assert "halted" not in render_transcript(traj, "output")


async def test_judge_scores_carry_a_config_hash():
    from vantage.judge import Judge
    from vantage.models.mock import MockClient
    from vantage.scorers.judge import JudgeScorer

    client = MockClient(lambda req: '{"verdict": "pass", "confidence": 90, "rationale": "ok"}')
    scorer = JudgeScorer("Is it right?", client=client, model="m")
    score = await scorer.score(Trajectory.single_turn("c", "m", "q", "a"), Case("c", "q"))
    assert score.meta["judge"] == scorer.judge.config_hash and len(score.meta["judge"]) == 12
    other = Judge(client, "m", "Is it right?", cot=True)
    assert other.config_hash != scorer.judge.config_hash  # prompt template differs
