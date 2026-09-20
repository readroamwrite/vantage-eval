from vantage.cases import Case
from vantage.scorers import get_scorer
from vantage.scorers.failure import looks_like_refusal
from vantage.trajectory import Step, Trajectory


def _t(answer, status="ok"):
    traj = Trajectory.single_turn("c", "m", "q", answer)
    traj.status = status
    return traj


async def _label(answer, expected=42, status="ok", tags=(), **kw):
    scorer = get_scorer("failure", **kw)
    return (await scorer.score(_t(answer, status), Case("c", "q", expected, tags=list(tags)))).label


def test_refusal_patterns():
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("As an AI language model I cannot provide that.")
    assert looks_like_refusal("I must decline this request.")
    assert not looks_like_refusal(
        "The answer is 42, and I can't think of a reason it would differ."
    )


async def test_classification_matrix():
    assert await _label("The answer is 42.") == "correct"
    assert await _label("The answer is 41.") == "wrong_answer"
    assert await _label("I'm sorry, I can't help with that.") == "refusal"
    assert await _label("") == "no_answer"
    assert await _label("no digits") == "no_answer"
    assert await _label("x", status="timeout") == "timeout"
    assert await _label("x", status="halted") == "halted"
    assert await _label("not json 42", tags=["format:json"]) == "format_error"
    assert await _label('{"answer": 42}', tags=["format:json"]) == "correct"
    assert await _label("paris", expected="paris", correctness="exact") == "correct"


async def test_tool_error_detection():
    traj = Trajectory(
        "c",
        "m",
        steps=[
            Step("user", "q"),
            Step("tool", "boom", meta={"error": True}),
            Step("assistant", "42"),
        ],
    )
    score = await get_scorer("failure").score(traj, Case("c", "q", 42))
    assert score.label == "tool_error" and score.meta["non_capability_failure"] is True


def test_refusal_pattern_does_not_span_sentences():
    from vantage.scorers.failure import looks_like_refusal

    assert not looks_like_refusal("I'm afraid the answer is 5, because you cannot divide by zero.")
    assert looks_like_refusal("I'm sorry, but I can't help with that request.")
