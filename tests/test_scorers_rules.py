import pytest

from vantage.cases import Case
from vantage.scorers import get_scorer, list_scorers, parse_spec
from vantage.scorers.rules import extract_number, parse_json
from vantage.trajectory import Trajectory


def _t(answer: str) -> Trajectory:
    return Trajectory.single_turn("c", "m", "q", answer)


async def _run(spec: str, answer: str, expected=None, **kw):
    return await get_scorer(spec, **kw).score(_t(answer), Case("c", "q", expected))


def test_parse_spec_grammar():
    assert parse_spec("exact") == ("exact", [], {})
    assert parse_spec("numeric:0.01") == ("numeric", [0.01], {})
    assert parse_spec("numeric:0.01,relative=true") == ("numeric", [0.01], {"relative": True})
    assert parse_spec("contains:hello world") == ("contains", ["hello world"], {})
    with pytest.raises(ValueError):
        parse_spec("  ")
    with pytest.raises(KeyError):
        get_scorer("nope")


def test_registry_lists_builtins():
    assert {"exact", "contains", "regex", "json_valid", "numeric", "choice", "max_length"} <= set(
        list_scorers()
    )


async def test_exact_normalises_by_default():
    assert (await _run("exact", "  Paris \n", "paris")).passed is True
    assert (await _run("exact", "Paris", "paris", normalize=False)).passed is False
    assert (await _run("exact", "Rome", "paris")).value == 0.0


async def test_contains_and_regex():
    assert (await _run("contains", "The capital is PARIS.", "paris")).passed is True
    assert (await _run("contains:Berlin", "paris", "paris")).passed is False
    assert (await _run("regex", "answer: 42", r"\d+")).passed is True
    assert (await _run("regex:^\\d+$", "abc 42")).passed is False
    assert (await _run("regex", "42", r"\d+", full=True)).passed is True


async def test_json_valid_lenient_and_keys():
    assert (await _run("json_valid", '{"a": 1}')).passed is True
    assert (await _run("json_valid", 'Sure! ```json\n{"a": 1}\n```')).passed is True
    assert (await _run("json_valid", 'here: {"a": 1} done')).passed is True
    assert (await _run("json_valid", "not json")).passed is False
    assert (await _run("json_valid", 'x {"a": 1}', lenient=False)).passed is False
    res = await _run("json_valid", '{"a": 1}', require_keys=["a", "b"])
    assert res.passed is False and res.meta["missing_keys"] == ["b"]
    assert parse_json("[1, 2]") == [1, 2]


def test_extract_number_prefers_cues_then_last_number():
    assert extract_number("3 apples and 4 pears make 7") == 7
    assert extract_number("The answer is 12, not 13.") == 12
    assert extract_number("Total: $1,250.50") == 1250.5
    assert extract_number("Final answer = -3") == -3
    assert extract_number("no digits here") is None


async def test_numeric_tolerances():
    assert (await _run("numeric", "the answer is 42", 42)).passed is True
    assert (await _run("numeric", "42.004", 42, tol=0.01)).passed is True
    assert (await _run("numeric", "43", 42)).passed is False
    assert (await _run("numeric:0.1,relative=true", "105", 100)).passed is True
    res = await _run("numeric", "I refuse", 42)
    assert res.passed is False and res.label == "no_number"
    assert (await _run("numeric", "42", "not a number")).passed is False


async def test_choice_letter():
    assert (await _run("choice", "The answer is (B).", "b")).passed is True
    assert (await _run("choice", "B", "B")).passed is True
    res = await _run("choice", "Because it is A, not B", "B")
    assert res.passed is False and res.label == "A"


async def test_max_length():
    assert (await _run("max_length:3,unit=words", "one two three")).passed is True
    assert (await _run("max_length:3,unit=words", "one two three four")).passed is False
    assert (await _run("max_length", "x" * 501)).passed is False
    with pytest.raises(ValueError):
        get_scorer("max_length", unit="lines")


async def test_overlap_is_rouge_l_on_words():
    full = await _run("overlap", "the cat sat on the mat", "the cat sat on the mat")
    assert full.value == 1.0 and full.passed
    partial = await _run("overlap", "the cat stood on a mat", "the cat sat on the mat")
    assert 0.5 < partial.value < 1.0
    beyond = await _run("overlap", "the cat sat on the mat and then left", "the cat sat on the mat")
    assert beyond.value == 1.0  # extra words past the reference are ignored
    strict = await _run(
        "overlap:0.5,truncate=false",
        "the cat sat on the mat and then left",
        "the cat sat on the mat",
    )
    assert strict.value < 1.0
    none = await _run("overlap", "anything", None)
    assert none.value == 0.0 and not none.passed
    assert "overlap" in list_scorers()


async def test_extract_number_prefers_final_answer_over_earlier_cues():
    assert extract_number("The total is 5 apples per box, so the answer is 10.") == 10.0
    assert extract_number("total: 41. Final answer: $9.00") == 9.0
    assert extract_number("Final answer: **1,250**") == 1250.0
    assert extract_number("just numbers 3 and 4") == 4.0


async def test_choice_ignores_lowercase_article():
    res = await _run("choice", "It is a tie between B and C", "B")
    assert res.passed is True and res.label == "B"
    assert (await _run("choice", "the option (c) is right", "C")).passed is True
