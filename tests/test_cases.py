from pathlib import Path

import pytest

from vantage.cases import Case, Dataset

DATA = Path(__file__).parent / "data"


def test_load_jsonl():
    ds = Dataset.load(DATA / "cases.jsonl")
    assert ds.name == "cases"
    assert ds.ids == ["a1", "a2", "chat1"]
    assert ds.get("a2").expected == 144
    assert ds.get("a2").meta["base_id"] == "m12"


def test_load_json_with_name_and_string_tags():
    ds = Dataset.load(DATA / "cases.json")
    assert ds.name == "tiny"
    assert ds.get("x").tags == ["a", "b"]


def test_load_csv_parses_expected_and_tags():
    ds = Dataset.load(DATA / "cases.csv")
    c1 = ds.get("c1")
    assert c1.expected == 7
    assert c1.tags == ["tier:1", "cap:arithmetic"]
    assert c1.system == "Answer with a number."
    assert ds.get("c2").expected == "hello"
    assert ds.get("c2").system is None


def test_unsupported_extension():
    with pytest.raises(ValueError):
        Dataset.load(DATA / "cases.txt")


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        Dataset("d", [Case("1", "a"), Case("1", "b")])


def test_filter_requires_all_tags():
    ds = Dataset.load(DATA / "cases.jsonl")
    assert ds.filter("cap:arithmetic").ids == ["a1", "a2"]
    assert ds.filter("cap:arithmetic", "tier:1").ids == ["a1"]
    assert ds.filter("nope").ids == []


def test_tag_value_and_messages():
    case = Case("c", "hi", tags=["tier:3"], system="sys")
    assert case.tag_value("tier") == "3"
    assert case.tag_value("missing") is None
    assert case.messages() == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_multi_turn_messages_and_prompt_text():
    case = Dataset.load(DATA / "cases.jsonl").get("chat1")
    assert case.messages()[-1] == {"role": "user", "content": "2+2?"}
    assert "assistant: hello" in case.prompt_text


def test_with_system_sets_condition_and_new_name():
    ds = Dataset.load(DATA / "cases.jsonl").with_system("Underperform.", "sandbag")
    assert ds.name == "cases:sandbag"
    assert all(c.system == "Underperform." for c in ds)
    assert all(c.meta["condition"] == "sandbag" for c in ds)


def test_hash_is_stable_and_content_sensitive(tmp_path):
    ds = Dataset.load(DATA / "cases.jsonl")
    assert ds.hash() == Dataset.load(DATA / "cases.jsonl").hash()
    assert ds.hash() != ds.with_system("x", "y").hash()
    out = tmp_path / "copy.jsonl"
    ds.save(out)
    assert Dataset.load(out, name="cases").hash() == ds.hash()
