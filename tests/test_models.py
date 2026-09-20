import pytest

from vantage.models import ChatRequest, ChatResponse
from vantage.models.cache import CachedClient
from vantage.models.mock import MockClient
from vantage.store import Store


def _req(text="hi", **kw):
    return ChatRequest.create("m", [{"role": "user", "content": text}], **kw)


def test_request_key_is_stable_and_content_sensitive():
    assert _req().key() == _req().key()
    assert _req().key() != _req("other").key()
    assert _req().key() != _req(temperature=0.5).key()
    assert _req().key(0) != _req().key(1)


def test_request_round_trip_dict():
    req = _req(stop=("END",))
    data = req.to_dict()
    assert data["messages"] == [{"role": "user", "content": "hi"}]
    assert data["stop"] == ["END"]


async def test_mock_rules_default_and_callable():
    client = MockClient(rules={"2+2": "4"}, default="dunno")
    assert (await client.chat(_req("what is 2+2?"))).content == "4"
    assert (await client.chat(_req("who?"))).content == "dunno"
    assert len(client.calls) == 2
    echo = MockClient(lambda r: r.messages[-1]["content"].upper())
    assert (await echo.chat(_req("abc"))).content == "ABC"


async def test_mock_can_fail():
    client = MockClient(fail_with=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        await client.chat(_req())


async def test_cached_client_hits_on_second_call_and_respects_sample_idx():
    inner = MockClient(default="x", latency_ms=5)
    with Store(":memory:") as store:
        cached = CachedClient(inner, store)
        first = await cached.chat(_req())
        second = await cached.chat(_req())
        assert first.cached is False and first.latency_ms == 5
        assert second.cached is True and second.latency_ms == 0
        assert second.content == "x"
        assert (cached.hits, cached.misses) == (1, 1)
        assert len(inner.calls) == 1

        other = cached.with_sample(1)
        third = await other.chat(_req())
        assert third.cached is False
        assert len(inner.calls) == 2


def test_response_step_meta():
    meta = ChatResponse("a", tokens_in=1, tokens_out=2, latency_ms=3.0, model="m").step_meta()
    assert meta == {
        "tokens_in": 1,
        "tokens_out": 2,
        "latency_ms": 3.0,
        "cached": False,
        "model": "m",
    }
