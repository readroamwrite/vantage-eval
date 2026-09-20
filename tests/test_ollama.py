import json

import httpx
import pytest

from vantage.models import ChatRequest, ModelError
from vantage.models.ollama import OllamaClient


def _client(handler, retries=0):
    return OllamaClient("http://fake", transport=httpx.MockTransport(handler), retries=retries)


async def test_chat_builds_payload_and_parses_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "m",
                "message": {"role": "assistant", "content": "4"},
                "prompt_eval_count": 7,
                "eval_count": 1,
            },
        )

    client = _client(handler)
    req = ChatRequest.create(
        "m", [{"role": "user", "content": "2+2?"}], max_tokens=8, json_mode=True, stop=("END",)
    )
    resp = await client.chat(req)
    assert resp.content == "4" and resp.tokens_in == 7 and resp.tokens_out == 1
    assert resp.latency_ms >= 0 and resp.model == "m"
    assert seen["path"] == "/api/chat"
    payload = seen["payload"]
    assert payload["model"] == "m" and payload["stream"] is False and payload["format"] == "json"
    assert payload["options"]["num_predict"] == 8 and payload["options"]["stop"] == ["END"]
    assert payload["options"]["seed"] == 0
    await client.aclose()


async def test_missing_model_is_a_clear_error():
    client = _client(lambda r: httpx.Response(404, text="model not found"))
    with pytest.raises(ModelError, match="ollama pull"):
        await client.chat(ChatRequest.create("nope", [{"role": "user", "content": "x"}]))


async def test_server_errors_are_retried_then_raised():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500, text="boom")

    client = _client(handler, retries=1)
    with pytest.raises(ModelError, match="after 2 attempts"):
        await client.chat(ChatRequest.create("m", [{"role": "user", "content": "x"}]))
    assert len(calls) == 2


async def test_list_models_and_ping():
    ok = _client(lambda r: httpx.Response(200, json={"models": [{"name": "qwen2.5:3b"}]}))
    assert [m["name"] for m in await ok.list_models()] == ["qwen2.5:3b"]
    assert await ok.ping() is True

    def down(request):
        raise httpx.ConnectError("refused")

    assert await _client(down).ping() is False
