from types import SimpleNamespace

import pytest

from vantage.config import make_client
from vantage.models import ChatRequest, ModelError
from vantage.models.anthropic import AnthropicClient


class FakeMessages:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("rate limited")
        return SimpleNamespace(
            content=[SimpleNamespace(text="4")],
            usage=SimpleNamespace(input_tokens=12, output_tokens=1),
            model=kwargs["model"],
            stop_reason="end_turn",
        )


async def test_adapter_maps_request_and_response():
    fake = FakeMessages()
    client = AnthropicClient(sdk_client=SimpleNamespace(messages=fake))
    req = ChatRequest.create(
        "claude-sonnet-5",
        [{"role": "system", "content": "be terse"}, {"role": "user", "content": "2+2?"}],
        max_tokens=16,
        json_mode=True,
        stop=("END",),
    )
    resp = await client.chat(req)
    assert (
        resp.content == "4"
        and resp.tokens_in == 12
        and resp.tokens_out == 1
        and resp.model == "claude-sonnet-5"
    )
    call = fake.calls[0]
    assert (
        call["system"] == "be terse"
        and call["stop_sequences"] == ["END"]
        and call["max_tokens"] == 16
        and call["thinking"] == {"type": "disabled"}
        and "temperature" not in call
    )
    assert (
        call["messages"][-1]["role"] == "user"
        and "JSON object only" in call["messages"][-1]["content"]
    )


async def test_adapter_errors_are_model_errors(monkeypatch):
    with pytest.raises(ModelError, match="rate limited"):
        await AnthropicClient(sdk_client=SimpleNamespace(messages=FakeMessages(fail=True))).chat(
            ChatRequest.create("m", [{"role": "user", "content": "x"}])
        )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ModelError, match="ANTHROPIC_API_KEY"):
        await AnthropicClient().chat(ChatRequest.create("m", [{"role": "user", "content": "x"}]))


def test_provider_registered():
    assert isinstance(make_client("anthropic:claude-sonnet-5"), AnthropicClient)
    with pytest.raises(ValueError):
        make_client("anthropic")
