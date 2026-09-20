"""Anthropic client adapter (optional extra ``vantage-eval[anthropic]``).

Selected with model specs such as ``anthropic:claude-sonnet-5``. Needs the
``ANTHROPIC_API_KEY`` environment variable. The adapter keeps the same
``ChatRequest``/``ChatResponse`` contract as every other provider, so judges,
targets and the cache work unchanged. ``ChatRequest.temperature`` is ignored:
current Claude models do not accept sampling parameters, so repeated samples
are independent draws at the API's default sampling.
"""

from __future__ import annotations

import os
import time
from typing import Any

from vantage.models import ChatRequest, ChatResponse, ModelError


class AnthropicClient:
    """Chat with Claude models through the official SDK.

    Args:
        api_key: Overrides ``ANTHROPIC_API_KEY``.
        sdk_client: A pre-built ``anthropic.AsyncAnthropic`` (or a test
            double exposing ``messages.create``). Built lazily when omitted.
        max_retries: SDK-level retries on transient failures.
    """

    def __init__(
        self, *, api_key: str | None = None, sdk_client: Any | None = None, max_retries: int = 2
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._sdk = sdk_client
        self._max_retries = max_retries

    def _client(self) -> Any:
        if self._sdk is None:
            if not self._api_key:
                raise ModelError("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the extra
                raise ModelError(
                    "install the anthropic extra: pip install 'vantage-eval[anthropic]'"
                ) from exc
            self._sdk = anthropic.AsyncAnthropic(
                api_key=self._api_key, max_retries=self._max_retries
            )
        return self._sdk

    @staticmethod
    def _split(request: ChatRequest) -> tuple[str | None, list[dict[str, str]]]:
        system = (
            "\n\n".join(m["content"] for m in request.messages if m["role"] == "system") or None
        )
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in request.messages
            if m["role"] != "system"
        ]
        if request.json_mode and messages:
            messages[-1] = {
                **messages[-1],
                "content": messages[-1]["content"] + "\n\nReply with a single JSON object only.",
            }
        return system, messages

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send one request to the Messages API.

        Raises:
            ModelError: On missing configuration or API failure.
        """
        system, messages = self._split(request)
        # Current Claude models reject sampling parameters, so ``temperature`` is
        # not sent. Thinking is switched off so that ``max_tokens`` goes to the
        # visible answer and an "immediate verdict" judge really is immediate.
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
            "thinking": {"type": "disabled"},
        }
        if system:
            kwargs["system"] = system
        if request.stop:
            kwargs["stop_sequences"] = list(request.stop)
        started = time.perf_counter()
        try:
            response = await self._client().messages.create(**kwargs)
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"anthropic: {type(exc).__name__}: {exc}") from exc
        text = "".join(getattr(block, "text", "") for block in getattr(response, "content", []))
        usage = getattr(response, "usage", None)
        return ChatResponse(
            content=text,
            tokens_in=int(getattr(usage, "input_tokens", 0) or 0),
            tokens_out=int(getattr(usage, "output_tokens", 0) or 0),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            model=getattr(response, "model", request.model),
            raw={"stop_reason": getattr(response, "stop_reason", None)},
        )
