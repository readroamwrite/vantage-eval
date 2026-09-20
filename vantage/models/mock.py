"""Deterministic model client for tests and offline demos."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from vantage.models import ChatRequest, ChatResponse

Responder = Callable[[ChatRequest], str]


class MockClient:
    """A model that answers from rules instead of weights.

    Resolution order for each request: the ``responder`` callable if given,
    then the first ``rules`` entry whose key is a substring of the last user
    message, then ``default``.

    Args:
        responder: Function from request to response text.
        rules: Substring-of-last-user-message to response text.
        default: Fallback response.
        latency_ms: Simulated latency recorded on responses (no real sleep).
        fail_with: If set, every call raises this exception instead.
    """

    def __init__(
        self,
        responder: Responder | None = None,
        *,
        rules: dict[str, str] | None = None,
        default: str = "mock response",
        latency_ms: float = 1.0,
        fail_with: Exception | None = None,
    ) -> None:
        self._responder = responder
        self._rules = dict(rules or {})
        self._default = default
        self._latency_ms = latency_ms
        self._fail_with = fail_with
        self.calls: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Answer ``request`` deterministically and record it in ``calls``."""
        self.calls.append(request)
        if self._fail_with is not None:
            raise self._fail_with
        await asyncio.sleep(0)
        content = self._resolve(request)
        return ChatResponse(
            content=content,
            tokens_in=sum(len(m["content"].split()) for m in request.messages),
            tokens_out=len(content.split()),
            latency_ms=self._latency_ms,
            model=request.model,
        )

    def _resolve(self, request: ChatRequest) -> str:
        if self._responder is not None:
            return self._responder(request)
        last_user = next(
            (m["content"] for m in reversed(request.messages) if m["role"] == "user"), ""
        )
        for needle, answer in self._rules.items():
            if needle in last_user:
                return answer
        return self._default
