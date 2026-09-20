"""Model client abstraction.

Every model provider (Ollama, Anthropic, the test mock) implements one method,
``chat``, that takes a ``ChatRequest`` and returns a ``ChatResponse``. Requests
are immutable and hashable by content, which is what makes the response cache
and reproducible reruns possible.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

Message = dict[str, str]


class ModelError(RuntimeError):
    """Raised when a provider call fails after retries."""


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """One chat completion request.

    Attributes:
        model: Provider-specific model name, for example ``"qwen2.5:3b"``.
        messages: ``{"role", "content"}`` dicts in conversation order.
        temperature: Sampling temperature. ``0`` is greedy on most providers.
        max_tokens: Output token cap.
        seed: Sampling seed where the provider supports one.
        json_mode: Ask the provider to constrain output to JSON.
        stop: Stop sequences.
    """

    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 256
    seed: int | None = 0
    json_mode: bool = False
    stop: tuple[str, ...] = ()

    @classmethod
    def create(cls, model: str, messages: Iterable[Message], **kwargs: Any) -> ChatRequest:
        """Build a request from any iterable of messages.

        Args:
            model: Model name.
            messages: Messages; copied into an immutable tuple.
            **kwargs: Other ``ChatRequest`` fields.
        """
        msgs = tuple({"role": m["role"], "content": m["content"]} for m in messages)
        return cls(model=model, messages=msgs, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        data = asdict(self)
        data["messages"] = list(self.messages)
        data["stop"] = list(self.stop)
        return data

    def key(self, sample_idx: int = 0) -> str:
        """Return a content hash that identifies this request in the cache.

        Args:
            sample_idx: Distinguishes repeated samples of the same request, so
                that self-consistency runs get independent draws instead of
                cache hits.
        """
        payload = json.dumps(
            {"request": self.to_dict(), "sample_idx": sample_idx},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ChatResponse:
    """One chat completion result.

    Attributes:
        content: The model's text.
        tokens_in: Prompt tokens, when reported.
        tokens_out: Completion tokens, when reported.
        latency_ms: Wall-clock time of the provider call. ``0`` for cache hits.
        cached: Whether the response came from the cache.
        model: Model that produced it.
        raw: Provider-specific payload kept for debugging.
    """

    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResponse:
        """Deserialize from ``to_dict`` output."""
        return cls(**data)

    def step_meta(self) -> dict[str, Any]:
        """Metadata to attach to the assistant ``Step`` this response produced."""
        return {
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "model": self.model,
        }


@runtime_checkable
class ModelClient(Protocol):
    """Anything that can answer a ``ChatRequest``."""

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Return the model's response to ``request``.

        Raises:
            ModelError: If the provider fails.
        """
        ...
