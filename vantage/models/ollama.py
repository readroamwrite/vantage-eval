"""Ollama client: local models over the ``/api/chat`` HTTP endpoint."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from vantage.models import ChatRequest, ChatResponse, ModelError

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaClient:
    """Chat with models served by a local Ollama daemon.

    Args:
        base_url: Daemon address.
        timeout_s: Per-request timeout.
        num_ctx: Context window to request; small values keep 3B models fast.
        keep_alive: How long the daemon keeps the model loaded between calls.
        retries: Extra attempts on connection errors and 5xx responses.
        transport: Optional httpx transport, used by tests to fake the daemon.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_HOST,
        *,
        timeout_s: float = 180.0,
        num_ctx: int = 4096,
        keep_alive: str = "30m",
        retries: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self.retries = retries
        self._http = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_s, transport=transport
        )

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._http.aclose()

    def _payload(self, request: ChatRequest) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
            "num_ctx": self.num_ctx,
        }
        if request.seed is not None:
            options["seed"] = request.seed
        if request.stop:
            options["stop"] = list(request.stop)
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": list(request.messages),
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if request.json_mode:
            payload["format"] = "json"
        return payload

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send one chat request.

        Raises:
            ModelError: If the daemon is unreachable, the model is missing, or
                the call keeps failing after retries.
        """
        payload = self._payload(request)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            try:
                resp = await self._http.post("/api/chat", json=payload)
            except httpx.HTTPError as exc:
                last_error = ModelError(f"{type(exc).__name__}: {exc or 'no detail'}")
                await asyncio.sleep(min(2**attempt, 8))
                continue
            if resp.status_code == 404:
                raise ModelError(
                    f"ollama: model {request.model!r} not found; run `ollama pull {request.model}`"
                )
            if resp.status_code >= 500:
                last_error = ModelError(f"ollama: HTTP {resp.status_code}: {resp.text[:200]}")
                await asyncio.sleep(min(2**attempt, 8))
                continue
            if resp.status_code >= 400:
                raise ModelError(f"ollama: HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            latency_ms = (time.perf_counter() - started) * 1000
            return ChatResponse(
                content=data.get("message", {}).get("content", ""),
                tokens_in=int(data.get("prompt_eval_count", 0) or 0),
                tokens_out=int(data.get("eval_count", 0) or 0),
                latency_ms=round(latency_ms, 1),
                model=data.get("model", request.model),
                raw={k: v for k, v in data.items() if k != "message"},
            )
        raise ModelError(
            f"ollama: request failed after {self.retries + 1} attempts ({last_error}); "
            "the daemon may be overloaded or the request too slow for the timeout"
        )

    async def list_models(self) -> list[dict[str, Any]]:
        """Return the models the daemon has installed (``/api/tags``).

        Raises:
            ModelError: If the daemon is unreachable.
        """
        try:
            resp = await self._http.get("/api/tags")
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelError(f"ollama: cannot reach {self.base_url}: {exc}") from exc
        return list(resp.json().get("models", []))

    async def ping(self) -> bool:
        """Return whether the daemon answers."""
        try:
            await self.list_models()
        except ModelError:
            return False
        return True
