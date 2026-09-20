"""Response cache wrapper around any model client."""

from __future__ import annotations

from vantage.models import ChatRequest, ChatResponse, ModelClient
from vantage.store import Store


class CachedClient:
    """Serve repeated requests from the store instead of the provider.

    Args:
        inner: The real client.
        store: Where responses are cached.
        sample_idx: Included in the cache key so that repeated sampling of the
            same prompt (for self-consistency studies) gets fresh draws.
    """

    def __init__(self, inner: ModelClient, store: Store, *, sample_idx: int = 0) -> None:
        self.inner = inner
        self.store = store
        self.sample_idx = sample_idx
        self.hits = 0
        self.misses = 0

    def with_sample(self, sample_idx: int) -> CachedClient:
        """Return a sibling client that keys the cache with ``sample_idx``."""
        return CachedClient(self.inner, self.store, sample_idx=sample_idx)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Return the cached response for ``request`` or fetch and cache it."""
        key = request.key(self.sample_idx)
        hit = self.store.cache_get(key)
        if hit is not None:
            self.hits += 1
            response = ChatResponse.from_dict(hit)
            response.cached = True
            response.latency_ms = 0.0
            return response
        self.misses += 1
        response = await self.inner.chat(request)
        self.store.cache_put(key, request.model, request.to_dict(), response.to_dict())
        return response
