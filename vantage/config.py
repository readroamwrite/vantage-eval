"""Configuration and factories for clients and targets.

Model specs are ``provider:model`` strings such as ``"ollama:qwen2.5:3b"`` or
``"anthropic:claude-sonnet-5"``. The special provider ``mock`` needs no
network and is used by tests, demos and smoke checks.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vantage.models import ChatRequest, ModelClient
from vantage.models.cache import CachedClient
from vantage.models.mock import MockClient
from vantage.store import Store
from vantage.targets import LLMTarget, Target

DEFAULT_DB = Path(os.environ.get("VANTAGE_DB", "vantage.db"))

ClientFactory = Callable[[str], ModelClient]
_PROVIDERS: dict[str, ClientFactory] = {}


def register_provider(name: str) -> Callable[[ClientFactory], ClientFactory]:
    """Register a factory that builds a client for ``provider:model`` specs."""

    def decorator(factory: ClientFactory) -> ClientFactory:
        _PROVIDERS[name] = factory
        return factory

    return decorator


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Split ``"provider:model"`` into its parts.

    Args:
        spec: For example ``"ollama:qwen2.5:3b"``; only the first colon splits.

    Returns:
        ``(provider, model)``. A bare provider gives an empty model.
    """
    provider, _, model = spec.strip().partition(":")
    return provider.lower(), model


_ARITH_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_EXPR = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-+*/%]\s*-?\d+(?:\.\d+)?)+")


def _safe_eval(expr: str) -> float:
    node = ast.parse(expr, mode="eval").body

    def walk(n: ast.AST) -> float:
        if isinstance(n, ast.Constant) and isinstance(n.value, int | float):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -walk(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _ARITH_OPS:
            return _ARITH_OPS[type(n.op)](walk(n.left), walk(n.right))
        raise ValueError("unsupported expression")

    return walk(node)


def arithmetic_mock(request: ChatRequest) -> str:
    """Answer bare arithmetic expressions found in the prompt; otherwise decline.

    This gives the mock provider believable, partly-wrong behaviour on the
    smoke dataset: expressions are answered, word problems are not.
    """
    prompt = next((m["content"] for m in reversed(request.messages) if m["role"] == "user"), "")
    match = _EXPR.search(prompt.replace("x", "*").replace("\u00d7", "*"))
    if match is None:
        return "I'm not sure how to answer that."
    try:
        value = _safe_eval(match.group(0))
    except (ValueError, ZeroDivisionError, SyntaxError):
        return "I'm not sure how to answer that."
    return f"The answer is {value:g}."


@register_provider("ollama")
def _ollama_client(model: str) -> ModelClient:
    from vantage.models.ollama import OllamaClient

    if not model:
        raise ValueError("ollama spec needs a model, e.g. 'ollama:qwen2.5:3b'")
    return OllamaClient()


@register_provider("anthropic")
def _anthropic_client(model: str) -> ModelClient:
    from vantage.models.anthropic import AnthropicClient

    if not model:
        raise ValueError("anthropic spec needs a model, e.g. 'anthropic:claude-sonnet-5'")
    return AnthropicClient()


@register_provider("mock")
def _mock_client(model: str) -> ModelClient:
    if model in ("", "arith"):
        return MockClient(arithmetic_mock)
    if model == "echo":
        return MockClient(lambda r: r.messages[-1]["content"])
    raise ValueError(f"unknown mock model {model!r}; use 'mock', 'mock:arith' or 'mock:echo'")


def make_client(spec: str, store: Store | None = None, *, cached: bool = True) -> ModelClient:
    """Build a model client from a ``provider:model`` spec.

    Args:
        spec: Provider and model.
        store: Needed for caching; when omitted no cache is used.
        cached: Wrap the client in a response cache.

    Raises:
        KeyError: If the provider is unknown.
    """
    provider, model = parse_model_spec(spec)
    try:
        factory = _PROVIDERS[provider]
    except KeyError as exc:
        raise KeyError(f"unknown provider {provider!r}; known: {sorted(_PROVIDERS)}") from exc
    client = factory(model)
    if cached and store is not None:
        return CachedClient(client, store)
    return client


def make_target(
    spec: str,
    store: Store | None = None,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
    cached: bool = True,
    **kwargs: Any,
) -> Target:
    """Build a single-call LLM target from a ``provider:model`` spec."""
    _, model = parse_model_spec(spec)
    client = make_client(spec, store, cached=cached)
    return LLMTarget(
        client,
        model or spec,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        target_id=spec,
        **kwargs,
    )


def read_text_or_path(value: str | None) -> str | None:
    """Return the contents of ``value`` if it names a file, else ``value`` itself."""
    if value is None:
        return None
    path = Path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return value
