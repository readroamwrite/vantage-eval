"""Scorers: functions from a trajectory (and its case) to a ``Score``.

Scorers are registered by name so that the CLI and config files can refer to
them as short specs such as ``"exact"``, ``"numeric:0.01"`` or
``"contains:needle,case_sensitive=true"``. The spec grammar is
``name[:arg[,arg|key=value]...]``; positional args and keyword args are passed
to the scorer's constructor, with values parsed as JSON when they are valid
JSON and kept as strings otherwise.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from vantage.cases import Case
from vantage.results import Score
from vantage.trajectory import Trajectory

__all__ = ["Score", "Scorer", "get_scorer", "list_scorers", "parse_spec", "register"]


@runtime_checkable
class Scorer(Protocol):
    """Anything that can score a trajectory."""

    name: str

    async def score(self, traj: Trajectory, case: Case) -> Score:
        """Score ``traj`` against ``case``."""
        ...


ScorerFactory = Callable[..., Scorer]
_REGISTRY: dict[str, ScorerFactory] = {}
_BUILTINS_LOADED = False


def register(name: str) -> Callable[[ScorerFactory], ScorerFactory]:
    """Register a scorer class or factory under ``name``.

    Args:
        name: The spec name users will write, for example ``"exact"``.

    Returns:
        A decorator that records the factory and returns it unchanged.
    """

    def decorator(factory: ScorerFactory) -> ScorerFactory:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def _load_builtins() -> None:
    global _BUILTINS_LOADED
    if not _BUILTINS_LOADED:
        _BUILTINS_LOADED = True
        from vantage.scorers import (  # noqa: F401  (registers on import)
            behaviour,
            failure,
            judge,
            rules,
        )


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_spec(spec: str) -> tuple[str, list[Any], dict[str, Any]]:
    """Split a scorer spec into name, positional args and keyword args.

    Args:
        spec: For example ``"numeric:0.01,relative=true"``.

    Returns:
        ``(name, args, kwargs)``.

    Raises:
        ValueError: If the spec is empty.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty scorer spec")
    name, _, rest = spec.partition(":")
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    if rest:
        for part in rest.split(","):
            part = part.strip()
            if not part:
                continue
            key, eq, value = part.partition("=")
            if eq and key.isidentifier():
                kwargs[key] = _parse_value(value)
            else:
                args.append(_parse_value(part))
    return name.strip(), args, kwargs


def get_scorer(spec: str, **overrides: Any) -> Scorer:
    """Build a scorer from a spec string.

    Args:
        spec: ``name[:args]`` as described in the module docstring.
        **overrides: Keyword arguments that take precedence over the spec.
            Overrides the factory cannot accept (for example ``store`` passed
            to a rule scorer) are dropped silently, so callers can hand every
            scorer the same context.

    Returns:
        A ready-to-use scorer.

    Raises:
        KeyError: If the name is not registered.
    """
    _load_builtins()
    name, args, kwargs = parse_spec(spec)
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"unknown scorer {name!r}; known: {sorted(_REGISTRY)}") from exc
    return factory(*args, **{**kwargs, **_accepted(factory, overrides)})


def _accepted(factory: ScorerFactory, overrides: dict[str, Any]) -> dict[str, Any]:
    """Keep only the overrides the factory can take (context such as ``store``)."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return overrides
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return overrides
    return {k: v for k, v in overrides.items() if k in params}


def list_scorers() -> list[str]:
    """Return registered scorer names, sorted."""
    _load_builtins()
    return sorted(_REGISTRY)
