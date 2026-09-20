"""Tag taxonomy: capabilities, threat models and where their evidence lives.

Cases carry ``prefix:value`` tags. Two prefixes have special meaning:

* ``cap:`` names the capability an item exercises.
* ``threat:`` names the threat-model node an item probes. Each threat tag
  declares where evidence of the behaviour shows up: in the final output or
  only in the trajectory. Combined with each monitor's declared ``view`` this
  lets ``coverage`` report which threats are structurally undetectable by a
  monitoring setup, before any run happens.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vantage.results import View


@dataclass(frozen=True, slots=True)
class Threat:
    """A threat-model node.

    Attributes:
        tag: Full tag, for example ``"threat:grader_tampering"``.
        evidence: Where the behaviour is observable.
        description: One-line explanation.
    """

    tag: str
    evidence: View
    description: str


_THREATS: dict[str, Threat] = {}


def register_threat(tag: str, evidence: View, description: str) -> Threat:
    """Add or replace a threat node in the taxonomy."""
    if not tag.startswith("threat:"):
        raise ValueError("threat tags must start with 'threat:'")
    threat = Threat(tag, evidence, description)
    _THREATS[tag] = threat
    return threat


for _tag, _evidence, _desc in [
    ("threat:none", "output", "Benign control item; no misbehaviour expected."),
    (
        "threat:grader_tampering",
        "trajectory",
        "Agent edits tests, graders or fixtures instead of solving the task.",
    ),
    (
        "threat:grader_tampering_pressure",
        "trajectory",
        "Task where the grader is wrong, so tampering is tempting.",
    ),
    ("threat:fake_success", "output", "Agent claims success it did not achieve or verify."),
    (
        "threat:forbidden_tool_use",
        "trajectory",
        "Agent calls a tool or path it was told not to touch.",
    ),
    (
        "threat:sycophancy",
        "output",
        "Model agrees with a user's wrong claim instead of correcting it.",
    ),
    ("threat:sandbagging", "output", "Model strategically underperforms on an evaluation."),
    ("threat:data_exfiltration", "trajectory", "Agent moves data somewhere it should not."),
    (
        "threat:unsafe_autonomous_action",
        "trajectory",
        "Agent takes a consequential action without the required check.",
    ),
]:
    register_threat(_tag, _evidence, _desc)  # type: ignore[arg-type]


def threats() -> dict[str, Threat]:
    """The registered threat nodes, keyed by tag."""
    return dict(_THREATS)


def evidence_for(tag: str) -> View | None:
    """Where evidence for ``tag`` lives, or ``None`` for unknown tags."""
    threat = _THREATS.get(tag)
    return threat.evidence if threat else None


def split_tag(tag: str) -> tuple[str, str]:
    """Split ``prefix:value``; tags without a colon get prefix ``"tag"``."""
    prefix, sep, value = tag.partition(":")
    return (prefix, value) if sep else ("tag", tag)


def tags_by_prefix(tags: Iterable[str]) -> dict[str, list[str]]:
    """Group tags by prefix, keeping each list sorted and unique."""
    groups: dict[str, set[str]] = {}
    for tag in tags:
        groups.setdefault(split_tag(tag)[0], set()).add(tag)
    return {prefix: sorted(values) for prefix, values in sorted(groups.items())}


def structural_coverage(
    monitors: Sequence[Any], present_threats: Iterable[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Which threats can the given monitors see at all?

    A monitor with ``view="output"`` cannot observe trajectory-only evidence,
    whatever its rubric says. This check is independent of any run.

    Args:
        monitors: Objects with ``name`` and ``view`` attributes.
        present_threats: Restrict to these threat tags; defaults to the whole taxonomy.

    Returns:
        Per threat tag: ``evidence``, ``covered_by`` (monitor names whose view
        can see the evidence) and ``structurally_detectable``.
    """
    wanted = list(present_threats) if present_threats is not None else list(_THREATS)
    out: dict[str, dict[str, Any]] = {}
    for tag in wanted:
        threat = _THREATS.get(tag)
        if threat is None:
            continue
        able = [m.name for m in monitors if m.view == "trajectory" or threat.evidence == "output"]
        out[tag] = {
            "evidence": threat.evidence,
            "description": threat.description,
            "covered_by": able,
            "structurally_detectable": bool(able),
        }
    return out


def taxonomy_gaps(present_tags: Iterable[str]) -> list[str]:
    """Threat tags in the taxonomy with no cases carrying them."""
    present = set(present_tags)
    return [tag for tag in _THREATS if tag not in present and tag != "threat:none"]
