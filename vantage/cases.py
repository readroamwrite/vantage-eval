"""Test cases and datasets.

A ``Case`` is one prompt (or conversation) with an optional expected answer
and tags. A ``Dataset`` is a named list of cases loaded from JSON, JSONL or CSV.
Tags are ``prefix:value`` strings such as ``tier:3`` or
``threat:grader_tampering`` and drive filtering, coverage reports and
diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

Message = dict[str, str]


@dataclass(slots=True)
class Case:
    """One evaluation item.

    Attributes:
        id: Unique id within its dataset.
        input: A prompt string, or a list of ``{"role", "content"}`` messages
            for multi-turn inputs.
        expected: Reference answer or grading target. Scorer-specific.
        tags: ``prefix:value`` labels used for filtering and coverage.
        system: Optional per-case system prompt.
        meta: Free-form data such as ``base_id``, ``files`` or ``tier``.
    """

    id: str
    input: str | list[Message]
    expected: Any = None
    tags: list[str] = field(default_factory=list)
    system: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def has_tag(self, tag: str) -> bool:
        """Return whether the case carries ``tag`` exactly."""
        return tag in self.tags

    def tag_value(self, prefix: str) -> str | None:
        """Return the value of the first ``prefix:value`` tag, or ``None``.

        Args:
            prefix: The part before the colon, for example ``"tier"``.
        """
        needle = f"{prefix}:"
        for tag in self.tags:
            if tag.startswith(needle):
                return tag[len(needle) :]
        return None

    def messages(self) -> list[Message]:
        """Return the chat messages for this case, including the system prompt.

        Returns:
            A list of ``{"role", "content"}`` dicts ready to send to a model.
        """
        msgs: list[Message] = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        if isinstance(self.input, str):
            msgs.append({"role": "user", "content": self.input})
        else:
            msgs.extend({"role": m["role"], "content": m["content"]} for m in self.input)
        return msgs

    @property
    def prompt_text(self) -> str:
        """The input as a single string, for display and rule scorers."""
        if isinstance(self.input, str):
            return self.input
        return "\n".join(f"{m['role']}: {m['content']}" for m in self.input)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain JSON-compatible data."""
        return {
            "id": self.id,
            "input": self.input,
            "expected": self.expected,
            "tags": list(self.tags),
            "system": self.system,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Case:
        """Build a case from a mapping with at least ``id`` and ``input``.

        Raises:
            KeyError: If ``id`` or ``input`` is missing.
        """
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = _split_tags(tags)
        return cls(
            id=str(data["id"]),
            input=data["input"],
            expected=data.get("expected"),
            tags=list(tags),
            system=data.get("system"),
            meta=dict(data.get("meta", {})),
        )


@dataclass(slots=True)
class Dataset:
    """A named, ordered collection of test cases.

    Attributes:
        name: Human-readable name, usually the file stem.
        cases: The cases, in order. Ids must be unique.
    """

    name: str
    cases: list[Case] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r} in dataset {self.name!r}")
            seen.add(case.id)

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    @property
    def ids(self) -> list[str]:
        """Case ids in order."""
        return [c.id for c in self.cases]

    def get(self, case_id: str) -> Case:
        """Return the case with ``case_id``.

        Raises:
            KeyError: If no case has that id.
        """
        for case in self.cases:
            if case.id == case_id:
                return case
        raise KeyError(case_id)

    @classmethod
    def from_dicts(cls, name: str, rows: Iterable[dict[str, Any]]) -> Dataset:
        """Build a dataset from mappings accepted by ``Case.from_dict``."""
        return cls(name=name, cases=[Case.from_dict(r) for r in rows])

    @classmethod
    def load(cls, path: str | Path, name: str | None = None) -> Dataset:
        """Load a dataset from a ``.json``, ``.jsonl`` or ``.csv`` file.

        JSON files hold either a list of cases or ``{"name": ..., "cases": [...]}``.
        JSONL files hold one case per line. CSV files need ``id`` and ``input``
        columns; ``expected``, ``tags`` (``;`` or ``,`` separated) and ``system``
        are optional, and ``expected`` is parsed as JSON when it is valid JSON.

        Args:
            path: File to read.
            name: Dataset name; defaults to the file stem.

        Returns:
            The loaded dataset.

        Raises:
            ValueError: If the extension is not supported or the content is malformed.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            rows, inner_name = _read_json(path)
        elif suffix == ".jsonl":
            rows, inner_name = _read_jsonl(path), None
        elif suffix == ".csv":
            rows, inner_name = _read_csv(path), None
        else:
            raise ValueError(f"unsupported dataset format {suffix!r} for {path}")
        return cls.from_dicts(name or inner_name or path.stem, rows)

    def save(self, path: str | Path) -> None:
        """Write the dataset as JSONL, one case per line."""
        path = Path(path)
        with path.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    def filter(self, *tags: str) -> Dataset:
        """Return the cases carrying every one of ``tags``.

        Args:
            *tags: Exact tags that must all be present.

        Returns:
            A new dataset named ``"<name>[tag1,tag2]"``.
        """
        wanted = set(tags)
        cases = [c for c in self.cases if wanted.issubset(c.tags)]
        return Dataset(name=f"{self.name}[{','.join(tags)}]", cases=cases)

    def with_system(self, system: str, suffix: str) -> Dataset:
        """Return a copy where every case uses ``system`` as its system prompt.

        Used to derive experimental conditions, for example a sandbagging
        prompt, from one shared set of items. Each copied case records the
        condition in ``meta["condition"]``.

        Args:
            system: The system prompt to apply.
            suffix: Condition name appended to the dataset name.
        """
        cases = [
            Case(
                id=c.id,
                input=c.input,
                expected=c.expected,
                tags=list(c.tags),
                system=system,
                meta={**c.meta, "condition": suffix},
            )
            for c in self.cases
        ]
        return Dataset(name=f"{self.name}:{suffix}", cases=cases)

    def hash(self) -> str:
        """Return a short content hash that changes when any case changes."""
        payload = json.dumps([c.to_dict() for c in self.cases], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _split_tags(text: str) -> list[str]:
    sep = ";" if ";" in text else ","
    return [t.strip() for t in text.split(sep) if t.strip()]


def _read_json(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return data["cases"], data.get("name")
    raise ValueError(f"{path}: expected a JSON list of cases or an object with a 'cases' list")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not {"id", "input"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: CSV needs 'id' and 'input' columns")
        for raw in reader:
            row: dict[str, Any] = {"id": raw["id"], "input": raw["input"]}
            expected = raw.get("expected")
            if expected not in (None, ""):
                row["expected"] = _parse_scalar(expected)
            if raw.get("tags"):
                row["tags"] = _split_tags(raw["tags"])
            if raw.get("system"):
                row["system"] = raw["system"]
            rows.append(row)
    return rows


def _parse_scalar(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text
