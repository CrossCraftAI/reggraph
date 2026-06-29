"""The Domain plugin abstraction.

A ``Domain`` packages everything domain-specific about a regulation: its source
document, the label for its units (article / section / rule), and where its
knowledge store and benchmark live. Adding a new domain is "define a Domain +
drop in a markdown document" — no core code changes.
"""

from dataclasses import dataclass, field
from pathlib import Path

from ..config import PROJECT_ROOT


@dataclass(frozen=True)
class RequiredCitationRule:
    rule_id: str
    trigger_groups: tuple[tuple[str, ...], ...]
    required_citations: tuple[str, ...]
    passed_message: str
    missing_message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "trigger_groups": self.trigger_groups,
            "required_citations": self.required_citations,
            "passed_message": self.passed_message,
            "missing_message": self.missing_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RequiredCitationRule":
        return cls(
            rule_id=str(data["rule_id"]),
            trigger_groups=_trigger_groups(data["trigger_groups"]),
            required_citations=_strings(data["required_citations"]),
            passed_message=str(data["passed_message"]),
            missing_message=str(data["missing_message"]),
        )


@dataclass(frozen=True)
class DeadlineRule:
    rule_id: str
    trigger_groups: tuple[tuple[str, ...], ...]
    citation: str
    deadline_pattern: str
    passed_message: str
    missing_message: str
    missing_deadline_message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "trigger_groups": self.trigger_groups,
            "citation": self.citation,
            "deadline_pattern": self.deadline_pattern,
            "passed_message": self.passed_message,
            "missing_message": self.missing_message,
            "missing_deadline_message": self.missing_deadline_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DeadlineRule":
        return cls(
            rule_id=str(data["rule_id"]),
            trigger_groups=_trigger_groups(data["trigger_groups"]),
            citation=str(data["citation"]),
            deadline_pattern=str(data["deadline_pattern"]),
            passed_message=str(data["passed_message"]),
            missing_message=str(data["missing_message"]),
            missing_deadline_message=str(data["missing_deadline_message"]),
        )


@dataclass(frozen=True)
class SymbolicRules:
    required_citation_rules: tuple[RequiredCitationRule, ...] = ()
    deadline_rules: tuple[DeadlineRule, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "required_citation_rules": [rule.to_dict() for rule in self.required_citation_rules],
            "deadline_rules": [rule.to_dict() for rule in self.deadline_rules],
        }

    @classmethod
    def from_dict(cls, data: object) -> "SymbolicRules":
        if not isinstance(data, dict):
            return cls()
        required = data.get("required_citation_rules", [])
        deadlines = data.get("deadline_rules", [])
        return cls(
            required_citation_rules=tuple(
                RequiredCitationRule.from_dict(item) for item in required if isinstance(item, dict)
            ),
            deadline_rules=tuple(
                DeadlineRule.from_dict(item) for item in deadlines if isinstance(item, dict)
            ),
        )


def _trigger_groups(value: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(_strings(group) for group in value)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class Domain:
    name: str  # short id, e.g. "gdpr"
    title: str  # human title, e.g. "EU GDPR (excerpt)"
    description: str
    source_path: Path  # the regulatory document (markdown)
    unit_label: str = "article"  # "article" | "section" | "rule" ... (UX + citation example)
    chunk_size: int = 512
    chunk_overlap: int = 64
    symbolic_rules: SymbolicRules = field(default_factory=SymbolicRules)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Domain name must not be empty")

    @property
    def store_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "store" / self.name

    @property
    def chroma_dir(self) -> Path:
        return self.store_dir / "chroma"

    @property
    def graph_path(self) -> Path:
        return self.store_dir / "graph.json"

    @property
    def benchmark_path(self) -> Path:
        return PROJECT_ROOT / "data" / "benchmark" / f"{self.name}.jsonl"
