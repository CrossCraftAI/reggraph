"""Small deterministic checks for high-confidence regulatory rules."""

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .._internal import extract_citations
from ..domains import SymbolicRules


class GraphLike(Protocol):
    def has_node(self, node_id: str) -> bool:
        """Return whether ``node_id`` exists in the regulatory graph."""


@dataclass
class SymbolicFinding:
    rule_id: str
    passed: bool
    message: str
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _has_any(text: str, words: tuple[str, ...] | set[str]) -> bool:
    haystack = text.lower()
    return any(word in haystack for word in words)


def _matches(text: str, trigger_groups: tuple[tuple[str, ...], ...]) -> bool:
    return all(_has_any(text, group) for group in trigger_groups)


def _available_required(graph: GraphLike, candidates: tuple[str, ...]) -> list[str]:
    return [node_id for node_id in candidates if graph.has_node(node_id)]


def _missing(required: list[str], cited: set[str]) -> list[str]:
    return [node_id for node_id in required if node_id not in cited]


def run_symbolic_checks(question: str, answer: str, graph: GraphLike) -> list[SymbolicFinding]:
    text = f"{question}\n{answer}"
    citations = extract_citations(answer)
    cited = set(citations)
    rules = getattr(graph, "symbolic_rules", None) or SymbolicRules()
    findings: list[SymbolicFinding] = []

    invalid = [citation for citation in citations if not graph.has_node(citation)]
    findings.append(
        SymbolicFinding(
            "citation_validity",
            not invalid,
            "All citations resolve to graph nodes."
            if not invalid
            else "Unknown citation(s): " + ", ".join(invalid),
            citations,
        )
    )

    for rule in rules.required_citation_rules:
        if not _matches(text, rule.trigger_groups):
            continue
        required = _available_required(graph, rule.required_citations)
        if not required:
            continue
        missing = _missing(required, cited)
        findings.append(
            SymbolicFinding(
                rule.rule_id,
                not missing,
                rule.passed_message
                if not missing
                else rule.missing_message + ": " + ", ".join(missing),
                required,
            )
        )

    for rule in rules.deadline_rules:
        if not _matches(text, rule.trigger_groups) or not graph.has_node(rule.citation):
            continue
        has_deadline = bool(re.search(rule.deadline_pattern, answer, flags=re.IGNORECASE))
        passed = has_deadline and rule.citation in cited
        if passed:
            message = rule.passed_message
        else:
            detail = []
            if not has_deadline:
                detail.append(rule.missing_deadline_message)
            if rule.citation not in cited:
                detail.append(f"cite {rule.citation}")
            message = rule.missing_message + " " + " and ".join(detail) + "."
        findings.append(
            SymbolicFinding(
                rule.rule_id,
                passed,
                message,
                [rule.citation],
            )
        )

    return findings
