"""Small deterministic checks for high-confidence regulatory rules."""

import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .._internal import extract_citations


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


def _has_any(text: str, words: set[str]) -> bool:
    haystack = text.lower()
    return any(word in haystack for word in words)


def _available_required(graph: GraphLike, candidates: list[str]) -> list[str]:
    return [node_id for node_id in candidates if graph.has_node(node_id)]


def _missing(required: list[str], cited: set[str]) -> list[str]:
    return [node_id for node_id in required if node_id not in cited]


def _deadline_clause(graph: GraphLike) -> str | None:
    for node_id in ("article-33", "section-67"):
        if graph.has_node(node_id):
            return node_id
    return None


def run_symbolic_checks(question: str, answer: str, graph: GraphLike) -> list[SymbolicFinding]:
    text = f"{question}\n{answer}"
    citations = extract_citations(answer)
    cited = set(citations)
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

    if _has_any(text, {"special category", "special categories", "health", "biometric"}):
        required = _available_required(graph, ["article-9", "section-10"]) + _available_required(
            graph, ["article-6", "section-8"]
        )
        if required:
            missing = _missing(required, cited)
            findings.append(
                SymbolicFinding(
                    "special_category_requires_basis",
                    not missing,
                    "Special-category processing cites both the specific condition "
                    "and lawful basis."
                    if not missing
                    else "Special-category answer should cite: " + ", ".join(missing),
                    required,
                )
            )

    if _has_any(text, {"withdraw", "withdraws", "withdrawal"}) and _has_any(
        text, {"erase", "erasure", "deleted"}
    ):
        required = _available_required(graph, ["article-17", "article-7", "article-6"])
        if required:
            missing = _missing(required, cited)
            findings.append(
                SymbolicFinding(
                    "withdrawal_erasure_chain",
                    not missing,
                    "Withdrawal/erasure answer cites erasure, consent, and lawful-basis clauses."
                    if not missing
                    else "Withdrawal/erasure answer should cite: " + ", ".join(missing),
                    required,
                )
            )

    if _has_any(text, {"breach", "notify", "notification"}):
        clause = _deadline_clause(graph)
        if clause:
            has_deadline = bool(re.search(r"\b72\s+hours?\b", answer, flags=re.IGNORECASE))
            passed = has_deadline and clause in cited
            detail = []
            if not has_deadline:
                detail.append("mention the 72 hour deadline")
            if clause not in cited:
                detail.append(f"cite {clause}")
            findings.append(
                SymbolicFinding(
                    "breach_notification_deadline",
                    passed,
                    "Breach notification answer includes the deadline and source clause."
                    if passed
                    else "Breach notification answer should " + " and ".join(detail) + ".",
                    [clause],
                )
            )

    return findings
