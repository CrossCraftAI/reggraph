"""State and data types for the multi-agent team."""

from dataclasses import dataclass, field
from typing import TypedDict

from ..trace import ReasoningTrace


@dataclass
class SubQuestion:
    text: str
    role: str


@dataclass
class AgentTask:
    id: str
    text: str
    role: str
    parent_id: str | None
    depth: int
    children: list[str] = field(default_factory=list)
    finding: str = ""
    citations: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    graph_node_ids: list[str] = field(default_factory=list)
    status: str = "pending"


@dataclass
class Finding:
    role: str
    sub_question: str
    text: str
    citations: list[str]
    retrieved_ids: list[str]
    graph_node_ids: list[str] = field(default_factory=list)
    graph_edges: list[dict[str, object]] = field(default_factory=list)
    multi_hop_paths: list[dict[str, object]] = field(default_factory=list)


@dataclass
class Verdict:
    ok: bool
    invalid_citations: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    symbolic_findings: list[dict[str, object]] = field(default_factory=list)
    llm_note: str = ""

    def issues(self) -> list[str]:
        items: list[str] = []
        items += [f"Invalid citation: [{citation}]" for citation in self.invalid_citations]
        items += [f"Unsupported claim: {claim}" for claim in self.unsupported_claims]
        items += [f"Contradiction: {claim}" for claim in self.contradictions]
        items += [
            f"Symbolic check failed ({item.get('rule_id', 'rule')}): {item.get('message', '')}"
            for item in self.symbolic_findings
            if item.get("passed") is False
        ]
        return items


class TeamState(TypedDict):
    question: str
    plan: list[SubQuestion]
    tasks: list[AgentTask]
    findings: list[Finding]
    graph_proposals: list[dict[str, object]]
    draft: str
    verdict: Verdict | None
    iteration: int
    trace: ReasoningTrace
