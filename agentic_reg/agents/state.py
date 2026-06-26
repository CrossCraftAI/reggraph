"""State and data types for the multi-agent team."""

import re
from dataclasses import dataclass, field
from typing import TypedDict

from ..trace import ReasoningTrace

_CITATION_RE = re.compile(r"\[([a-z][a-z-]*-\d+)\]", flags=re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _CITATION_RE.findall(text):
        seen.setdefault(match.lower(), None)
    return list(seen)


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


class TeamState(TypedDict):
    question: str
    plan: list[SubQuestion]
    tasks: list[AgentTask]
    findings: list[Finding]
    draft: str
    trace: ReasoningTrace
