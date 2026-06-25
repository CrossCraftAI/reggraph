"""Small shared types for the basic Phase 0 team orchestrator."""

import re
from dataclasses import dataclass, field

_CITATION_RE = re.compile(r"\[([a-z][a-z-]*-\d+)\]", flags=re.IGNORECASE)


def extract_citations(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _CITATION_RE.findall(text):
        seen.setdefault(match.lower(), None)
    return list(seen)


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
