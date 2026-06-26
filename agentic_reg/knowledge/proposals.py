"""Reviewable graph-update proposals.

Agents may suggest nodes or edges, but the graph only changes after deterministic
validation and an explicit apply step. The default runtime path writes reviewed
proposals to JSONL so a maintainer can inspect them before accepting changes.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import PROJECT_ROOT, Settings


class ProposalGraph(Protocol):
    def has_node(self, node_id: str) -> bool:
        """Return whether ``node_id`` exists."""

    def has_edge(self, source_id: str, target_id: str, relation: str | None = None) -> bool:
        """Return whether an edge exists, optionally matching relation."""

    def add_node(self, node_id: str, *, label: str, kind: str) -> None:
        """Add a graph node."""

    def add_edge(self, source_id: str, target_id: str, relation: str) -> None:
        """Add a graph edge."""


ALLOWED_NODE_KINDS = {
    "clause",
    "concept",
    "obligation",
    "definition",
    "right",
    "condition",
    "principle",
    "prohibition",
    "temporal_constraint",
}
ALLOWED_RELATIONS = {
    "references",
    "requires",
    "overrides",
    "depends_on",
    "exception_to",
    "applies_to",
    "implies",
    "conflicts_with",
    "temporal_constraint",
    "imposes",
    "defines",
    "grants",
    "sets_condition",
    "establishes",
    "prohibits",
    "sets_deadline",
    "introduces",
}
STATUSES = {"pending", "accepted", "rejected", "applied"}


@dataclass
class GraphUpdateProposal:
    action: str  # node | edge
    evidence: str
    citations: list[str] = field(default_factory=list)
    node_id: str = ""
    label: str = ""
    kind: str = ""
    source_id: str = ""
    target_id: str = ""
    relation: str = ""
    status: str = "pending"
    reason: str = ""
    proposal_id: str = ""

    def __post_init__(self) -> None:
        self.action = self.action.strip().lower()
        self.kind = self.kind.strip().lower()
        self.relation = self.relation.strip().lower()
        self.node_id = self.node_id.strip().lower()
        self.source_id = self.source_id.strip().lower()
        self.target_id = self.target_id.strip().lower()
        self.citations = [
            citation.strip().lower() for citation in self.citations if citation.strip()
        ]
        if self.status not in STATUSES:
            self.status = "pending"
        if not self.proposal_id:
            self.proposal_id = _stable_id(self)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stable_id(proposal: GraphUpdateProposal) -> str:
    key = "|".join(
        [
            proposal.action,
            proposal.node_id,
            proposal.source_id,
            proposal.target_id,
            proposal.relation,
            proposal.label,
            proposal.evidence[:120],
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def proposal_store_path(settings: Settings) -> Path:
    if settings.graph_proposals_path:
        return Path(settings.graph_proposals_path)
    return PROJECT_ROOT / "data" / "store" / settings.domain / "graph_proposals.jsonl"


def proposal_from_dict(data: dict[str, Any]) -> GraphUpdateProposal | None:
    action = str(data.get("action", "")).strip().lower()
    if action not in {"node", "edge"}:
        return None
    citations = data.get("citations") or []
    if not isinstance(citations, list):
        citations = []
    return GraphUpdateProposal(
        action=action,
        node_id=str(data.get("node_id", "")).strip(),
        label=str(data.get("label", "")).strip(),
        kind=str(data.get("kind", "")).strip(),
        source_id=str(data.get("source_id", "")).strip(),
        target_id=str(data.get("target_id", "")).strip(),
        relation=str(data.get("relation", "")).strip(),
        evidence=str(data.get("evidence", "")).strip(),
        citations=[str(item) for item in citations],
    )


def validate_proposal(proposal: GraphUpdateProposal, graph: ProposalGraph) -> GraphUpdateProposal:
    problems: list[str] = []
    if not proposal.evidence:
        problems.append("evidence is required")

    missing_citations = [
        citation for citation in proposal.citations if not graph.has_node(citation)
    ]
    if missing_citations:
        problems.append("unknown citation(s): " + ", ".join(missing_citations))

    if proposal.action == "node":
        if not proposal.node_id:
            problems.append("node_id is required")
        elif graph.has_node(proposal.node_id):
            problems.append(f"node already exists: {proposal.node_id}")
        if not proposal.label:
            problems.append("label is required")
        if proposal.kind not in ALLOWED_NODE_KINDS:
            problems.append(f"unsupported node kind: {proposal.kind or '(empty)'}")

    if proposal.action == "edge":
        if not graph.has_node(proposal.source_id):
            problems.append(f"unknown source node: {proposal.source_id or '(empty)'}")
        if not graph.has_node(proposal.target_id):
            problems.append(f"unknown target node: {proposal.target_id or '(empty)'}")
        if proposal.relation not in ALLOWED_RELATIONS:
            problems.append(f"unsupported relation: {proposal.relation or '(empty)'}")
        if graph.has_edge(proposal.source_id, proposal.target_id, proposal.relation):
            problems.append("edge already exists")

    proposal.status = "rejected" if problems else "accepted"
    proposal.reason = "; ".join(problems) if problems else "accepted"
    return proposal


def apply_proposal(proposal: GraphUpdateProposal, graph: ProposalGraph) -> bool:
    validate_proposal(proposal, graph)
    if proposal.status != "accepted":
        return False
    if proposal.action == "node":
        graph.add_node(proposal.node_id, label=proposal.label, kind=proposal.kind)
    else:
        graph.add_edge(proposal.source_id, proposal.target_id, proposal.relation)
    proposal.status = "applied"
    proposal.reason = "applied"
    return True


def write_proposals(path: str | Path, proposals: list[GraphUpdateProposal]) -> None:
    if not proposals:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for proposal in proposals:
            handle.write(json.dumps(proposal.to_dict(), ensure_ascii=False) + "\n")


def load_proposals(path: str | Path) -> list[GraphUpdateProposal]:
    path = Path(path)
    if not path.exists():
        return []
    proposals: list[GraphUpdateProposal] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        proposal = proposal_from_dict(data)
        if proposal is None:
            continue
        proposal.status = str(data.get("status", proposal.status))
        proposal.reason = str(data.get("reason", proposal.reason))
        proposal.proposal_id = str(data.get("proposal_id", proposal.proposal_id))
        proposals.append(proposal)
    return proposals
