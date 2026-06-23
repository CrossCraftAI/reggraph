from agentic_reg.knowledge.proposals import (
    GraphUpdateProposal,
    apply_proposal,
    load_proposals,
    proposal_from_dict,
    validate_proposal,
    write_proposals,
)


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, str]] = {}
        self.edges: set[tuple[str, str, str]] = set()

    def add_node(self, node_id: str, *, label: str, kind: str) -> None:
        self.nodes[node_id] = {"label": label, "kind": kind}

    def add_edge(self, source_id: str, target_id: str, relation: str) -> None:
        self.edges.add((source_id, target_id, relation))

    def has_node(self, node_id: str) -> bool:
        return node_id in self.nodes

    def has_edge(self, source_id: str, target_id: str, relation: str | None = None) -> bool:
        if relation is None:
            return any(
                source_id == source and target_id == target for source, target, _ in self.edges
            )
        return (source_id, target_id, relation) in self.edges


def _graph() -> _Graph:
    graph = _Graph()
    graph.add_node("article-33", label="Article 33", kind="clause")
    graph.add_node("article-32", label="Article 32", kind="clause")
    return graph


def test_validate_accepts_supported_edge_proposal():
    proposal = GraphUpdateProposal(
        action="edge",
        source_id="article-33",
        target_id="article-32",
        relation="depends_on",
        evidence="Article 33 says notification is shaped by Article 32.",
        citations=["article-33", "article-32"],
    )

    reviewed = validate_proposal(proposal, _graph())

    assert reviewed.status == "accepted"
    assert reviewed.reason == "accepted"


def test_validate_rejects_unknown_citation_and_duplicate_edge():
    graph = _graph()
    graph.add_edge("article-33", "article-32", "depends_on")
    proposal = GraphUpdateProposal(
        action="edge",
        source_id="article-33",
        target_id="article-32",
        relation="depends_on",
        evidence="Duplicate relation.",
        citations=["article-99"],
    )

    reviewed = validate_proposal(proposal, graph)

    assert reviewed.status == "rejected"
    assert "unknown citation" in reviewed.reason
    assert "edge already exists" in reviewed.reason


def test_apply_proposal_mutates_graph_only_after_validation():
    graph = _graph()
    proposal = GraphUpdateProposal(
        action="node",
        node_id="article-33::temporal::review",
        label="72 hours",
        kind="temporal_constraint",
        evidence="Article 33 sets a 72 hour deadline.",
        citations=["article-33"],
    )

    assert apply_proposal(proposal, graph)
    assert graph.has_node("article-33::temporal::review")
    assert proposal.status == "applied"


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "proposals.jsonl"
    proposal = GraphUpdateProposal(
        action="edge",
        source_id="article-33",
        target_id="article-32",
        relation="depends_on",
        evidence="Supported.",
        citations=["article-33", "article-32"],
    )
    validate_proposal(proposal, _graph())

    write_proposals(path, [proposal])
    loaded = load_proposals(path)

    assert len(loaded) == 1
    assert loaded[0].proposal_id == proposal.proposal_id
    assert loaded[0].status == "accepted"


def test_proposal_from_dict_filters_bad_actions():
    assert proposal_from_dict({"action": "delete"}) is None
    assert proposal_from_dict({"action": "edge", "citations": "article-33"}).citations == []
