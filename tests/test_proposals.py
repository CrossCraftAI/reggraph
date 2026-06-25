import pytest

from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.proposals import (
    GraphUpdateProposal,
    apply_proposal,
    load_proposals,
    proposal_from_dict,
    validate_proposal,
    write_proposals,
)


def _graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
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


@pytest.mark.parametrize(
    ("proposal", "message"),
    [
        (
            GraphUpdateProposal(
                action="edge",
                source_id="article-33",
                target_id="article-32",
                relation="not_a_relation",
                evidence="Unsupported relation.",
                citations=["article-33"],
            ),
            "unsupported relation",
        ),
        (
            GraphUpdateProposal(
                action="edge",
                source_id="article-99",
                target_id="article-32",
                relation="depends_on",
                evidence="Unknown source.",
                citations=["article-32"],
            ),
            "unknown source node",
        ),
        (
            GraphUpdateProposal(
                action="edge",
                source_id="",
                target_id="article-32",
                relation="depends_on",
                evidence="Missing source.",
                citations=["article-32"],
            ),
            "unknown source node: (empty)",
        ),
        (
            GraphUpdateProposal(
                action="edge",
                source_id="article-33",
                target_id="article-99",
                relation="depends_on",
                evidence="Unknown target.",
                citations=["article-33"],
            ),
            "unknown target node",
        ),
        (
            GraphUpdateProposal(
                action="edge",
                source_id="article-33",
                target_id="",
                relation="depends_on",
                evidence="Missing target.",
                citations=["article-33"],
            ),
            "unknown target node: (empty)",
        ),
        (
            GraphUpdateProposal(
                action="node",
                node_id="article-33::bad",
                label="Bad node",
                kind="unsupported_kind",
                evidence="Unsupported kind.",
                citations=["article-33"],
            ),
            "unsupported node kind",
        ),
        (
            GraphUpdateProposal(
                action="node",
                node_id="",
                label="",
                kind="concept",
                evidence="Missing required node fields.",
                citations=["article-33"],
            ),
            "node_id is required",
        ),
        (
            GraphUpdateProposal(
                action="node",
                node_id="article-33",
                label="Duplicate",
                kind="clause",
                evidence="Duplicate node.",
                citations=["article-33"],
            ),
            "node already exists",
        ),
        (
            GraphUpdateProposal(
                action="edge",
                source_id="article-33",
                target_id="article-32",
                relation="depends_on",
                evidence="",
                citations=["article-33"],
            ),
            "evidence is required",
        ),
    ],
)
def test_validate_rejects_invalid_proposals(proposal, message):
    reviewed = validate_proposal(proposal, _graph())

    assert reviewed.status == "rejected"
    assert message in reviewed.reason


def test_validate_rejects_node_with_empty_label():
    proposal = GraphUpdateProposal(
        action="node",
        node_id="article-33::concept",
        label="",
        kind="concept",
        evidence="Missing label.",
        citations=["article-33"],
    )

    reviewed = validate_proposal(proposal, _graph())

    assert reviewed.status == "rejected"
    assert "label is required" in reviewed.reason


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


def test_apply_proposal_rejects_invalid_proposal_without_mutating_graph():
    graph = _graph()
    proposal = GraphUpdateProposal(
        action="edge",
        source_id="article-33",
        target_id="article-32",
        relation="not_a_relation",
        evidence="Unsupported relation.",
        citations=["article-33", "article-32"],
    )

    assert not apply_proposal(proposal, graph)
    assert graph.edge_count == 0
    assert not graph.has_edge("article-33", "article-32")
    assert proposal.status == "rejected"
    assert "unsupported relation" in proposal.reason


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
