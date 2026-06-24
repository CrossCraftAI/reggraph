from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph, _article_id, _extract_citations
from agentic_reg.knowledge.proposals import GraphUpdateProposal, validate_proposal
from agentic_reg.knowledge.symbolic import run_symbolic_checks


def test_article_id_from_heading():
    assert _article_id("Article 5 — Principles") == "article-5"
    assert _article_id("Article 17 — Right to erasure") == "article-17"


def test_extract_citations_deduplicates():
    text = "See [article-6] and [article-6] and [ARTICLE-7]."
    citations = _extract_citations(text)
    assert sorted(citations) == ["article-6", "article-7"]


def test_graph_add_and_check_node():
    graph = KnowledgeGraph()
    graph.add_node("article-1", label="Article 1", kind="clause")
    assert graph.has_node("article-1")
    assert graph.has_node("ARTICLE-1")  # case-insensitive
    assert not graph.has_node("article-99")


def test_graph_add_and_check_edge():
    graph = KnowledgeGraph()
    graph.add_node("article-1", label="A1", kind="clause")
    graph.add_node("article-2", label="A2", kind="clause")
    graph.add_edge("article-1", "article-2", "references")

    assert graph.has_edge("article-1", "article-2")
    assert graph.has_edge("article-1", "article-2", "references")
    assert not graph.has_edge("article-1", "article-2", "depends_on")
    assert not graph.has_edge("article-2", "article-1")


def test_build_from_gdpr_domain():
    domain = get_domain("gdpr")
    graph = KnowledgeGraph.build(domain)

    # Every article in the seed data becomes a node.
    expected = {
        "article-5",
        "article-6",
        "article-7",
        "article-9",
        "article-15",
        "article-17",
        "article-33",
    }
    for nid in expected:
        assert graph.has_node(nid), f"Missing node: {nid}"

    # Cross-reference edges should exist.
    assert graph.has_edge("article-6", "article-5")
    assert graph.has_edge("article-9", "article-6")
    assert graph.has_edge("article-7", "article-6")


def test_save_and_load_roundtrip(tmp_path):
    domain = get_domain("gdpr")
    original = KnowledgeGraph.build(domain)
    path = tmp_path / "graph.json"
    original.save(path)

    loaded = KnowledgeGraph.load(path)
    assert loaded.node_count == original.node_count
    assert loaded.edge_count == original.edge_count
    for nid in original.nodes():
        assert loaded.has_node(nid)


def test_expand_returns_reachable_nodes():
    graph = KnowledgeGraph()
    for nid in ("a", "b", "c", "d", "e"):
        graph.add_node(nid, label=nid, kind="clause")
    graph.add_edge("a", "b", "references")
    graph.add_edge("b", "c", "references")
    graph.add_edge("c", "d", "references")
    graph.add_edge("d", "e", "references")

    # 1 hop from a reaches a, b
    reached = graph.expand({"a"}, hops=1)
    assert reached == {"a", "b"}

    # 2 hops from a reaches a, b, c
    reached = graph.expand({"a"}, hops=2)
    assert reached == {"a", "b", "c"}

    # 0 hops returns only the start set
    reached = graph.expand({"a"}, hops=0)
    assert reached == {"a"}


def test_expand_traverses_inbound_edges():
    graph = KnowledgeGraph()
    graph.add_node("a", label="A", kind="clause")
    graph.add_node("b", label="B", kind="clause")
    graph.add_edge("a", "b", "references")

    # b can reach a via inbound edge.
    reached = graph.expand({"b"}, hops=1)
    assert "a" in reached


def test_graph_satisfies_proposal_graph_protocol():
    domain = get_domain("gdpr")
    graph = KnowledgeGraph.build(domain)

    proposal = GraphUpdateProposal(
        action="edge",
        source_id="article-33",
        target_id="article-17",
        relation="depends_on",
        evidence="Breach notification connects to erasure obligations.",
        citations=["article-33", "article-17"],
    )
    reviewed = validate_proposal(proposal, graph)
    assert reviewed.status == "accepted"


def test_graph_satisfies_graphlike_protocol():
    domain = get_domain("gdpr")
    graph = KnowledgeGraph.build(domain)

    findings = run_symbolic_checks(
        "What must happen after a breach?",
        "Notify within 72 hours [article-33].",
        graph,
    )
    validity = next(f for f in findings if f.rule_id == "citation_validity")
    assert validity.passed

    breach = next(f for f in findings if f.rule_id == "breach_notification_deadline")
    assert breach.passed

    # special_category and withdrawal_erasure rules don't fire for breach questions.
    triggered_ids = {f.rule_id for f in findings}
    assert "special_category_requires_basis" not in triggered_ids
    assert "withdrawal_erasure_chain" not in triggered_ids


def test_empty_graph_expand_returns_empty():
    graph = KnowledgeGraph()
    assert graph.expand(set(), hops=2) == set()


def test_build_produces_no_self_edges():
    domain = get_domain("gdpr")
    graph = KnowledgeGraph.build(domain)
    for source, target, _ in graph.edges():
        assert source != target, f"Self-edge: {source} -> {target}"
