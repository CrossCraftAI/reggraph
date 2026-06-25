from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph


def _sample_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for article in ("article-5", "article-6", "article-7", "article-9"):
        graph.add_node(article, label=article.replace("-", " ").title(), kind="clause")
    graph.add_edge("article-6", "article-7", "references")
    graph.add_edge("article-6", "article-9", "references")
    return graph


def test_expand_is_undirected_and_respects_hops():
    graph = _sample_graph()

    nodes, _ = graph.expand(["article-7"], hops=1)
    ids = {node["id"] for node in nodes}
    assert "article-7" in ids
    assert "article-6" in ids
    assert "article-9" not in ids

    nodes_2hop, _ = graph.expand(["article-7"], hops=2)
    assert "article-9" in {node["id"] for node in nodes_2hop}


def test_clause_paths_are_multi_hop_and_clause_only():
    graph = KnowledgeGraph()
    for article in ("article-33", "article-32", "article-5"):
        graph.add_node(article, label=article, kind="clause")
    graph.add_node("c1", label="a concept", kind="obligation")
    graph.add_edge("article-33", "article-32", "references")
    graph.add_edge("article-32", "article-5", "references")
    graph.add_edge("article-32", "c1", "imposes")

    paths = graph.clause_paths(["article-33", "article-5"], cutoff=3)

    assert ["article-33", "article-32", "article-5"] in paths
    assert all("c1" not in path for path in paths)
    assert graph.edge_relation("article-33", "article-32") == "references"
    assert graph.label("article-5") == "article-5"


def test_save_and_load_roundtrip(tmp_path):
    graph = _sample_graph()
    path = tmp_path / "graph.json"
    graph.save(path)

    loaded = KnowledgeGraph.load(path)

    assert loaded.num_nodes == 4
    assert loaded.node_count == 4
    assert loaded.num_edges == 2
    assert loaded.edge_count == 2
    nodes, edges = loaded.expand(["article-6"], hops=1)
    assert {node["id"] for node in nodes} == {"article-6", "article-7", "article-9"}
    assert {edge["relation"] for edge in edges} == {"references"}


def test_build_compatibility_uses_domain_chunks():
    graph = KnowledgeGraph.build(get_domain("gdpr"))

    assert graph.has_node("article-6")
    assert graph.has_edge("article-6", "article-7", "references")
    assert "article-6" in graph.nodes()
    assert all(source != target for source, target, _ in graph.edges())


def test_view_can_filter_clause_nodes():
    graph = _sample_graph()
    graph.add_node("concept-1", label="Concept", kind="concept")
    graph.add_edge("article-6", "concept-1", "introduces")

    nodes, edges = graph.view(kinds={"clause"})

    assert {node["kind"] for node in nodes} == {"clause"}
    assert all(edge["target"] != "concept-1" for edge in edges)
