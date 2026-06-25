from agentic_reg.config import Settings
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorHit
from agentic_reg.retrieval import RetrievedContext, hybrid_retrieve


class _FakeIndex:
    def search(self, query, top_k):
        return [
            VectorHit(id="article-6", title="Article 6", text="Lawful bases.", score=0.9),
            VectorHit(id="article-7", title="Article 7", text="Consent.", score=0.7),
        ]


def _graph() -> KnowledgeGraph:
    g = KnowledgeGraph()
    g.add_node("article-6", label="Article 6", kind="clause")
    g.add_node("article-7", label="Article 7", kind="clause")
    g.add_node("article-9", label="Article 9", kind="clause")
    g.add_edge("article-6", "article-9", "references")
    g.add_edge("article-7", "article-6", "depends_on")
    return g


def _settings(use_graph=True):
    return Settings(_env_file=None, use_graph=use_graph, graph_hops=2, vector_top_k=4)


def test_hybrid_retrieve_with_graph():
    ctx = hybrid_retrieve("consent bases", _FakeIndex(), _graph(), _settings(use_graph=True))

    assert len(ctx.vector_hits) == 2
    assert ctx.vector_hits[0].id == "article-6"

    # graph expansion should pull in article-9 (within 2 hops of article-6)
    node_ids = {n["id"] for n in ctx.graph_nodes}
    assert "article-9" in node_ids
    assert "article-6" in node_ids

    # edges between selected nodes
    sources = {e["source"] for e in ctx.graph_edges}
    assert "article-6" in sources


def test_hybrid_retrieve_vector_only():
    """use_graph=False should skip graph expansion entirely."""
    ctx = hybrid_retrieve("consent", _FakeIndex(), _graph(), _settings(use_graph=False))

    assert len(ctx.vector_hits) == 2
    assert ctx.graph_nodes == []
    assert ctx.graph_edges == []
    assert ctx.multi_hop_paths == []


def test_retrieved_context_to_prompt():
    ctx = RetrievedContext(
        vector_hits=[VectorHit(id="article-6", title="Art 6", text="Lawful.", score=0.9)],
        graph_nodes=[],
        graph_edges=[],
        multi_hop_paths=[],
    )
    prompt = ctx.to_prompt_context()
    assert "Art 6" in prompt
    assert "Lawful" in prompt


def test_retrieved_context_includes_multi_hop():
    ctx = RetrievedContext(
        vector_hits=[],
        graph_nodes=[],
        graph_edges=[],
        multi_hop_paths=[{"nodes": ["article-6", "article-9"], "text": "article-6 -> article-9"}],
    )
    prompt = ctx.to_prompt_context()
    assert "REASONING CHAINS" in prompt
    assert "article-6 -> article-9" in prompt
