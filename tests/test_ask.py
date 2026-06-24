import json
from unittest.mock import patch

from agentic_reg.ask import (
    Trace,
    _execute_tool,
    _tool_expand,
    _tool_search,
    _tool_verify,
    _try_parse_tool_call,
    answer,
)
from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.vectors import VectorIndex, _chunk_articles


def test_try_parse_tool_call_returns_none_for_plain_text():
    assert _try_parse_tool_call("This is a plain answer.") is None


def test_try_parse_tool_call_parses_json_block():
    text = '```json\n{"name": "search", "arguments": {"query": "consent"}}\n```'
    result = _try_parse_tool_call(text)
    assert result is not None
    assert result["name"] == "search"
    assert result["arguments"]["query"] == "consent"


def test_trace_to_json():
    trace = Trace(question="What is consent?")
    trace.answer = "Consent is defined in [article-7]."
    trace.steps.append({"role": "answer", "content": trace.answer})  # type: ignore[arg-type]
    result = trace.to_json()
    assert "What is consent?" in result
    assert "article-7" in result


def test_tool_search_returns_chunks(tmp_path):
    domain = get_domain("gdpr")
    model = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir = tmp_path / "chroma"
    index = VectorIndex(chroma_dir, model)
    text = domain.source_path.read_text(encoding="utf-8")
    chunks = _chunk_articles(text)
    ids = [c.id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [{"article_ref": c.article_ref} for c in chunks]
    index._collection.add(ids=ids, documents=docs, metadatas=metas)

    result = _tool_search("consent", index)
    assert "article-" in result


def test_tool_expand():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")
    graph.add_node("article-7", label="A7", kind="clause")
    graph.add_edge("article-6", "article-7", "references")

    result = _tool_expand(["article-6"], graph, hops=1)
    assert "article-7" in result


def test_tool_verify():
    graph = KnowledgeGraph()
    for nid in ("article-6", "article-7", "article-9", "article-17", "article-33"):
        graph.add_node(nid, label=nid, kind="clause")

    result = _tool_verify(
        "Can health data be processed?",
        "Yes, if Article 9 conditions are met [article-9] and there is a lawful basis [article-6].",
        graph,
    )
    assert "special_category_requires_basis" in result


def test_execute_tool_handles_unknown(tmp_path):
    graph = KnowledgeGraph()
    chroma_dir = tmp_path / "chroma"
    index = VectorIndex(str(chroma_dir), "sentence-transformers/all-MiniLM-L6-v2")
    result = _execute_tool("nonexistent", {}, graph, index)
    assert "Unknown tool" in result


def test_answer_with_mock_provider(tmp_path):
    """Simulate a full ReAct loop with a fake provider."""
    # Build graph at the path answer() expects: PROJECT_ROOT/data/store/<domain>/graph.json
    domain = get_domain("gdpr")
    graph = KnowledgeGraph.build(domain)
    store_dir = tmp_path / "data" / "store" / "gdpr"
    store_dir.mkdir(parents=True, exist_ok=True)
    graph.save(store_dir / "graph.json")

    # Build vector index in the expected chroma dir.
    chroma_dir = store_dir / "chroma"
    model = "sentence-transformers/all-MiniLM-L6-v2"
    index = VectorIndex(str(chroma_dir), model)
    text = domain.source_path.read_text(encoding="utf-8")
    chunks = _chunk_articles(text)
    index._collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[{"article_ref": c.article_ref} for c in chunks],
    )

    # Mock provider — returns tool call then final answer.
    call_count = [0]

    def fake_chat(messages, tools=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return json.dumps({"name": "search", "arguments": {"query": "lawful basis consent"}})
        return "Processing requires a lawful basis such as consent [article-6] [article-7]."

    with (
        patch("agentic_reg.ask.get_provider") as mock_provider,
        patch("agentic_reg.ask.PROJECT_ROOT", tmp_path),
    ):
        mock_provider.return_value.chat = fake_chat
        trace = answer("What is the lawful basis for consent?", domain_name="gdpr")

    assert trace.answer
    assert "article-6" in trace.answer
    assert len(trace.steps) >= 2
