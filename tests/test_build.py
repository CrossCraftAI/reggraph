import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_reg import build as build_module
from agentic_reg.domains import Domain
from agentic_reg.domains import base as domain_base
from agentic_reg.ingest import Chunk
from agentic_reg.knowledge.graph import KnowledgeGraph


class _FakeVectorIndex:
    def __init__(self, chroma_dir, embedding_model):
        self._chroma_dir = Path(chroma_dir)
        self._chroma_dir.mkdir(parents=True, exist_ok=True)
        self._chunks = []

    def add(self, chunks):
        self._chunks = list(chunks)

    def count(self):
        return len(self._chunks)


def test_build_no_enrich_writes_store_without_requesting_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(domain_base, "PROJECT_ROOT", tmp_path)
    domain = Domain(
        name="test_reg",
        title="Test Regulation",
        description="x",
        source_path=tmp_path / "source.md",
        unit_label="section",
    )
    chunks = [
        Chunk(id="section-1", title="Section 1", text="See Section 2."),
        Chunk(id="section-2", title="Section 2", text="Definitions."),
    ]

    monkeypatch.setattr(
        build_module, "get_settings", lambda: SimpleNamespace(embedding_model="fake", domain="x")
    )
    monkeypatch.setattr(build_module, "get_domain", lambda name: domain)
    monkeypatch.setattr(build_module, "load_chunks", lambda path: chunks)
    monkeypatch.setattr(build_module, "VectorIndex", _FakeVectorIndex)
    monkeypatch.setattr(
        build_module,
        "extract_graph_elements",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("enrichment called")),
    )

    original_import = builtins.__import__

    def _guard_provider_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "agentic_reg.providers" or (level and name == "providers"):
            raise AssertionError("provider requested")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _guard_provider_import)

    domain.chroma_dir.mkdir(parents=True)
    stale = domain.chroma_dir / "stale.txt"
    stale.write_text("old", encoding="utf-8")

    build_module.build("test_reg", enrich=False)

    assert domain.graph_path.exists()
    assert domain.chroma_dir.exists()
    assert not stale.exists()
    graph = KnowledgeGraph.load(domain.graph_path)
    assert graph.has_node("section-1")
    assert graph.has_node("section-2")
    assert graph.g.nodes["section-1"]["text"] == "See Section 2."
    assert graph.g.nodes["section-2"]["text"] == "Definitions."
    assert graph.has_edge("section-1", "section-2", "references")


def test_main_rejects_unknown_domain():
    with pytest.raises(ValueError):
        build_module.main(["--domain", "nonexistent", "--no-enrich"])
