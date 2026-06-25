from agentic_reg.domains import get_domain
from agentic_reg.ingest import load_chunks
from agentic_reg.knowledge.vectors import VectorIndex, _chunk_articles, _split_by_article


MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_split_by_article_extracts_chunk_bodies():
    text = get_domain("gdpr").source_path.read_text(encoding="utf-8")
    articles = _split_by_article(text)

    ids = [article_id for article_id, _ in articles]
    assert "article-5" in ids
    assert "article-33" in ids
    assert all(body for _, body in articles)


def test_chunk_articles_keeps_article_refs():
    text = get_domain("gdpr").source_path.read_text(encoding="utf-8")
    chunks = _chunk_articles(text, max_chars=300, overlap=50)

    assert chunks
    assert all(chunk.article_ref.startswith("article-") for chunk in chunks)
    assert all(0 < len(chunk.text) <= 300 for chunk in chunks)


def test_vector_index_add_count_load_and_search(tmp_path):
    chunks = load_chunks(get_domain("gdpr").source_path)
    index = VectorIndex(tmp_path / "chroma", MODEL)

    index.add(chunks)
    loaded = VectorIndex.load(tmp_path / "chroma", MODEL)
    results = loaded.search("lawful basis for processing", top_k=3)

    assert loaded.count() == len(chunks)
    assert loaded.chunk_count == len(chunks)
    assert 1 <= len(results) <= 3
    assert all(hit.id.startswith("article-") for hit in results)
    assert all(hit.article_ref == hit.id for hit in results)
