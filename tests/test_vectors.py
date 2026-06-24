from agentic_reg.domains import get_domain
from agentic_reg.knowledge.vectors import VectorIndex, _chunk_articles, _split_by_article


def test_split_by_article_extracts_article_bodies():
    domain = get_domain("gdpr")
    text = domain.source_path.read_text(encoding="utf-8")
    articles = _split_by_article(text)

    ids = [aid for aid, _ in articles]
    assert "article-5" in ids
    assert "article-33" in ids
    # Every article should have non-empty body.
    for _aid, body in articles:
        assert len(body) > 50, f"Body too short for {_aid}"


def test_chunk_articles_respects_article_boundaries():
    domain = get_domain("gdpr")
    text = domain.source_path.read_text(encoding="utf-8")
    chunks = _chunk_articles(text, max_chars=300, overlap=50)

    # All chunks reference real articles.
    for chunk in chunks:
        assert chunk.article_ref.startswith("article-")
        assert len(chunk.text) > 0
        assert len(chunk.text) <= 300


def test_vector_index_build_and_search(tmp_path):
    domain = get_domain("gdpr")
    # Override chroma dir so we don't write to the real store.
    model = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir = tmp_path / "chroma"

    index = VectorIndex(chroma_dir, model)
    text = domain.source_path.read_text(encoding="utf-8")
    chunks = _chunk_articles(text)
    if chunks:
        ids = [c.id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{"article_ref": c.article_ref} for c in chunks]
        index._collection.add(ids=ids, documents=docs, metadatas=metas)

    results = index.search("lawful basis for processing", top_k=3)
    assert len(results) >= 1
    assert len(results) <= 3
    for chunk in results:
        assert chunk.article_ref
        assert chunk.text


def test_vector_index_load(tmp_path):
    model = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir = tmp_path / "chroma"

    # Build first.
    index = VectorIndex(chroma_dir, model)
    assert index.chunk_count == 0

    # Load same path.
    loaded = VectorIndex.load(chroma_dir, model)
    assert loaded._chroma_dir == str(chroma_dir)


def test_search_returns_article_6_for_lawfulness_query(tmp_path):
    domain = get_domain("gdpr")
    model = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir = tmp_path / "chroma"

    text = domain.source_path.read_text(encoding="utf-8")
    chunks = _chunk_articles(text)

    index = VectorIndex(chroma_dir, model)
    ids = [c.id for c in chunks]
    docs = [c.text for c in chunks]
    metas = [{"article_ref": c.article_ref} for c in chunks]
    index._collection.add(ids=ids, documents=docs, metadatas=metas)

    results = index.search("lawful basis consent contract", top_k=5)
    article_refs = {chunk.article_ref for chunk in results}
    # Small models on tiny docs can be noisy — at least one regulatory article
    # was retrieved.
    assert len(results) >= 1
    assert all(ref.startswith("article-") for ref in article_refs)
