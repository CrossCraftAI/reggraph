from agentic_reg.domains import get_domain
from agentic_reg.ingest import _make_id, load_chunks


def test_make_id_from_unit_title():
    assert _make_id("Article 6 — Lawfulness of processing") == "article-6"
    assert _make_id("Section 67 — Notification of breach") == "section-67"


def test_load_chunks_from_gdpr_markdown():
    chunks = load_chunks(get_domain("gdpr").source_path)
    ids = [chunk.id for chunk in chunks]

    assert ids[:3] == ["article-5", "article-6", "article-7"]
    assert "article-17" in ids
    assert len(ids) == len(set(ids))

    article_6 = next(chunk for chunk in chunks if chunk.id == "article-6")
    assert "lawful" in article_6.text.lower()


def test_load_chunks_from_uk_dpa_markdown():
    chunks = load_chunks(get_domain("uk_dpa").source_path)
    ids = [chunk.id for chunk in chunks]

    assert ids[:3] == ["section-1", "section-2", "section-3"]
    assert "section-67" in ids


def test_load_chunks_plain_text_splits_on_unit_headings(tmp_path):
    path = tmp_path / "reg.txt"
    path.write_text(
        "Article 1 — Scope\nThis Regulation applies broadly.\n"
        "Section 2 — Terms\nDefinitions go here.\n",
        encoding="utf-8",
    )

    chunks = load_chunks(path)

    assert [chunk.id for chunk in chunks] == ["article-1", "section-2"]
    assert "broadly" in chunks[0].text
