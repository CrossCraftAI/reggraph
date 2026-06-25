from agentic_reg._internal import parse_json_object
from agentic_reg.domains import get_domain
from agentic_reg.ingest import Chunk, load_chunks
from agentic_reg.knowledge.extract import extract_cross_references, extract_graph_elements
from agentic_reg.providers.base import LLMProvider


def test_cross_references_are_unit_agnostic():
    gdpr_refs = set(extract_cross_references(load_chunks(get_domain("gdpr").source_path)))
    uk_refs = set(extract_cross_references(load_chunks(get_domain("uk_dpa").source_path)))

    assert ("article-7", "article-6") in gdpr_refs
    assert ("article-9", "article-6") in gdpr_refs
    assert ("section-10", "section-8") in uk_refs
    assert all(source != target for source, target in gdpr_refs | uk_refs)


def test_parse_json_object_is_resilient():
    assert parse_json_object("no json here") == {}
    assert parse_json_object('prose {"a": 1} more prose') == {"a": 1}
    assert parse_json_object("{not valid") == {}


class _StubProvider(LLMProvider):
    name = "stub"

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def complete(self, prompt, *, system=None, temperature=0.0):
        return self._payload


def test_extract_graph_elements_typed_and_filtered():
    payload = (
        '{"concepts": [{"label": "explicit consent", "kind": "condition"}, '
        '{"label": "junk", "kind": "weird"}], '
        '"relations": [{"target_unit": 6, "type": "requires"}, '
        '{"target_unit": 999, "type": "requires"}, '
        '{"target_unit": 7, "type": "bogus"}]}'
    )
    chunk = Chunk(id="article-9", title="Article 9", text="...")
    known = {"article-6", "article-7", "article-9"}

    extraction = extract_graph_elements(chunk, _StubProvider(payload), known)

    assert {concept.kind for concept in extraction.concepts} == {"condition", "concept"}
    assert [(rel.target_id, rel.relation) for rel in extraction.relations] == [
        ("article-6", "requires")
    ]


def test_extract_graph_elements_accepts_non_article_units():
    payload = (
        '{"concepts": [], '
        '"relations": [{"target_unit": 3, "type": "depends_on"}, '
        '{"target_unit": 10, "type": "requires"}]}'
    )
    chunk = Chunk(id="section-8", title="Section 8", text="...")
    known = {"section-3", "section-8", "section-10"}

    extraction = extract_graph_elements(chunk, _StubProvider(payload), known, unit_label="section")

    assert [(rel.target_id, rel.relation) for rel in extraction.relations] == [
        ("section-3", "depends_on"),
        ("section-10", "requires"),
    ]


def test_extract_graph_elements_never_raises():
    class _Boom(LLMProvider):
        name = "boom"

        def complete(self, *args, **kwargs):
            raise RuntimeError("model unreachable")

    extraction = extract_graph_elements(
        Chunk(id="article-6", title="x", text="y"), _Boom(), {"article-6"}
    )

    assert extraction.concepts == []
    assert extraction.relations == []
