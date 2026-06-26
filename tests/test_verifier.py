from agentic_reg.agents.state import Finding
from agentic_reg.agents.verifier import verify
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.providers.base import LLMProvider


def _graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for node_id in ("article-6", "article-7", "article-9", "article-33"):
        graph.add_node(node_id, label=node_id, kind="clause")
    return graph


class _CleanProvider(LLMProvider):
    name = "clean"

    def complete(self, *args, **kwargs):
        return '{"unsupported_claims": [], "contradictions": []}'


def test_invalid_citation_detected_deterministically():
    findings = [Finding("clause_analyst", "q", "x", ["article-6"], ["article-6"])]
    verdict = verify(
        "Lawful per [article-6] and [article-99].", findings, _graph(), _CleanProvider()
    )

    assert verdict.invalid_citations == ["article-99"]
    assert not verdict.ok


def test_llm_flagged_issues_fail_the_verdict():
    class _IssuesProvider(LLMProvider):
        name = "issues"

        def complete(self, *args, **kwargs):
            return '{"unsupported_claims": ["claim X is not supported"], "contradictions": []}'

    verdict = verify("All fine [article-6].", [], _graph(), _IssuesProvider())

    assert verdict.unsupported_claims == ["claim X is not supported"]
    assert not verdict.ok


def test_symbolic_failures_fail_the_verdict():
    findings = [Finding("clause_analyst", "q", "x", ["article-9"], ["article-9"])]
    verdict = verify(
        "Health data may be processed with explicit consent [article-9].",
        findings,
        _graph(),
        _CleanProvider(),
        question="Can special category health data be processed?",
    )

    assert not verdict.ok
    assert any(
        item["rule_id"] == "special_category_requires_basis" for item in verdict.symbolic_findings
    )


def test_symbolic_checks_can_be_disabled():
    verdict = verify(
        "Health data may be processed with explicit consent [article-9].",
        [],
        _graph(),
        _CleanProvider(),
        question="Can special category health data be processed?",
        symbolic_checks=False,
    )

    assert verdict.ok
    assert verdict.symbolic_findings == []


def test_verifier_resilient_when_llm_raises():
    class _BoomProvider(LLMProvider):
        name = "boom"

        def complete(self, *args, **kwargs):
            raise RuntimeError("model down")

    verdict = verify("Only [article-6].", [], _graph(), _BoomProvider())

    assert verdict.ok
    assert verdict.llm_note
