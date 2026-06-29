"""Verification for team draft answers."""

from .._internal import extract_citations, parse_json_object
from ..knowledge.graph import KnowledgeGraph
from ..knowledge.symbolic import run_symbolic_checks
from ..providers.base import LLMProvider
from .state import Finding, Verdict

VERIFY_SYSTEM = "You are a strict verification agent. Respond with JSON only."
VERIFY_PROMPT = """Check this draft answer against the supporting findings.

DRAFT:
{draft}

SUPPORTING FINDINGS:
{findings}

Identify claims in the draft that are not supported by the findings and any
internal contradictions. Return ONLY JSON:
{{"unsupported_claims": ["..."], "contradictions": ["..."]}}
Use empty lists if there are none."""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def verify(
    draft: str,
    findings: list[Finding],
    graph: KnowledgeGraph,
    provider: LLMProvider,
    *,
    question: str = "",
    symbolic_checks: bool = True,
) -> Verdict:
    invalid = [citation for citation in extract_citations(draft) if not graph.has_node(citation)]

    unsupported: list[str] = []
    contradictions: list[str] = []
    llm_note = ""
    findings_text = "\n".join(f"- ({finding.role}) {finding.text}" for finding in findings)
    try:
        raw = provider.complete(
            VERIFY_PROMPT.format(draft=draft, findings=findings_text or "(none)"),
            system=VERIFY_SYSTEM,
        )
        data = parse_json_object(raw)
        unsupported = _string_list(data.get("unsupported_claims"))
        contradictions = _string_list(data.get("contradictions"))
    except Exception:
        llm_note = "LLM verification unavailable; deterministic checks only."

    symbolic = (
        [finding.to_dict() for finding in run_symbolic_checks(question, draft, graph)]
        if symbolic_checks
        else []
    )
    symbolic_failed = [item for item in symbolic if item.get("passed") is False]

    return Verdict(
        ok=not (invalid or unsupported or contradictions or symbolic_failed),
        invalid_citations=invalid,
        unsupported_claims=unsupported,
        contradictions=contradictions,
        symbolic_findings=symbolic,
        llm_note=llm_note,
    )
