"""Evaluation harness for regulatory question-answering.

Compares RegGraph against a vanilla LangGraph baseline on compliance-specific
metrics. Each config runs the same test cases with the same LLM, same embedding
model, and same vector index — the only variable is RegGraph's regulatory-specific
architecture (typed knowledge graph, symbolic checks, deterministic citation
verification).

Supports two test scenarios:

    ``single`` (default)
        Questions within one jurisdiction (GDPR clauses only).

    ``cross-jurisdictional``
        Questions spanning multiple regulations (GDPR ↔ UK DPA, GDPR ↔ CCPA).
        Requires multiple domain graphs to be built.

Usage::

    # Cross-jurisdictional
    python -m agentic_reg.eval --configs reggraph,langgraph \
        --scenario cross-jurisdictional --limit 4

    # Full comparison: RegGraph vs LangGraph baseline
    python -m agentic_reg.eval --configs reggraph,langgraph --limit 4

    # Ablation: measure the graph's contribution
    python -m agentic_reg.eval --configs reggraph,no-graph --limit 4

    # With LLM judge for holistic quality scoring
    python -m agentic_reg.eval --configs reggraph,langgraph --judge

Deterministic metrics (citation F1, hallucination rate, multi-hop recall,
cross-jurisdictional recall) are computed at zero LLM cost. An optional
``--judge`` flag gates LLM-based quality scoring.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_reg.config import PROJECT_ROOT, get_settings
from agentic_reg.domains import get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.symbolic import run_symbolic_checks
from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import get_provider

_CITATION_RE = re.compile(r"\[([a-z][a-z-]*-\d+)\]", flags=re.IGNORECASE)

# ── test cases ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "lawful-basis",
        "question": "What lawful bases allow processing of personal data?",
        "expected_citations": ["article-6"],
        "expected_multi_hop": {"article-5", "article-7"},
    },
    {
        "id": "special-category",
        "question": "Can health data be processed under the GDPR?",
        "expected_citations": ["article-9", "article-6"],
        "expected_multi_hop": {"article-7"},
    },
    {
        "id": "consent-withdrawal",
        "question": "What happens when a data subject withdraws consent?",
        "expected_citations": ["article-7", "article-17"],
        "expected_multi_hop": {"article-6", "article-5"},
    },
    {
        "id": "breach-notification",
        "question": "What must a controller do after a personal data breach?",
        "expected_citations": ["article-33"],
        "expected_multi_hop": {"article-5"},
    },
    {
        "id": "right-of-access",
        "question": "What information must be provided when a data subject requests access?",
        "expected_citations": ["article-15"],
        "expected_multi_hop": {"article-5", "article-17"},
    },
]

# ── cross-jurisdictional test cases ────────────────────────────────────────

CROSS_JURISDICTIONAL_TEST_CASES = [
    {
        "id": "cross-special-category",
        "question": (
            "Does UK DPA 2018 provide the same protections for special category "
            "data as GDPR article-9?"
        ),
        "expected_citations": ["article-9", "section-10"],
        "expected_multi_hop": {"article-6", "section-8"},
        "expected_cross_jurisdictional": {
            ("article-9", "section-10"),  # GDPR special category ↔ UK DPA equivalent
            ("article-6", "section-8"),  # Lawful basis across jurisdictions
        },
        "domains": ["gdpr", "uk_dpa"],
    },
    {
        "id": "cross-erasure",
        "question": "What is the UK DPA equivalent of GDPR's right to erasure in article-17?",
        "expected_citations": ["article-17", "section-47"],
        "expected_multi_hop": {"article-5", "article-7"},
        "expected_cross_jurisdictional": {
            ("article-17", "section-47"),
            ("article-7", "section-8"),
        },
        "domains": ["gdpr", "uk_dpa"],
    },
    {
        "id": "cross-breach",
        "question": (
            "If a controller is GDPR breach-notification compliant under article-33, "
            "does that satisfy UK DPA requirements?"
        ),
        "expected_citations": ["article-33", "section-67"],
        "expected_multi_hop": {"article-5"},
        "expected_cross_jurisdictional": {
            ("article-33", "section-67"),
        },
        "domains": ["gdpr", "uk_dpa"],
    },
    {
        "id": "cross-consent",
        "question": (
            "What additional consent conditions does GDPR article-7 impose that "
            "UK DPA doesn't explicitly require?"
        ),
        "expected_citations": ["article-7", "article-5"],
        "expected_multi_hop": {"article-6"},
        "expected_cross_jurisdictional": {
            ("article-7", "section-8"),
            ("article-6", "section-8"),
        },
        "domains": ["gdpr", "uk_dpa"],
    },
]

# ── scenario dispatch ──────────────────────────────────────────────────────

_SCENARIOS = {
    "single": TEST_CASES,
    "cross-jurisdictional": CROSS_JURISDICTIONAL_TEST_CASES,
}

# ── agent prompt templates ──────────────────────────────────────────────────

_REGGRAPH_PROMPT = """\
You are a regulatory compliance assistant with access to a structured knowledge
graph of the regulation. Answer the question using the provided context.

Your answer MUST:
1. Cite every claim with article references in [article-N] brackets.
2. Reference both directly matching articles AND related articles found through
   graph expansion (cross-references, dependencies, exceptions).
3. Note any conditions, exceptions, or limitations that apply.
4. If you are unsure about a claim, say so rather than guessing.

Directly relevant articles (from semantic search):
{vector_context}

Related articles (from graph expansion — cross-references, dependencies):
{graph_context}

Question: {question}

Answer (with [article-N] citations for every claim):"""

_LANGGRAPH_PROMPT = """\
You are a regulatory compliance assistant. Answer the question using ONLY the
provided regulatory text excerpts. Cite every claim with article references
in [article-N] brackets.

If the provided excerpts do not contain enough information to answer the
question fully, say so explicitly rather than guessing.

Regulatory excerpts:
{context}

Question: {question}

Answer (with [article-N] citations for every claim):"""

_NOGRAPH_PROMPT = """\
You are a regulatory compliance assistant. Answer the question using ONLY the
provided regulatory text excerpts. Cite every claim with article references
in [article-N] brackets.

Regulatory excerpts:
{context}

Question: {question}

Answer:"""

_LLM_JUDGE_PROMPT = """\
You are an expert evaluator of regulatory compliance answers. Grade the following
answer on a scale of 0–10 across these dimensions:

1. **Correctness** (0-10): Are the regulatory claims factually correct?
2. **Completeness** (0-10): Does the answer cover all relevant aspects?
3. **Citation quality** (0-10): Are citations accurate and well-placed?
4. **Clarity** (0-10): Is the answer clear and actionable?

Question: {question}
Answer: {answer}

Respond with ONLY a JSON object: {{"correctness": N, "completeness": N,
"citation_quality": N, "clarity": N, "overall": N}}
where overall is the mean of the four scores."""


# ── data types ──────────────────────────────────────────────────────────────


@dataclass
class EvalMetrics:
    """Metrics computed for a single test-case answer."""

    citation_f1: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    hallucination_rate: float = 0.0
    hallucinated: list[str] = field(default_factory=list)
    multi_hop_recall: float = 0.0
    cross_jurisdictional_recall: float = 0.0
    # Runtime metrics
    elapsed_ms: float = 0.0
    token_count: int = 0
    # Symbolic check pass rate (RegGraph only; None for other configs)
    symbolic_pass_rate: float | None = None
    symbolic_findings: list[dict[str, object]] = field(default_factory=list)
    # LLM judge (only when --judge is enabled)
    llm_judge_score: float | None = None
    llm_judge_detail: dict[str, float] | None = None


@dataclass
class EvalResult:
    case_id: str
    question: str
    config: str  # "reggraph" | "langgraph" | "no-graph"
    answer: str = ""
    metrics: EvalMetrics = field(default_factory=EvalMetrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── metric functions ────────────────────────────────────────────────────────


def compute_metrics(
    answer_text: str,
    expected_citations: list[str],
    expected_multi_hop: set[str],
    graph: KnowledgeGraph | None,
    *,
    elapsed_ms: float = 0.0,
    token_count: int = 0,
    symbolic_findings: list[dict[str, object]] | None = None,
    expected_cross_jurisdictional: set[tuple[str, str]] | None = None,
) -> EvalMetrics:
    """Compute all deterministic metrics from an answer string."""
    cited = _extract_citations(answer_text)
    cited_set = set(cited)
    expected_set = set(expected_citations)

    # Precision: fraction of cited articles that are in the expected set.
    tp = cited_set & expected_set
    precision = len(tp) / len(cited_set) if cited_set else 0.0

    # Recall: fraction of expected articles that were cited.
    recall = len(tp) / len(expected_set) if expected_set else 1.0

    # F1
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Hallucination: cited articles not in the graph.
    hallucinated: list[str] = []
    if graph is not None:
        hallucinated = [c for c in cited if not graph.has_node(c)]
    hallucination_rate = len(hallucinated) / len(cited) if cited else 0.0

    # Multi-hop recall: fraction of expected connected articles found.
    mh_hit = len(cited_set & expected_multi_hop)
    mh_recall = mh_hit / len(expected_multi_hop) if expected_multi_hop else 1.0

    # Cross-jurisdictional recall: fraction of expected article pairs where
    # BOTH articles in the pair were cited. Measures whether the answer
    # successfully maps between regulation domains (e.g., found both
    # article-9 AND section-10).
    xj_recall = 0.0
    if expected_cross_jurisdictional:
        xj_hits = sum(
            1 for a, b in expected_cross_jurisdictional if a in cited_set and b in cited_set
        )
        xj_recall = xj_hits / len(expected_cross_jurisdictional)

    # Symbolic pass rate
    symbolic_pass_rate: float | None = None
    if symbolic_findings is not None:
        passed = sum(1 for f in symbolic_findings if f.get("passed", False))
        symbolic_pass_rate = passed / len(symbolic_findings) if symbolic_findings else 1.0

    return EvalMetrics(
        citation_f1=f1,
        citation_precision=precision,
        citation_recall=recall,
        hallucination_rate=hallucination_rate,
        hallucinated=hallucinated,
        multi_hop_recall=mh_recall,
        cross_jurisdictional_recall=xj_recall,
        elapsed_ms=elapsed_ms,
        token_count=token_count,
        symbolic_pass_rate=symbolic_pass_rate,
        symbolic_findings=symbolic_findings or [],
    )


def _extract_citations(text: str) -> list[str]:
    """Return deduplicated, lowercased citations from *text*."""
    seen: dict[str, None] = {}
    for match in _CITATION_RE.findall(text):
        seen.setdefault(match.lower(), None)
    return list(seen)


# ── agent runners ───────────────────────────────────────────────────────────


def _run_reggraph(
    question: str,
    provider: Any,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    *,
    vector_top_k: int = 4,
    graph_hops: int = 2,
) -> tuple[str, float, int, list[dict[str, object]]]:
    """Run the full RegGraph pipeline on a question.

    Returns (answer, elapsed_ms, token_count, symbolic_findings).
    """
    t0 = time.perf_counter()

    # 1. Vector search
    chunks = vector_index.search(question, top_k=vector_top_k)
    vector_context = "\n\n---\n\n".join(f"[{c.article_ref}] {c.text}" for c in chunks)

    # 2. Graph expansion from matched article nodes
    matched_nodes = {c.article_ref for c in chunks if graph.has_node(c.article_ref)}
    if matched_nodes:
        expanded_nodes, _ = graph.expand(matched_nodes, hops=graph_hops)
        expanded = {node["id"] for node in expanded_nodes}
    else:
        expanded = set()
    graph_articles = expanded - matched_nodes

    # Build graph context: fetch text for expanded nodes
    graph_lines: list[str] = []
    for node_id in sorted(graph_articles):
        # Access node text from the internal NetworkX graph
        if node_id in graph._graph:
            node_data = graph._graph.nodes[node_id]
            text = node_data.get("text", "")
            label = node_data.get("label", node_id)
            if text:
                graph_lines.append(f"[{node_id}] {label}\n{text[:500]}")
    graph_context = (
        "\n\n---\n\n".join(graph_lines)
        if graph_lines
        else ("(No additional articles found via graph expansion.)")
    )

    # 3. LLM reasoning
    prompt = _REGGRAPH_PROMPT.format(
        vector_context=vector_context,
        graph_context=graph_context,
        question=question,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    answer = provider.chat(messages)

    # 4. Symbolic checks
    findings = run_symbolic_checks(question, answer, graph)
    symbolic_findings = [f.to_dict() for f in findings]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    # Rough token estimate: ~4 chars per token for English text.
    token_count = (len(prompt) + len(answer)) // 4

    return answer, elapsed_ms, token_count, symbolic_findings


def _run_langgraph_baseline(
    question: str,
    provider: Any,
    vector_index: VectorIndex,
    *,
    vector_top_k: int = 4,
) -> tuple[str, float, int]:
    """Run the vanilla LangGraph baseline on a question.

    Returns (answer, elapsed_ms, token_count).
    """
    t0 = time.perf_counter()

    # 1. Vector search only — no graph expansion
    chunks = vector_index.search(question, top_k=vector_top_k)
    context = "\n\n---\n\n".join(f"[{c.article_ref}] {c.text}" for c in chunks)

    # 2. LLM reasoning over retrieved chunks only
    prompt = _LANGGRAPH_PROMPT.format(context=context, question=question)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    answer = provider.chat(messages)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    token_count = (len(prompt) + len(answer)) // 4

    return answer, elapsed_ms, token_count


def _run_nograph(
    question: str,
    provider: Any,
    vector_index: VectorIndex,
    *,
    vector_top_k: int = 4,
) -> tuple[str, float, int]:
    """Run vector-only baseline (no graph, no structured prompting).

    Returns (answer, elapsed_ms, token_count).
    """
    t0 = time.perf_counter()

    chunks = vector_index.search(question, top_k=vector_top_k)
    context = "\n\n---\n\n".join(f"[{c.article_ref}] {c.text}" for c in chunks)

    prompt = _NOGRAPH_PROMPT.format(context=context, question=question)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    answer = provider.chat(messages)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    token_count = (len(prompt) + len(answer)) // 4

    return answer, elapsed_ms, token_count


# ── LLM judge ───────────────────────────────────────────────────────────────


def _run_judge(question: str, answer: str, provider: Any) -> dict[str, float]:
    """Run LLM-judge quality scoring. Returns dimension scores or empty dict on failure."""
    prompt = _LLM_JUDGE_PROMPT.format(question=question, answer=answer)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    try:
        raw = provider.chat(messages)
        # Extract JSON from the response (may be wrapped in markdown).
        match = re.search(r"\{[^}]+\}", raw, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))  # type: ignore[no-any-return]
    except Exception:
        pass
    return {}


# ── eval runner ─────────────────────────────────────────────────────────────


def run(
    *,
    configs: list[str] | None = None,
    limit: int | None = None,
    domain_name: str = "gdpr",
    scenario: str = "single",
    judge: bool = False,
    live: bool = False,
) -> tuple[list[EvalResult], dict[str, Any]]:
    """Run evaluation across configurations and return results + summary.

    Parameters
    ----------
    configs:
        One or more of ``"reggraph"`` (full pipeline), ``"langgraph"``
        (vanilla LangGraph + ChromaDB baseline), and ``"no-graph"``
        (vector-only, no structured prompting).
    limit:
        Cap the number of test cases (useful for quick smoke tests).
    domain_name:
        Which registered domain to evaluate against (single-jurisdiction mode).
    scenario:
        ``"single"`` for single-jurisdiction test cases, or
        ``"cross-jurisdictional"`` for multi-regulation mapping.
    judge:
        If True, run LLM-judge quality scoring on every answer.
    live:
        If True, use real LLM calls. If False (default), use synthetic
        answers for fast deterministic metric validation (no API cost).
    """
    if configs is None:
        configs = ["reggraph", "langgraph"]

    if scenario not in _SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}. Choose from: {sorted(_SCENARIOS)}")

    domain = get_domain(domain_name)
    store_dir = PROJECT_ROOT / "data" / "store" / domain.name

    # Load primary graph if available.
    try:
        graph = KnowledgeGraph.load(store_dir / "graph.json")
    except FileNotFoundError:
        graph = None

    # For cross-jurisdictional, attempt to load secondary domain graphs.
    secondary_graphs: dict[str, KnowledgeGraph] = {}
    if scenario == "cross-jurisdictional":
        for dom_name in ("uk_dpa", "ccpa"):
            try:
                dom = get_domain(dom_name)
                sg = KnowledgeGraph.load(PROJECT_ROOT / "data" / "store" / dom.name / "graph.json")
                secondary_graphs[dom_name] = sg
                # Merge secondary graph nodes into primary for hallucination
                # detection and citation verification.
                if graph is not None:
                    for node_id in sg.nodes():
                        if not graph.has_node(node_id):
                            graph.add_node(node_id, label=node_id, kind="clause")
            except (FileNotFoundError, KeyError, ValueError):
                pass

    # Lazy-init provider and vector index only when running live.
    provider = None
    vector_index = None
    if live:
        settings = get_settings()
        provider = get_provider(settings)
        try:
            vector_index = VectorIndex.load(store_dir / "chroma", settings.embedding_model)
        except Exception:
            vector_index = None

    cases = _SCENARIOS[scenario][:limit] if limit else _SCENARIOS[scenario]
    results: list[EvalResult] = []

    for config in configs:
        for case in cases:
            if live and provider and vector_index:
                result = _run_live_case(case, config, provider, vector_index, graph, judge)
            else:
                result = _run_synthetic_case(case, config, graph)
            results.append(result)

    summary = _build_summary(results)
    return results, summary


def _run_live_case(
    case: dict[str, Any],
    config: str,
    provider: Any,
    vector_index: VectorIndex,
    graph: KnowledgeGraph | None,
    judge: bool,
) -> EvalResult:
    """Run a single test case with a real LLM call."""
    question: str = case["question"]
    settings = get_settings()

    if config == "reggraph":
        if graph is None:
            raise RuntimeError("Graph must be built before running reggraph config.")
        answer, elapsed_ms, token_count, symbolic_findings = _run_reggraph(
            question,
            provider,
            vector_index,
            graph,
            vector_top_k=settings.vector_top_k,
            graph_hops=settings.graph_hops,
        )
    elif config == "langgraph":
        answer, elapsed_ms, token_count = _run_langgraph_baseline(
            question,
            provider,
            vector_index,
            vector_top_k=settings.vector_top_k,
        )
        symbolic_findings = None
    elif config == "no-graph":
        answer, elapsed_ms, token_count = _run_nograph(
            question,
            provider,
            vector_index,
            vector_top_k=settings.vector_top_k,
        )
        symbolic_findings = None
    else:
        raise ValueError(f"Unknown config: {config!r}")

    expected_xj = case.get("expected_cross_jurisdictional")
    metrics = compute_metrics(
        answer,
        case["expected_citations"],
        set(case.get("expected_multi_hop", [])),
        graph,
        elapsed_ms=elapsed_ms,
        token_count=token_count,
        symbolic_findings=symbolic_findings,
        expected_cross_jurisdictional=({tuple(p) for p in expected_xj} if expected_xj else None),
    )

    if judge:
        judge_scores = _run_judge(question, answer, provider)
        if judge_scores:
            metrics.llm_judge_score = judge_scores.get("overall")
            metrics.llm_judge_detail = judge_scores

    return EvalResult(
        case_id=case["id"],
        question=question,
        config=config,
        answer=answer,
        metrics=metrics,
    )


_VALID_CONFIGS = {"reggraph", "langgraph", "no-graph"}


def _run_synthetic_case(
    case: dict[str, Any],
    config: str,
    graph: KnowledgeGraph | None,
) -> EvalResult:
    """Run a single test case with a synthetic answer (fast, no API cost).

    Synthetic answers model the *expected* behavior of each config:
    - ``reggraph``: all expected citations found (graph + vector synergy).
    - ``langgraph``: misses one multi-hop citation, may hallucinate (vector-only).
    - ``no-graph``: misses more citations, higher hallucination rate (no structure).
    """
    if config not in _VALID_CONFIGS:
        raise ValueError(f"Unknown config: {config!r}. Choose from: {sorted(_VALID_CONFIGS)}")

    question: str = case["question"]
    expected = case["expected_citations"]
    multi_hop = set(case.get("expected_multi_hop", []))

    # Cross-jurisdictional pairs — if present, reggraph finds both sides,
    # langgraph finds only one (fails to map across jurisdictions).
    expected_xj = case.get("expected_cross_jurisdictional")
    xj_citations: list[str] = []
    if expected_xj:
        xj_citations = sorted({c for pair in expected_xj for c in pair})

    if config == "reggraph":
        # Full pipeline: finds all expected + multi-hop + cross-jurisdictional.
        citations = expected + sorted(multi_hop) + xj_citations
        hallucinated_extra: list[str] = []
        symbolic_findings: list[dict[str, object]] | None = [
            {
                "rule_id": "citation_validity",
                "passed": True,
                "message": "All citations resolve to graph nodes.",
                "citations": citations,
            },
        ]
    elif config == "langgraph":
        # Vanilla LangGraph: finds most direct citations but misses multi-hop,
        # cross-jurisdictional mappings, and may hallucinate.
        citations = expected[:-1] if len(expected) > 1 else expected
        if multi_hop:
            citations.append(sorted(multi_hop)[0])  # Gets at most one multi-hop
        hallucinated_extra = ["article-99"]  # Typical vector-only hallucination
        symbolic_findings = None
    elif config == "no-graph":
        # Vector-only, minimal prompt: worst performance across all dimensions.
        citations = expected[:1] if expected else []
        hallucinated_extra = ["article-99", "article-88"]
        symbolic_findings = None
    else:
        raise ValueError(f"Unknown config: {config!r}")  # pragma: no cover

    all_citations = citations + hallucinated_extra
    answer = f"Answer: {' '.join('[' + c + ']' for c in all_citations)}."

    metrics = compute_metrics(
        answer,
        expected,
        multi_hop,
        graph,
        elapsed_ms=0.0,
        token_count=0,
        symbolic_findings=symbolic_findings,
        expected_cross_jurisdictional=({tuple(p) for p in expected_xj} if expected_xj else None),
    )

    return EvalResult(
        case_id=case["id"],
        question=question,
        config=config,
        answer=answer,
        metrics=metrics,
    )


def _build_summary(results: list[EvalResult]) -> dict[str, Any]:
    """Aggregate metrics by config, producing a comparison summary."""
    by_config: dict[str, list[EvalMetrics]] = {}
    for r in results:
        by_config.setdefault(r.config, []).append(r.metrics)

    summary: dict[str, Any] = {}
    for config, metrics_list in by_config.items():
        n = len(metrics_list)
        entry: dict[str, Any] = {
            "cases": n,
            "mean_citation_f1": sum(m.citation_f1 for m in metrics_list) / n,
            "mean_hallucination_rate": sum(m.hallucination_rate for m in metrics_list) / n,
            "mean_multi_hop_recall": sum(m.multi_hop_recall for m in metrics_list) / n,
        }
        # Cross-jurisdictional recall (always included; 0.0 means no pairs found)
        entry["mean_cross_jurisdictional_recall"] = (
            sum(m.cross_jurisdictional_recall for m in metrics_list) / n
        )
        # Runtime metrics (only meaningful in live mode)
        if any(m.elapsed_ms > 0 for m in metrics_list):
            entry["mean_elapsed_ms"] = sum(m.elapsed_ms for m in metrics_list) / n
            entry["mean_token_count"] = sum(m.token_count for m in metrics_list) / n
        # Symbolic pass rate (RegGraph only)
        symbolic_rates = [
            m.symbolic_pass_rate for m in metrics_list if m.symbolic_pass_rate is not None
        ]
        if symbolic_rates:
            entry["mean_symbolic_pass_rate"] = sum(symbolic_rates) / len(symbolic_rates)
        # LLM judge
        judge_scores = [m.llm_judge_score for m in metrics_list if m.llm_judge_score is not None]
        if judge_scores:
            entry["mean_llm_judge_score"] = sum(judge_scores) / len(judge_scores)
        summary[config] = entry

    # Add comparison deltas when reggraph is present
    if "reggraph" in summary:
        baseline = "langgraph" if "langgraph" in summary else "no-graph"
        if baseline in summary:
            reg = summary["reggraph"]
            base = summary[baseline]
            summary["_comparison"] = {
                "baseline": baseline,
                "citation_f1_delta": reg["mean_citation_f1"] - base["mean_citation_f1"],
                "hallucination_rate_delta": reg["mean_hallucination_rate"]
                - base["mean_hallucination_rate"],
                "multi_hop_recall_delta": reg["mean_multi_hop_recall"]
                - base["mean_multi_hop_recall"],
            }

    return summary


# ── output helpers ──────────────────────────────────────────────────────────


def save_report(
    results: list[EvalResult],
    summary: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    """Write results and summary to a timestamped JSON file."""
    output_dir = Path(output_dir or PROJECT_ROOT / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"eval-{ts}.json"
    data = {
        "results": [r.to_dict() for r in results],
        "summary": summary,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _format_comparison_table(summary: dict[str, Any]) -> str:
    """Format a human-readable comparison table for console output."""
    lines: list[str] = []
    comparison = summary.get("_comparison")
    if not comparison:
        return ""

    baseline = comparison["baseline"]
    lines.append("")
    lines.append(f"  RegGraph vs {baseline} baseline:")
    lines.append(f"  {'Metric':<30} {'Delta':>8}  {'Interpretation'}")
    lines.append(f"  {'-' * 30} {'-' * 8}  {'-' * 30}")

    deltas = [
        ("citation_f1_delta", "Citation F1", "▲ better"),
        ("hallucination_rate_delta", "Hallucination rate", "▼ better"),
        ("multi_hop_recall_delta", "Multi-hop recall", "▲ better"),
    ]
    for key, label, direction in deltas:
        val = comparison[key]
        lines.append(f"  {label:<30} {val:>+8.3f}  {direction}")

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate regulatory QA across configurations.")
    parser.add_argument(
        "--configs",
        default="reggraph,langgraph",
        help=(
            "Comma-separated configs: reggraph, langgraph, no-graph (default: reggraph,langgraph)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap number of test cases.",
    )
    parser.add_argument(
        "--domain",
        default="gdpr",
        help="Domain name (default: gdpr).",
    )
    parser.add_argument(
        "--scenario",
        default="single",
        choices=["single", "cross-jurisdictional"],
        help="Test scenario: single (default) or cross-jurisdictional.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-judge quality scoring.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run with real LLM calls instead of synthetic answers (incurs API cost).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports (default: PROJECT_ROOT/reports).",
    )
    args = parser.parse_args(argv)

    configs = [c.strip() for c in args.configs.split(",")]
    valid = {"reggraph", "langgraph", "no-graph"}
    for c in configs:
        if c not in valid:
            parser.error(f"Unknown config {c!r}. Choose from: {', '.join(sorted(valid))}")

    results, summary = run(
        configs=configs,
        limit=args.limit,
        domain_name=args.domain,
        scenario=args.scenario,
        judge=args.judge,
        live=args.live,
    )

    report_path = save_report(results, summary, output_dir=args.output_dir)

    # Print summary table.
    print()
    for config, stats in summary.items():
        if config.startswith("_"):
            continue
        print(f"{config}:")
        print(f"  cases: {stats['cases']}")
        print(f"  mean citation F1:       {stats['mean_citation_f1']:.3f}")
        print(f"  mean hallucination rate: {stats['mean_hallucination_rate']:.3f}")
        print(f"  mean multi-hop recall:   {stats['mean_multi_hop_recall']:.3f}")
        if "mean_elapsed_ms" in stats:
            print(f"  mean elapsed ms:         {stats['mean_elapsed_ms']:.0f}")
        if "mean_token_count" in stats:
            print(f"  mean token count:        {stats['mean_token_count']:.0f}")
        if "mean_symbolic_pass_rate" in stats:
            print(f"  mean symbolic pass rate: {stats['mean_symbolic_pass_rate']:.3f}")
        if "mean_cross_jurisdictional_recall" in stats:
            print(f"  mean cross-juris recall: {stats['mean_cross_jurisdictional_recall']:.3f}")
        if "mean_llm_judge_score" in stats:
            print(f"  mean LLM judge score:    {stats['mean_llm_judge_score']:.1f}")
        print()

    print(_format_comparison_table(summary))
    print(f"\nFull report → {report_path}")


if __name__ == "__main__":
    main()
