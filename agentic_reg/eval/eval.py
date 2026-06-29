"""Evaluation smoke harness for the Phase 0/1 agent stack.

Default runs are deterministic fixture checks for metric plumbing only. They
are explicitly marked as non-benchmark output. Use ``--live`` to run the real
single/team orchestrators against a built local store.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_reg._internal import extract_citations, parse_json_object
from agentic_reg.agents import get_orchestrator
from agentic_reg.config import PROJECT_ROOT, Settings, get_settings
from agentic_reg.domains import Domain, get_domain
from agentic_reg.knowledge.graph import KnowledgeGraph
from agentic_reg.knowledge.symbolic import run_symbolic_checks
from agentic_reg.knowledge.vectors import VectorIndex
from agentic_reg.providers import get_provider
from agentic_reg.providers.base import LLMProvider

VALID_CONFIGS = {"single", "team", "team-no-graph"}
CONFIG_ALIASES = {
    "reggraph": "team",
    "langgraph": "single",
    "no-graph": "team-no-graph",
}

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


@dataclass
class EvalMetrics:
    citation_f1: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    hallucination_rate: float = 0.0
    hallucinated: list[str] = field(default_factory=list)
    multi_hop_recall: float = 0.0
    elapsed_ms: float = 0.0
    token_count: int = 0
    symbolic_pass_rate: float | None = None
    symbolic_findings: list[dict[str, object]] = field(default_factory=list)
    llm_judge_score: float | None = None
    llm_judge_detail: dict[str, float] | None = None


@dataclass
class EvalResult:
    case_id: str
    question: str
    config: str
    answer: str = ""
    mode: str = "fixture"
    metrics: EvalMetrics = field(default_factory=EvalMetrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_citations(text: str) -> list[str]:
    """Backward-compatible wrapper around the shared citation parser."""
    return extract_citations(text)


def canonical_config(config: str) -> str:
    """Return the canonical Phase 1 config name."""
    normalized = config.strip().lower()
    normalized = CONFIG_ALIASES.get(normalized, normalized)
    if normalized not in VALID_CONFIGS:
        raise ValueError(f"Unknown config: {config!r}. Choose from: {sorted(VALID_CONFIGS)}")
    return normalized


def _canonical_configs(configs: list[str] | None) -> list[str]:
    requested = configs or ["single", "team"]
    canonical: list[str] = []
    for config in requested:
        name = canonical_config(config)
        if name not in canonical:
            canonical.append(name)
    return canonical


def compute_metrics(
    answer_text: str,
    expected_citations: list[str],
    expected_multi_hop: set[str],
    graph: KnowledgeGraph | None,
    *,
    elapsed_ms: float = 0.0,
    token_count: int = 0,
    symbolic_findings: list[dict[str, object]] | None = None,
) -> EvalMetrics:
    """Compute deterministic citation and graph-recall metrics."""
    cited = extract_citations(answer_text)
    cited_set = set(cited)
    expected_set = set(expected_citations)

    tp = cited_set & expected_set
    precision = len(tp) / len(cited_set) if cited_set else 0.0
    recall = len(tp) / len(expected_set) if expected_set else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    hallucinated: list[str] = []
    if graph is not None:
        hallucinated = [citation for citation in cited if not graph.has_node(citation)]
    hallucination_rate = len(hallucinated) / len(cited) if cited else 0.0

    multi_hop_hit = len(cited_set & expected_multi_hop)
    multi_hop_recall = multi_hop_hit / len(expected_multi_hop) if expected_multi_hop else 1.0

    symbolic_pass_rate: float | None = None
    if symbolic_findings is not None:
        passed = sum(1 for item in symbolic_findings if item.get("passed", False))
        symbolic_pass_rate = passed / len(symbolic_findings) if symbolic_findings else 1.0

    return EvalMetrics(
        citation_f1=f1,
        citation_precision=precision,
        citation_recall=recall,
        hallucination_rate=hallucination_rate,
        hallucinated=hallucinated,
        multi_hop_recall=multi_hop_recall,
        elapsed_ms=elapsed_ms,
        token_count=token_count,
        symbolic_pass_rate=symbolic_pass_rate,
        symbolic_findings=symbolic_findings or [],
    )


def run(
    *,
    configs: list[str] | None = None,
    limit: int | None = None,
    domain_name: str = "gdpr",
    judge: bool = False,
    live: bool = False,
) -> tuple[list[EvalResult], dict[str, Any]]:
    """Run fixture or live eval for canonical Phase 1 configs."""
    canonical = _canonical_configs(configs)
    cases = TEST_CASES[:limit] if limit else TEST_CASES

    graph: KnowledgeGraph | None = None
    provider: LLMProvider | None = None
    vector_index: VectorIndex | None = None
    settings: Settings | None = None

    if live:
        settings, provider, vector_index, graph = _load_live_dependencies(domain_name)
    else:
        graph = _load_graph_if_present(domain_name)

    results: list[EvalResult] = []
    for config in canonical:
        for case in cases:
            if live:
                assert settings is not None and provider is not None and vector_index is not None
                assert graph is not None
                result = _run_live_case(
                    case, config, settings, provider, vector_index, graph, judge
                )
            else:
                result = _run_fixture_case(case, config, graph)
            results.append(result)

    return results, _build_summary(results)


def _load_graph_if_present(domain_name: str) -> KnowledgeGraph | None:
    domain = get_domain(domain_name)
    try:
        return KnowledgeGraph.load(domain.graph_path)
    except FileNotFoundError:
        return None


def _load_live_dependencies(
    domain_name: str,
) -> tuple[Settings, LLMProvider, VectorIndex, KnowledgeGraph]:
    settings = get_settings()
    domain = get_domain(domain_name)
    settings.domain = domain.name

    if not domain.graph_path.exists():
        raise RuntimeError(_store_error(domain, "graph.json"))
    if not domain.chroma_dir.exists():
        raise RuntimeError(_store_error(domain, "chroma/"))

    graph = KnowledgeGraph.load(domain.graph_path)
    try:
        vector_index = VectorIndex.load(domain.chroma_dir, settings.embedding_model)
    except Exception as exc:
        raise RuntimeError(f"Could not load vector store at {domain.chroma_dir}: {exc}") from exc
    if vector_index.count() == 0:
        raise RuntimeError(_store_error(domain, "non-empty Chroma collection"))

    try:
        provider = get_provider(settings)
    except Exception as exc:
        raise RuntimeError(f"Could not initialize LLM provider for live eval: {exc}") from exc

    return settings, provider, vector_index, graph


def _store_error(domain: Domain, missing: str) -> str:
    return (
        f"Live eval requires a built {domain.name!r} store with {missing}. "
        f"Run `uv run python -m agentic_reg.build --domain {domain.name} --no-enrich` first."
    )


def _settings_for_config(settings: Settings, config: str) -> Settings:
    configured = settings.model_copy()
    configured.agent_mode = "single" if config == "single" else "team"
    configured.use_graph = config != "team-no-graph"
    configured.graph_update_mode = "off"
    return configured


def _run_live_case(
    case: dict[str, Any],
    config: str,
    settings: Settings,
    provider: LLMProvider,
    vector_index: VectorIndex,
    graph: KnowledgeGraph,
    judge: bool,
) -> EvalResult:
    question = str(case["question"])
    configured = _settings_for_config(settings, config)

    t0 = time.perf_counter()
    trace = get_orchestrator(configured, provider, vector_index, graph).answer(question)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    token_count = (len(trace.to_json()) + len(question)) // 4

    symbolic_findings = [
        item.to_dict() for item in run_symbolic_checks(question, trace.answer, graph)
    ]
    metrics = compute_metrics(
        trace.answer,
        list(case["expected_citations"]),
        set(case.get("expected_multi_hop", [])),
        graph,
        elapsed_ms=elapsed_ms,
        token_count=token_count,
        symbolic_findings=symbolic_findings,
    )

    if judge:
        scores = _run_judge(question, trace.answer, provider)
        if scores:
            metrics.llm_judge_score = scores.get("overall")
            metrics.llm_judge_detail = scores

    return EvalResult(
        case_id=str(case["id"]),
        question=question,
        config=config,
        answer=trace.answer,
        mode="live",
        metrics=metrics,
    )


def _run_fixture_case(
    case: dict[str, Any],
    config: str,
    graph: KnowledgeGraph | None,
) -> EvalResult:
    """Run a deterministic metric smoke case.

    The same fixture answer is used for every config so default output cannot be
    mistaken for a model comparison.
    """
    expected = list(case["expected_citations"])
    answer = "Fixture answer: " + " ".join(f"[{citation}]" for citation in expected) + "."
    symbolic_findings = [
        {
            "rule_id": "fixture_mode",
            "passed": True,
            "message": "Fixture mode validates metric plumbing only; it is not a benchmark.",
            "citations": expected,
        }
    ]
    metrics = compute_metrics(
        answer,
        expected,
        set(case.get("expected_multi_hop", [])),
        graph,
        symbolic_findings=symbolic_findings,
    )
    return EvalResult(
        case_id=str(case["id"]),
        question=str(case["question"]),
        config=config,
        answer=answer,
        mode="fixture",
        metrics=metrics,
    )


def _run_judge(question: str, answer: str, provider: LLMProvider) -> dict[str, float]:
    prompt = f"""\
Grade this regulatory answer from 0-10 for correctness, completeness, citation_quality,
clarity, and overall. Return JSON only.

Question: {question}
Answer: {answer}
"""
    try:
        data = parse_json_object(provider.complete(prompt, system="You are a strict evaluator."))
    except Exception:
        return {}
    scores: dict[str, float] = {}
    for key in ("correctness", "completeness", "citation_quality", "clarity", "overall"):
        value = data.get(key)
        if isinstance(value, int | float):
            scores[key] = float(value)
    return scores


def _build_summary(results: list[EvalResult]) -> dict[str, Any]:
    by_config: dict[str, list[EvalMetrics]] = {}
    for result in results:
        by_config.setdefault(result.config, []).append(result.metrics)

    modes = {result.mode for result in results}
    mode = "live" if modes == {"live"} else "fixture"
    summary: dict[str, Any] = {
        "_metadata": {
            "mode": mode,
            "benchmark": mode == "live",
            "notice": (
                "Live eval ran the configured orchestrators."
                if mode == "live"
                else "Fixture mode validates metrics only; do not use as a benchmark."
            ),
        }
    }

    for config, metrics_list in by_config.items():
        n = len(metrics_list)
        entry: dict[str, Any] = {
            "cases": n,
            "mean_citation_f1": sum(m.citation_f1 for m in metrics_list) / n,
            "mean_hallucination_rate": sum(m.hallucination_rate for m in metrics_list) / n,
            "mean_multi_hop_recall": sum(m.multi_hop_recall for m in metrics_list) / n,
        }
        if any(m.elapsed_ms > 0 for m in metrics_list):
            entry["mean_elapsed_ms"] = sum(m.elapsed_ms for m in metrics_list) / n
            entry["mean_token_count"] = sum(m.token_count for m in metrics_list) / n
        symbolic_rates = [
            m.symbolic_pass_rate for m in metrics_list if m.symbolic_pass_rate is not None
        ]
        if symbolic_rates:
            entry["mean_symbolic_pass_rate"] = sum(symbolic_rates) / len(symbolic_rates)
        judge_scores = [m.llm_judge_score for m in metrics_list if m.llm_judge_score is not None]
        if judge_scores:
            entry["mean_llm_judge_score"] = sum(judge_scores) / len(judge_scores)
        summary[config] = entry

    if mode == "live" and "team" in summary:
        baseline = "single" if "single" in summary else "team-no-graph"
        if baseline in summary:
            team = summary["team"]
            base = summary[baseline]
            summary["_comparison"] = {
                "baseline": baseline,
                "citation_f1_delta": team["mean_citation_f1"] - base["mean_citation_f1"],
                "hallucination_rate_delta": team["mean_hallucination_rate"]
                - base["mean_hallucination_rate"],
                "multi_hop_recall_delta": team["mean_multi_hop_recall"]
                - base["mean_multi_hop_recall"],
            }

    return summary


def save_report(
    results: list[EvalResult],
    summary: dict[str, Any],
    output_dir: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir or PROJECT_ROOT / "reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"eval-{ts}.json"
    data = {
        "metadata": summary.get("_metadata", {}),
        "results": [result.to_dict() for result in results],
        "summary": summary,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _format_comparison_table(summary: dict[str, Any]) -> str:
    comparison = summary.get("_comparison")
    if not comparison:
        return ""
    baseline = comparison["baseline"]
    lines = [
        "",
        f"  team vs {baseline}:",
        f"  {'Metric':<30} {'Delta':>8}",
        f"  {'-' * 30} {'-' * 8}",
    ]
    for key, label in [
        ("citation_f1_delta", "Citation F1"),
        ("hallucination_rate_delta", "Hallucination rate"),
        ("multi_hop_recall_delta", "Multi-hop recall"),
    ]:
        lines.append(f"  {label:<30} {comparison[key]:>+8.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 0/1 regulatory QA configs.")
    parser.add_argument(
        "--configs",
        default="single,team",
        help="Comma-separated configs: single, team, team-no-graph (default: single,team).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of test cases.")
    parser.add_argument("--domain", default="gdpr", help="Domain name (default: gdpr).")
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge in live mode.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run real orchestrators instead of deterministic fixture mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for reports (default: PROJECT_ROOT/reports).",
    )
    args = parser.parse_args(argv)

    try:
        configs = _canonical_configs([config.strip() for config in args.configs.split(",")])
    except ValueError as exc:
        parser.error(str(exc))

    if args.judge and not args.live:
        parser.error("--judge requires --live")

    results, summary = run(
        configs=configs,
        limit=args.limit,
        domain_name=args.domain,
        judge=args.judge,
        live=args.live,
    )
    report_path = save_report(results, summary, output_dir=args.output_dir)

    metadata = summary["_metadata"]
    print()
    print(f"mode: {metadata['mode']} (benchmark: {str(metadata['benchmark']).lower()})")
    print(metadata["notice"])
    for config, stats in summary.items():
        if config.startswith("_"):
            continue
        print(f"\n{config}:")
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
        if "mean_llm_judge_score" in stats:
            print(f"  mean LLM judge score:    {stats['mean_llm_judge_score']:.1f}")
    comparison = _format_comparison_table(summary)
    if comparison:
        print(comparison)
    print(f"\nFull report -> {report_path}")


if __name__ == "__main__":
    main()
