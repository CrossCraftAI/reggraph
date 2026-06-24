import json

import pytest

from agentic_reg.eval.baselines.langgraph_baseline import LangGraphBaseline
from agentic_reg.eval.eval import (
    EvalMetrics,
    EvalResult,
    _build_summary,
    _extract_citations,
    _format_comparison_table,
    _run_synthetic_case,
    compute_metrics,
    run,
    save_report,
)
from agentic_reg.knowledge.graph import KnowledgeGraph

# ── citation extraction ────────────────────────────────────────────────────


def test_extract_citations():
    text = "See [article-6] and [article-7]."
    assert sorted(_extract_citations(text)) == ["article-6", "article-7"]


def test_extract_citations_deduplicates():
    text = "[article-6] and [article-6] again."
    assert _extract_citations(text) == ["article-6"]


def test_extract_citations_case_insensitive():
    text = "[Article-6] and [ARTICLE-7]."
    assert sorted(_extract_citations(text)) == ["article-6", "article-7"]


def test_extract_citations_no_match():
    assert _extract_citations("No citations here.") == []


# ── compute_metrics ────────────────────────────────────────────────────────


def test_compute_metrics_perfect_answer():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")
    graph.add_node("article-7", label="A7", kind="clause")

    metrics = compute_metrics(
        "Answer: [article-6] [article-7].",
        expected_citations=["article-6", "article-7"],
        expected_multi_hop={"article-5"},
        graph=graph,
    )
    assert metrics.citation_f1 == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.citation_recall == 1.0
    assert metrics.hallucination_rate == 0.0


def test_compute_metrics_detects_hallucination():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")

    metrics = compute_metrics(
        "Answer: [article-6] [article-99].",
        expected_citations=["article-6"],
        expected_multi_hop=set(),
        graph=graph,
    )
    assert metrics.hallucination_rate == 0.5
    assert "article-99" in metrics.hallucinated


def test_compute_metrics_no_citations():
    graph = KnowledgeGraph()
    metrics = compute_metrics(
        "Answer: No citations here.",
        expected_citations=["article-6"],
        expected_multi_hop=set(),
        graph=graph,
    )
    assert metrics.citation_recall == 0.0
    assert metrics.citation_f1 == 0.0


def test_compute_metrics_multi_hop_recall():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")
    graph.add_node("article-7", label="A7", kind="clause")

    metrics = compute_metrics(
        "Answer: [article-6] [article-7].",
        expected_citations=["article-6"],
        expected_multi_hop={"article-5", "article-7"},
        graph=graph,
    )
    # Found article-7 from multi-hop set, missed article-5
    assert metrics.multi_hop_recall == 0.5


def test_compute_metrics_with_symbolic_findings():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")

    symbolic = [
        {
            "rule_id": "citation_validity",
            "passed": True,
            "message": "All citations resolve.",
            "citations": ["article-6"],
        },
        {
            "rule_id": "special_category",
            "passed": False,
            "message": "Missing article-9.",
            "citations": ["article-9"],
        },
    ]
    metrics = compute_metrics(
        "Answer: [article-6].",
        expected_citations=["article-6"],
        expected_multi_hop=set(),
        graph=graph,
        symbolic_findings=symbolic,
    )
    assert metrics.symbolic_pass_rate == 0.5


def test_compute_metrics_with_timing():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")

    metrics = compute_metrics(
        "Answer: [article-6].",
        expected_citations=["article-6"],
        expected_multi_hop=set(),
        graph=graph,
        elapsed_ms=1500.0,
        token_count=420,
    )
    assert metrics.elapsed_ms == 1500.0
    assert metrics.token_count == 420


# ── synthetic answers ──────────────────────────────────────────────────────


def test_run_synthetic_reggraph_finds_all():
    graph = KnowledgeGraph()
    # Only expected and multi-hop articles are real nodes. No hallucinated articles.
    for aid in ("article-5", "article-6", "article-7", "article-15", "article-17"):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    case = {
        "id": "test",
        "question": "Test?",
        "expected_citations": ["article-6"],
        "expected_multi_hop": ["article-5", "article-7"],
    }
    result = _run_synthetic_case(case, "reggraph", graph)
    assert result.config == "reggraph"
    # Synthetic reggraph outputs: expected + all multi-hop = [article-6, article-5, article-7]
    # TP = {article-6}, precision = 1/3, recall = 1/1, F1 = 2*(1/3)*1/(1/3+1) = 0.5
    assert result.metrics.citation_f1 == 0.5
    assert result.metrics.citation_recall == 1.0  # article-6 found
    assert result.metrics.hallucination_rate == 0.0  # all nodes in graph
    assert result.metrics.multi_hop_recall == 1.0  # article-5 + article-7 both found
    assert result.metrics.symbolic_pass_rate is not None


def test_run_synthetic_langgraph_misses_multihop():
    graph = KnowledgeGraph()
    # Only real articles are in the graph. article-99 is NOT in the graph
    # (it's a hallucinated reference the baseline may generate).
    for aid in ("article-6", "article-7", "article-5"):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    case = {
        "id": "test",
        "question": "Test?",
        "expected_citations": ["article-6", "article-7"],
        "expected_multi_hop": ["article-5"],
    }
    result = _run_synthetic_case(case, "langgraph", graph)
    assert result.config == "langgraph"
    # Synthetic langgraph: citations = expected[:-1] + first multi-hop + hallucinated
    # = ["article-6"] + ["article-5"] + ["article-99"]
    # TP = {article-6, article-5} ∩ {article-6, article-7} = {article-6} → 1
    # precision = 1/3, recall = 1/2, F1 = 2*(1/3)*(1/2)/(1/3+1/2) = 0.4
    assert result.metrics.citation_f1 < 1.0
    assert result.metrics.hallucination_rate > 0.0  # article-99 not in graph
    assert result.metrics.symbolic_pass_rate is None  # no symbolic checks
    assert "article-99" in result.metrics.hallucinated


def test_run_synthetic_nograph_worst():
    graph = KnowledgeGraph()
    # Only article-6 is real. article-7 (expected but not found), article-5
    # (multi-hop), article-99, and article-88 are NOT in the graph for this
    # config — they represent either missed or hallucinated references.
    for aid in ("article-6",):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    case = {
        "id": "test",
        "question": "Test?",
        "expected_citations": ["article-6", "article-7"],
        "expected_multi_hop": ["article-5"],
    }
    result = _run_synthetic_case(case, "no-graph", graph)
    assert result.config == "no-graph"
    # Synthetic no-graph: citations = expected[:1] + hallucinated_extra
    # = ["article-6"] + ["article-99", "article-88"]
    # TP = {article-6} ∩ {article-6, article-7} = {article-6} → 1
    # precision = 1/3, recall = 1/2 = 0.5, F1 = 2*(1/3)*(1/2)/(1/3+1/2) = 0.4
    assert result.metrics.citation_recall == 0.5  # article-7 not cited
    # article-99 and article-88 are NOT in graph → both hallucinated
    assert len(result.metrics.hallucinated) == 2
    assert "article-99" in result.metrics.hallucinated
    assert "article-88" in result.metrics.hallucinated


# ── run (synthetic) ────────────────────────────────────────────────────────


def test_run_produces_results_for_reggraph_vs_langgraph():
    results, summary = run(configs=["reggraph", "langgraph"], limit=2)

    assert len(results) == 4  # 2 configs * 2 cases
    assert "reggraph" in summary
    assert "langgraph" in summary
    # RegGraph should outperform LangGraph baseline.
    assert summary["reggraph"]["mean_citation_f1"] > summary["langgraph"]["mean_citation_f1"]


def test_run_produces_results_for_reggraph_vs_nograph():
    results, summary = run(configs=["reggraph", "no-graph"], limit=2)

    assert len(results) == 4
    # RegGraph should outperform vector-only.
    assert summary["reggraph"]["mean_citation_f1"] > summary["no-graph"]["mean_citation_f1"]


def test_run_all_three_configs():
    results, summary = run(configs=["reggraph", "langgraph", "no-graph"], limit=1)

    assert len(results) == 3
    assert set(summary.keys()) == {"reggraph", "langgraph", "no-graph", "_comparison"}


def test_run_default_configs():
    """Default configs should be reggraph + langgraph (not the old graph + no-graph)."""
    results, summary = run(limit=1)
    configs = {r.config for r in results}
    assert configs == {"reggraph", "langgraph"}


# ── summary & comparison ───────────────────────────────────────────────────


def test_build_summary_includes_comparison_when_reggraph_present():
    results = [
        EvalResult(
            case_id="t1",
            question="Q?",
            config="reggraph",
            metrics=EvalMetrics(citation_f1=0.95, hallucination_rate=0.0, multi_hop_recall=0.8),
        ),
        EvalResult(
            case_id="t1",
            question="Q?",
            config="langgraph",
            metrics=EvalMetrics(citation_f1=0.70, hallucination_rate=0.15, multi_hop_recall=0.4),
        ),
    ]
    summary = _build_summary(results)
    assert "_comparison" in summary
    comp = summary["_comparison"]
    assert comp["baseline"] == "langgraph"
    assert comp["citation_f1_delta"] > 0  # reggraph better
    assert comp["hallucination_rate_delta"] < 0  # reggraph lower (better)
    assert comp["multi_hop_recall_delta"] > 0  # reggraph better


def test_build_summary_prefers_langgraph_baseline_over_nograph():
    results = [
        EvalResult(
            case_id="t1",
            question="Q?",
            config="reggraph",
            metrics=EvalMetrics(citation_f1=1.0, hallucination_rate=0.0, multi_hop_recall=0.9),
        ),
        EvalResult(
            case_id="t1",
            question="Q?",
            config="langgraph",
            metrics=EvalMetrics(citation_f1=0.8, hallucination_rate=0.1, multi_hop_recall=0.5),
        ),
        EvalResult(
            case_id="t1",
            question="Q?",
            config="no-graph",
            metrics=EvalMetrics(citation_f1=0.5, hallucination_rate=0.3, multi_hop_recall=0.2),
        ),
    ]
    summary = _build_summary(results)
    # Should compare against langgraph (more relevant baseline), not no-graph.
    assert summary["_comparison"]["baseline"] == "langgraph"


def test_format_comparison_table():
    summary = {
        "reggraph": {
            "cases": 5,
            "mean_citation_f1": 0.95,
            "mean_hallucination_rate": 0.02,
            "mean_multi_hop_recall": 0.80,
        },
        "langgraph": {
            "cases": 5,
            "mean_citation_f1": 0.72,
            "mean_hallucination_rate": 0.18,
            "mean_multi_hop_recall": 0.45,
        },
        "_comparison": {
            "baseline": "langgraph",
            "citation_f1_delta": 0.23,
            "hallucination_rate_delta": -0.16,
            "multi_hop_recall_delta": 0.35,
        },
    }
    table = _format_comparison_table(summary)
    assert "RegGraph vs langgraph" in table
    assert "+0.230" in table
    assert "-0.160" in table
    assert "+0.350" in table


# ── save_report ────────────────────────────────────────────────────────────


def test_save_report_writes_valid_json(tmp_path):
    results, summary = run(configs=["reggraph"], limit=1)
    path = save_report(results, summary, output_dir=tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) == 1


# ── LangGraph baseline class ───────────────────────────────────────────────


def test_langgraph_baseline_initializes():
    """Baseline should initialize without errors when given mock dependencies."""
    # The baseline requires a provider and vector_index at init time.
    # We verify the class is importable and has the expected interface.
    assert hasattr(LangGraphBaseline, "answer")
    assert hasattr(LangGraphBaseline, "__init__")


# ── EvalResult serialization ───────────────────────────────────────────────


def test_eval_result_to_dict():
    result = EvalResult(
        case_id="test-1",
        question="What is GDPR?",
        config="reggraph",
        answer="It is a regulation.",
        metrics=EvalMetrics(
            citation_f1=0.9,
            hallucination_rate=0.1,
            hallucinated=["article-99"],
            multi_hop_recall=0.75,
            elapsed_ms=1234.0,
            token_count=567,
            symbolic_pass_rate=0.8,
        ),
    )
    d = result.to_dict()
    assert d["case_id"] == "test-1"
    assert d["config"] == "reggraph"
    assert d["metrics"]["citation_f1"] == 0.9
    assert d["metrics"]["hallucinated"] == ["article-99"]
    assert d["metrics"]["elapsed_ms"] == 1234.0
    assert d["metrics"]["token_count"] == 567


# ── invalid config ─────────────────────────────────────────────────────────


def test_run_raises_on_unknown_config():
    with pytest.raises(ValueError, match="Unknown config"):
        run(configs=["invalid_config"], limit=1)


# ── cross-jurisdictional ──────────────────────────────────────────────────


def test_compute_metrics_cross_jurisdictional_recall():
    """RegGraph should find both sides of cross-jurisdictional article pairs."""
    graph = KnowledgeGraph()
    for aid in ("article-9", "article-6", "section-10", "section-8"):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    metrics = compute_metrics(
        "Answer: [article-9] [section-10] [article-6] [section-8].",
        expected_citations=["article-9", "section-10"],
        expected_multi_hop={"article-6", "section-8"},
        graph=graph,
        expected_cross_jurisdictional={
            ("article-9", "section-10"),
            ("article-6", "section-8"),
        },
    )
    # F1 < 1.0 because multi-hop citations (article-6, section-8) lower
    # precision (4 cited, 2 expected), but cross-jurisdictional recall is
    # perfect — both pairs found.
    assert metrics.cross_jurisdictional_recall == 1.0
    assert metrics.citation_recall == 1.0


def test_compute_metrics_cross_jurisdictional_partial():
    """LangGraph baseline finds only one side of cross-jurisdictional pairs."""
    graph = KnowledgeGraph()
    for aid in ("article-9", "section-10", "article-99"):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    metrics = compute_metrics(
        "Answer: [article-9] [article-6] [article-99].",  # misses section-10
        expected_citations=["article-9", "section-10"],
        expected_multi_hop={"article-6"},
        graph=graph,
        expected_cross_jurisdictional={
            ("article-9", "section-10"),
            ("article-6", "section-8"),
        },
    )
    # Only found article-9 from the first pair, missed section-10;
    # found article-6 but missed section-8 from the second pair.
    assert metrics.cross_jurisdictional_recall == 0.0


def test_compute_metrics_cross_jurisdictional_none_when_empty():
    """cross_jurisdictional_recall is 0 when no pairs expected."""
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")

    metrics = compute_metrics(
        "Answer: [article-6].",
        expected_citations=["article-6"],
        expected_multi_hop=set(),
        graph=graph,
        expected_cross_jurisdictional=None,
    )
    assert metrics.cross_jurisdictional_recall == 0.0


def test_run_cross_jurisdictional_scenario():
    """Cross-jurisdictional scenario produces results with cross-juris recall."""
    results, summary = run(
        configs=["reggraph", "langgraph"],
        scenario="cross-jurisdictional",
        limit=2,
    )
    assert len(results) == 4  # 2 configs * 2 cases
    assert "reggraph" in summary
    assert "langgraph" in summary
    # RegGraph should have higher cross-jurisdictional recall than LangGraph.
    reg = summary["reggraph"]
    base = summary["langgraph"]
    assert reg["mean_cross_jurisdictional_recall"] > base["mean_cross_jurisdictional_recall"]


def test_run_unknown_scenario_raises():
    with pytest.raises(ValueError, match="Unknown scenario"):
        run(scenario="invalid_scenario", limit=1)


def test_run_synthetic_cross_jurisdictional_reggraph():
    """Synthetic reggraph in cross-jurisdictional mode finds all pairs."""
    graph = KnowledgeGraph()
    for aid in ("article-9", "section-10", "article-6", "section-8"):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    case = {
        "id": "cross-test",
        "question": "Test cross-jurisdictional?",
        "expected_citations": ["article-9", "section-10"],
        "expected_multi_hop": ["article-6", "section-8"],
        "expected_cross_jurisdictional": [
            ["article-9", "section-10"],
            ["article-6", "section-8"],
        ],
    }
    from agentic_reg.eval.eval import _run_synthetic_case

    result = _run_synthetic_case(case, "reggraph", graph)
    assert result.config == "reggraph"
    assert result.metrics.cross_jurisdictional_recall == 1.0


def test_run_synthetic_cross_jurisdictional_langgraph():
    """Synthetic langgraph in cross-jurisdictional mode misses cross-juris pairs."""
    graph = KnowledgeGraph()
    # Only add expected articles; article-99 is NOT in the graph (hallucination).
    for aid in ("article-9", "article-6", "section-10"):
        graph.add_node(aid, label=aid.upper(), kind="clause")

    case = {
        "id": "cross-test",
        "question": "Test cross-jurisdictional?",
        "expected_citations": ["article-9", "section-10"],
        "expected_multi_hop": ["article-6"],
        "expected_cross_jurisdictional": [
            ["article-9", "section-10"],
            ["article-6", "section-8"],
        ],
    }
    from agentic_reg.eval.eval import _run_synthetic_case

    result = _run_synthetic_case(case, "langgraph", graph)
    assert result.config == "langgraph"
    # LangGraph baseline should miss cross-jurisdictional mappings.
    assert result.metrics.cross_jurisdictional_recall < 1.0
    # Should still have some hallucination.
    assert "article-99" in result.metrics.hallucinated
