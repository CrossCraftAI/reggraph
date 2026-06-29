import json
import shutil
import subprocess

import pytest

from agentic_reg.domains import Domain
from agentic_reg.eval import eval as eval_module
from agentic_reg.eval.eval import (
    EvalMetrics,
    EvalResult,
    _build_summary,
    _extract_citations,
    _run_fixture_case,
    canonical_config,
    compute_metrics,
    run,
    save_report,
)
from agentic_reg.knowledge.graph import KnowledgeGraph


def test_extract_citations_uses_shared_parser():
    assert _extract_citations("See [Article-6], [article-6], and [section-67].") == [
        "article-6",
        "section-67",
    ]


def test_compute_metrics_detects_missing_and_hallucinated_citations():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")

    metrics = compute_metrics(
        "Answer [article-6] [article-99].",
        expected_citations=["article-6", "article-7"],
        expected_multi_hop={"article-5"},
        graph=graph,
    )

    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 0.5
    assert metrics.hallucination_rate == 0.5
    assert metrics.hallucinated == ["article-99"]


def test_canonical_config_accepts_phase_1_names_and_legacy_aliases():
    assert canonical_config("single") == "single"
    assert canonical_config("team") == "team"
    assert canonical_config("team-no-graph") == "team-no-graph"
    assert canonical_config("reggraph") == "team"
    assert canonical_config("langgraph") == "single"
    assert canonical_config("no-graph") == "team-no-graph"


def test_canonical_config_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown config"):
        canonical_config("swarm")


def test_fixture_case_is_not_a_benchmark_comparison():
    graph = KnowledgeGraph()
    graph.add_node("article-6", label="A6", kind="clause")
    case = {
        "id": "lawful",
        "question": "What is lawful?",
        "expected_citations": ["article-6"],
        "expected_multi_hop": ["article-5"],
    }

    single = _run_fixture_case(case, "single", graph)
    team = _run_fixture_case(case, "team", graph)

    assert single.mode == "fixture"
    assert team.mode == "fixture"
    assert single.answer == team.answer
    assert single.metrics.citation_f1 == team.metrics.citation_f1


def test_run_defaults_to_canonical_fixture_mode():
    results, summary = run(limit=1)

    assert {result.config for result in results} == {"single", "team"}
    assert {result.mode for result in results} == {"fixture"}
    assert summary["_metadata"]["benchmark"] is False
    assert "_comparison" not in summary


def test_run_deduplicates_legacy_aliases():
    results, summary = run(configs=["reggraph", "team", "langgraph"], limit=1)

    assert [result.config for result in results] == ["team", "single"]
    assert summary["_metadata"]["mode"] == "fixture"


def test_live_mode_fails_loudly_when_store_is_missing(monkeypatch, tmp_path):
    domain = Domain(
        name="missing",
        title="Missing",
        description="x",
        source_path=tmp_path / "source.md",
    )
    monkeypatch.setattr(eval_module, "get_domain", lambda name: domain)

    with pytest.raises(RuntimeError, match="Live eval requires"):
        run(configs=["single"], domain_name="missing", live=True, limit=1)


def test_save_report_marks_fixture_output_as_non_benchmark(tmp_path):
    result = EvalResult(
        case_id="case",
        question="Q?",
        config="single",
        answer="A [article-6].",
        mode="fixture",
        metrics=EvalMetrics(citation_f1=1.0),
    )
    summary = _build_summary([result])

    path = save_report([result], summary, output_dir=tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["metadata"]["mode"] == "fixture"
    assert data["metadata"]["benchmark"] is False
    assert data["results"][0]["mode"] == "fixture"


def test_reggraph_eval_console_script_runs_once(tmp_path):
    script = shutil.which("reggraph-eval")
    if script is None:
        pytest.skip("console script is not installed in this environment")

    result = subprocess.run(
        [script, "--configs", "single", "--limit", "1", "--output-dir", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.count("Full report ->") == 1


def test_judge_requires_live_mode():
    with pytest.raises(SystemExit):
        eval_module.main(["--judge"])
