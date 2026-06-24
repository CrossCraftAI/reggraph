"""Evaluation harness for regulatory question-answering.

Compares RegGraph against a vanilla LangGraph baseline on compliance-specific
metrics: citation F1, hallucination rate, multi-hop recall, answer quality,
wall-clock time, and token usage.

Usage::

    python -m agentic_reg.eval --configs reggraph,langgraph --limit 4
"""

from agentic_reg.eval.baselines.langgraph_baseline import LangGraphBaseline
from agentic_reg.eval.eval import (
    EvalMetrics,
    EvalResult,
    compute_metrics,
    run,
    save_report,
)

__all__ = [
    "EvalMetrics",
    "EvalResult",
    "LangGraphBaseline",
    "compute_metrics",
    "run",
    "save_report",
]
