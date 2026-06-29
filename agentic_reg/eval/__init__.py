"""Evaluation helpers for Phase 0/1 regulatory QA smoke checks.

Usage::

    python -m agentic_reg.eval --configs single,team --limit 4
"""

from agentic_reg.eval.eval import (
    EvalMetrics,
    EvalResult,
    canonical_config,
    compute_metrics,
    run,
    save_report,
)

__all__ = [
    "EvalMetrics",
    "EvalResult",
    "canonical_config",
    "compute_metrics",
    "run",
    "save_report",
]
