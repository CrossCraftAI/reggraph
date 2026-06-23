from agentic_reg.knowledge.symbolic import run_symbolic_checks


class _Graph:
    def __init__(self, node_ids: set[str]) -> None:
        self._node_ids = node_ids

    def has_node(self, node_id: str) -> bool:
        return node_id in self._node_ids


def _gdpr_graph() -> _Graph:
    return _Graph({"article-6", "article-7", "article-9", "article-17", "article-33"})


def test_citation_validity_flags_unknown_nodes():
    findings = run_symbolic_checks(
        "What makes processing lawful?",
        "Processing can be lawful [article-6] [article-99].",
        _gdpr_graph(),
    )
    validity = next(item for item in findings if item.rule_id == "citation_validity")

    assert not validity.passed
    assert "article-99" in validity.message


def test_special_category_requires_specific_condition_and_lawful_basis():
    findings = run_symbolic_checks(
        "Can health data be processed?",
        "Yes, if Article 9 conditions are met [article-9].",
        _gdpr_graph(),
    )
    special = next(item for item in findings if item.rule_id == "special_category_requires_basis")

    assert not special.passed
    assert "article-6" in special.message


def test_withdrawal_erasure_chain_passes_when_all_clauses_cited():
    findings = run_symbolic_checks(
        "If consent is withdrawn, can data be erased?",
        (
            "Yes: withdrawal, lawful basis, and erasure are connected "
            "[article-7] [article-6] [article-17]."
        ),
        _gdpr_graph(),
    )
    erasure = next(item for item in findings if item.rule_id == "withdrawal_erasure_chain")

    assert erasure.passed


def test_breach_notification_requires_deadline_and_source_clause():
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller should notify the authority [article-33].",
        _gdpr_graph(),
    )
    deadline = next(item for item in findings if item.rule_id == "breach_notification_deadline")

    assert not deadline.passed
    assert "72 hour" in deadline.message


def test_breach_notification_passes_with_72_hours_and_citation():
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller must notify within 72 hours where feasible [article-33].",
        _gdpr_graph(),
    )
    deadline = next(item for item in findings if item.rule_id == "breach_notification_deadline")

    assert deadline.passed
