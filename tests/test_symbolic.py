from agentic_reg.knowledge.symbolic import run_symbolic_checks


class _Graph:
    def __init__(self, node_ids: set[str]) -> None:
        self._node_ids = node_ids

    def has_node(self, node_id: str) -> bool:
        return node_id in self._node_ids


def _gdpr_graph() -> _Graph:
    return _Graph({"article-6", "article-7", "article-9", "article-17", "article-33"})


def _graph_with_nodes(*nodes: str) -> _Graph:
    return _Graph(set(nodes))


def _finding(findings, rule_id: str):
    return next(item for item in findings if item.rule_id == rule_id)


def test_citation_validity_passes_for_known_citations():
    findings = run_symbolic_checks(
        "What lawful basis applies?",
        "Processing can rely on lawful basis [article-6] and consent [article-7].",
        _gdpr_graph(),
    )
    validity = _finding(findings, "citation_validity")

    assert validity.passed
    assert validity.citations == ["article-6", "article-7"]
    assert validity.message == "All citations resolve to graph nodes."


def test_citation_validity_reports_unknown_citations():
    findings = run_symbolic_checks(
        "What lawful basis applies?",
        "Processing can rely on [article-6] and [article-99].",
        _gdpr_graph(),
    )
    validity = _finding(findings, "citation_validity")

    assert not validity.passed
    assert validity.citations == ["article-6", "article-99"]
    assert "article-99" in validity.message


def test_citation_validity_deduplicates_and_normalizes_citations():
    findings = run_symbolic_checks(
        "What lawful basis applies?",
        "Processing can rely on [Article-6], [article-6], and [ARTICLE-7].",
        _gdpr_graph(),
    )
    validity = _finding(findings, "citation_validity")

    assert validity.passed
    assert validity.citations == ["article-6", "article-7"]


def test_symbolic_finding_dict_shape_is_stable():
    findings = run_symbolic_checks(
        "What lawful basis applies?",
        "Processing can rely on [article-6].",
        _gdpr_graph(),
    )

    assert set(findings[0].to_dict()) == {"rule_id", "passed", "message", "citations"}


def test_special_category_passes_with_specific_condition_and_lawful_basis():
    findings = run_symbolic_checks(
        "Can health data be processed?",
        "Yes, if Article 9 conditions and Article 6 lawful basis apply [article-9] [article-6].",
        _gdpr_graph(),
    )
    special = _finding(findings, "special_category_requires_basis")

    assert special.passed


def test_special_category_requires_specific_condition_and_lawful_basis():
    findings = run_symbolic_checks(
        "Can health data be processed?",
        "Yes, if Article 9 conditions are met [article-9].",
        _gdpr_graph(),
    )
    special = _finding(findings, "special_category_requires_basis")

    assert not special.passed
    assert "article-6" in special.message


def test_special_category_uses_uk_dpa_condition_and_lawful_basis():
    graph = _graph_with_nodes("section-8", "section-10")
    findings = run_symbolic_checks(
        "Can biometric data be processed?",
        "Yes, if the condition and lawful basis are satisfied [section-10] [section-8].",
        graph,
    )
    special = _finding(findings, "special_category_requires_basis")

    assert special.passed
    assert special.citations == ["section-10", "section-8"]


def test_special_category_rule_skips_when_graph_has_no_required_clauses():
    findings = run_symbolic_checks(
        "Can health data be processed?",
        "The graph only has this placeholder clause for the test [article-33].",
        _graph_with_nodes("article-33"),
    )

    assert [item.rule_id for item in findings] == ["citation_validity"]


def test_withdrawal_erasure_chain_passes_when_all_clauses_cited():
    findings = run_symbolic_checks(
        "If consent is withdrawn, can data be erased?",
        (
            "Yes: withdrawal, lawful basis, and erasure are connected "
            "[article-7] [article-6] [article-17]."
        ),
        _gdpr_graph(),
    )
    erasure = _finding(findings, "withdrawal_erasure_chain")

    assert erasure.passed


def test_withdrawal_erasure_chain_fails_when_lawful_basis_clause_is_missing():
    findings = run_symbolic_checks(
        "If consent is withdrawn, can data be erased?",
        "Yes, withdrawal and erasure are connected [article-7] [article-17].",
        _gdpr_graph(),
    )
    erasure = _finding(findings, "withdrawal_erasure_chain")

    assert not erasure.passed
    assert "article-6" in erasure.message


def test_breach_notification_fails_when_source_cited_but_deadline_missing():
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller should notify the authority [article-33].",
        _gdpr_graph(),
    )
    deadline = _finding(findings, "breach_notification_deadline")

    assert not deadline.passed
    assert "72 hour" in deadline.message


def test_breach_notification_fails_when_deadline_present_but_source_missing():
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller must notify within 72 hours after assessing lawful basis [article-6].",
        _gdpr_graph(),
    )
    deadline = _finding(findings, "breach_notification_deadline")

    assert not deadline.passed
    assert "cite article-33" in deadline.message


def test_breach_notification_passes_with_72_hours_and_citation():
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller must notify within 72 hours where feasible [article-33].",
        _gdpr_graph(),
    )
    deadline = _finding(findings, "breach_notification_deadline")

    assert deadline.passed


def test_breach_notification_deadline_is_case_insensitive():
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller must notify within 72 Hours where feasible [article-33].",
        _gdpr_graph(),
    )
    deadline = _finding(findings, "breach_notification_deadline")

    assert deadline.passed


def test_breach_notification_uses_uk_dpa_deadline_clause():
    graph = _graph_with_nodes("section-67")
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller must notify within 72 hours where feasible [section-67].",
        graph,
    )
    deadline = _finding(findings, "breach_notification_deadline")

    assert deadline.passed
    assert deadline.citations == ["section-67"]


def test_breach_notification_skips_deadline_rule_when_no_deadline_clause_exists():
    graph = _graph_with_nodes("article-6")
    findings = run_symbolic_checks(
        "What must happen after a personal data breach notification?",
        "The controller should assess the breach [article-6].",
        graph,
    )

    assert [item.rule_id for item in findings] == ["citation_validity"]


def test_overlapping_rule_triggers_are_reported_independently():
    findings = run_symbolic_checks(
        "Can health data be erased after consent is withdrawn?",
        (
            "Health data processing needs a condition and lawful basis "
            "[article-9] [article-6]. Withdrawal can support erasure "
            "analysis [article-7] [article-17]."
        ),
        _gdpr_graph(),
    )
    by_rule = {item.rule_id: item for item in findings}

    assert by_rule["special_category_requires_basis"].passed
    assert by_rule["withdrawal_erasure_chain"].passed
