from pathlib import Path

from agentic_reg.domains import Domain, get_domain, list_domains, register, registry


def test_domain_defaults():
    domain = Domain(
        name="test",
        title="Test Regulation",
        description="A test.",
        source_path=Path("/tmp/test.md"),
    )
    assert domain.name == "test"
    assert domain.unit_label == "article"
    assert domain.chunk_size == 512
    assert domain.chunk_overlap == 64
    assert domain.symbolic_rules.required_citation_rules == ()
    assert domain.symbolic_rules.deadline_rules == ()


def test_domain_raises_on_empty_name():
    try:
        Domain(name="", title="T", description="D", source_path=Path("/tmp/t.md"))
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_register_and_get_domain():
    register(
        Domain(
            name="test-reg",
            title="Test",
            description="Desc.",
            source_path=Path("/tmp/reg.md"),
        )
    )
    domain = get_domain("test-reg")
    assert domain.name == "test-reg"
    assert domain.title == "Test"


def test_get_domain_raises_for_unknown():
    try:
        get_domain("nonexistent")
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass


def test_list_domains_includes_gdpr():
    names = [domain.name for domain in list_domains()]
    assert "gdpr" in names
    assert "uk_dpa" in names


def test_gdpr_domain_exists_and_points_to_source():
    domain = get_domain("gdpr")
    assert domain.title == "General Data Protection Regulation"
    assert domain.unit_label == "article"
    assert domain.source_path.exists()
    assert domain.source_path.suffix == ".md"
    assert domain.symbolic_rules.required_citation_rules[0].rule_id == (
        "special_category_requires_basis"
    )


def test_uk_dpa_domain_uses_section_units_and_isolated_store():
    gdpr = get_domain("gdpr")
    uk_dpa = get_domain("uk_dpa")

    assert uk_dpa.unit_label == "section"
    assert uk_dpa.source_path.exists()
    assert gdpr.chroma_dir != uk_dpa.chroma_dir
    assert gdpr.graph_path != uk_dpa.graph_path
    assert uk_dpa.symbolic_rules.deadline_rules[0].citation == "section-67"


def test_discover_plugins_registers_entry_point_domain(monkeypatch):
    domain = Domain(
        name="entry_point_reg",
        title="Entry Point Regulation",
        description="Loaded from a fake entry point.",
        source_path=Path("/tmp/entry-point.md"),
    )

    class _Point:
        def load(self):
            return domain

    monkeypatch.setattr(registry, "entry_points", lambda group: [_Point()])

    registry.discover_plugins()

    assert get_domain("entry_point_reg") is domain
