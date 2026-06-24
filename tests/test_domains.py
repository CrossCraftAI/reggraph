from pathlib import Path

from agentic_reg.domains import Domain, get_domain, list_domains, register


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
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass


def test_list_domains_includes_gdpr():
    names = list_domains()
    assert "gdpr" in names


def test_gdpr_domain_exists_and_points_to_source():
    domain = get_domain("gdpr")
    assert domain.title == "General Data Protection Regulation"
    assert domain.unit_label == "article"
    assert domain.source_path.exists()
    assert domain.source_path.suffix == ".md"
