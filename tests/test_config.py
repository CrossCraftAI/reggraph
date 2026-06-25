import pytest
from pydantic import ValidationError

from agentic_reg.config import Settings


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("llm_provider", "openai"),
        ("agent_mode", "swarm"),
        ("graph_update_mode", "auto"),
    ],
)
def test_settings_reject_unknown_enum_values(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vector_top_k", 0),
        ("graph_hops", -1),
        ("max_subquestions", 0),
        ("max_revisions", -1),
        ("max_agent_depth", 0),
        ("max_agent_tasks", 0),
    ],
)
def test_settings_reject_invalid_numeric_bounds(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("github_base_url", "not-a-url"),
        ("github_base_url", "ftp://models.example.com"),
        ("ollama_host", "not-a-url"),
        ("ollama_host", "file:///tmp/ollama.sock"),
    ],
)
def test_settings_reject_malformed_provider_urls(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("github_model", ""),
        ("github_base_url", " "),
        ("ollama_model", ""),
        ("ollama_host", " "),
        ("anthropic_model", ""),
        ("embedding_model", " "),
        ("domain", ""),
    ],
)
def test_settings_reject_empty_required_strings(field, value):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_env_example_values_validate():
    settings = Settings(_env_file=".env.example")

    assert settings.llm_provider == "github"
    assert settings.agent_mode == "team"
    assert settings.graph_update_mode == "propose"


def test_settings_accept_boundary_values():
    settings = Settings(
        _env_file=None,
        vector_top_k=1,
        graph_hops=0,
        max_subquestions=1,
        max_revisions=0,
        max_agent_depth=1,
        max_agent_tasks=1,
    )

    assert settings.vector_top_k == 1
    assert settings.graph_hops == 0
    assert settings.max_subquestions == 1
    assert settings.max_revisions == 0
    assert settings.max_agent_depth == 1
    assert settings.max_agent_tasks == 1


def test_optional_blank_strings_normalize_to_none():
    settings = Settings(
        _env_file=None,
        github_token=" ",
        ANTHROPIC_API_KEY="",
        graph_proposals_path="   ",
    )

    assert settings.github_token is None
    assert settings.anthropic_api_key is None
    assert settings.graph_proposals_path is None


def test_provider_urls_are_stripped_validated_and_kept_as_strings():
    settings = Settings(
        _env_file=None,
        github_base_url=" https://models.github.ai/inference ",
        ollama_host=" http://localhost:11434 ",
    )

    assert settings.github_base_url == "https://models.github.ai/inference"
    assert isinstance(settings.github_base_url, str)
    assert settings.ollama_host == "http://localhost:11434"
    assert isinstance(settings.ollama_host, str)
