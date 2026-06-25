"""Central configuration.

All tunables live here and are overridable via environment variables (prefixed
``AGENTIC_REG_``) or a local ``.env`` file. This is the one place that knows
which LLM backend is active, so switching from local Ollama to the Claude API
is a config change, not a code change.
"""

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, AnyHttpUrl, Field, TypeAdapter, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
HTTP_URL_ADAPTER = TypeAdapter(AnyHttpUrl)

LLMProviderName = Literal["github", "ollama", "anthropic"]
AgentMode = Literal["single", "team"]
GraphUpdateMode = Literal["off", "propose", "apply"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_REG_",
        env_file=".env",
        extra="ignore",
        validate_default=True,
    )

    # --- LLM backend ---
    # "github" (GitHub Models, strong + free for students), "ollama" (local), or
    # "anthropic" (Claude API).
    llm_provider: LLMProviderName = "github"

    # GitHub Models — OpenAI-compatible, free tier via your GitHub token.
    github_model: str = "openai/gpt-4o"
    github_base_url: str = "https://models.github.ai/inference"
    # Read the conventional GITHUB_TOKEN; if unset, the provider falls back to
    # `gh auth token` so a logged-in gh user needs no extra setup.
    github_token: str | None = Field(
        default=None, validation_alias=AliasChoices("github_token", "GITHUB_TOKEN")
    )

    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    anthropic_model: str = "claude-sonnet-4-6"
    # Read the conventional ANTHROPIC_API_KEY (no AGENTIC_REG_ prefix).
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("anthropic_api_key", "ANTHROPIC_API_KEY"),
    )

    # --- Embeddings (local, free) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Domain ---
    # Which regulatory domain to use; see agentic_reg.domains (gdpr, uk_dpa, ...).
    # Source document + store paths live on the resolved Domain, not here.
    domain: str = "gdpr"

    # --- Retrieval tuning ---
    # How many chunks vector search returns.
    vector_top_k: int = Field(default=4, ge=1)
    # How far to expand from matched nodes in the graph.
    graph_hops: int = Field(default=2, ge=0)
    use_graph: bool = True  # False = vector-only retrieval baseline

    # --- Agent orchestration ---
    agent_mode: AgentMode = "team"  # "single" agent or hierarchical "team"
    # Cap on how many sub-questions the supervisor creates.
    max_subquestions: int = Field(default=3, ge=1)
    # How many self-correction passes the team may run.
    max_revisions: int = Field(default=1, ge=0)
    # Root supervisor -> mid-level supervisor -> leaf specialist.
    max_agent_depth: int = Field(default=2, ge=1)
    # Cap on spawned non-root tasks across the hierarchy.
    max_agent_tasks: int = Field(default=8, ge=1)
    symbolic_checks: bool = True  # deterministic high-confidence verification rules
    graph_update_mode: GraphUpdateMode = "propose"  # off | propose | apply
    graph_proposals_path: str | None = None  # optional override for review JSONL output

    @field_validator(
        "github_model",
        "github_base_url",
        "ollama_model",
        "ollama_host",
        "anthropic_model",
        "embedding_model",
        "domain",
        mode="before",
    )
    @classmethod
    def _strip_required_strings(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("must not be empty")
            return value
        return value

    @field_validator("github_token", "anthropic_api_key", "graph_proposals_path", mode="before")
    @classmethod
    def _blank_optional_strings_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator("github_base_url", "ollama_host")
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        HTTP_URL_ADAPTER.validate_python(value)
        return value


def get_settings() -> Settings:
    """Return freshly-loaded settings (re-reads env / .env each call)."""
    return Settings()
