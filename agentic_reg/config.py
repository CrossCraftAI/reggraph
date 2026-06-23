"""Central configuration.

All tunables live here and are overridable via environment variables (prefixed
``AGENTIC_REG_``) or a local ``.env`` file. This is the one place that knows
which LLM backend is active, so switching from local Ollama to the Claude API
is a config change, not a code change.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_REG_",
        env_file=".env",
        extra="ignore",
    )

    # --- LLM backend ---
    # "github" (GitHub Models, strong + free for students), "ollama" (local), or
    # "anthropic" (Claude API).
    llm_provider: str = "github"

    # GitHub Models — OpenAI-compatible, free tier via your GitHub token.
    github_model: str = "openai/gpt-4o"
    github_base_url: str = "https://models.github.ai/inference"
    # Read the conventional GITHUB_TOKEN; if unset, the provider falls back to
    # `gh auth token` so a logged-in gh user needs no extra setup.
    github_token: str | None = Field(default=None, validation_alias="GITHUB_TOKEN")

    ollama_model: str = "llama3.1:8b"
    ollama_host: str = "http://localhost:11434"

    anthropic_model: str = "claude-sonnet-4-6"
    # Read the conventional ANTHROPIC_API_KEY (no AGENTIC_REG_ prefix).
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # --- Embeddings (local, free) ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Domain ---
    # Which regulatory domain to use; see agentic_reg.domains (gdpr, uk_dpa, ...).
    # Source document + store paths live on the resolved Domain, not here.
    domain: str = "gdpr"

    # --- Retrieval tuning ---
    vector_top_k: int = 4  # how many chunks vector search returns
    graph_hops: int = 2  # how far to expand from matched nodes in the graph
    use_graph: bool = True  # False = vector-only retrieval baseline

    # --- Agent orchestration ---
    agent_mode: str = "team"  # "single" agent or hierarchical "team"
    max_subquestions: int = 3  # cap on how many sub-questions the supervisor creates
    max_revisions: int = 1  # how many self-correction passes the team may run
    max_agent_depth: int = 2  # root supervisor -> mid-level supervisor -> leaf specialist
    max_agent_tasks: int = 8  # cap on spawned non-root tasks across the hierarchy
    symbolic_checks: bool = True  # deterministic high-confidence verification rules
    graph_update_mode: str = "propose"  # off | propose | apply
    graph_proposals_path: str | None = None  # optional override for review JSONL output


def get_settings() -> Settings:
    """Return freshly-loaded settings (re-reads env / .env each call)."""
    return Settings()
