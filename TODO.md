# TODO

> Generated from the 2026-06-23 full-repo audit. Items are ordered by dependency: each
> phase unblocks the next. "P0" = required for the Quick Start to work; "P1" = required
> for the full README-described experience; "P2" = polish.

---

## Phase 0 — Build the missing foundation (P0)

These modules are referenced throughout the README and config but do not exist yet.
Without them, **nothing in the Quick Start works**.

- [ ] **Create `agentic_reg/__init__.py` and `agentic_reg/knowledge/__init__.py`**
  — The package and subpackage need init files so Hatchling can build a working wheel.
- [ ] **Implement `agentic_reg/providers/`** — LLM backend abstraction.
  - `__init__.py` with `get_provider(settings) -> Provider`
  - `_github.py` — GitHub Models (OpenAI-compatible endpoint, `GITHUB_TOKEN` auth, fallback to `gh auth token`)
  - `_ollama.py` — Ollama chat/completions
  - `_anthropic.py` — Claude API (behind the `anthropic` extra)
  - Ref: config fields `llm_provider`, `github_model`, `ollama_model`, `anthropic_model`
- [ ] **Implement `agentic_reg/domains/`** — Pluggable regulatory domains.
  - `__init__.py` with `Domain` dataclass, `get_domain(name)`, `register(domain)`, and entry-point discovery
  - `_gdpr.py` — GDPR articles markdown + chunking config (or ship `data/gdpr.md`)
  - `_uk_dpa.py` — UK DPA 2018 markdown + chunking config (or ship `data/uk_dpa.md`)
  - Ref: `.env.example` `AGENTIC_REG_DOMAIN`, README "Plug in Your Own Domain"
- [ ] **Implement `agentic_reg/knowledge/graph.py`** — `KnowledgeGraph` class (NetworkX-backed).
  - `build(domain) -> KnowledgeGraph`
  - `load(path) -> KnowledgeGraph`
  - `save(path)`
  - `expand(node_ids, hops) -> set[node_ids]`
  - Must satisfy the `GraphLike` and `ProposalGraph` Protocols already defined in `symbolic.py` and `proposals.py`
- [ ] **Implement `agentic_reg/knowledge/vectors.py`** — `VectorIndex` class (ChromaDB-backed).
  - `build(domain, embedding_model)`
  - `search(query, top_k) -> list[Chunk]`
  - `load(chroma_dir, embedding_model) -> VectorIndex`
- [ ] **Implement `agentic_reg/build.py`** — CLI entry point for `python -m agentic_reg.build`.
  - `--domain <name>` (required)
  - `--no-enrich` (skip LLM concept typing, build deterministic graph only)
  - Orchestrates: domain loading → chunking → vector indexing → graph construction → typed concept enrichment (unless `--no-enrich`)
- [ ] **Implement `agentic_reg/agents/`** — Multi-agent orchestration (LangGraph).
  - `__init__.py` with `get_orchestrator(settings, provider, vector_index, graph) -> Orchestrator`
  - Supervisor agent (decomposes question into sub-questions, max `max_subquestions`)
  - Specialist agents (clause analyst, cross-reference analyst)
  - Synthesis agent
  - Verification agent (uses `run_symbolic_checks` + LLM review)
  - Self-correction loop (max `max_revisions` passes)
  - `Orchestrator.answer(question) -> Trace` with `trace.answer` and `trace.to_json()`
  - Ref: config fields `agent_mode`, `max_subquestions`, `max_revisions`, `max_agent_depth`, `max_agent_tasks`, `symbolic_checks`, `graph_update_mode`
- [ ] **Implement `app.py`** — Streamlit UI.
  - Domain selector, question input, answer display, trace viewer, JSON export
  - Ref: README Quick Start `uv run streamlit run app.py`

---

## Phase 1 — Evaluation, CI, hardening (P1)

- [ ] **Implement `agentic_reg/eval.py`** — Evaluation harness (`python -m agentic_reg.eval`).
  - `--configs single,team,team-no-graph` (agent mode + graph ablation)
  - `--limit N` (cap test cases)
  - Deterministic metrics: citation F1, hallucination rate, multi-hop recall
  - LLM-judge quality scores (excluded from mean on failure)
  - Output to `reports/`
  - Ref: README "Evaluation" section
- [ ] **Add `[project.scripts]` entry points** to `pyproject.toml` — `reggraph-build`, `reggraph-eval` console scripts.
- [ ] **Create `.github/workflows/ci.yml`** — CI pipeline (lint → test → type-check).
  - Runs on push/PR to `main`
  - Steps: checkout → install uv → `uv sync` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run pytest`
  - Fix the CI badge in README once the workflow file exists
- [ ] **Add config validation** — Use `Literal` types or Pydantic validators on `llm_provider`, `agent_mode`, and `graph_update_mode` so misconfiguration fails fast at startup rather than deep in provider/agent code.
- [ ] **Increase test coverage** on `validate_proposal` — exercise rejected paths: invalid node kinds, invalid relations, missing evidence, non-existent source/target nodes, empty citations.
- [ ] **Increase test coverage** on `run_symbolic_checks` — UK DPA `section-67` deadline path, graph with no breach-deadline clause at all, answers with overlapping rule triggers.

---

## Phase 2 — Polish (P2)

- [ ] **Make symbolic rules domain-pluggable** — `_deadline_clause` and rule triggers are hardcoded for GDPR/UK DPA. Extract into the `Domain` object or a rule-registry so adding a new domain does not require editing `symbolic.py`.
- [ ] **Add a "Work in Progress" note** to the README Quick Start section until Phase 0 is complete. The current instructions do not work.
- [ ] **Add `[project.entry-points]` for domain plugins** so external packages can ship new regulations via `agentic_reg.domains` entry point group (the config comment already references this, but the wiring isn't built).
- [ ] **Add `mypy` or `pyright` to dev dependencies and CI** — the code uses Protocols which benefit from static checking.
