# TODO

> Generated from the 2026-06-23 full-repo audit. Items are ordered by dependency: each
> phase unblocks the next. "P0" = required for the Quick Start to work; "P1" = required
> for the full README-described experience; "P2" = polish.
>
> **Checked items** were completed as part of the initial audit follow-up
> (commit `1c29cfb`).

---

## Prerequisites — Repo admin actions (manual, one-time)

These steps are **required** for the PR gating infrastructure (`CODEOWNERS`,
`CONTRIBUTING.md`, `.github/workflows/ci.yml`) to be enforceable. Without them,
every file in this repo is just documentation — nobody is actually gated.

### 1. Create the `reggraph-maintainers` team

- Go to: `https://github.com/orgs/CrossCraftAI/new-team`
- Team name: `reggraph-maintainers`
- Add at least **one person who is not `yrnrkv`**.
- The `CODEOWNERS` file already references `@CrossCraftAI/reggraph-maintainers`. Until
  this team exists with members, CODEOWNERS is inert — every PR can self-merge without
  review.

### 2. Enable branch protection on `main`

- Go to: `https://github.com/CrossCraftAI/reggraph/settings/branch_protection_rules`
- Click **"New branch protection rule"** (or "Add rule").
- Branch name pattern: `main`
- Enable these checks:

  | Setting | Value |
  |---------|-------|
  | **Require a pull request before merging** | ✅ on |
  | └ Require approvals | **1** |
  | └ Dismiss stale pull request approvals when new commits are pushed | ✅ on |
  | **Require status checks to pass before merging** | ✅ on |
  | └ Search for `check` (the CI job name) | add it |
  | **Require branches to be up to date before merging** | ✅ on |
  | **Require conversation resolution before merging** | ✅ on |
  | **Do not allow bypassing the above settings** | ✅ on (if available on your plan) |

- **Do not** check "Require approval from someone other than the last pusher" — leave
  it off unless the maintainers team has 2+ people. With only one maintainer and
  yrnrkv as the author, that rule would make every PR unmergeable.

These rules mean: every change to `main` must come through a PR → pass CI → get an
approving review from a maintainer → have all review threads resolved. This is the
gate.

---

## Phase 0 — Build the missing foundation (P0)

These modules are referenced throughout the README and config but do not exist yet.
Without them, **nothing in the Quick Start works**.

- [x] **Create `agentic_reg/__init__.py` and `agentic_reg/knowledge/__init__.py`**
  — The package and subpackage need init files so Hatchling can build a working wheel.
- [x] **Implement `agentic_reg/providers/`** — LLM backend abstraction.
  - `__init__.py` with `get_provider(settings) -> Provider`
  - `github_provider.py` — GitHub Models (OpenAI-compatible endpoint, `GITHUB_TOKEN` auth, fallback to `gh auth token`)
  - `ollama_provider.py` — Ollama chat/completions
  - `anthropic_provider.py` — Claude API (behind the `anthropic` extra)
  - Ref: config fields `llm_provider`, `github_model`, `ollama_model`, `anthropic_model`
- [x] **Implement `agentic_reg/domains/`** — Pluggable regulatory domains.
  - `__init__.py` with `Domain` dataclass, `get_domain(name)`, `register(domain)`, and entry-point discovery
  - `builtin.py` — built-in GDPR and UK DPA domain registrations
  - `data/gdpr.md` and `data/uk_dpa_excerpt.md` — bundled domain source excerpts
  - Ref: `.env.example` `AGENTIC_REG_DOMAIN`, README "Plug in Your Own Domain"
- [x] **Implement `agentic_reg/knowledge/graph.py`** — `KnowledgeGraph` class (NetworkX-backed).
  - `build(domain) -> KnowledgeGraph`
  - `load(path) -> KnowledgeGraph`
  - `save(path)`
  - `expand(node_ids, hops) -> (nodes, edges)`
  - Must satisfy the `GraphLike` and `ProposalGraph` Protocols already defined in `symbolic.py` and `proposals.py`
- [x] **Implement `agentic_reg/knowledge/vectors.py`** — `VectorIndex` class (ChromaDB-backed).
  - `build(domain, embedding_model)`
  - `search(query, top_k) -> list[VectorHit]`
  - `load(chroma_dir, embedding_model) -> VectorIndex`
- [x] **Implement `agentic_reg/build.py`** — CLI entry point for `python -m agentic_reg.build`.
  - `--domain <name>` (required)
  - `--no-enrich` (skip LLM concept typing, build deterministic graph only)
  - Orchestrates: domain loading → chunking → vector indexing → graph construction → typed concept enrichment (unless `--no-enrich`)
- [x] **Implement `agentic_reg/agents/`** — Multi-agent orchestration (LangGraph).
  - `__init__.py` with `get_orchestrator(settings, provider, vector_index, graph) -> Orchestrator`
  - Supervisor agent (decomposes question into sub-questions, max `max_subquestions`)
  - Specialist agents (clause analyst, cross-reference analyst)
  - Synthesis agent
  - Verification agent (uses `run_symbolic_checks` + LLM review)
  - Self-correction loop (max `max_revisions` passes)
  - `Orchestrator.answer(question) -> Trace` with `trace.answer` and `trace.to_json()`
  - Ref: config fields `agent_mode`, `max_subquestions`, `max_revisions`, `max_agent_depth`, `max_agent_tasks`, `symbolic_checks`, `graph_update_mode`
- [x] **Implement `app.py`** — Streamlit UI.
  - Domain selector, question input, answer display, trace viewer, JSON export
  - Ref: README Quick Start `uv run streamlit run app.py`

---

## Phase 1 — Evaluation, CI, hardening (P1)

- [x] **Implement `agentic_reg/eval.py`** — Evaluation harness (`python -m agentic_reg.eval`).
  - `--configs single,team,team-no-graph` (agent mode + graph ablation)
  - `--limit N` (cap test cases)
  - Deterministic metrics: citation F1, hallucination rate, multi-hop recall
  - LLM-judge quality scores (excluded from mean on failure)
  - Output to `reports/`
  - Ref: README "Evaluation" section
- [x] **Add `[project.scripts]` entry points** to `pyproject.toml` — `reggraph-build`, `reggraph-eval` console scripts.
- [x] **Create `.github/workflows/ci.yml`** — CI pipeline (lint → format → test).
  - Runs on push/PR to `main`
  - Steps: checkout → install uv → `uv sync` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run pytest`
  - CI badge in README now points to the real workflow
- [x] **Add config validation** — Use `Literal` types or Pydantic validators on `llm_provider`, `agent_mode`, and `graph_update_mode` so misconfiguration fails fast at startup rather than deep in provider/agent code.
- [x] **Increase test coverage** on `validate_proposal` — exercise rejected paths: invalid node kinds, invalid relations, missing evidence, non-existent source/target nodes, empty citations.
- [x] **Increase test coverage** on `run_symbolic_checks` — UK DPA `section-67` deadline path, graph with no breach-deadline clause at all, answers with overlapping rule triggers.

---

## Phase 2 — Polish (P2)

- [x] **Make symbolic rules domain-pluggable** — `_deadline_clause` and rule triggers are hardcoded for GDPR/UK DPA. Extract into the `Domain` object or a rule-registry so adding a new domain does not require editing `symbolic.py`.
- [x] **Add a "Work in Progress" note** to the README Quick Start section until Phase 0 is complete. The current instructions do not work.
- [x] **Add `[project.entry-points]` for domain plugins** so external packages can ship new regulations via `agentic_reg.domains` entry point group (the config comment already references this, but the wiring isn't built).
- [x] **Add `mypy` or `pyright` to dev dependencies and CI** — the code uses Protocols which benefit from static checking.

---

## Phase 3 — Cross-jurisdictional reasoning (P2)

Cross-jurisdictional mapping with reasoning chains is the core differentiator.
Legal experts are bounded by jurisdiction; startups scale across borders with
no systematic mapping. This phase builds the primitives that make inter-domain
reasoning possible.

- [ ] **Inter-domain graph edges** — `equivalent_to`, `conflicts_with`, `extends`,
  `narrower_than`, `broader_than` relations between nodes in different domain
  graphs. Enables traversal like "GDPR article-9 → equivalent_to → UK DPA section-10
  → requires → UK DPA schedule-1."
- [ ] **Cross-jurisdictional agent specialist** — A jurisdiction-aware specialist
  that can load two domain graphs side by side, trace equivalence chains, and
  flag gaps where one regulation imposes obligations another doesn't.
- [ ] **Delta analysis output** — Structured output showing: (1) equivalent
  provisions, (2) gaps where jurisdiction A requires something B doesn't,
  (3) conflicts where A and B impose contradictory obligations, (4) recommended
  bridging actions.
- [ ] **Cross-jurisdictional eval harness** — Extend `--scenario cross-jurisdictional`
  with live LLM agent execution (currently synthetic-only). Load multiple domain
  graphs, run agents across them, measure cross-jurisdictional recall.
- [ ] **Domain pack metadata** — Domain markdown files gain a YAML frontmatter
  block with jurisdiction info (country, regulator, effective date, amendment
  history) so the engine can reason about temporal precedence and
  geographic scope.

---

## Phase 4 — Polyglot core (P3)

The long-term ambition is a compiled core (Rust or Zig) with a C FFI, exposed
via Python bindings and WASM. This follows the eyecite → incitez trajectory:
start in Python, prove the primitives, then rewrite the hot path in a systems
language for speed, portability, and zero-dependency embedding.

- [ ] **Extract `reggraph-core`** — Pure library: `GraphLike` Protocol, typed
  node/edge schemas, citation regex engine, and deterministic verification as
  a standalone Python package with zero LLM dependency. `pip install reggraph-core`.
- [ ] **Rust rewrite of graph ops** — `has_node`, `expand`, `_extract_citations`,
  and edge traversal in Rust with `pyo3` bindings. Target: <1ms for citation
  verification against a 500-node regulation graph.
- [ ] **WASM build for browser** — `reggraph-core` compiled to WASM so citation
  verification runs entirely client-side. No server, no API key, no data leaving
  the machine. Enable the "paste a regulation, verify citations in browser" demo.
- [ ] **C FFI header** — Stable C ABI so `reggraph-core` can be embedded in Go,
  Kotlin, Swift, or any language with C interop. Target: compliance checks in
  mobile apps, CI pipelines, and edge deployments.
- [ ] **Language-agnostic test corpus** — Port the eval test cases to a
  language-agnostic JSON format so every binding (Python, Rust, WASM, C)
  validates against the same expected outputs.
