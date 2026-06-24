<div align="center">
  <h1>RegGraph</h1>
  <p><strong>Regulatory QA over graph + vector knowledge stores</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
  [![CI](https://github.com/CrossCraftAI/reggraph/actions/workflows/ci.yml/badge.svg)](https://github.com/CrossCraftAI/reggraph/actions/workflows/ci.yml)
</div>

---

## Status

> **Work in progress.** The current repository contains the knowledge-store
> foundation: configuration, a GDPR domain excerpt, vector indexing, a NetworkX
> graph, deterministic citation/symbolic checks, graph-update proposals, a thin
> ask CLI, and an evaluation harness. The Streamlit UI, full multi-agent team,
> UK DPA domain package, and cross-jurisdictional reasoning are planned follow-up
> work tracked in [TODO.md](TODO.md).

RegGraph experiments with regulatory question-answering where citations are
checked against a local graph rather than trusted as free-form model output.
Vector search finds relevant clauses; graph traversal adds connected clauses; and
deterministic checks flag hallucinated citation IDs such as `[article-99]`.

## What Works Today

- Build a deterministic GDPR knowledge store from the bundled excerpt.
- Store semantic chunks in Chroma with local sentence-transformer embeddings.
- Build and persist a NetworkX clause graph.
- Run a thin ask CLI over the built store with an LLM provider.
- Run synthetic and live evaluation configs for RegGraph vs a LangGraph-style
  baseline.
- Validate graph-update proposals before they are written or applied.

## Quick Start

```bash
git clone https://github.com/CrossCraftAI/reggraph.git
cd reggraph
uv sync
uv run python -m agentic_reg.build --domain gdpr --no-enrich
```

`--no-enrich` skips best-effort LLM concept extraction and builds only the
deterministic graph/vector store.

To ask a live question, configure an LLM provider first:

```bash
gh auth login
uv run python -m agentic_reg.ask "What lawful bases allow processing of personal data?"
```

To run the deterministic eval smoke:

```bash
uv run python -m agentic_reg.eval --configs reggraph,langgraph --limit 4
```

## Configuration

Configuration is read from environment variables prefixed with `AGENTIC_REG_`
and from `.env`. Copy `.env.example` to start.

| Setting | Values |
| --- | --- |
| `AGENTIC_REG_LLM_PROVIDER` | `github`, `ollama`, `anthropic` |
| `AGENTIC_REG_DOMAIN` | `gdpr` |
| `AGENTIC_REG_USE_GRAPH` | `true`, `false` |

## Evaluation

The eval harness compares configs on deterministic metrics:

- citation precision/recall/F1
- hallucinated citation rate
- multi-hop recall
- optional LLM judge quality

Without `--live`, eval uses synthetic answers for quick metric validation. With
`--live`, it calls the configured provider.

## Current Roadmap

- Land the fuller domain/provider package layout.
- Add the bounded single/team orchestrator interface.
- Add Streamlit UI and deployment bootstrap.
- Add UK DPA as a second built-in domain.
- Make symbolic rules domain-pluggable.
- Add real cross-jurisdictional reasoning and evaluation.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Generated stores live under `data/store/` and are not committed.

## License

[MIT](LICENSE) (c) 2026 CrossCraftAI
