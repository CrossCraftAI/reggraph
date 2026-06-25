<div align="center">
  <h1>RegGraph</h1>
  <p><strong>Regulatory QA over graph + vector knowledge stores</strong></p>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
  [![CI](https://github.com/CrossCraftAI/reggraph/actions/workflows/ci.yml/badge.svg)](https://github.com/CrossCraftAI/reggraph/actions/workflows/ci.yml)
</div>

---

## Status

Phase 0 is implemented locally: RegGraph can build GDPR and UK DPA knowledge
stores, retrieve over graph + vector evidence, answer through a traced single
agent or a basic specialist team, and run in Streamlit. Richer hierarchy,
verification/revision, graph-curator proposals, deployment polish, and
cross-jurisdictional reasoning remain follow-up work tracked in [TODO.md](TODO.md).

RegGraph experiments with regulatory question-answering where citations are
checked against a local graph rather than trusted as free-form model output.
Vector search finds relevant clauses; graph traversal adds connected clauses; and
deterministic checks flag hallucinated citation IDs such as `[article-99]`.

## What Works Today

- Build deterministic GDPR and UK DPA knowledge stores from bundled excerpts.
- Store semantic chunks in Chroma with local sentence-transformer embeddings.
- Build and persist NetworkX clause graphs with cross-reference edges.
- Run a traced single agent or basic specialist team over graph-aware retrieval.
- Use the Streamlit UI to ask questions, inspect answers, view traces, and see
  the clause graph.
- Run the thin ask CLI over the built store with an LLM provider.
- Run synthetic and live evaluation configs for RegGraph vs a LangGraph-style
  baseline.
- Validate graph-update proposals before they are written or applied.

## Quick Start

```bash
git clone https://github.com/CrossCraftAI/reggraph.git
cd reggraph
uv sync
uv run python -m agentic_reg.build --domain gdpr --no-enrich
uv run python -m agentic_reg.build --domain uk_dpa --no-enrich
```

`--no-enrich` skips best-effort LLM concept extraction and builds only the
deterministic graph/vector store.

To ask a live question in the app or CLI, configure an LLM provider first:

```bash
gh auth login
uv run streamlit run app.py
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
| `AGENTIC_REG_DOMAIN` | `gdpr`, `uk_dpa` |
| `AGENTIC_REG_USE_GRAPH` | `true`, `false` |
| `AGENTIC_REG_AGENT_MODE` | `team`, `single` |

Settings construction validates enum values, numeric bounds, non-empty required
strings, and provider URL shape. Credentials and domain existence are checked
later by the provider and domain registry.

## Evaluation

The eval harness compares configs on deterministic metrics:

- citation precision/recall/F1
- hallucinated citation rate
- multi-hop recall
- optional LLM judge quality

Without `--live`, eval uses synthetic answers for quick metric validation. With
`--live`, it calls the configured provider.

## Current Roadmap

- Add hierarchical team orchestration, graph-curator proposals, verification,
  and bounded self-correction.
- Add hosted demo deployment bootstrap.
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
