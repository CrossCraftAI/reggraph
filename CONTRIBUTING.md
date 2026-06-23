# Contributing to RegGraph

Thanks for contributing. This document covers the workflow, code standards, and review
process for this repository.

---

## Quick reference

| Step | Command |
|------|---------|
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` |
| Test | `uv run pytest` |
| Full check | `uv run ruff check . && uv run ruff format --check . && uv run pytest` |

---

## Development workflow

1. **Pick or create an issue.** All work starts from an issue — bug report, feature
   request, or task from `TODO.md`. If you are starting something new, open an issue
   first to discuss the approach.

2. **Create a feature branch.** Branch from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b <scope>/<short-description>
   ```

   Use one of these prefixes:
   - `feat/` — new feature or module
   - `fix/` — bug fix
   - `test/` — test additions or improvements
   - `refactor/` — restructuring without behaviour change
   - `docs/` — documentation only

3. **Keep changes focused.** One PR = one logical change. If you find yourself writing
   "and also…" in the PR description, split it into separate PRs.

4. **Write tests.** Every new module, function, or fix needs tests. See the existing
   test style in `tests/` — use plain functions (not classes), descriptive names, and
   stub dependencies rather than real NetworkX/ChromaDB objects where possible.

5. **Run the full check before pushing:**

   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pytest
   ```

   All three must pass. A PR with lint failures or test failures will not be reviewed.

6. **Open a pull request.** Fill out the PR template completely. Link the issue your
   PR closes.

---

## Code review process

### For all contributors

Every PR goes through review before merging. The review is not a gate to slow you
down — it is a chance for a second pair of eyes to catch issues, suggest improvements,
and share knowledge across the team.

### What reviewers look for

| Dimension | Questions |
|-----------|-----------|
| **Correctness** | Does it do what it claims? Are edge cases handled? |
| **Tests** | Do the tests actually verify the behaviour, or just exercise the happy path? Would a bug in the changed code cause a test failure? |
| **Design** | Does it follow existing patterns in the codebase? Does it use Protocols rather than hard imports for external dependencies? Is the new API consistent with neighbouring modules? |
| **Config** | If a new setting is added, is it in `config.py` with a sensible default and documented in `.env.example`? |
| **Safety** | Does it validate inputs from external sources (LLM output, config, user input)? Are allowlists used where the value space is known? |
| **Clarity** | Are names descriptive? Are complex sections commented? Would a new contributor understand what this does? |
| **Scope** | Does the PR do one thing, or is it sneaking in unrelated changes? |

### Review approval requirements

- **At least one approving review** is required before merging.
- The reviewer must be a code owner (see `CODEOWNERS`).
- The author cannot approve their own PR.
- All review comments must be resolved (either addressed or discussed to agreement).

### If your PR receives change requests

1. Address each comment — either make the change or explain your reasoning.
2. Push additional commits to the same branch; do not force-push after review has
   started (it destroys review comment history).
3. Re-request review when ready.
4. Squash-and-merge is preferred once approved — keep the commit history on `main`
   clean, one commit per logical change.

---

## Code conventions

### Style

- Ruff enforces style automatically. Run `uv run ruff format .` before committing.
- Line length: 100 characters.
- Use descriptive variable names. Avoid single-letter names except in comprehensions
  or very short scoped loops.

### Type hints

- All public functions and methods must have type annotations on parameters and return
  values.
- Use `Protocol` classes (like the existing `GraphLike` and `ProposalGraph`) instead of
  importing concrete implementations — this keeps the code testable and avoids coupling.

### Docstrings

- Every public module, class, and function gets a docstring.
- First line: one-line summary in imperative mood (`"""Return whether ..."""` not
  `"""Returns whether ..."""`).
- Blank line, then longer description if needed.
- Use `Args:` / `Returns:` / `Raises:` sections for complex signatures (optional for
  simple ones).

### Imports

- Standard library first, then third-party, then local (`agentic_reg.*`).
- Ruff (`I` rule) enforces this automatically.

### Testing

- Tests live in `tests/` with filenames matching `test_<module>.py`.
- Use plain `def test_<description>():` functions, not classes.
- Use stub/fake objects (like `_Graph` in the existing tests) rather than real
  dependencies.
- Name tests to describe the behaviour: `test_validate_rejects_unknown_citation`
  rather than `test_validate_2`.

---

## Project structure

```
agentic_reg/
├── __init__.py
├── config.py          # Central settings (pydantic-settings)
├── build.py           # CLI: build a domain's knowledge store
├── eval.py            # CLI: evaluation harness
├── agents/            # LangGraph multi-agent orchestration
├── domains/           # Pluggable regulatory domains
├── knowledge/
│   ├── __init__.py
│   ├── graph.py       # NetworkX KnowledgeGraph
│   ├── vectors.py     # ChromaDB VectorIndex
│   ├── proposals.py   # Graph-update proposal lifecycle
│   └── symbolic.py    # Deterministic verification rules
├── providers/         # LLM backend abstraction
tests/
├── test_proposals.py
├── test_symbolic.py
└── ...
```

Modules marked above that do not yet exist are tracked in `TODO.md` Phase 0.

---

## Branch protection (repository administrators)

The following branch-protection rules are recommended for the `main` branch:

- **Require a pull request before merging** — no direct pushes to `main`.
- **Require approvals** — at least 1 approving review from a code owner.
- **Dismiss stale reviews** when new commits are pushed.
- **Require status checks to pass before merging** — lint + format + test (the CI
  workflow at `.github/workflows/ci.yml`).
- **Require branches to be up to date** before merging.
- **Require conversation resolution** before merging.

These rules can be configured at:
`https://github.com/CrossCraftAI/reggraph/settings/branch_protection_rules`.

---

## Questions?

Open a discussion or ask in an issue. If something in this document is unclear, that
is a documentation bug — please say so.
