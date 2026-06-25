# TODO

> Reconciled from the 2026-06-23 audit. Checked items are present on the
> current branch. Unchecked items should land as focused follow-up PRs.

---

## Prerequisites - Repo Admin Actions

These steps are required for `CODEOWNERS`, `CONTRIBUTING.md`, and CI to be
enforceable. Until GitHub settings are enabled, the files document process but
do not gate merges.

### 1. Create the `reggraph-maintainers` Team

- Go to: `https://github.com/orgs/CrossCraftAI/new-team`
- Team name: `reggraph-maintainers`
- Add at least one person who is not `yrnrkv`.
- `CODEOWNERS` references `@CrossCraftAI/reggraph-maintainers`; until this team
  exists, code owner review is inert.

### 2. Enable Branch Protection on `main`

- Go to: `https://github.com/CrossCraftAI/reggraph/settings/branch_protection_rules`
- Branch name pattern: `main`
- Enable:

| Setting | Value |
| --- | --- |
| Require a pull request before merging | on |
| Require approvals | 1 |
| Dismiss stale approvals after new commits | on |
| Require status checks to pass | on, job `check` |
| Require branches to be up to date | on |
| Require conversation resolution | on |
| Do not allow bypassing | on, if available |

Do not enable "Require approval from someone other than the last pusher" until
the maintainers team has at least two people.

---

## Implemented Foundation

- [x] Package init files for `agentic_reg` and `agentic_reg.knowledge`.
- [x] GDPR domain excerpt and domain registry.
- [x] GitHub/Ollama provider abstraction, with Anthropic deferred.
- [x] NetworkX `KnowledgeGraph` with JSON persistence and graph expansion.
- [x] Chroma-backed `VectorIndex`.
- [x] `python -m agentic_reg.build --domain gdpr --no-enrich`.
- [x] Thin `python -m agentic_reg.ask` ReAct-style CLI.
- [x] Evaluation harness with `reggraph`, `langgraph`, and `no-graph` configs.
- [x] CI workflow, PR template, `CODEOWNERS`, and `CONTRIBUTING.md`.

## Near-Term Follow-Up PRs

- [ ] **Config validation (#3).** Use `Literal` or validators for
  `llm_provider`, `agent_mode`, and `graph_update_mode`.
- [ ] **Proposal validation coverage (#4).** Add tests for invalid relations,
  missing source/target nodes, invalid node kinds, duplicate nodes, empty
  evidence, and bad citation inputs.
- [ ] **Symbolic-check coverage.** Add UK DPA `section-67`, no-deadline, and
  overlapping-trigger tests.
- [ ] **Console scripts.** Add `reggraph-build` and `reggraph-eval`.
- [ ] **Package layout reconciliation.** Move toward package directories for
  domains, providers, and eval before adding the fuller agent stack.

## Larger Follow-Up Work

- [ ] **Close or update issue #1.** The repo still lacks the full README-era
  app/team/domain surface; keep the issue open until the follow-up
  implementation PR lands and smoke commands pass from a clean checkout.
- [ ] **Streamlit UI.** Add `app.py` after the store/bootstrap path is stable.
- [ ] **Built-in UK DPA domain.** Add source excerpt, benchmark cases, and build
  smoke coverage.
- [ ] **Multi-agent orchestrator.** Add bounded single/team modes with structured
  reasoning traces.
- [ ] **Make symbolic rules domain-pluggable (#5).** Move hardcoded GDPR/UK DPA
  rule parameters into domain metadata or a rule registry.
- [ ] **Cross-jurisdictional reasoning.** Add inter-domain graph edges,
  jurisdiction-aware specialists, delta output, and live eval coverage.
