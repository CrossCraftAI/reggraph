## Summary

<!-- One or two sentences describing what this PR does and why. -->


## Checklist

### Before requesting review

- [ ] I have run `uv run ruff check .` and fixed all issues
- [ ] I have run `uv run ruff format --check .` (or `uv run ruff format .` to auto-fix)
- [ ] I have run `uv run pytest` and all tests pass (no regressions)
- [ ] I have added or updated tests for any new or changed behaviour
- [ ] I have updated `TODO.md` if this PR completes or supersedes a listed item
- [ ] I have verified my changes work end-to-end in the target use case (e.g., `uv run python -m agentic_reg.build --domain gdpr` if modifying the build pipeline)

### For new modules or significant changes

- [ ] New public classes/functions have docstrings explaining purpose, parameters, and return values
- [ ] Config changes are reflected in `.env.example` if a new setting was added
- [ ] New dependencies are added to `pyproject.toml` with a minimum-version pin
- [ ] New modules are added to `[tool.hatch.build.targets.wheel] packages` in `pyproject.toml` if they live outside `agentic_reg/`

### For bug fixes

- [ ] A regression test has been added that fails on `main` and passes on this branch

---

## How to test

<!-- Steps for a reviewer to verify this change locally. -->

```
# Example:
uv sync
uv run pytest tests/test_<module>.py -v
```

---

## Related issues

<!-- Link issues this PR closes or relates to. Use "Closes #N" to auto-close. -->

Closes #
