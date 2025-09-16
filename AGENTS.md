# AGENTS.md — Repository-wide Guidelines

## 1. Scope

These instructions apply to every file in the repository unless a deeper directory contains its own `AGENTS.md` with overriding rules.

## 2. Directory Layout

- `src/rldk/` – Library and CLI source code.
- `tests/` – Automated tests. Mirrors the `src` hierarchy; top-level test modules should be located here or in its subpackages.
- `docs/` – Documentation. Long-form reports now live in `docs/reports/`, and internal engineering notes live in `docs/internal/`.
- `examples/`, `recipes/`, `reference/` – Usage examples, sample configs, and API references.
- `tools/`, `scripts/` – Developer utilities and supporting CLIs.
- `artifacts/`, `assets/`, `templates/`, `tracking_demo_output/` – Generated or static resources. Commit new files here only when they are stable project assets.

**Do not create new modules or documentation at the repository root.** If a new category is required, document the rationale in `docs/` and reference it in the pull request.

## 3. Code Conventions

1. **Formatting & Linting**
   - Python code must satisfy `black`, `isort`, `ruff`, and `flake8`.
   - Use type hints on all public interfaces and add doctrings for public modules, classes, and functions.
   - Follow an 88-character soft line length unless a nested `AGENTS.md` specifies otherwise.
1. **Imports**
   - Prefer absolute imports from `rldk`. Avoid relative imports that cross package boundaries.
1. **Error Handling & Logging**
   - Raise specific exceptions; never use bare `except:`.
   - Use the `logging` module instead of `print` in production code.
1. **Configuration & Constants**
   - Centralize configuration under `src/rldk/config/`. New constants belong in the appropriate config module rather than scattered throughout the code.

## 4. Testing & Validation

1. Add or update tests in `tests/` whenever behavior under `src/` changes.
1. Run the following before requesting review:
   ```bash
   pre-commit run --files $(git diff --name-only --cached)
   pytest
   ```
   Run targeted pytest selections when appropriate (e.g., `pytest tests/path/to/module`).
1. Record skipped validations in the PR description when full coverage is impractical.

## 5. Pull Requests

1. Keep each PR focused on one logical change. When touching code, update the related documentation and tests in the same PR.
1. PR descriptions must include:
   - Summary of changes.
   - Motivation or linked issue.
   - Validation evidence (commands and outcomes).
   - Any follow-up tasks or cross-cutting impacts.
1. Use conventional commit prefixes (`feat:`, `fix:`, `docs:`, `chore:`, etc.).

## 6. Documentation Rules

- Place Markdown under `docs/`. Use `docs/reports/` for status reports and postmortems, and `docs/internal/` for internal engineering notes.
- Prefer relative links and update them when files move.
- Wrap prose at ~100 characters when editing existing documents unless doing so would damage tables or code blocks.
- Clear notebook outputs before committing (`nbstripout`).

## 7. Examples & Tutorials

- Keep runnable examples under `examples/` with concise README files.
- Document setup requirements for demos in `docs/`.
- Large datasets or generated outputs belong in `artifacts/` and should not be regenerated during tests.

## 8. Tooling & Scripts

- Place automation utilities in `tools/` or `scripts/` with usage documentation.
- Note any external dependencies in the script header and in `docs/` if users must install them separately.

## 9. Generated Files

- Update `.gitignore` if new generated artifacts appear.
- Do not commit temporary experiment output; instead, document reproducible steps in `docs/`.

## 10. Nested `AGENTS.md`

Directories may define additional rules. When multiple instructions apply, follow the most specific (deepest) `AGENTS.md` file.
