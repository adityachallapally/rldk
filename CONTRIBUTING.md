# Contributing to RLDK

Thank you for helping keep the Reinforcement Learning Debug Kit reliable and well-documented. This guide explains how to prepare your development environment and how to work with the new repository conventions described in `AGENTS.md`.

## Before You Start

- Review the root [`AGENTS.md`](AGENTS.md) and any nested `AGENTS.md` files in the directories you plan to touch. They contain binding rules for formatting, validation, and directory layout.
- Skim the documentation tree, especially `docs/reports/` and `docs/internal/`, so new material can be filed in the correct location.

## Development Setup

1. Fork the repository and clone your fork.
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .
   ```
1. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```
   Running hooks locally mirrors the checks enforced in CI.

## Repository Layout

- `src/rldk/` – Production code and configuration modules.
- `tests/` – Automated tests. Follow `tests/AGENTS.md` for naming and fixture guidance.
- `docs/` – Documentation. Long-form reports live in `docs/reports/`; internal engineering notes live in `docs/internal/`.
- `examples/`, `recipes/`, `reference/` – Usage patterns and reference material.
- `tools/`, `scripts/` – Developer utilities.
- Root-level files are limited to project-wide assets such as this document, `README.md`, and configuration files.

## Code Style and Quality

We rely on automated tooling to keep the codebase consistent:

- **Formatting**: [`black`](https://github.com/psf/black) and [`isort`](https://pycqa.github.io/isort/) format Python sources.
- **Linting**: [`ruff`](https://github.com/astral-sh/ruff) and [`flake8`](https://github.com/PyCQA/flake8) catch style and correctness issues.
- **Static typing**: [`mypy`](https://mypy-lang.org/) enforces type hints.
- **Spelling and docs**: [`codespell`](https://github.com/codespell-project/codespell) and `mdformat` handle documentation polish.

Run the combined toolchain with:

```bash
pre-commit run --files $(git diff --name-only --cached)
```

## Testing

Execute the automated tests before submitting a pull request:

```bash
pytest
```

Use markers (`-m "not slow"`) or targeted paths (`pytest tests/evals`) when a full suite run is impractical, and report any deviations in the PR description.

## Documentation

- Keep new documentation in `docs/`, using the structure described in `docs/AGENTS.md`.
- Update guides, examples, and reports when behavior or APIs change.
- When moving documents, fix relative links and verify the MkDocs site builds locally if the nav is affected.

## Pull Request Process

1. Create a feature branch from `main`.
1. Make focused changes that keep code, docs, and tests in sync.
1. Run the validation commands listed above.
1. Fill out the pull request template, including testing evidence and references to affected documentation.
1. Request review once the branch is ready and CI is green.

## Release Process

### Prerequisites

- PyPI access for the project
- Trusted publishing configured in PyPI (OIDC)
- GitHub Pages enabled for documentation

### Steps

1. **Update version** in `pyproject.toml`.
1. **Update release notes** in `CHANGELOG.md` or the release description.
1. **Commit changes** related to the release.
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: release prep"
   ```
1. **Tag the release**.
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```
1. **Publish the GitHub release** with the new tag and release notes.

### Automated Workflows

- `publish_pypi.yml` builds and publishes packages when a GitHub release is published.
- `deploy_docs.yml` rebuilds the MkDocs site on pushes to `main` and on tags.

### Troubleshooting

- Verify PyPI trusted publisher settings if publishing fails.
- Confirm GitHub Pages is enabled if documentation does not deploy.
- Ensure the version in `pyproject.toml` matches the git tag.

## Questions

If you need help:

- Open an issue or discussion on GitHub.
- Review existing documentation in the `docs/` directory.
- Reach out to maintainers or contributors listed in recent pull requests.

Thank you for contributing to RLDK!
