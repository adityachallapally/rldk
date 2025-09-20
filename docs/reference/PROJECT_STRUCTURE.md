# RLDK Project Structure Guidelines

## 🏗️ Overview

RLDK now groups library code, documentation, configuration, and generated data into predictable domains. Follow this layout whenever you add new modules or assets to keep the repository easy to navigate and audit.

## 📁 Root Layout

```
/workspace/
├── src/                        # Installable library code
│   └── rldk/
│       ├── core/               # Config, IO, shared utilities
│       ├── pipelines/          # Ingestion, determinism, replay flows
│       ├── evaluations/        # Evals, reward health, forensic tools
│       ├── monitoring/         # Live monitors, adapters, tracking
│       ├── integrations/       # TRL, OpenRLHF, and third-party bridges
│       ├── cli/                # Typer CLI entry points
│       └── support/            # Diffing, bisect, emit helpers
├── tests/                      # Pytest suites (mirrors src domains)
│   ├── unit/
│   ├── integration/
│   │   └── legacy/             # Archived wide-scope regressions
│   ├── e2e/
│   ├── reward_health/
│   ├── fixtures/
│   ├── data/
│   ├── smoke/
│   └── conftest.py, helpers
├── docs/                       # MkDocs content
│   ├── guides/                 # How-to and onboarding guides
│   ├── change-logs/            # Fix summaries and historical notes
│   ├── reference/              # API docs, architecture records, tasks
│   └── getting-started/, evals/, blog/, ...
├── examples/                   # Tutorials, demos, notebooks
├── scripts/                    # Operational scripts and utilities
│   ├── reference/              # Legacy baseline tooling
│   └── utils/                  # Shared script helpers
├── configs/                    # Versioned configuration bundles
│   ├── recipes/                # Health presets and sample configs
│   ├── rules/                  # Monitor rule packs
│   └── rules.yaml              # Default monitor configuration
├── data/                       # Checked-in fixtures and benchmarks
│   ├── fixtures/               # JSONL, CSV, PNG baselines
│   └── benchmarks/             # Heavier reference outputs
├── tooling/                    # Profilers, dashboards, Streamlit apps
├── assets/                     # Images used by docs and marketing
├── var/                        # Generated artifacts (gitignored)
├── requirements*.txt
├── pyproject.toml
├── mkdocs.yml
├── Makefile
├── Dockerfile
├── README.md
└── CONTRIBUTING.md, PR_GUIDELINES.md
```

## 🧠 Source Package (`src/rldk`)

```
src/rldk/
├── __init__.py                # Public facade + lazy imports
├── cli/                       # Typer commands (`main.py`, __init__.py)
├── core/                      # Config schemas, IO, utility helpers
├── pipelines/                 # Ingestion, determinism, replay orchestration
├── evaluations/               # Reward health, eval suites, forensic scanners
├── monitoring/                # Monitor engine, adapters, tracking
├── integrations/              # TRL/OpenRLHF shims, rlhf_core
└── support/                   # Diffing, bisect, emit, shared templates
```

## ✅ File Placement Rules

- New runtime modules belong under the domain-matched subpackage in `src/rldk/`.
- CLI additions live in `src/rldk/cli/` and export Typer commands through `app`.
- Shared helpers should go in `src/rldk/support/` (for library code) or `scripts/utils/` (for operational scripts).
- Packaging metadata and configuration stay in `configs/` and `pyproject.toml`.

## 🧪 Test Layout

Mirror the source package when adding tests:

- **Unit tests:** `tests/unit/<domain>/test_*.py`
- **Integration tests:** `tests/integration/<domain>/...`
- **Legacy broad tests:** keep or refactor files in `tests/integration/legacy/`
- **Fixtures & data:** `tests/fixtures/` and `tests/data/`
- **Smoke & reward health suites:** dedicated subdirectories

Generate ephemeral outputs inside `var/` or a temporary directory, not alongside tests.

## 🗂️ Documentation

- User-facing guides → `docs/guides/`
- Historical change summaries → `docs/change-logs/`
- Architectural references, task specs, baseline expectations → `docs/reference/`
- Keep MkDocs navigation (`mkdocs.yml`) in sync when relocating files.

## ⚙️ Configurations & Data

- Default monitor rules: `configs/rules.yaml`
- Detailed rule packs: `configs/rules/`
- Health presets and recipes: `configs/recipes/`
- Long-lived benchmark outputs: `data/benchmarks/`
- Test fixtures copied into CI: `data/fixtures/`

## 🧾 Generated Artifacts

- Runtime outputs, experiment logs, and large JSON/CSV dumps should live under `var/`.
- Ensure `.gitignore` excludes `var/` so CI and contributors do not commit transient files.

Adhering to this structure keeps imports clean, shortens review cycles, and avoids accidental churn from regenerated assets.
