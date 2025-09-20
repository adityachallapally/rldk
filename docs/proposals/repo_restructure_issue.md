# Repo Restructure Tracking Issue

## Summary
The current repository layout scatters production code, test suites, generated assets, and narrative documentation across the root directory. This issue proposes a staged reorganization that centralizes importable code under `src/`, groups documentation by purpose, and relocates generated artifacts into a single, ignored area. The goal is a predictable structure that speeds onboarding, clarifies ownership, and reduces noise in diffs and reviews.

## Goals
- Preserve existing functionality while clarifying the purpose of each top-level directory.
- Align module namespaces with their domains (monitoring, evaluations, pipelines, integrations, etc.).
- Reduce the number of root-level Markdown summaries by moving them into the documentation tree.
- Separate versioned assets (fixtures, benchmarks) from ephemeral runtime output.
- Keep CI, packaging, and documentation tooling working after the move.

## Proposed Top-Level Layout
```
.
├── docs/
│   ├── guides/
│   ├── reference/
│   └── change-logs/
├── src/
│   └── rldk/
│       ├── core/                # config, io, utils primitives
│       ├── pipelines/           # acceptance, ingest, determinism, replay
│       ├── evaluations/         # evals, reward, testing cards
│       ├── monitoring/          # monitor, tracking, adapters
│       ├── integrations/        # openrlhf, trl (remains namespaced)
│       ├── cli/                 # CLI entry points and cards
│       └── support/             # emit.py, diff/, bisect/, templating helpers
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── golden/
├── examples/                    # retain demos; move *_demo directories here
├── scripts/                     # helper CLIs (current utils/, tools/, scripts/reference)
├── configs/                     # recipes/, rules/, mkdocs extras
├── data/
│   ├── fixtures/                # data/fixtures/test_artifacts/, data/fixtures/reference_expected/
│   └── benchmarks/              # curated CSV/JSON fixtures
├── var/                         # artifacts/, runs/, tracking_demo_output/ (gitignored)
└── tooling/                     # profiling dashboards, generated dashboards
```

## Migration Phases
1. **Planning (this issue):** confirm directory mapping, owners, and acceptance criteria. Document import namespace changes (`rldk.monitor` → `rldk.monitoring`, etc.).
2. **Automated move:** create a script to relocate directories, adjust package `__init__` files, and rewrite imports. Update `pyproject.toml` and MkDocs nav.
3. **Verification:** run the full `pytest` suite and existing smoke tests (`run_tests.py`, shell scripts in `tests/`). Confirm CI configs still point at valid paths. Ensure `mkdocs build` works.
4. **Cleanup:** update `.gitignore`, delete legacy stubs (e.g., `PROJECT_STRUCTURE.md`), and document the new layout in `README.md`.

## Acceptance Criteria
- `pyproject.toml` and packaging continue to install `rldk` correctly.
- `pytest` passes locally with new import paths.
- No tracked generated artifacts remain outside `var/` or `data/`.
- Documentation build succeeds and navigation references the relocated files.
- Project README reflects the new structure.

## Open Questions
- Which generated assets are still required for regression baselines vs. safe to ignore?
- Should `rlhf_core/` live under `src/rldk/integrations/` or be exposed as a top-level namespace?
- Are any of the `test_*` top-level scripts relied on by CI jobs that will need path tweaks?
- Do we need to preserve symlinks or compatibility shims for third-party consumers?
