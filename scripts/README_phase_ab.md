# Phase A/B Acceptance Runner

This directory contains a one-shot acceptance runner and supporting instructions for
validating the Phase A and Phase B normalization and reward health features.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

If the project does not expose a `dev` extra, use `pip install -e .` instead.

## Running the checks

```bash
pytest -q tests/integration
python scripts/phase_ab_acceptance.py
```

Both commands should complete without failures. The acceptance script writes a JSON summary
under `artifacts/phase_ab_acceptance/summary.json` and prints a colorized table of results.
