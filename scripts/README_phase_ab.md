# Phase A/B Acceptance Test Pack

This directory contains the acceptance test pack for Phase A and Phase B features of RLDK.

## Phase A Features
- PR1: Stream JSONL to TrainingMetrics normalizer
- PR2: Numeric coercion and None guards in schema standardizer  
- PR3: Table to normalized events and reverse helpers

## Phase B Features
- PR4: Reward health CLI uses the normalizer for files and dirs
- PR5: Reward drift supports score file mode in addition to model folders
- PR6: Reward health API accepts DataFrame, list of dicts, and JSONL path

## Running the Tests

### Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

If the project has dev extras:
```bash
pip install -e .[dev]
```

### Run Integration Tests

```bash
pytest -q tests/integration/test_phase_ab_*.py
```

### Run Acceptance Script

```bash
python scripts/phase_ab_acceptance.py
```

## Test Structure

### Integration Tests
- `tests/integration/test_phase_ab_normalization.py` - Phase A normalization features
- `tests/integration/test_phase_ab_reward_cli.py` - Phase B reward CLI features  
- `tests/integration/test_phase_ab_api.py` - Phase B reward API features

### Fixtures
- `tests/fixtures/phase_ab/stream_small.jsonl` - 20 steps of EventWriter-style JSONL
- `tests/fixtures/phase_ab/stream_mixed_types.jsonl` - Mixed types for coercion testing
- `tests/fixtures/phase_ab/table_small.csv` - 5 rows of tabular training metrics
- `tests/fixtures/phase_ab/scores_a.jsonl` - 20 prompts with scores for drift testing
- `tests/fixtures/phase_ab/scores_b.jsonl` - Same prompts with shifted scores

### Utilities
- `rldk/testing/cli_detect.py` - CLI command discovery helper

## Expected Runtime

All tests should complete in under 1 minute on a laptop with no GPU.

## Acceptance Criteria

- All integration tests pass with `pytest`
- Acceptance script shows green summary and exits with code 0
- Reward CLI commands work on valid inputs and fail clearly on invalid inputs
- API accepts all specified input formats with equivalent results
- No test takes more than a few seconds to complete
