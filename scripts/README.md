# Scripts

This directory contains utility scripts for the RL Debug Kit project.

## setup_reference_runs.py

Creates the missing reference runs required for the test suite to pass.

### Usage

```bash
# Setup reference runs
python3 scripts/setup_reference_runs.py

# Or use the Makefile target
make reference:setup

# Clean and recreate reference runs
python3 scripts/setup_reference_runs.py --clean
```

### What it does

- Creates `var/runs/reference/summarization/good/` directory with training logs
- Creates `var/runs/reference/summarization/tokenizer_changed/` directory with training logs  
- Generates minimal dataset manifests in `data/benchmarks/reference_datasets/`
- Creates properly formatted JSONL training logs that match the expected schema

### Fixes

This script fixes the failing test:
- `tests/test_first_divergence.py::test_divergence_cause_identification`

The test was failing because it expected reference runs to exist at:
- `var/runs/reference/summarization/good`
- `var/runs/reference/summarization/tokenizer_changed`

### Generated Files

- `data/benchmarks/reference_datasets/ag_news_manifest.jsonl` - Minimal dataset manifest
- `var/runs/reference/summarization/good/training_log.jsonl` - Good run training log
- `var/runs/reference/summarization/tokenizer_changed/training_log.jsonl` - Tokenizer changed run training log

The training logs contain properly differentiated parameters:
- **Good run**: `pad_direction: "right"`, `truncate_at: 512`
- **Tokenizer changed run**: `pad_direction: "left"`, `truncate_at: 256`