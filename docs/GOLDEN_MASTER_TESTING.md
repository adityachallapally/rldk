# Golden Master Testing System

This document describes the golden master testing system for RL Debug Kit, which ensures zero behavior change during refactoring.

## Overview

The golden master testing system consists of:

1. **Capture Script** (`scripts/capture_golden_master.py`) - Runs every public CLI command and programmatic entry point with synthetic inputs
2. **Replay Script** (`scripts/replay_golden_master.py`) - Replays the same commands and compares results
3. **JSON Schemas** (`scripts/artifact_schemas.py`) - Minimal schemas for each artifact type
4. **End-to-End Runner** (`scripts/run_golden_master_test.sh`) - Complete workflow in fresh virtual environment

## Quick Start

Run the complete golden master test:

```bash
make golden-master-test
```

This will:
- Create a fresh virtual environment
- Install the package in development mode
- Capture golden master
- Replay and compare
- Generate a comprehensive report

## What Gets Tested

### CLI Commands
- `rldk version`
- `rldk --help`
- `rldk ingest`
- `rldk diff`
- `rldk check-determinism`
- `rldk reward-health`
- `rldk eval`
- `rldk track`
- `rldk forensics compare-runs`
- `rldk reward reward-drift`

### Programmatic Entry Points
- `ingest_runs()`
- `first_divergence()`
- `check()` (determinism)
- `health()` (reward health)
- `run()` (evaluation)

## What Gets Captured

For each command/entry point:

1. **Exit Codes** - Exact match required
2. **STDOUT** - Exact match required
3. **STDERR** - Exact match required
4. **JSON Artifacts** - Checksum match for deterministic artifacts, field-level comparison for non-deterministic
5. **JSON Schema Validation** - All artifacts must validate against their schemas

## Artifact Types

The system defines schemas for these artifact types:

- `ingest_result` - Training data ingestion results
- `diff_result` - Run comparison results
- `determinism_result` - Determinism check results
- `reward_health_result` - Reward model health analysis
- `eval_result` - Evaluation suite results
- `golden_master_summary` - Overall test summary
- `determinism_card` - Determinism trust card
- `drift_card` - Drift detection card
- `reward_card` - Reward health card
- `diff_report` - Detailed diff report
- `reward_health_summary` - Reward health summary
- `eval_summary` - Evaluation summary
- `run_comparison` - Run comparison report
- `reward_drift` - Reward drift analysis
- `ckpt_diff` - Checkpoint difference report
- `ppo_scan` - PPO anomaly scan
- `env_audit` - Environment audit
- `replay_comparison` - Replay comparison results
- `tracking_data` - Experiment tracking data

## Output Files

After running the test, you'll get:

- `golden_master_output/` - Directory containing captured golden master
- `golden_master.zip` - Compressed golden master for version control
- `replay_output/` - Directory containing replay comparison results
- `golden_master_test_report.md` - Human-readable test report

## Deterministic vs Non-Deterministic Artifacts

### Deterministic Artifacts
These must have exact checksum matches:
- All core analysis results
- Trust cards
- Summary reports
- Configuration data

### Non-Deterministic Artifacts
These allow field-level differences for stable fields:
- Timestamps
- Duration measurements
- Random seeds (when not explicitly controlled)

## Running Individual Components

### Capture Only
```bash
python scripts/capture_golden_master.py
```

### Replay Only
```bash
python scripts/replay_golden_master.py --golden-master golden_master_output --output-dir replay_output
```

### Schema Validation
```bash
python scripts/artifact_schemas.py
```

## Synthetic Test Data

The system creates synthetic test data including:

- Training run metrics (loss, reward, steps)
- Two slightly different runs for comparison
- Reward model configurations
- Evaluation prompts
- Model checkpoints

This ensures consistent, reproducible testing without external dependencies.

## Integration with CI/CD

The golden master test can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Golden Master Test
  run: make golden-master-test
```

The test will:
- Exit with code 0 if all comparisons pass
- Exit with code 1 if any comparisons fail
- Generate detailed reports for debugging

## Updating the Golden Master

When you make intentional changes that affect behavior:

1. Run the test: `make golden-master-test`
2. Review the differences in `replay_output/replay_comparison_report.txt`
3. If the changes are expected, commit the new `golden_master.zip`
4. Update this documentation if new commands or artifacts are added

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```
   pip install jsonschema
   ```

2. **Virtual Environment Issues**
   ```bash
   # Clean and recreate
   rm -rf rldk_golden_master_test/
   make golden-master-test
   ```

3. **Permission Issues**
   ```bash
   chmod +x scripts/*.sh scripts/*.py
   ```

4. **Package Installation Issues**
   ```bash
   # Install in development mode
   pip install -e .
   ```

### Debugging Failed Tests

1. Check `replay_output/replay_comparison_report.txt` for detailed differences
2. Look at `replay_output/replay_summary.json` for overall statistics
3. Compare artifacts in `golden_master_output/artifacts/` vs current run
4. Check if new commands were added that aren't in the golden master

## Schema Evolution

When adding new artifact types:

1. Add schema to `scripts/artifact_schemas.py`
2. Update `is_deterministic_artifact()` function if needed
3. Add test cases to capture/replay scripts
4. Update this documentation

## Performance Considerations

- Test duration: 2-5 minutes
- Memory usage: ~100MB
- Disk usage: ~50MB for artifacts
- Network: None (all synthetic data)

## Best Practices

1. **Run Before Committing** - Always run golden master test before committing changes
2. **Review Differences** - Understand why tests failed before updating golden master
3. **Keep Schemas Minimal** - Only include required fields in schemas
4. **Document Changes** - Update this document when adding new commands or artifacts
5. **Version Control** - Commit `golden_master.zip` with code changes

## Architecture

```
scripts/
├── capture_golden_master.py      # Main capture logic
├── replay_golden_master.py       # Main replay logic
├── artifact_schemas.py          # JSON schemas
└── run_golden_master_test.sh    # End-to-end runner

Makefile                          # Build system integration
GOLDEN_MASTER_TESTING.md          # This documentation
```

The system is designed to be:
- **Reproducible** - Same results every time
- **Comprehensive** - Tests all public interfaces
- **Fast** - Uses synthetic data and minimal runs
- **Maintainable** - Clear separation of concerns
- **Extensible** - Easy to add new commands and artifacts