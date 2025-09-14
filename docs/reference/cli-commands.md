# CLI Commands Reference

This document provides comprehensive reference for all RLDK CLI commands, generated from actual help text.

## Main CLI

```
Usage: rldk [OPTIONS] COMMAND [ARGS]...

RL Debug Kit - Library and CLI for debugging reinforcement learning training runs
```

### Global Options
- `--help` - Show help message and exit

### Available Commands

#### Core Analysis Commands
- `ingest` - Ingest training runs from various sources
- `diff` - Find first divergence between two training runs
- `check-determinism` - Check if a training command is deterministic
- `bisect` - Find regression using git bisect
- `replay` - Replay a training run with the original seed and verify reproducibility

#### Evaluation Commands
- `eval` - Run evaluation suite with statistical analysis
- `evals` - Evaluation suite commands (subcommand group)

#### Forensics Commands
- `compare-runs` - Compare two training runs and identify divergences
- `diff-ckpt` - Compare two model checkpoints and identify parameter differences
- `env-audit` - Audit environment for determinism and reproducibility
- `log-scan` - Scan training logs for PPO anomalies and issues
- `forensics` - Forensics commands for RL training analysis (subcommand group)

#### Reward Analysis Commands
- `reward-health` - Analyze reward model health and detect pathologies
- `reward-drift` - Compare two reward models and detect drift
- `reward` - Reward model analysis commands (subcommand group)

#### Utility Commands
- `track` - Start tracking an experiment with W&B (default) or file logging
- `doctor` - Run comprehensive diagnostics on a training run or repository
- `format-info` - Show data format information for adapters
- `validate-format` - Validate data format and suggest appropriate adapter
- `version` - Show version information
- `seed` - Manage global seed for reproducible experiments
- `card` - Generate trust cards for RL training runs

## Command Groups

### Forensics Commands (`rldk forensics`)

```
Usage: rldk forensics [OPTIONS] COMMAND [ARGS]...

Forensics commands for RL training analysis
```

#### Available Commands
- `compare-runs` - Compare two training runs and identify divergences
- `diff-ckpt` - Compare two model checkpoints and identify parameter differences
- `env-audit` - Audit environment for determinism and reproducibility
- `log-scan` - Scan training logs for PPO anomalies and issues
- `doctor` - Run comprehensive diagnostics on a training run or repository

### Evaluation Commands (`rldk evals`)

```
Usage: rldk evals [OPTIONS] COMMAND [ARGS]...

Evaluation suite commands
```

#### Available Commands
- `evaluate` - Run evaluation suite on JSONL data
- `list-suites` - List available evaluation suites
- `validate-data` - Validate JSONL file structure and data
- `validate` - Validate JSONL file structure and data (alias for validate-data)

### Reward Analysis Commands (`rldk reward`)

```
Usage: rldk reward [OPTIONS] COMMAND [ARGS]...

Reward model analysis commands
```

#### Available Commands
- `reward-drift` - Compare two reward models and detect drift
- `reward-health` - Reward health analysis commands

## Detailed Command Reference

### `rldk check-determinism`

```
Usage: rldk check-determinism [OPTIONS]

Check if a training command is deterministic.
```

#### Options
- `--cmd, -c TEXT` - Command to run for testing
- `--compare, -m TEXT` - Metrics to compare (comma-separated)
- `--steps, -s TEXT` - Specific steps to compare (comma-separated)
- `--stride INTEGER` - Step interval for comparison if steps not specified [default: 50]
- `--replicas, -r INTEGER` - Number of replicas to run [default: 5]
- `--runs INTEGER` - Number of runs for determinism check (alias for replicas)
- `--tolerance, -t FLOAT` - Tolerance for metric differences [default: 0.01]
- `--device, -d TEXT` - Device to use (auto-detected if None)
- `--output-dir, -o TEXT` - Output directory for reports [default: determinism_analysis]
- `--gate` - Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)

#### Examples
```bash
# Basic determinism check
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean

# Check with specific tolerance and replicas
rldk check-determinism --cmd "python train.py --seed 42" --compare loss,reward_mean --tolerance 0.001 --replicas 10

# CI gate mode for automated testing
rldk check-determinism --cmd "python train.py" --compare loss --gate
```

### `rldk ingest`

```
Usage: rldk ingest [OPTIONS] RUNS

Ingest training runs from various sources.
```

#### Arguments
- `runs` (required) - Path to runs directory, file, or wandb:// URI

#### Options
- `--adapter, -a TEXT` - Adapter type (trl, openrlhf, wandb, custom_jsonl, flexible)
- `--output, -o TEXT` - Output file path [default: metrics.jsonl]
- `--field-map TEXT` - JSON string or file path with field mapping (e.g., '{"step": "global_step"}')
- `--config-file, -c TEXT` - YAML/JSON config file with field mapping
- `--validation-mode TEXT` - Validation mode: strict, flexible, or lenient [default: flexible]
- `--required-fields TEXT` - Comma-separated list of required fields (default: step,reward)
- `--validate / --no-validate` - Validate input data before processing [default: validate]
- `--verbose, -v` - Enable verbose output

#### Examples
```bash
# Ingest TRL logs
rldk ingest /path/to/logs --adapter trl

# Ingest from WandB
rldk ingest wandb://entity/project/run_id --adapter wandb

# Ingest with custom output
rldk ingest data.jsonl --adapter custom_jsonl --output results.jsonl

# Ingest with flexible adapter and field mapping
rldk ingest data.jsonl --adapter flexible --field-map '{"step": "global_step"}'

# Ingest with config file
rldk ingest data.jsonl --config-file field_mapping.yaml --validation-mode strict

# Ingest with required fields
rldk ingest data.jsonl --required-fields step,reward,kl --validation-mode lenient
```

### `rldk diff`

Find first divergence between two training runs.

#### Examples
```bash
# Compare two runs
rldk diff --a run_a --b run_b --signals loss,reward_mean

# Compare with custom threshold
rldk diff --a run_a.jsonl --b run_b.jsonl --threshold 0.1
```

### `rldk replay`

Replay a training run with the original seed and verify reproducibility.

#### Examples
```bash
# Replay with original seed
rldk replay ./run --command "python train.py --seed {seed}" --metrics loss,reward_mean

# Replay with tolerance
rldk replay ./run --command "python train.py --seed {seed}" --metrics loss --tolerance 0.01
```

### `rldk evals evaluate`

Run evaluation suite on JSONL data.

#### Examples
```bash
# Run comprehensive evaluation
rldk evals evaluate data.jsonl --suite comprehensive --output results.json

# Run quick evaluation
rldk evals evaluate data.jsonl --suite quick --output results.json

# Run safety evaluation
rldk evals evaluate data.jsonl --suite safety --output results.json
```

### `rldk evals list-suites`

List available evaluation suites.

#### Examples
```bash
# List all available suites
rldk evals list-suites
```

### `rldk forensics compare-runs`

Compare two training runs and identify divergences.

#### Examples
```bash
# Compare two training runs
rldk forensics compare-runs run_a run_b

# Compare with custom output
rldk forensics compare-runs run_a run_b --output comparison_report.json
```

### `rldk forensics env-audit`

Audit environment for determinism and reproducibility.

#### Examples
```bash
# Audit current environment
rldk forensics env-audit ./my_training_run

# Audit with detailed output
rldk forensics env-audit ./my_training_run --verbose
```

### `rldk forensics log-scan`

Scan training logs for PPO anomalies and issues.

#### Examples
```bash
# Scan logs for anomalies
rldk forensics log-scan ./my_training_run

# Scan with custom thresholds
rldk forensics log-scan ./my_training_run --kl-threshold 0.5
```

### `rldk forensics diff-ckpt`

Compare two model checkpoints and identify parameter differences.

#### Examples
```bash
# Compare two checkpoints
rldk forensics diff-ckpt model_a.pt model_b.pt

# Compare with detailed analysis
rldk forensics diff-ckpt model_a.pt model_b.pt --detailed
```

### `rldk forensics doctor`

Run comprehensive diagnostics on a training run or repository.

#### Examples
```bash
# Run diagnostics on training run
rldk forensics doctor ./my_training_run

# Run diagnostics on repository
rldk forensics doctor ./my_repo --repo-mode
```

### `rldk reward reward-drift`

Compare two reward models and detect drift.

#### Examples
```bash
# Compare reward models
rldk reward reward-drift model_a model_b --prompts prompts.jsonl

# Compare with custom output
rldk reward reward-drift model_a model_b --prompts prompts.jsonl --output drift_report.json
```

### `rldk reward reward-health`

Analyze reward model health and detect pathologies.

#### Examples
```bash
# Analyze reward health
rldk reward reward-health run --scores scores.jsonl --out ./reports

# Health analysis with custom thresholds
rldk reward reward-health run --scores scores.jsonl --threshold 0.8
```

### `rldk seed`

Manage global seed for reproducible experiments.

#### Examples
```bash
# Set global seed
rldk seed --seed 42 --deterministic

# Show current seed
rldk seed --show

# Validate environment
rldk seed --env --validate
```

### `rldk card`

Generate trust cards for RL training runs.

#### Examples
```bash
# Generate determinism card
rldk card determinism run_a

# Generate drift card
rldk card drift run_a run_b

# Generate reward card
rldk card reward run_a
```

### `rldk track`

Start tracking an experiment with W&B (default) or file logging.

#### Examples
```bash
# Start interactive tracking
rldk track "my_experiment" --interactive

# Start tracking with file backend
rldk track "my_experiment" --backend file
```

### `rldk version`

Show version information.

#### Examples
```bash
# Show version
rldk version
```

## Exit Codes

Most RLDK commands use standard exit codes:
- `0` - Success
- `1` - General error
- `2` - Critical failure (when using `--gate` mode)

## Configuration

RLDK can be configured through:
- Environment variables (prefixed with `RLDK_`)
- Configuration files
- Command-line arguments

See the [Configuration Guide](../user-guide/configuration.md) for details.

## Integration Examples

### CI/CD Integration

```bash
# In your CI pipeline
rldk check-determinism --cmd "python train.py" --compare loss --gate
if [ $? -eq 0 ]; then
    echo "✅ Training is deterministic"
elif [ $? -eq 1 ]; then
    echo "⚠️ Training has minor non-determinism"
else
    echo "❌ Training is non-deterministic"
    exit 1
fi
```

### Automated Analysis Pipeline

```bash
#!/bin/bash
# Complete analysis pipeline

# 1. Ingest training data
rldk ingest ./logs --adapter trl --output metrics.jsonl

# 2. Run evaluations
rldk evals evaluate metrics.jsonl --suite comprehensive --output eval_results.json

# 3. Check determinism
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean

# 4. Generate cards
rldk card determinism ./run
rldk card reward ./run

# 5. Run forensics
rldk forensics env-audit ./run
rldk forensics log-scan ./run
```

For more examples and detailed usage, see the [User Guide](../user-guide/) and [Examples](../examples/).
