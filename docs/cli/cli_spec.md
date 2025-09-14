# RLDK CLI Specification

This document provides a comprehensive specification of the RLDK command-line interface, including all commands, options, exit codes, and behavior.

## Overview

The RLDK CLI is built using Typer and provides a consistent interface for running reinforcement learning diagnostics and evaluations. All commands support shared options for configuration, logging, and output formatting.

## Global Options

All commands support these shared options:

- `--config TEXT`: Path to configuration file (TOML or YAML)
- `--json`: Print machine-readable JSON output
- `--verbose, -v`: Enable verbose logging
- `--quiet`: Reduce logs to warnings and errors only
- `--log-file TEXT`: Write full logs to specified file

## Exit Codes

The CLI uses a consistent exit code policy:

- `0`: Success
- `2`: Invalid arguments or usage errors
- `3`: Threshold/gate failures (CI integration)
- `4`: Runtime errors (evaluation failures, data issues)
- `5`: Internal errors (unexpected exceptions)

## Commands

### Main CLI

```bash
rldk [OPTIONS] COMMAND [ARGS]...
```

**Description**: Main entry point for RLDK CLI.

**Available Commands**:
- `forensics`: Forensic analysis tools
- `reward`: Reward model analysis and health checks
- `evals`: Evaluation suite management

### Reward Health Commands

#### `rldk reward-health`

**Description**: Reward model health analysis and CI gating.

**Subcommands**:
- `run`: Run reward health analysis
- `gate`: Gate CI based on health results

#### `rldk reward-health run`

**Description**: Run reward health analysis on training data.

**Usage**:
```bash
rldk reward-health run --scores SCORES_FILE --out OUTPUT_DIR [OPTIONS]
```

**Required Arguments**:
- `--scores TEXT`: Path to scores JSONL file

**Required Options**:
- `--out TEXT`: Output directory for reports

**Optional Options**:
- `--config TEXT`: Path to configuration file (TOML or YAML)
- `--adapter TEXT`: Adapter type for data ingestion (custom_jsonl, trl, openrlhf, wandb)
- `--gate`: Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)
- `--json`: Print machine-readable JSON output
- `--verbose, -v`: Enable verbose logging
- `--quiet`: Reduce logs to warnings and errors only
- `--log-file TEXT`: Write full logs to specified file

**Examples**:
```bash
# Basic analysis
rldk reward-health run --scores results.jsonl --out ./reports

# With configuration and JSON output
rldk reward-health run --scores results.jsonl --config config.yaml --json

# With specific adapter
rldk reward-health run --scores results.jsonl --out ./reports --adapter trl
```

**Output Files**:
- `reward_health_card.md`: Human-readable health report
- `reward_health_summary.json`: Machine-readable health summary
- `calibration_plots.png`: Calibration visualization plots

**Exit Codes**:
- `0`: Analysis completed successfully
- `2`: Invalid arguments or file paths
- `4`: Runtime errors during analysis
- `5`: Internal errors

#### `rldk reward-health gate`

**Description**: Gate CI based on health.json results.

**Usage**:
```bash
rldk reward-health gate --from HEALTH_FILE [OPTIONS]
```

**Required Options**:
- `--from TEXT`: Path to health.json file

**Optional Options**:
- `--min-pass-rate FLOAT`: Minimum pass rate threshold (default: 0.8)
- `--json`: Print machine-readable JSON output
- `--verbose, -v`: Enable verbose logging
- `--quiet`: Reduce logs to warnings and errors only
- `--log-file TEXT`: Write full logs to specified file

**Examples**:
```bash
# Basic gating with default threshold
rldk reward-health gate --from health.json

# Custom threshold
rldk reward-health gate --from health.json --min-pass-rate 0.9

# JSON output for CI integration
rldk reward-health gate --from health.json --json
```

**Exit Codes**:
- `0`: Health check passed
- `2`: Invalid arguments or file paths
- `3`: Health check failed (threshold not met)
- `5`: Internal errors

### Evaluation Commands

#### `rldk evals evaluate`

**Description**: Run evaluation suite on JSONL data.

**Usage**:
```bash
rldk evals evaluate INPUT_FILE [OPTIONS]
```

**Required Arguments**:
- `INPUT_FILE PATH`: Path to JSONL input file

**Optional Options**:
- `--suite, -s TEXT`: Evaluation suite (quick, comprehensive, safety) (default: output)
- `--events-column TEXT`: Column name containing event logs (default: events)
- `--min-samples INTEGER`: Minimum samples required for evaluation (default: 10)
- `--timeout INTEGER`: Timeout in seconds for evaluation (default: 300)
- `--gate`: Enable CI gate mode with exit codes (0=pass, 3=fail)
- `--json`: Print machine-readable JSON output
- `--verbose, -v`: Enable verbose logging
- `--quiet`: Reduce logs to warnings and errors only
- `--log-file TEXT`: Write full logs to specified file

**Examples**:
```bash
# Quick evaluation
rldk evals evaluate data.jsonl --suite quick

# Comprehensive evaluation with gate mode
rldk evals evaluate data.jsonl --suite comprehensive --gate

# Custom timeout and minimum samples
rldk evals evaluate data.jsonl --timeout 600 --min-samples 50
```

**Exit Codes**:
- `0`: Evaluation completed successfully
- `2`: Invalid arguments or file paths
- `3`: Gate mode failures (insufficient samples, threshold failures)
- `4`: Runtime errors during evaluation
- `5`: Internal errors

**Available Suites**:
- `quick`: Fast evaluation with core metrics
- `comprehensive`: Full evaluation with all available metrics
- `safety`: Safety-focused evaluation suite

## Data Formats

### Custom JSONL Format

The `custom_jsonl` adapter expects data with these fields:

**Required Fields**:
- `global_step`: Training step number
- `reward_scalar`: Reward value
- `kl_to_ref`: KL divergence to reference model

**Optional Fields**:
- `entropy`: Model entropy
- `loss`: Training loss
- `learning_rate`: Learning rate
- `grad_norm`: Gradient norm
- `clip_frac`: Clipping fraction
- `reward_std`: Reward standard deviation
- `tokens_in`: Input tokens
- `tokens_out`: Output tokens
- `wall_time`: Wall clock time
- `seed`: Random seed
- `run_id`: Run identifier
- `git_sha`: Git commit hash
- `phase`: Training phase (e.g., "train")

**Example**:
```json
{"global_step": 100, "reward_scalar": 0.8, "kl_to_ref": 0.05, "entropy": 2.3, "loss": 0.15, "learning_rate": 1e-4, "grad_norm": 1.2, "clip_frac": 0.1, "reward_std": 0.2, "tokens_in": 512, "tokens_out": 128, "wall_time": 100.5, "seed": 42, "run_id": "test_run_1", "git_sha": "abc123", "phase": "train"}
```

### Evaluation Data Format

Evaluation data should be in JSONL format with these fields:

**Required Fields**:
- `output`: Model output text

**Optional Fields**:
- `events`: List of events for detailed analysis

**Example**:
```json
{"output": "This is a model response", "events": []}
{"output": "Another response", "events": []}
```

## Error Handling

### Error Messages

All error messages follow this format:
- Prefix: `ERROR:`
- Description: Clear explanation of the problem
- Hint: Suggested fix or additional information

### Common Error Scenarios

1. **Invalid Arguments** (Exit Code 2):
   - Missing required arguments
   - Invalid option values
   - File not found

2. **Runtime Errors** (Exit Code 4):
   - Data parsing failures
   - Evaluation failures
   - Insufficient data for analysis

3. **Threshold Failures** (Exit Code 3):
   - Health check failures
   - Gate mode threshold violations

4. **Internal Errors** (Exit Code 5):
   - Unexpected exceptions
   - System errors

## Configuration

### Configuration Files

RLDK supports configuration files in TOML or YAML format. Configuration can be specified with the `--config` option.

### Environment Variables

Some options can be set via environment variables:
- `RLDK_CONFIG`: Default configuration file path
- `RLDK_LOG_LEVEL`: Default log level

## Logging

### Log Levels

- `DEBUG`: Detailed diagnostic information
- `INFO`: General information about program execution
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error messages for recoverable errors
- `CRITICAL`: Critical errors that may cause program termination

### Log Output

- Console output: Colored, formatted logs
- File output: Plain text logs when `--log-file` is specified
- JSON output: Machine-readable format when `--json` is specified

## Testing

The CLI includes comprehensive tests covering:
- Command help text and options
- Error handling and exit codes
- JSON output format
- Shared options across commands
- Data validation and processing

Run tests with:
```bash
python -m pytest tests/test_cli.py -v
```

## Examples

### Basic Workflow

1. **Run reward health analysis**:
   ```bash
   rldk reward-health run --scores training_data.jsonl --out ./health_reports
   ```

2. **Gate CI based on health**:
   ```bash
   rldk reward-health gate --from ./health_reports/reward_health_summary.json
   ```

3. **Run evaluations**:
   ```bash
   rldk evals evaluate test_data.jsonl --suite quick --gate
   ```

### CI Integration

```bash
# In CI pipeline
if rldk reward-health gate --from health.json --min-pass-rate 0.9; then
    echo "Health check passed"
else
    echo "Health check failed"
    exit 1
fi
```

### JSON Output for Automation

```bash
# Get machine-readable results
rldk reward-health run --scores data.jsonl --out ./reports --json > results.json
rldk reward-health gate --from ./reports/reward_health_summary.json --json > gate_results.json
```

## Future Enhancements

- Additional data adapters (HuggingFace, Weights & Biases)
- More evaluation suites
- Configuration file templates
- Interactive mode for exploration
- Plugin system for custom metrics