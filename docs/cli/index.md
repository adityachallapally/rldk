# RLDK CLI Reference

The RLDK (Reinforcement Learning Diagnostics Kit) command-line interface provides tools for analyzing reward models, running evaluations, and performing forensic analysis on reinforcement learning training runs.

## Quick Start

```bash
# Install RLDK
pip install rldk

# Check installation
rldk --help

# Run reward health analysis
rldk reward-health run --scores training_data.jsonl --out ./reports

# Gate CI based on health results
rldk reward-health gate --from ./reports/reward_health_summary.json

# Run evaluations
rldk evals evaluate test_data.jsonl --suite quick --gate
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `rldk reward-health run` | Analyze reward model health from training data |
| `rldk reward-health gate` | Gate CI based on health analysis results |
| `rldk evals evaluate` | Run evaluation suites on model outputs |
| `rldk forensics` | Forensic analysis tools (coming soon) |

## Global Options

All commands support these shared options:

| Option | Description |
|--------|-------------|
| `--config TEXT` | Path to configuration file (TOML or YAML) |
| `--json` | Print machine-readable JSON output |
| `--verbose, -v` | Enable verbose logging |
| `--quiet` | Reduce logs to warnings and errors only |
| `--log-file TEXT` | Write full logs to specified file |

## Exit Codes

| Code | Meaning | Usage |
|------|---------|-------|
| `0` | Success | Normal operation |
| `2` | Invalid arguments | Usage errors, missing files |
| `3` | Threshold/gate failure | CI integration failures |
| `4` | Runtime error | Evaluation failures, data issues |
| `5` | Internal error | Unexpected exceptions |

## Reward Health Analysis

### `rldk reward-health run`

Analyzes reward model health from training logs to detect issues like:
- Reward hacking and gaming
- Distribution shifts
- Calibration problems
- Shortcut learning

**Usage**:
```bash
rldk reward-health run --scores SCORES_FILE --out OUTPUT_DIR [OPTIONS]
```

**Required Arguments**:
- `--scores TEXT`: Path to scores JSONL file

**Required Options**:
- `--out TEXT`: Output directory for reports

**Optional Options**:
- `--config TEXT`: Configuration file path
- `--adapter TEXT`: Data adapter (custom_jsonl, trl, openrlhf, wandb)
- `--gate`: Enable CI gate mode
- `--json`: JSON output format
- `--verbose, -v`: Verbose logging
- `--quiet`: Quiet mode
- `--log-file TEXT`: Log file path

**Examples**:
```bash
# Basic analysis
rldk reward-health run --scores results.jsonl --out ./reports

# With configuration
rldk reward-health run --scores results.jsonl --config config.yaml --out ./reports

# JSON output for automation
rldk reward-health run --scores results.jsonl --out ./reports --json
```

**Output Files**:
- `reward_health_card.md`: Human-readable health report
- `reward_health_summary.json`: Machine-readable summary
- `calibration_plots.png`: Calibration visualizations

### `rldk reward-health gate`

Gates CI pipelines based on reward health analysis results.

**Usage**:
```bash
rldk reward-health gate --from HEALTH_FILE [OPTIONS]
```

**Required Options**:
- `--from TEXT`: Path to health.json file

**Optional Options**:
- `--min-pass-rate FLOAT`: Minimum pass rate (default: 0.8)
- `--json`: JSON output format
- `--verbose, -v`: Verbose logging
- `--quiet`: Quiet mode
- `--log-file TEXT`: Log file path

**Examples**:
```bash
# Default threshold (80%)
rldk reward-health gate --from health.json

# Custom threshold
rldk reward-health gate --from health.json --min-pass-rate 0.9

# CI integration
rldk reward-health gate --from health.json --json
```

## Evaluation Suite

### `rldk evals evaluate`

Runs comprehensive evaluation suites on model outputs.

**Usage**:
```bash
rldk evals evaluate INPUT_FILE [OPTIONS]
```

**Required Arguments**:
- `INPUT_FILE PATH`: Path to JSONL input file

**Optional Options**:
- `--suite, -s TEXT`: Evaluation suite (quick, comprehensive, safety)
- `--events-column TEXT`: Column name for event logs (default: events)
- `--min-samples INTEGER`: Minimum samples required (default: 10)
- `--timeout INTEGER`: Timeout in seconds (default: 300)
- `--gate`: Enable CI gate mode
- `--json`: JSON output format
- `--verbose, -v`: Verbose logging
- `--quiet`: Quiet mode
- `--log-file TEXT`: Log file path

**Available Suites**:
- `quick`: Fast evaluation with core metrics
- `comprehensive`: Full evaluation with all metrics
- `safety`: Safety-focused evaluation suite

**Examples**:
```bash
# Quick evaluation
rldk evals evaluate data.jsonl --suite quick

# Comprehensive evaluation with gate
rldk evals evaluate data.jsonl --suite comprehensive --gate

# Custom parameters
rldk evals evaluate data.jsonl --timeout 600 --min-samples 50
```

## Data Formats

### Custom JSONL Format

For reward health analysis, use this format:

```json
{
  "global_step": 100,
  "reward_scalar": 0.8,
  "kl_to_ref": 0.05,
  "entropy": 2.3,
  "loss": 0.15,
  "learning_rate": 1e-4,
  "grad_norm": 1.2,
  "clip_frac": 0.1,
  "reward_std": 0.2,
  "tokens_in": 512,
  "tokens_out": 128,
  "wall_time": 100.5,
  "seed": 42,
  "run_id": "test_run_1",
  "git_sha": "abc123",
  "phase": "train"
}
```

### Evaluation Data Format

For evaluations, use this format:

```json
{"output": "Model response text", "events": []}
{"output": "Another response", "events": []}
```

## Configuration

### Configuration Files

Create a configuration file in TOML or YAML format:

```toml
# config.toml
[reward_health]
min_samples = 100
calibration_threshold = 0.8

[evaluations]
timeout = 600
min_samples = 50
```

```yaml
# config.yaml
reward_health:
  min_samples: 100
  calibration_threshold: 0.8

evaluations:
  timeout: 600
  min_samples: 50
```

Use with:
```bash
rldk reward-health run --scores data.jsonl --config config.toml --out ./reports
```

## CI/CD Integration

### GitHub Actions

```yaml
name: RLDK Health Check
on: [push, pull_request]

jobs:
  health-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install RLDK
        run: pip install rldk
      - name: Run health analysis
        run: |
          rldk reward-health run --scores training_data.jsonl --out ./health_reports
          rldk reward-health gate --from ./health_reports/reward_health_summary.json --min-pass-rate 0.9
```

### GitLab CI

```yaml
health_check:
  stage: test
  script:
    - pip install rldk
    - rldk reward-health run --scores training_data.jsonl --out ./health_reports
    - rldk reward-health gate --from ./health_reports/reward_health_summary.json
  artifacts:
    reports:
      junit: ./health_reports/reward_health_summary.json
```

## Troubleshooting

### Common Issues

1. **"No such file or directory"**
   - Check file paths are correct
   - Ensure files exist and are readable

2. **"Invalid JSON"**
   - Validate JSONL format
   - Check for missing commas or brackets

3. **"Insufficient samples"**
   - Increase `--min-samples` parameter
   - Add more data to input file

4. **"Evaluation timeout"**
   - Increase `--timeout` parameter
   - Reduce data size or use `quick` suite

### Debug Mode

Use verbose logging for detailed information:

```bash
rldk reward-health run --scores data.jsonl --out ./reports --verbose --log-file debug.log
```

### Getting Help

```bash
# General help
rldk --help

# Command-specific help
rldk reward-health --help
rldk reward-health run --help
rldk evals evaluate --help
```

## Examples

### Complete Workflow

```bash
# 1. Analyze reward model health
rldk reward-health run --scores training_logs.jsonl --out ./health_reports --adapter custom_jsonl

# 2. Check if health passes threshold
if rldk reward-health gate --from ./health_reports/reward_health_summary.json --min-pass-rate 0.9; then
    echo "Health check passed"
    
    # 3. Run evaluations
    rldk evals evaluate test_data.jsonl --suite comprehensive --gate
    
    # 4. Generate reports
    echo "All checks passed"
else
    echo "Health check failed"
    exit 1
fi
```

### Batch Processing

```bash
# Process multiple datasets
for dataset in data1.jsonl data2.jsonl data3.jsonl; do
    echo "Processing $dataset"
    rldk reward-health run --scores "$dataset" --out "./reports/$(basename "$dataset" .jsonl)"
done

# Aggregate results
rldk reward-health gate --from ./reports/data1/reward_health_summary.json
```

### JSON Output for Automation

```bash
# Get structured results
rldk reward-health run --scores data.jsonl --out ./reports --json > health_results.json
rldk evals evaluate data.jsonl --suite quick --json > eval_results.json

# Process in Python
python -c "
import json
with open('health_results.json') as f:
    health = json.load(f)
    print(f'Health passed: {health[\"passed\"]}')
"
```

## Advanced Usage

### Custom Adapters

RLDK supports multiple data adapters:

- `custom_jsonl`: Custom JSONL format (default)
- `trl`: TRL training logs
- `openrlhf`: OpenRLHF format
- `wandb`: Weights & Biases logs

```bash
rldk reward-health run --scores trl_logs.jsonl --adapter trl --out ./reports
```

### Parallel Processing

For large datasets, consider parallel processing:

```bash
# Split data and process in parallel
split -l 1000 large_data.jsonl chunk_
for chunk in chunk_*; do
    rldk reward-health run --scores "$chunk" --out "./reports/$chunk" &
done
wait
```

### Custom Configuration

```bash
# Use environment variables
export RLDK_CONFIG=./my_config.toml
rldk reward-health run --scores data.jsonl --out ./reports

# Override specific settings
rldk reward-health run --scores data.jsonl --out ./reports --min-samples 200
```

## Contributing

To contribute to RLDK CLI:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

Run tests locally:
```bash
python -m pytest tests/test_cli.py -v
```

## Support

- **Documentation**: [Full CLI Specification](cli_spec.md)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@rldk.dev

## Changelog

### v1.0.0
- Initial CLI release
- Reward health analysis
- Evaluation suites
- CI/CD integration
- Comprehensive testing