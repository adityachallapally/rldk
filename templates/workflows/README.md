# RL Debug Kit CI Gates

This directory contains GitHub Actions workflow templates for using RL Debug Kit's CI gate functionality.

## Overview

The CI gates provide reliable pass/fail checks for reinforcement learning training runs with the following exit codes:

- **0 (PASS)**: All checks passed
- **1 (WARN)**: Non-critical issues detected, but within acceptable thresholds
- **2 (FAIL)**: Critical issues detected that require attention

## Available Gates

### 1. Reward Health Gate

Checks reward model health and detects pathologies:

```bash
rldk reward-health run \
  --scores artifacts/scores.jsonl \
  --config .rldk/health.yaml \
  --out artifacts/health \
  --gate
```

**What it checks:**
- Reward drift detection
- Saturation issues
- Calibration quality
- Shortcut signal detection
- Label leakage risk

**Exit codes:**
- 0: All health checks passed
- 1: Non-critical issues (saturation, calibration, shortcuts)
- 2: Critical issues (drift, label leakage)

### 2. Determinism Gate

Checks if training runs are deterministic:

```bash
rldk check-determinism \
  --runs 2 \
  --tolerance 0.01 \
  --gate
```

**What it checks:**
- Metric consistency across multiple runs
- Non-deterministic operation detection
- RNG state management
- DataLoader determinism

**Exit codes:**
- 0: All runs are deterministic
- 1: Minor variations within tolerance
- 2: Significant non-determinism detected

## Usage

### GitHub Actions

Copy the workflow template to your repository:

```bash
cp templates/workflows/rldk-gates.yml .github/workflows/
```

The workflow will run on every push and pull request, checking both reward health and determinism.

### Local Testing

Test gates locally before pushing:

```bash
# Test reward health gate
rldk reward-health run --scores data/scores.jsonl --out reports/health --gate

# Test determinism gate
rldk check-determinism --runs 3 --tolerance 0.001 --gate
```

### Configuration

Create a `.rldk/health.yaml` file to customize thresholds:

```yaml
threshold_drift: 0.05
threshold_saturation: 0.9
threshold_calibration: 0.8
threshold_shortcut: 0.5
threshold_leakage: 0.2
```

## Integration

The gates are designed to be:

- **Fast**: Optimized for CI environments
- **Reliable**: Consistent exit codes and clear messaging
- **Configurable**: Adjustable thresholds for different use cases
- **Drop-in**: Easy to integrate into existing workflows

## Troubleshooting

### Common Issues

1. **Gate fails with exit code 2**: Check the detailed output for specific issues
2. **False positives**: Adjust thresholds in configuration file
3. **Slow execution**: Reduce number of runs or steps for determinism checks

### Getting Help

- Check the detailed reports in the output directory
- Review the gate output for specific error messages
- Adjust thresholds based on your model's characteristics