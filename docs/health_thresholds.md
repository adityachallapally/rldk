# Health Thresholds Guide

This guide describes the default health thresholds used by RL Debug Kit's reward health analysis system. These thresholds provide opinionated defaults that give good out-of-the-box pass/fail behavior without requiring manual tuning for most use cases.

## Overview

The health analysis system includes several detectors that monitor different aspects of reward model behavior. Each detector has two thresholds:

- **Warn**: Issues that should be noted but don't necessarily indicate failure
- **Fail**: Critical issues that indicate the reward model has serious problems

## Detectors

### 1. Reward Length Correlation (`reward_length_correlation`)

**Purpose**: Detects correlation between reward scores and response length.

**Statistic**: Pearson correlation coefficient between reward scores and token count.

**Rationale**: Strong correlation with length often indicates the reward model is biased toward longer or shorter responses rather than quality. This can lead to models that generate unnecessarily verbose or terse outputs.

**Thresholds**:
- **Warn**: `|correlation| > 0.25`
- **Fail**: `|correlation| > 0.40`

**How to Override**:
```yaml
detectors:
  reward_length_correlation:
    enabled: true
    thresholds:
      warn: 0.20  # More sensitive
      fail: 0.35
```

### 2. Saturation High Tail (`saturation_high_tail`)

**Purpose**: Detects high tail saturation in reward distribution.

**Statistic**: Percentage of rewards above the 95th percentile.

**Rationale**: When too many rewards cluster at the high end, it suggests the reward model lacks discrimination between good and excellent responses. This can lead to training instability and poor convergence.

**Thresholds**:
- **Warn**: `> 15%` of rewards above 95th percentile
- **Fail**: `> 30%` of rewards above 95th percentile

**How to Override**:
```yaml
detectors:
  saturation_high_tail:
    enabled: true
    thresholds:
      warn: 0.10  # More strict
      fail: 0.25
```

### 3. Expected Calibration Error (`ece_pairwise`)

**Purpose**: Measures calibration quality using Expected Calibration Error.

**Statistic**: Expected Calibration Error between predicted and actual reward rankings.

**Rationale**: Poor calibration means the reward model's confidence doesn't match its accuracy. This leads to unreliable reward signals during training and can cause the policy to learn suboptimal behaviors.

**Thresholds**:
- **Warn**: `ECE > 0.08`
- **Fail**: `ECE > 0.12`

**How to Override**:
```yaml
detectors:
  ece_pairwise:
    enabled: true
    thresholds:
      warn: 0.06  # More strict
      fail: 0.10
```

### 4. Slice Drift Domain (`slice_drift_domain`)

**Purpose**: Detects drift between different data domains/slices.

**Statistic**: Maximum drift score across different data slices (e.g., different topics, sources, or time periods).

**Rationale**: Domain drift indicates the reward model behaves differently across different types of data. This can lead to inconsistent training signals and poor generalization.

**Thresholds**:
- **Warn**: `drift > 0.20`
- **Fail**: `drift > 0.30`

**How to Override**:
```yaml
detectors:
  slice_drift_domain:
    enabled: true
    thresholds:
      warn: 0.15  # More sensitive
      fail: 0.25
```

### 5. Label Flip Sensitivity (`label_flip_sensitivity`)

**Purpose**: Measures sensitivity to label flips in training data.

**Statistic**: Change in reward scores when training labels are randomly flipped.

**Rationale**: High sensitivity to label noise indicates the reward model is overfitting to the training data and may not generalize well. This can lead to brittle reward signals.

**Thresholds**:
- **Warn**: `sensitivity > 0.10`
- **Fail**: `sensitivity > 0.20`

**How to Override**:
```yaml
detectors:
  label_flip_sensitivity:
    enabled: true
    thresholds:
      warn: 0.08  # More strict
      fail: 0.15
```

## Configuration

### Using Default Thresholds

By default, RL Debug Kit uses the thresholds defined in `recipes/health_default.yaml`. No configuration is needed:

```bash
rldk reward-health run --scores data/scores.jsonl --out artifacts/health
```

### Overriding Thresholds

Create a custom configuration file to override specific thresholds:

```yaml
# my_health_config.yaml
detectors:
  reward_length_correlation:
    enabled: true
    thresholds:
      warn: 0.20  # More sensitive to length bias
      fail: 0.35
  saturation_high_tail:
    enabled: false  # Disable this detector
```

Then use it with the CLI:

```bash
rldk reward-health run --scores data/scores.jsonl --out artifacts/health --config my_health_config.yaml
```

### Configuration Merging

User configurations are merged with the defaults, so you only need to specify the detectors you want to override. The system will:

1. Load the default configuration from `recipes/health_default.yaml`
2. Load your custom configuration
3. Merge them (your settings take precedence)
4. Validate the final configuration

### Legacy Thresholds

For backward compatibility, the system also supports legacy threshold names:

```yaml
# Legacy format (still supported)
threshold_drift: 0.1
threshold_saturation: 0.8
threshold_calibration: 0.7
threshold_shortcut: 0.6
threshold_leakage: 0.3
threshold_length_bias: 0.4
enable_length_bias_detection: true
```

## Best Practices

### When to Adjust Thresholds

1. **Domain-specific requirements**: If your use case has different quality standards
2. **Data characteristics**: If your training data has known biases or patterns
3. **Model architecture**: If you're using a different reward model architecture
4. **Iterative refinement**: Start with defaults and adjust based on results

### Threshold Selection Guidelines

1. **Start with defaults**: The provided thresholds work well for most cases
2. **Monitor over time**: Track how often you hit warn vs fail thresholds
3. **Consider trade-offs**: Stricter thresholds catch more issues but may be too sensitive
4. **Validate with experts**: Have domain experts review the detected issues

### Disabling Detectors

You can disable detectors that aren't relevant for your use case:

```yaml
detectors:
  reward_length_correlation:
    enabled: false  # Disable if length bias is acceptable

# Legacy switch to disable the dedicated detector
enable_length_bias_detection: false
```

## Troubleshooting

### Common Issues

1. **Too many warnings**: Consider relaxing warn thresholds
2. **Missing failures**: Consider tightening fail thresholds
3. **Configuration errors**: Use the validation in the config loader to check your YAML

### Getting Help

- Check the CLI output for specific error messages
- Review the generated health reports for detailed analysis
- Consult the [RL Debug Kit documentation](../README.md) for more information

## Examples

### Strict Quality Control

For applications requiring high-quality reward models:

```yaml
detectors:
  reward_length_correlation:
    thresholds:
      warn: 0.15
      fail: 0.25
  ece_pairwise:
    thresholds:
      warn: 0.05
      fail: 0.08
```

### Relaxed Monitoring

For experimental or research settings:

```yaml
detectors:
  reward_length_correlation:
    thresholds:
      warn: 0.35
      fail: 0.50
  saturation_high_tail:
    thresholds:
      warn: 0.25
      fail: 0.40
```

### Custom Domain

For domain-specific applications:

```yaml
detectors:
  reward_length_correlation:
    enabled: false  # Length bias acceptable in this domain
  slice_drift_domain:
    thresholds:
      warn: 0.10  # More sensitive to domain drift
      fail: 0.20
```