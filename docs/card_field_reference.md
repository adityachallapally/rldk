# RLDK Card Field Reference

This document provides a comprehensive reference for all fields in RLDK trust cards, including their meanings, data types, and interpretation guidelines.

## Overview

RLDK generates three types of trust cards as first-class artifacts:

1. **Determinism Cards** - Assess reproducibility and consistency
2. **Drift Cards** - Compare runs and detect divergences  
3. **Reward Cards** - Analyze reward model health and behavior

All cards are saved as both JSON files and PNG visualizations in the `runs/{run_id}/rldk_cards/` directory.

## Common Fields

All cards share these common fields:

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Card schema version (e.g., "1.0") |
| `run_id` | string | Unique identifier for the training run |
| `generated_at` | string | ISO timestamp when card was generated |

## Determinism Card Fields

### Core Assessment Fields

| Field | Type | Description |
|-------|------|-------------|
| `passed` | boolean | Overall determinism assessment (true = deterministic) |
| `replicas` | integer | Number of replica runs analyzed |
| `metrics_compared` | array[string] | List of metrics used in comparison |
| `replica_variance` | object | Variance for each metric across replicas |

### RNG Configuration

| Field | Type | Description |
|-------|------|-------------|
| `rng_map.python_hash_seed` | string/null | Python hash seed setting |
| `rng_map.torch_deterministic` | boolean | Whether torch deterministic mode is enabled |
| `rng_map.torch_seed` | string/null | Torch seed configuration |
| `rng_map.numpy_seed` | string/null | NumPy seed configuration |
| `rng_map.random_seed` | string/null | Python random seed configuration |

### Issues and Fixes

| Field | Type | Description |
|-------|------|-------------|
| `mismatches` | array[object] | Metric mismatches between replicas |
| `fixes` | array[string] | Recommended fixes for determinism issues |
| `nondeterminism_hints` | array[string] | Patterns suggesting non-determinism |
| `flags` | object | Environment flags affecting determinism |

### Mismatch Object Structure

```json
{
  "step": 5,
  "metric": "reward_mean", 
  "replica_1": 0.5234,
  "replica_2": 0.5241,
  "variance": 0.0012
}
```

### Flags Object Structure

```json
{
  "cudnn_deterministic": true,
  "cudnn_benchmark": false,
  "tokenizers_parallelism": "false"
}
```

## Drift Card Fields

### Divergence Detection

| Field | Type | Description |
|-------|------|-------------|
| `run_a` | string | Identifier for first run |
| `run_b` | string | Identifier for second run |
| `diverged` | boolean | Whether divergence was detected |
| `first_step` | integer/null | Step where divergence first occurred |
| `tripped_signals` | array[string] | Metrics that triggered divergence detection |
| `suspected_causes` | array[string] | Potential causes of divergence |

### Reproducibility Information

| Field | Type | Description |
|-------|------|-------------|
| `repro.command` | string | Command to reproduce the analysis |
| `repro.changes` | array[string] | Changes detected between runs |

### Detailed Analysis

| Field | Type | Description |
|-------|------|-------------|
| `details.kl_divergence` | object | KL divergence measurements by step |
| `details.reward_drift` | object | Reward drift analysis |
| `details.metric_correlations` | object | Correlations between metrics |
| `details.drift_patterns` | object | Specific drift patterns detected |
| `notes` | array[string] | Additional analysis notes |

### Reward Drift Object Structure

```json
{
  "correlation": 0.92,
  "mae": 0.08
}
```

## Reward Card Fields

### Health Assessment

| Field | Type | Description |
|-------|------|-------------|
| `passed` | boolean | Overall reward health assessment |
| `drift_detected` | boolean | Whether reward drift was detected |
| `calibration_score` | float | Reward model calibration score (0-1) |
| `saturation_detected` | boolean | Whether reward model is saturated |
| `shortcut_signals` | array[string] | Signs of shortcut learning |
| `label_noise` | float | Estimated label noise level (0-1) |

### Metrics Analysis

| Field | Type | Description |
|-------|------|-------------|
| `metrics.correlation` | float | Correlation between reward and other metrics |
| `metrics.mae` | float | Mean absolute error |
| `metrics.l2_distance` | float | L2 distance from baseline |

### Slice Analysis

| Field | Type | Description |
|-------|------|-------------|
| `slice_analysis` | object | Analysis by data slices (e.g., phases, domains) |

### Slice Object Structure

```json
{
  "math": {
    "delta_mean": 0.02,
    "n_samples": 150
  },
  "code": {
    "delta_mean": -0.01, 
    "n_samples": 200
  }
}
```

### Recommendations

| Field | Type | Description |
|-------|------|-------------|
| `recommendations` | array[string] | Actionable recommendations for improvement |

## File Naming Convention

Cards are saved with stable filenames:

- **Determinism Card**: `determinism_card.json` and `determinism_card.png`
- **Drift Card**: `drift_card.json` and `drift_card.png`  
- **Reward Card**: `reward_card.json` and `reward_card.png`

## Interpretation Guidelines

### Determinism Card

- **PASS**: All replicas produce consistent results
- **FAIL**: Significant variance between replicas detected
- **High variance** (>0.1) in any metric indicates potential issues
- **Multiple seeds** suggest intentional variation for robustness testing

### Drift Card

- **Diverged = true**: Runs have significantly different behavior
- **First step**: Pinpoints exact moment of divergence
- **Tripped signals**: Which metrics detected the divergence
- **Correlation < 0.8**: Suggests significant drift

### Reward Card

- **PASS**: Reward model appears healthy and well-calibrated
- **Calibration score > 0.7**: Good calibration
- **Drift detected**: Reward model behavior has changed over time
- **Saturation**: Rewards are stuck at extreme values
- **Shortcut signals**: Model may be exploiting unintended patterns

## CLI Usage

Generate cards using the `rldk card` command:

```bash
# Determinism card
rldk card determinism runs/clean_ppo

# Drift card  
rldk card drift runs/clean_ppo runs/doctored_ppo

# Reward card
rldk card reward runs/clean_ppo

# Reward card from JSONL stream with preset mapping
rldk card reward logs/reward_stream.jsonl --preset trl
```

The command accepts run directories or standalone metrics files (JSONL, CSV, TSV,
or Parquet) and supports `--preset` / `--field-map` options to align custom
column names with the TrainingMetrics schema.

## Schema Validation

All cards are validated against JSON schemas defined in `src/rldk/io/schemas.py`:

- `DeterminismCardV2` - Determinism card schema
- `DriftCardV1` - Drift card schema  
- `RewardCardV1` - Reward card schema

## Best Practices

1. **Generate cards early**: Create cards during development to catch issues
2. **Compare systematically**: Use drift cards to compare before/after changes
3. **Monitor trends**: Track reward cards over time to detect degradation
4. **Act on recommendations**: Follow the suggested fixes and recommendations
5. **Version control**: Commit cards alongside code changes for reproducibility

## Troubleshooting

### Common Issues

- **Empty cards**: Ensure training data contains required metrics
- **Missing visualizations**: Check matplotlib installation and backend
- **Schema validation errors**: Verify card data matches expected format
- **High variance**: Review RNG settings and environment configuration

### Debugging Tips

- Check the `notes` field for additional context
- Examine `nondeterminism_hints` for specific issues
- Review `suspected_causes` for potential root causes
- Use `repro.command` to rerun analysis