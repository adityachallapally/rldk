# KL Drift Detection Guide

This guide explains how the RL Debug Kit (RLDK) detects, monitors, and visualizes KL drift during PPO-style training. Use it as a quick reference when configuring the new drift trackers, monitoring presets, and visualization cards that ship with the toolkit.

## Why KL Drift Matters

* **Policy stability** – sustained KL divergence increases typically signal runaway policy updates.
* **Reward regressions** – drift often coincides with reward collapse or reward hacking.
* **Safety envelopes** – most production PPO stacks enforce explicit KL bounds; drift monitoring automates those guardrails.

## How Detection Works

1. **Reference window** – the first `reference_period` steps form a baseline KL distribution.
2. **Rolling window** – recent KL values (controlled by `drift_window_size`) are compared against the reference.
3. **Histogram KL divergence** – reference and current windows are binned and a discrete KL divergence is computed.
4. **Severity scoring** – divergence is mapped to a 0–1 score via an exponential curve. Higher means more severe drift.
5. **Trend analysis** – a short history of drift scores is fit with a linear model to classify trends as increasing, decreasing, or stable.
6. **Anomaly surfacing** – when divergence exceeds the configured threshold, the tracker raises critical anomalies.

## Configuration Cheat Sheet

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enable_kl_drift_tracking` | `True` | Toggle drift detection in `ComprehensivePPOForensics` |
| `kl_drift_threshold` | `0.15` | Divergence level that triggers anomalies |
| `kl_drift_window_size` | `100` | Number of recent steps compared against the reference |
| `kl_drift_reference_period` | `500` | Steps used to build the baseline KL distribution |
| `drift_threshold` (monitor preset) | `0.08 – 0.20` | Preset-specific warning/critical gates |

### Comprehensive Forensics Example

```python
from rldk.forensics import ComprehensivePPOForensics

forensics = ComprehensivePPOForensics(
    kl_target=0.1,
    enable_kl_drift_tracking=True,
    kl_drift_threshold=0.12,
    kl_drift_window_size=80,
    kl_drift_reference_period=400,
)

metrics = forensics.update(step=1280, kl=0.18, kl_coef=0.35, entropy=2.1, reward_mean=0.74, reward_std=0.28)
kl_drift = forensics.get_kl_drift_analysis()

if kl_drift["detected"]:
    print(f"Drift score={kl_drift['score']:.3f} (trend={kl_drift['trend']})")
```

### CLI Quick Start

```bash
# Analyze an existing run and emit a KL drift card
rldk forensics kl-drift ./runs/my_run

# Monitor a live training loop with the built-in drift preset
rldk monitor --stream artifacts/run.jsonl --rules kl_drift
```

### Visualization Highlights

`generate_kl_drift_card` produces a multi-panel visualization with:

* Drift score timeline (KL vs severity)
* Reference vs current KL histograms
* Severity heatmap with color-coded bands
* Trend summary and mitigation recommendations
* Export in PNG, SVG, and PDF out of the box

## Troubleshooting

| Symptom | Likely Cause | Suggested Fix |
|---------|--------------|---------------|
| `MISSING_KL_COLUMN` error | Metrics source missing KL field | Map the correct column using `--field-map` or `--kl-col` |
| Drift score always zero | Not enough data to build reference window | Lower `kl_drift_reference_period` or ingest more steps |
| Frequent false positives | Threshold too low for the environment | Increase `kl_drift_threshold` / preset warning levels |
| Visualization lacks histogram | KL column contains NaNs | Clean the source metrics or adjust ingestion filters |

## Best Practices

* **Warm-up period** – allow the reference period to complete before acting on anomalies.
* **Pair with reward checks** – combine KL drift with reward drift to catch coupled failures.
* **Tune per environment** – highly stochastic tasks may need a larger window and higher thresholds.
* **Export artifacts** – keep the generated PNG/SVG/PDF attachments with your training reports for auditability.

For deeper integration details, review `src/rldk/forensics/kl_schedule_tracker.py` and `src/rldk/cards/drift.py` in the repository.
