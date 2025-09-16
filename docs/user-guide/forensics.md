# Forensics Analysis

RLDK provides comprehensive forensics analysis to detect training anomalies and debug issues in RL training runs.

## Overview

The forensics system analyzes training logs and checkpoints to identify:
- **PPO training anomalies** with 30+ detection rules
- **Gradient instabilities** and exploding/vanishing gradients
- **KL divergence issues** and schedule problems
- **Advantage function problems** and distribution shifts
- **Environment inconsistencies** and non-deterministic behavior

## Command Line Usage

### Environment Audit
Detect non-deterministic behavior and environment issues:

```bash
# Basic environment audit
rldk forensics env-audit ./my_training_run

# Detailed audit with specific checks
rldk forensics env-audit ./my_training_run --verbose --check-cuda --check-threading
```

### Log Scanning
Analyze training logs for PPO anomalies:

```bash
# Scan logs for anomalies
rldk forensics log-scan ./my_training_run

# Scan with custom thresholds
rldk forensics log-scan ./my_training_run --kl-threshold 0.1 --grad-threshold 10.0
```

### Checkpoint Comparison
Compare model checkpoints to track changes:

```bash
# Compare two checkpoints
rldk forensics diff-ckpt checkpoint_100.pt checkpoint_200.pt

# Compare with detailed analysis
rldk forensics diff-ckpt checkpoint_100.pt checkpoint_200.pt --detailed --output diff_report.json
```

### Comprehensive Diagnostics
Run all forensics checks:

```bash
# Complete forensics analysis
rldk forensics doctor ./my_training_run

# Doctor with custom output
rldk forensics doctor ./my_training_run --output forensics_report.json --verbose
```

## Python API

### Comprehensive PPO Forensics

```python
from rldk.forensics import ComprehensivePPOForensics

# Initialize forensics
forensics = ComprehensivePPOForensics(
    kl_target=0.1,
    enable_kl_schedule_tracking=True,
    enable_gradient_norms_analysis=True,
    enable_advantage_statistics=True
)

# Update with training data
metrics = forensics.update(
    step=100,
    kl=0.15,
    kl_coef=0.2,
    entropy=2.5,
    reward_mean=0.8,
    reward_std=0.3,
    policy_grad_norm=1.2,
    value_grad_norm=0.8,
    advantage_mean=0.1,
    advantage_std=0.5
)

# Check for anomalies
if metrics.has_anomalies:
    print(f"Found {len(metrics.anomalies)} anomalies:")
    for anomaly in metrics.anomalies:
        print(f"  - {anomaly.type}: {anomaly.description}")
```

### Gradient Analysis

```python
from rldk.forensics import GradientNormsAnalyzer

analyzer = GradientNormsAnalyzer(
    exploding_threshold=10.0,
    vanishing_threshold=1e-6
)

# Analyze gradient norms
result = analyzer.analyze(
    policy_grad_norm=15.0,  # High gradient norm
    value_grad_norm=0.5,
    step=100
)

if result.exploding_gradients:
    print("Warning: Exploding gradients detected!")
    print(f"Recommendations: {result.recommendations}")
```

### KL Schedule Tracking

```python
from rldk.forensics import KLScheduleTracker

tracker = KLScheduleTracker(
    target_kl=0.1,
    kl_coef_min=0.01,
    kl_coef_max=1.0
)

# Track KL divergence and coefficient
status = tracker.update(
    kl_divergence=0.15,
    kl_coefficient=0.2,
    step=100
)

if status.needs_adjustment:
    print(f"KL coefficient should be adjusted to: {status.recommended_kl_coef}")
```

## Anomaly Detection Rules

### PPO-Specific Anomalies

1. **KL Divergence Issues**
   - KL spike detection (sudden increases)
   - KL plateau detection (stuck values)
   - KL oscillation detection (unstable training)

2. **Gradient Problems**
   - Exploding gradients (norm > threshold)
   - Vanishing gradients (norm < threshold)
   - Gradient instability (high variance)

3. **Advantage Function Issues**
   - Advantage distribution shifts
   - Extreme advantage values
   - Advantage-reward correlation problems

4. **Entropy Problems**
   - Entropy collapse (too low)
   - Entropy explosion (too high)
   - Entropy instability

5. **Reward Issues**
   - Reward hacking detection
   - Reward distribution shifts
   - Reward-policy misalignment

### Environment Issues

1. **Non-determinism Detection**
   - Random seed inconsistencies
   - Environment state variations
   - Hardware-specific differences

2. **Configuration Problems**
   - Missing deterministic settings
   - Inconsistent framework versions
   - Hardware configuration issues

## Integration with Training

### TRL Integration

```python
from trl import PPOTrainer
from rldk.integrations.trl import RLDKCallback

# Create forensics callback
callback = RLDKCallback(
    enable_forensics=True,
    forensics_config={
        "kl_target": 0.1,
        "enable_anomaly_detection": True,
        "enable_gradient_analysis": True
    }
)

# Add to trainer
trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    callbacks=[callback]
)

# Train with real-time forensics
trainer.train()

# Get forensics report
report = callback.get_forensics_report()
print(f"Anomalies detected: {len(report.anomalies)}")
```

### Custom Integration

```python
from rldk.forensics import ComprehensivePPOForensics

# Initialize forensics
forensics = ComprehensivePPOForensics()

# In your training loop
for step in range(num_steps):
    # ... training code ...
    
    # Update forensics
    metrics = forensics.update(
        step=step,
        kl=kl_divergence,
        kl_coef=kl_coefficient,
        entropy=entropy,
        reward_mean=reward_mean,
        reward_std=reward_std,
        policy_grad_norm=policy_grad_norm,
        value_grad_norm=value_grad_norm,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std
    )
    
    # Check for issues
    if metrics.has_anomalies:
        print(f"Step {step}: {len(metrics.anomalies)} anomalies detected")
        for anomaly in metrics.anomalies:
            print(f"  - {anomaly.type}: {anomaly.description}")
            
        # Optionally stop training or adjust parameters
        if any(a.severity == "critical" for a in metrics.anomalies):
            print("Critical anomaly detected, stopping training")
            break
```

## Output Reports

### Forensics Report Structure

```json
{
  "summary": {
    "total_anomalies": 3,
    "critical_anomalies": 1,
    "warning_anomalies": 2,
    "training_health_score": 0.75
  },
  "anomalies": [
    {
      "type": "kl_spike",
      "severity": "critical",
      "step": 150,
      "description": "KL divergence spiked to 0.25 (target: 0.1)",
      "recommendations": ["Reduce learning rate", "Adjust KL coefficient"]
    }
  ],
  "metrics_analysis": {
    "kl_divergence": {
      "mean": 0.08,
      "std": 0.03,
      "max": 0.25,
      "trend": "increasing"
    },
    "gradient_norms": {
      "policy_mean": 2.1,
      "value_mean": 1.8,
      "exploding_episodes": 2
    }
  },
  "recommendations": [
    "Consider reducing learning rate due to gradient instability",
    "Monitor KL divergence more closely",
    "Review reward function for potential hacking"
  ]
}
```

## Best Practices

1. **Enable Early**: Start forensics analysis from the beginning of training
2. **Monitor Continuously**: Use callbacks for real-time monitoring
3. **Set Appropriate Thresholds**: Adjust thresholds based on your specific use case
4. **Act on Anomalies**: Don't ignore warnings - investigate and address issues
5. **Save Reports**: Keep forensics reports for post-training analysis

## Troubleshooting

### Common Issues

1. **False Positives**: Adjust thresholds if getting too many warnings
2. **Missing Metrics**: Ensure your training loop logs all required metrics
3. **Performance Impact**: Disable detailed analysis for very long training runs
4. **Memory Usage**: Use sampling for large-scale analysis

### Performance Tips

- Use `--max-steps` to limit analysis scope
- Disable detailed gradient analysis for faster processing
- Use JSON output format for programmatic analysis
- Consider running forensics post-training for very long runs

For more examples, see the [Examples](../examples/basic-ppo-cartpole.md) section.
