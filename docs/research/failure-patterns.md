# Common RL Training Failure Patterns

This guide documents common failure patterns in reinforcement learning training and how to detect and resolve them using RLDK.

## Overview

RL training can fail in many subtle ways. RLDK's forensics system is designed to detect these patterns early and provide actionable recommendations for resolution.

## PPO-Specific Failure Patterns

### 1. KL Divergence Explosion

**Symptoms:**
- KL divergence suddenly spikes above target (e.g., > 0.5 when target is 0.1)
- Policy updates become unstable
- Training loss oscillates wildly

**Detection with RLDK:**
```python
from rldk.forensics import ComprehensivePPOForensics

forensics = ComprehensivePPOForensics(kl_target=0.1)
metrics = forensics.update(step=100, kl=0.8, ...)  # High KL

if metrics.has_anomalies:
    for anomaly in metrics.anomalies:
        if anomaly.type == "kl_spike":
            print(f"KL spike detected: {anomaly.description}")
```

**Common Causes:**
- Learning rate too high
- Batch size too small
- Insufficient KL penalty
- Poor initialization

**Solutions:**
- Reduce learning rate by 2-5x
- Increase KL coefficient
- Use adaptive KL scheduling
- Implement gradient clipping

### 2. Reward Hacking

**Symptoms:**
- Reward increases rapidly but model behavior degrades
- High reward variance
- Policy exploits reward function loopholes

**Detection with RLDK:**
```bash
# Monitor reward patterns
rldk forensics log-scan ./training_logs --reward-threshold 10.0

# Generate reward analysis card
rldk card reward ./model --prompts test_prompts.jsonl
```

**Common Causes:**
- Poorly designed reward function
- Insufficient reward model training
- Overfitting to specific prompts

**Solutions:**
- Improve reward function design
- Add regularization terms
- Use diverse training data
- Implement reward clipping

### 3. Policy Collapse

**Symptoms:**
- Entropy drops to near zero
- Model generates repetitive outputs
- Loss of exploration

**Detection with RLDK:**
```python
# Monitor entropy trends
forensics.update(step=100, entropy=0.1, ...)  # Very low entropy

# Check for entropy collapse anomaly
if "entropy_collapse" in [a.type for a in metrics.anomalies]:
    print("Policy collapse detected!")
```

**Solutions:**
- Increase entropy coefficient
- Add exploration bonuses
- Use temperature scaling
- Implement diversity regularization

## Gradient-Related Failures

### 4. Exploding Gradients

**Symptoms:**
- Gradient norms suddenly spike (> 10.0)
- Training becomes unstable
- Loss values become NaN

**Detection with RLDK:**
```python
from rldk.forensics import GradientNormsAnalyzer

analyzer = GradientNormsAnalyzer(exploding_threshold=10.0)
result = analyzer.analyze(
    policy_grad_norm=25.0,  # High gradient norm
    value_grad_norm=15.0,
    step=100
)

if result.exploding_gradients:
    print("Exploding gradients detected!")
```

**Solutions:**
- Implement gradient clipping
- Reduce learning rate
- Use gradient accumulation
- Check for numerical instabilities

### 5. Vanishing Gradients

**Symptoms:**
- Gradient norms approach zero (< 1e-6)
- Training stagnates
- No learning progress

**Detection with RLDK:**
```python
analyzer = GradientNormsAnalyzer(vanishing_threshold=1e-6)
result = analyzer.analyze(
    policy_grad_norm=1e-8,  # Very small gradient
    value_grad_norm=1e-7,
    step=100
)

if result.vanishing_gradients:
    print("Vanishing gradients detected!")
```

**Solutions:**
- Increase learning rate
- Use residual connections
- Implement proper weight initialization
- Check activation functions

## Data-Related Failures

### 6. Distribution Shift

**Symptoms:**
- Performance degrades over time
- Model behavior changes unexpectedly
- Metrics drift from baseline

**Detection with RLDK:**
```bash
# Compare training runs
rldk diff run1.jsonl run2.jsonl --threshold 0.1

# Generate drift analysis
rldk card drift run1.jsonl run2.jsonl --output drift_analysis.html
```

**Solutions:**
- Monitor data distribution
- Use domain adaptation techniques
- Implement continual learning
- Regular model retraining

### 7. Overfitting to Training Data

**Symptoms:**
- Training metrics improve but validation degrades
- Poor generalization to new prompts
- High variance in performance

**Detection with RLDK:**
```python
# Track training vs validation metrics
tracker.add_metadata("train_reward", train_reward)
tracker.add_metadata("val_reward", val_reward)

# Monitor overfitting indicators
if train_reward - val_reward > threshold:
    print("Potential overfitting detected!")
```

**Solutions:**
- Increase dataset diversity
- Implement regularization
- Use early stopping
- Cross-validation

## Environment-Related Failures

### 8. Non-Deterministic Training

**Symptoms:**
- Results vary between runs with same seed
- Reproducibility issues
- Inconsistent performance

**Detection with RLDK:**
```bash
# Check determinism
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean --replicas 5

# Audit environment
rldk forensics env-audit ./training_run
```

**Solutions:**
- Set all random seeds properly
- Use deterministic algorithms
- Fix hardware differences
- Control environment variables

### 9. Memory Leaks

**Symptoms:**
- Memory usage increases over time
- Training slows down progressively
- Out-of-memory errors

**Detection with RLDK:**
```python
# Monitor memory usage
from rldk.utils.progress import ProgressTracker

tracker = ProgressTracker(total=1000, monitor_memory=True)
for step in range(1000):
    # Training step
    tracker.update(1)
    
    if tracker.memory_usage > 0.9:  # 90% memory usage
        print("High memory usage detected!")
```

**Solutions:**
- Use gradient checkpointing
- Clear unused variables
- Implement proper cleanup
- Use memory profiling tools

## Model Architecture Failures

### 10. Poor Value Function Learning

**Symptoms:**
- Value loss remains high
- Poor advantage estimation
- Unstable policy updates

**Detection with RLDK:**
```python
# Monitor value function performance
forensics.update(
    step=100,
    value_loss=2.5,  # High value loss
    advantage_mean=0.0,
    advantage_std=2.0,  # High variance
    ...
)
```

**Solutions:**
- Increase value function capacity
- Adjust value loss coefficient
- Use separate learning rates
- Implement value function regularization

## Hyperparameter-Related Failures

### 11. Learning Rate Issues

**Symptoms:**
- Too high: Unstable training, oscillating loss
- Too low: Slow convergence, no progress

**Detection with RLDK:**
```python
# Monitor learning rate effects
tracker.add_metadata("learning_rate", lr)

# Track convergence metrics
if step > 1000 and improvement < threshold:
    print("Possible learning rate issue!")
```

**Solutions:**
- Use learning rate scheduling
- Implement adaptive learning rates
- Grid search or Bayesian optimization
- Monitor gradient-to-parameter ratios

### 12. Batch Size Problems

**Symptoms:**
- Too small: Noisy gradients, unstable training
- Too large: Poor generalization, memory issues

**Detection with RLDK:**
```python
# Monitor batch size effects
tracker.add_metadata("batch_size", batch_size)
tracker.add_metadata("gradient_noise", grad_noise)

# Check for batch size issues
if grad_noise > threshold:
    print("Consider increasing batch size")
```

**Solutions:**
- Experiment with different batch sizes
- Use gradient accumulation
- Implement batch size scheduling
- Consider hardware constraints

## Framework-Specific Issues

### 13. TRL Integration Problems

**Common Issues:**
- Version compatibility
- Memory management
- Callback integration

**Detection and Solutions:**
```python
from rldk.integrations.trl import fix_generation_config

# Fix TRL compatibility issues
fix_generation_config(model, tokenizer)

# Use RLDK callback for monitoring
from rldk.integrations.trl import RLDKCallback
callback = RLDKCallback(enable_forensics=True)
```

### 14. OpenRLHF Distributed Issues

**Common Issues:**
- Network bottlenecks
- Load imbalancing
- Communication overhead

**Detection and Solutions:**
```python
from rldk.integrations.openrlhf import NetworkMonitor

# Monitor distributed training
monitor = NetworkMonitor()
metrics = monitor.get_current_metrics()

if metrics.latency_ms > 100:
    print("Network latency issue detected!")
```

## Prevention Strategies

### 1. Comprehensive Monitoring

```python
# Set up complete monitoring
from rldk.tracking import ExperimentTracker
from rldk.forensics import ComprehensivePPOForensics

tracker = ExperimentTracker(config)
forensics = ComprehensivePPOForensics()

# Monitor everything
tracker.start_experiment()
for step in training_loop:
    metrics = forensics.update(...)
    if metrics.has_anomalies:
        handle_anomalies(metrics.anomalies)
```

### 2. Early Detection

```python
# Set up alerts for early detection
def check_training_health(metrics):
    critical_issues = [
        a for a in metrics.anomalies 
        if a.severity == "critical"
    ]
    
    if critical_issues:
        print("Critical issues detected - consider stopping training")
        return False
    return True
```

### 3. Automated Recovery

```python
# Implement automated recovery strategies
def auto_recovery(anomaly):
    if anomaly.type == "kl_spike":
        # Reduce learning rate
        optimizer.param_groups[0]['lr'] *= 0.5
    elif anomaly.type == "exploding_gradients":
        # Enable gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

## Best Practices

1. **Start with Conservative Settings**: Use lower learning rates and smaller batch sizes initially
2. **Monitor Continuously**: Use RLDK forensics throughout training
3. **Validate Frequently**: Regular validation on held-out data
4. **Document Everything**: Track all hyperparameters and configurations
5. **Plan for Failure**: Have recovery strategies ready
6. **Learn from Failures**: Analyze failed runs to improve future training

## Troubleshooting Workflow

1. **Identify Symptoms**: What specific issues are you observing?
2. **Run Diagnostics**: Use RLDK forensics to analyze the problem
3. **Check Common Causes**: Review the patterns above
4. **Implement Solutions**: Apply appropriate fixes
5. **Monitor Recovery**: Verify that solutions work
6. **Document Learnings**: Record what worked for future reference

For more specific guidance, see the [Forensics Analysis Guide](../user-guide/forensics.md) and [API Reference](../reference/api.md).
