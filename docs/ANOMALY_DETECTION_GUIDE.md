# Advanced Anomaly Detection System

This guide explains how to use the Advanced Anomaly Detection System for RLHF training monitoring.

## Overview

The Advanced Anomaly Detection System provides sophisticated detection rules for monitoring various aspects of RLHF training:

- **Gradient explosion/vanishing detection**: Monitors gradient norms and detects when gradients become too large or too small
- **Learning rate schedule anomalies**: Detects unexpected changes in learning rate schedules
- **Batch size impact analysis**: Analyzes the impact of batch size changes on training performance
- **Model convergence tracking**: Tracks convergence metrics and detects plateaus or divergence
- **Reward model calibration drift**: Monitors reward model calibration and detects drift over time

## Quick Start

### Basic Usage

```python
from profiler.anomaly_detection import AdvancedAnomalyDetector
from profiler.hooks import AnomalyDetectionHook

# Initialize the anomaly detector
anomaly_detector = AdvancedAnomalyDetector(
    output_dir="anomaly_detection_results",
    save_alerts=True
)

# Initialize the hook for integration
anomaly_hook = AnomalyDetectionHook(anomaly_detector)
anomaly_hook.register_with_profiler()

# In your training loop
for step in range(num_steps):
    # ... your training code ...
    
    # Analyze for anomalies
    alerts = anomaly_detector.analyze_training_step(
        model=model,
        optimizer=optimizer,
        loss=loss.item(),
        batch_size=batch_size,
        rewards=rewards,  # Optional
        predictions=predictions  # Optional
    )
    
    # Handle alerts
    for alert in alerts:
        if alert.severity == 'critical':
            print(f"CRITICAL: {alert.message}")
        elif alert.severity == 'high':
            print(f"HIGH: {alert.message}")
```

### Configuration Options

```python
anomaly_detector = AdvancedAnomalyDetector(
    output_dir="anomaly_detection_results",
    save_alerts=True,
    gradient={
        'explosion_threshold': 10.0,      # Threshold for gradient explosion
        'vanishing_threshold': 1e-6,      # Threshold for gradient vanishing
        'window_size': 100                # Window size for gradient analysis
    },
    learning_rate={
        'change_threshold': 0.3,          # Threshold for LR change detection
        'min_lr': 1e-8,                  # Minimum acceptable LR
        'max_lr': 1.0                    # Maximum acceptable LR
    },
    batch_size={
        'performance_threshold': 0.1,     # Threshold for performance impact
        'window_size': 20                 # Window size for batch analysis
    },
    convergence={
        'plateau_threshold': 0.001,       # Threshold for plateau detection
        'plateau_window': 50              # Window size for plateau detection
    },
    reward_drift={
        'drift_threshold': 0.1,           # Threshold for drift detection
        'calibration_threshold': 0.7      # Threshold for calibration quality
    }
)
```

## Detection Categories

### 1. Gradient Anomalies

Detects gradient explosion and vanishing problems:

```python
# Gradient explosion detection
if total_norm > explosion_threshold:
    # Alert: Gradient explosion detected

# Gradient vanishing detection  
if total_norm < vanishing_threshold:
    # Alert: Gradient vanishing detected

# Gradient variance detection
if norm_std > norm_mean * alert_threshold:
    # Alert: High gradient variance detected
```

**Alert Types:**
- `gradient_explosion`: Gradients are too large
- `gradient_vanishing`: Gradients are too small
- `gradient_variance`: High variance in gradient norms

### 2. Learning Rate Anomalies

Monitors learning rate schedules for unexpected changes:

```python
# Out-of-range learning rates
if current_lr < min_lr or current_lr > max_lr:
    # Alert: Learning rate out of range

# Sudden changes
if max_change > change_threshold:
    # Alert: Sudden learning rate change
```

**Alert Types:**
- `lr_too_high`: Learning rate exceeds maximum
- `lr_too_low`: Learning rate below minimum
- `lr_sudden_change`: Sudden change in learning rate

### 3. Batch Size Impact Analysis

Analyzes the impact of batch size changes on training:

```python
# Performance impact detection
if abs(performance_change) > performance_threshold:
    # Alert: Batch size change impact detected
```

**Alert Types:**
- `batch_size_impact`: Performance degradation from batch size change

### 4. Convergence Tracking

Tracks model convergence and detects issues:

```python
# Plateau detection
if plateau_detected and improvement_rate < min_improvement:
    # Alert: Training plateau detected

# Divergence detection
if convergence_rate < -0.1:  # Loss increasing
    # Alert: Loss increasing (divergence)
```

**Alert Types:**
- `convergence_plateau`: Training has plateaued
- `convergence_divergence`: Loss is increasing

### 5. Reward Calibration Drift

Monitors reward model calibration and drift:

```python
# Calibration quality
if calibration_score < calibration_threshold:
    # Alert: Poor calibration detected

# Calibration drift
if abs(calibration_trend) > drift_threshold:
    # Alert: Calibration drift detected

# Reward distribution drift
if mean_change > drift_threshold:
    # Alert: Reward distribution drift
```

**Alert Types:**
- `calibration_poor`: Poor calibration quality
- `calibration_drift`: Calibration drift over time
- `reward_distribution_drift`: Reward distribution changes

## Alert Severity Levels

- **Critical**: Immediate action required (e.g., gradient explosion)
- **High**: Significant issue that needs attention (e.g., gradient vanishing)
- **Medium**: Moderate issue to monitor (e.g., learning rate changes)
- **Low**: Minor issue for awareness (e.g., small performance changes)

## Integration Examples

### With PyTorch Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim
from profiler.anomaly_detection import AdvancedAnomalyDetector

# Initialize
model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
anomaly_detector = AdvancedAnomalyDetector()

# Training loop
for step in range(1000):
    # Forward pass
    x = torch.randn(32, 100)
    y = torch.randn(32, 10)
    output = model(x)
    loss = nn.MSELoss()(output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Anomaly detection
    alerts = anomaly_detector.analyze_training_step(
        model=model,
        optimizer=optimizer,
        loss=loss.item(),
        batch_size=32
    )
    
    # Handle alerts
    for alert in alerts:
        print(f"[{alert.severity.upper()}] {alert.message}")
    
    # Optimizer step
    optimizer.step()
```

### With Hugging Face Transformers

```python
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from profiler.anomaly_detection import AdvancedAnomalyDetector

# Initialize
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
anomaly_detector = AdvancedAnomalyDetector()

# Custom trainer with anomaly detection
class AnomalyDetectionTrainer(Trainer):
    def training_step(self, model, inputs):
        # Get training step result
        result = super().training_step(model, inputs)
        
        # Analyze for anomalies
        alerts = anomaly_detector.analyze_training_step(
            model=model,
            optimizer=self.optimizer,
            loss=result['loss'].item(),
            batch_size=inputs['input_ids'].size(0)
        )
        
        # Log alerts
        for alert in alerts:
            self.log({f"anomaly/{alert.category}_{alert.severity}": 1})
        
        return result

# Use the custom trainer
trainer = AnomalyDetectionTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)
trainer.train()
```

## Output and Reporting

### Alert Files

The system saves alerts to JSON files in the output directory:

```json
{
  "severity": "high",
  "category": "gradient",
  "message": "Gradient explosion detected: norm=15.2341 > 10.0",
  "step": 150,
  "value": 15.2341,
  "threshold": 10.0,
  "metadata": {
    "gradient_stats": [...]
  }
}
```

### Summary Report

A final summary report is generated:

```json
{
  "total_alerts": 45,
  "by_category": {
    "gradient": 20,
    "learning_rate": 15,
    "convergence": 10
  },
  "by_severity": {
    "critical": 5,
    "high": 15,
    "medium": 25
  },
  "latest_alerts": [...]
}
```

## Best Practices

1. **Start with default thresholds** and adjust based on your specific use case
2. **Monitor alerts regularly** during training
3. **Set up automated responses** for critical alerts
4. **Use the hook system** for seamless integration
5. **Save alerts** for post-training analysis
6. **Adjust thresholds** based on your model and data characteristics

## Troubleshooting

### Common Issues

1. **Too many alerts**: Increase thresholds or adjust window sizes
2. **Missing alerts**: Decrease thresholds or check data quality
3. **Memory issues**: Reduce window sizes or disable auto-saving
4. **Serialization errors**: Ensure all metadata is JSON-serializable

### Performance Considerations

- The system adds minimal overhead to training
- Memory usage scales with window sizes
- Consider disabling auto-saving for very long training runs
- Use appropriate batch sizes for your hardware

## Advanced Usage

### Custom Detectors

You can extend the system with custom detectors:

```python
from profiler.anomaly_detection import AdvancedAnomalyDetector

class CustomAnomalyDetector(AdvancedAnomalyDetector):
    def analyze_training_step(self, **kwargs):
        # Call parent analysis
        alerts = super().analyze_training_step(**kwargs)
        
        # Add custom analysis
        custom_alerts = self.custom_analysis(kwargs)
        alerts.extend(custom_alerts)
        
        return alerts
    
    def custom_analysis(self, kwargs):
        # Your custom anomaly detection logic
        return []
```

### Real-time Monitoring

For real-time monitoring, you can set up alert handlers:

```python
def alert_handler(alerts, step):
    for alert in alerts:
        if alert.severity == 'critical':
            # Send notification, stop training, etc.
            print(f"CRITICAL ALERT: {alert.message}")
        elif alert.severity == 'high':
            # Log warning
            print(f"HIGH ALERT: {alert.message}")

# Register the handler
profiler_registry.register_hook("anomaly_detection", alert_handler)
```

This system provides comprehensive monitoring for RLHF training, helping you detect and respond to various types of anomalies that can occur during training.