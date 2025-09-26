# TRL Integration Guide

This guide explains how to use RLDK with TRL (Transformer Reinforcement Learning) for monitoring and analyzing PPO training runs.

**Note:** RLDK requires TRL 0.23 or newer.

## Overview

RLDK provides comprehensive monitoring and analysis capabilities for TRL training runs through:

- **Real-time metrics collection** during training
- **Standardized JSONL logging** for event tracking
- **Robust parsing** of training logs
- **Integration with analysis tools** for post-training evaluation

## Quick Start

### 1. Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install RLDK with development dependencies
pip install -e .[dev] trl
```

### 2. Basic Usage

```python
from rldk.integrations.trl import RLDKCallback
from trl import PPOTrainer, PPOConfig

# Initialize RLDK callback
rldk_callback = RLDKCallback(
    output_dir="./rldk_logs",
    log_interval=10,
    run_id="my_ppo_run"
)

# Add to your PPO trainer
trainer = PPOTrainer(
    args=ppo_config,
    model=model,
    ref_model=ref_model,
    callbacks=[rldk_callback],  # Add RLDK callback
    # ... other parameters
)

# Train your model
trainer.train()
```

### 3. Lightweight Event Logging

For minimal setups that only need raw scalar metrics, RLDK ships a dedicated
`EventWriterCallback`. It mirrors TRL's log payloads into the canonical
`EventWriter` JSONL format using RLDK's field presets, so downstream tooling can
ingest KL, reward, and gradient metrics without the heavier monitoring stack.

```python
from pathlib import Path

from rldk.integrations.trl import EventWriterCallback, create_ppo_trainer
from trl import PPOConfig

event_log_path = Path("./artifacts/my_run/events.jsonl")

trainer = create_ppo_trainer(
    model_name="sshleifer/tiny-gpt2",
    ppo_config=ppo_config,
    train_dataset=train_dataset,
    callbacks=[],
    event_log_path=event_log_path,
)

trainer.train()
```

When ``event_log_path`` is provided, :func:`create_ppo_trainer` automatically
attaches the callback (unless one is already supplied) and appends events to the
specified JSONL file.

## JSONL Event Logging

RLDK automatically logs training events in a standardized JSONL format that includes:

### Event Schema Structure

Each training step generates a JSONL line with the following structure:

```json
{
  "step": 10,
  "wall_time": 100.5,
  "metrics": {
    "reward_mean": 0.8,
    "reward_std": 0.2,
    "kl_mean": 0.05,
    "entropy_mean": 0.9,
    "clip_frac": 0.1,
    "grad_norm": 1.0,
    "lr": 0.001,
    "loss": 0.5
  },
  "rng": {
    "seed": 42
  },
  "data_slice": {
    "tokens_in": 1000,
    "tokens_out": 500
  },
  "model_info": {
    "run_id": "my_ppo_run",
    "git_sha": "abc123",
    "phase": "train"
  },
  "notes": ["High clipping fraction detected"]
}
```

### Key Features

- **Standardized format**: All events follow the same schema
- **Rich metadata**: Includes RNG state, model info, and data slice details
- **Automatic notes**: System generates notes for training health indicators
- **Real-time writing**: Events are written immediately to avoid data loss

## Configuration Options

### RLDKCallback Parameters

```python
RLDKCallback(
    output_dir="./rldk_logs",        # Directory for output files
    log_interval=10,                 # Steps between detailed logging
    jsonl_log_interval=1,            # Steps between JSONL event logging
    enable_jsonl_logging=True,       # Enable/disable JSONL logging
    alert_thresholds={               # Custom alert thresholds
        "kl_divergence": 0.1,
        "clip_fraction": 0.2,
        "gradient_norm": 1.0
    },
    run_id="my_run",                # Unique run identifier
    enable_resource_monitoring=True, # Monitor GPU/CPU usage
    enable_checkpoint_analysis=True  # Analyze model checkpoints
)
```

### Alert Thresholds

RLDK automatically monitors training health and generates alerts for:

- **High KL divergence**: When KL divergence exceeds threshold
- **High clipping fraction**: When policy clipping is excessive
- **Large gradient norms**: When gradients become too large
- **Memory usage**: When GPU memory usage is high

## Data Analysis

### Loading Training Data

```python
from rldk.adapters.trl import TRLAdapter
from rldk.ingest import ingest_runs

# Load JSONL events
adapter = TRLAdapter("rldk_logs/my_run_events.jsonl")
df = adapter.load()

# Or use the ingest function
df = ingest_runs("rldk_logs/my_run_events.jsonl", adapter_hint="trl")
```

### Analyzing Training Metrics

```python
import pandas as pd
import matplotlib.pyplot as plt

# Plot training progress
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df['step'], df['reward_mean'])
plt.title('Reward Progress')
plt.xlabel('Step')
plt.ylabel('Reward Mean')

plt.subplot(2, 2, 2)
plt.plot(df['step'], df['kl_mean'])
plt.title('KL Divergence')
plt.xlabel('Step')
plt.ylabel('KL Mean')

plt.subplot(2, 2, 3)
plt.plot(df['step'], df['loss'])
plt.title('Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.subplot(2, 2, 4)
plt.plot(df['step'], df['clip_frac'])
plt.title('Clipping Fraction')
plt.xlabel('Step')
plt.ylabel('Clip Fraction')

plt.tight_layout()
plt.show()
```

## File Outputs

RLDK generates several output files in the specified directory:

### JSONL Events File
- **Format**: `{run_id}_events.jsonl` or the path passed to ``event_log_path``
- **Content**: Per-step training events in standardized format with canonical
  metric names (``kl``, ``reward_mean``, ``grad_norm_policy`` and more)
- **Usage**: Primary data source for analysis or ingestion tools

### Metrics CSV File
- **Format**: `{run_id}_metrics.csv`
- **Content**: Aggregated training metrics
- **Usage**: Quick overview of training progress

### Alerts File
- **Format**: `{run_id}_alerts.json`
- **Content**: Training health alerts and warnings
- **Usage**: Identify potential training issues

### Final Report
- **Format**: `{run_id}_final_report.json`
- **Content**: Summary statistics and training health indicators
- **Usage**: Post-training analysis and comparison

## Error Handling

### Malformed JSONL

The TRL adapter gracefully handles malformed JSONL files:

```python
# Malformed JSONL is automatically skipped
adapter = TRLAdapter("malformed_events.jsonl")
df = adapter.load()  # Only loads valid lines
```

### Unknown Schemas

Logs that use custom field names now fall back to the flexible adapter with a helpful
message suggesting ``--field-map``. This keeps ingestion resilient while guiding you to
map bespoke metrics into the shared TrainingMetrics schema.

### Missing Dependencies

If TRL is not available, RLDK will raise a clear error:

```python
try:
    from rldk.integrations.trl import RLDKCallback
except ImportError as e:
    print("TRL not available. Install with: pip install trl")
```

## Best Practices

### 1. Run ID Management

Use descriptive, unique run IDs:

```python
import time
run_id = f"ppo_gpt2_{int(time.time())}"
```

### 2. Logging Intervals

Balance between detail and performance:

```python
# For debugging: log every step
jsonl_log_interval=1

# For production: log every 10 steps
jsonl_log_interval=10
```

### 3. Resource Monitoring

Enable resource monitoring for long training runs:

```python
callback = RLDKCallback(
    enable_resource_monitoring=True,
    alert_thresholds={"memory_usage": 0.9}
)
```

### 4. Checkpoint Analysis

Enable checkpoint analysis for model health tracking:

```python
callback = RLDKCallback(
    enable_checkpoint_analysis=True
)
```

## Troubleshooting

### Common Issues

1. **JSONL file not created**: Check that `enable_jsonl_logging=True`
2. **Missing metrics**: Ensure TRL is properly installed and configured
3. **Permission errors**: Check write permissions for output directory
4. **Memory issues**: Adjust `log_interval` to reduce memory usage

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

callback = RLDKCallback(
    output_dir="./debug_logs",
    log_interval=1
)
```

## Integration Examples

### With Weights & Biases

```python
from rldk.integrations.trl import RLDKCallback
import wandb

# Initialize W&B
wandb.init(project="ppo_training")

# Add RLDK callback
callback = RLDKCallback(
    output_dir="./rldk_logs",
    run_id=wandb.run.id
)

# Your training code here
```

### With Custom Metrics

```python
class CustomRLDKCallback(RLDKCallback):
    def on_log(self, args, state, control, logs, **kwargs):
        # Add custom metrics
        logs['custom_metric'] = compute_custom_metric()
        
        # Call parent method
        super().on_log(args, state, control, logs, **kwargs)
```

## API Reference

### RLDKCallback

Main callback class for TRL integration.

**Methods:**
- `on_train_begin()`: Called at training start
- `on_step_end()`: Called at each training step
- `on_log()`: Called when logs are generated
- `on_save()`: Called when checkpoints are saved
- `on_train_end()`: Called at training end

### TRLAdapter

Adapter for parsing TRL JSONL files.

**Methods:**
- `can_handle()`: Check if file can be processed
- `load()`: Load and parse JSONL file
- `_extract_trl_metric()`: Extract metrics from JSON data

## Contributing

To contribute to TRL integration:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:

1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include relevant logs and configuration