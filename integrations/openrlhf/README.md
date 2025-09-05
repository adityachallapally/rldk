# OpenRLHF Integration

This module provides callbacks and monitoring tools for OpenRLHF training loops with real-time monitoring and distributed training support.

## Features

- Real-time training monitoring and analysis
- Resource usage tracking (GPU, CPU, memory)
- Distributed training monitoring
- Network performance metrics
- Training health indicators and alerts
- **Standardized JSONL event logging**

## JSONL Event Logging

The OpenRLHF integration now provides standardized JSONL event logging that creates per-step training records compatible with the Event schema.

### Configuration

JSONL logging is enabled by default. You can configure it when initializing the callback:

```python
from rldk.integrations.openrlhf import OpenRLHFCallback

callback = OpenRLHFCallback(
    output_dir="./rldk_logs",
    enable_jsonl_logging=True,  # Default: True
    run_id="my_training_run",
    model_name="gpt2",
    dataset_name="squad"
)
```

### Output Files

The integration creates several output files:

- `{run_id}_events.jsonl` - Per-step training events (JSONL format)
- `metrics_{run_id}.csv` - Aggregated metrics (CSV format)
- `metrics_{run_id}.parquet` - Aggregated metrics (Parquet format)
- `summary_{run_id}.json` - Training summary statistics
- `events_{run_id}.jsonl` - Additional event logs

### JSONL Event Schema

Each JSONL line contains a complete Event object with the following structure:

```json
{
  "step": 0,
  "wall_time": 10.5,
  "metrics": {
    "reward_mean": 0.5,
    "reward_std": 0.1,
    "kl_mean": 0.1,
    "entropy_mean": 0.8,
    "clip_frac": 0.2,
    "grad_norm": 1.0,
    "lr": 0.001,
    "loss": 0.4
  },
  "rng": {
    "seed": 42,
    "python_hash_seed": 42,
    "torch_seed": 42,
    "numpy_seed": 42,
    "random_seed": 42
  },
  "data_slice": {
    "tokens_in": 1000,
    "tokens_out": 500,
    "batch_size": 32,
    "sequence_length": 512
  },
  "model_info": {
    "run_id": "my_training_run",
    "git_sha": "abc123",
    "phase": "train",
    "model_name": "gpt2",
    "model_size": 124000000,
    "optimizer": "AdamW",
    "scheduler": "CosineAnnealingLR"
  },
  "notes": []
}
```

### Usage

```python
from openrlhf.trainer import PPOTrainer
from rldk.integrations.openrlhf import OpenRLHFCallback

# Initialize callback
callback = OpenRLHFCallback(
    output_dir="./rldk_logs",
    run_id="ppo_training_run",
    model_name="gpt2",
    dataset_name="squad",
    enable_jsonl_logging=True
)

# Add to trainer
trainer = PPOTrainer(
    model=model,
    # ... other trainer arguments
)

# Register callback
trainer.add_callback(callback)

# Train - JSONL events will be written automatically
trainer.train()
```

### Distributed Training

For distributed training, each node will create its own JSONL file:

```python
from rldk.integrations.openrlhf import DistributedTrainingMonitor

# Initialize distributed monitor
monitor = DistributedTrainingMonitor(
    output_dir=f"./rldk_logs/node_{rank}",
    run_id="distributed_training_run",
    enable_jsonl_logging=True
)

# Each node will create: {run_id}_events.jsonl
```

### Environment Variables

You can disable JSONL logging using environment variables:

```bash
export RLDK_DISABLE_JSONL=1
```

### Log Rotation

JSONL files can grow large during long training runs. Consider implementing log rotation or cleanup:

```python
import os
from pathlib import Path

# Example: Keep only last 1000 lines
def cleanup_old_logs(log_file: Path, max_lines: int = 1000):
    if log_file.exists() and log_file.stat().st_size > 10 * 1024 * 1024:  # 10MB
        with open(log_file, 'r') as f:
            lines = f.readlines()
        with open(log_file, 'w') as f:
            f.writelines(lines[-max_lines:])
```

### Converting JSONL to CSV

You can convert JSONL logs to CSV using standard tools:

```bash
# Using jq
jq -r '[.step, .wall_time, .metrics.reward_mean, .metrics.loss] | @csv' events.jsonl > events.csv

# Using Python
import pandas as pd
import json

events = []
with open('events.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))
df = pd.DataFrame(events)
df.to_csv('events.csv', index=False)
```

## Compatibility

The JSONL events are compatible with:
- OpenRLHFAdapter for ingestion
- Event schema for normalization
- Standard JSONL processing tools
- Distributed training setups

## Performance Considerations

- JSONL logging adds minimal overhead (< 1ms per step)
- Events are written immediately with `flush()` for real-time access
- Consider buffering for high-frequency training (> 1000 steps/second)
- For distributed training, each node writes independently

## Network Monitoring

The integration includes network monitoring for distributed training:

```python
from rldk.integrations.openrlhf import DistributedTrainingMonitor

monitor = DistributedTrainingMonitor(
    network_monitoring=True,
    enable_jsonl_logging=True
)

# Network metrics will be included in JSONL events:
# - network_bandwidth
# - network_latency
# - allreduce_time
```

## Troubleshooting

### JSONL File Not Created
- Check that `enable_jsonl_logging=True`
- Verify output directory is writable
- Check for Event schema import errors
- For distributed training, check node-specific directories

### Malformed JSONL Lines
- Check for concurrent writes
- Verify all metrics are primitive types (no tensors)
- Ensure proper JSON serialization
- Check for network interruptions in distributed training

### Large File Sizes
- Implement log rotation
- Consider reducing logging frequency
- Use compression for archival
- Monitor disk space in distributed setups

### Distributed Training Issues
- Verify each node has unique output directory
- Check network connectivity between nodes
- Monitor allreduce performance
- Ensure consistent run_id across nodes