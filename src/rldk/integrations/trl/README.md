# TRL Integration

This module provides callbacks and monitoring tools for TRL (Transformer Reinforcement Learning) training loops.

**Note:** RLDK requires TRL 0.23 or newer.

## Features

- Real-time training monitoring and analysis
- Resource usage tracking (GPU, CPU, memory)
- Training health indicators and alerts
- Checkpoint analysis
- **Standardized JSONL event logging**

## JSONL Event Logging

The TRL integration now provides standardized JSONL event logging that creates per-step training records compatible with the Event schema.

### Configuration

JSONL logging is enabled by default. You can configure it when initializing the callback:

```python
from rldk.integrations.trl import RLDKCallback

callback = RLDKCallback(
    output_dir="./rldk_logs",
    enable_jsonl_logging=True,  # Default: True
    run_id="my_training_run"
)
```

### Output Files

The integration creates several output files:

- `{run_id}_events.jsonl` - Per-step training events (JSONL format)
- `{run_id}_metrics.csv` - Aggregated metrics (CSV format)
- `{run_id}_metrics.json` - Aggregated metrics (JSON format)
- `{run_id}_alerts.json` - Training alerts and warnings
- `{run_id}_final_report.json` - Final training summary

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
from transformers import Trainer, TrainingArguments
from rldk.integrations.trl import RLDKCallback

# Initialize callback
callback = RLDKCallback(
    output_dir="./rldk_logs",
    run_id="ppo_training_run",
    enable_jsonl_logging=True
)

# Add to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[callback]
)

# Train - JSONL events will be written automatically
trainer.train()
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
- TRLAdapter for ingestion
- Event schema for normalization
- Standard JSONL processing tools

## Performance Considerations

- JSONL logging adds minimal overhead (< 1ms per step)
- Events are written immediately with `flush()` for real-time access
- Consider buffering for high-frequency training (> 1000 steps/second)

## Troubleshooting

### JSONL File Not Created
- Check that `enable_jsonl_logging=True`
- Verify output directory is writable
- Check for Event schema import errors

### Malformed JSONL Lines
- Check for concurrent writes
- Verify all metrics are primitive types (no tensors)
- Ensure proper JSON serialization

### Large File Sizes
- Implement log rotation
- Consider reducing logging frequency
- Use compression for archival