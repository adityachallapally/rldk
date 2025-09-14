# Seeded Replay

The Seeded Replay utility enables you to verify the reproducibility of training runs by re-running them with the original seeds and comparing metrics.

## Overview

Reproducibility is crucial for trustworthy machine learning research and development. The seeded replay utility:

1. **Extracts seeds** from saved training runs
2. **Re-runs training commands** with identical seeds and deterministic settings
3. **Compares metrics** between original and replay runs
4. **Reports violations** when metrics differ beyond specified tolerance
5. **Generates detailed analysis** for debugging non-deterministic behavior

## Features

- **Automatic seed extraction** from training run data
- **Deterministic environment setup** (CPU/GPU agnostic)
- **Configurable tolerance** for metric comparisons
- **Comprehensive reporting** with detailed statistics
- **Integration with existing CLI** and library functions
- **Support for various training frameworks** (TRL, OpenRLHF, WandB)

## Command Line Usage

```bash
# Basic replay with default tolerance
rldk replay --run runs/ppo_training.jsonl \
           --command "python train_ppo.py --model gpt2" \
           --metrics "reward_mean,kl_mean,entropy_mean"

# Custom tolerance and step limit
rldk replay --run runs/ppo_training.jsonl \
           --command "python train_ppo.py --model gpt2" \
           --metrics "reward_mean,kl_mean" \
           --tolerance 0.005 \
           --max-steps 100 \
           --output-dir replay_analysis

# Specify device
rldk replay --run runs/ppo_training.jsonl \
           --command "python train_ppo.py --model gpt2" \
           --metrics "reward_mean,kl_mean" \
           --device cpu
```

## Python API

```python
from rldk.replay import replay

# Run replay
report = replay(
    run_path="runs/ppo_training.jsonl",
    training_command="python train_ppo.py --model gpt2",
    metrics_to_compare=["reward_mean", "kl_mean", "entropy_mean"],
    tolerance=0.01,
    max_steps=50,
    output_dir="replay_results"
)

# Check results
if report.passed:
    print("✅ Reproducibility verified!")
else:
    print(f"🚨 {len(report.mismatches)} tolerance violations found")
    
    # Analyze violations
    for metric in report.metrics_compared:
        stats = report.comparison_stats[metric]
        violations = stats['tolerance_violations']
        max_diff = stats['max_diff']
        print(f"  {metric}: {violations} violations, max diff: {max_diff:.6f}")
```

## How It Works

### 1. Seed Extraction
The utility reads the original training run data and extracts the seed value used for training. This seed is essential for reproducibility.

### 2. Command Preparation
The training command is modified to include the original seed:
- If `--seed` is already present, it's replaced
- If not present, `--seed <value>` is appended

### 3. Deterministic Execution
The replay runs with deterministic settings:
- **Python**: `PYTHONHASHSEED=42`
- **NumPy**: Fixed random seed
- **PyTorch**: Deterministic algorithms, fixed seeds
- **CUDA**: Deterministic operations (when available)
- **Threading**: Single-threaded execution

### 4. Metric Comparison
Metrics are compared step-by-step:
- **Relative differences** are calculated to handle different scales
- **Tolerance checking** identifies violations
- **Statistical analysis** provides insights into differences

### 5. Reporting
Comprehensive reports are generated:
- **Replay metrics** in JSONL format
- **Comparison statistics** with violation details
- **Mismatch analysis** for debugging

## Output Files

The replay utility generates several output files:

```
replay_results/
├── replay_metrics.jsonl      # Metrics from replay run
├── replay_comparison.json    # Summary of comparison results
└── replay_mismatches.json    # Detailed violation information (if any)
```

## Training Script Requirements

For the replay utility to work correctly, your training script should:

1. **Accept `--seed` argument** for setting random seeds
2. **Output metrics in JSONL format** to the path specified by `RLDK_METRICS_PATH` environment variable
3. **Use deterministic operations** where possible

### Example Training Script Integration

```python
import argparse
import json
import os
import random
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set deterministic seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Get output path from environment
    metrics_path = os.environ.get('RLDK_METRICS_PATH', 'metrics.jsonl')
    
    # Training loop
    for step in range(num_steps):
        # ... training code ...
        
        # Log metrics
        metrics = {
            "step": step,
            "seed": args.seed,
            "reward_mean": reward_mean,
            "kl_mean": kl_mean,
            # ... other metrics ...
        }
        
        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

if __name__ == "__main__":
    main()
```

## Tolerance Configuration

The tolerance parameter controls how strict the comparison is:

- **0.001 (0.1%)**: Very strict, for highly deterministic systems
- **0.01 (1%)**: Standard tolerance, good for most use cases
- **0.05 (5%)**: Relaxed tolerance, for systems with some inherent variability
- **0.1 (10%)**: Very relaxed, for systems with significant variability

Choose tolerance based on:
- **Expected precision** of your training system
- **Hardware differences** between original and replay environments
- **Framework version** differences
- **Acceptable variation** for your use case

## Best Practices

1. **Test determinism first**: Use `rldk check-determinism` before replay
2. **Start small**: Begin with limited steps to verify setup
3. **Monitor resources**: Ensure sufficient memory and storage for replay
4. **Version control**: Track training script versions for reproducibility
5. **Document environment**: Record hardware and software configurations

For more examples and troubleshooting, see the [Examples](../examples/basic-ppo-cartpole.md) section.
