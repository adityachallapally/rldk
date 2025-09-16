# Determinism Checking

RLDK provides comprehensive determinism checking to ensure your training runs are reproducible across different environments and runs.

## Overview

Determinism checking verifies that:
- **Training produces identical results** when run with the same seed
- **Metrics match exactly** across multiple replicas
- **Environment setup** is properly configured for reproducibility
- **Framework settings** enable deterministic operations

## Command Line Usage

### Basic Determinism Check

```bash
# Check if training is deterministic
rldk check-determinism --cmd "python train.py --seed 42" --compare loss,reward_mean --replicas 5

# Check with custom tolerance
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean,kl --replicas 3 --tolerance 0.001

# Check on specific device
rldk check-determinism --cmd "python train.py" --compare loss --device cpu --replicas 5
```

### Advanced Options

```bash
# Check with timeout and custom output
rldk check-determinism \
  --cmd "python train.py --epochs 10" \
  --compare loss,accuracy,reward_mean \
  --replicas 5 \
  --tolerance 0.01 \
  --timeout 300 \
  --output determinism_report.json

# Check with environment variables
rldk check-determinism \
  --cmd "python train.py" \
  --compare loss \
  --replicas 3 \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env PYTHONHASHSEED=42
```

## Python API

### Basic Usage

```python
from rldk.determinism import check

# Check if training is deterministic
report = check(
    cmd="python train.py --seed 42",
    compare=["loss", "reward_mean", "kl"],
    replicas=5,
    device="cuda"  # Optional device specification
)

print(f"Deterministic: {report.passed}")
print(f"Issues found: {len(report.mismatches)}")
print(f"Recommended fixes: {report.fixes}")
```

### Advanced Configuration

```python
from rldk.determinism import check, DeterminismConfig

# Custom configuration
config = DeterminismConfig(
    tolerance=0.001,
    timeout=600,
    max_steps=1000,
    environment_vars={
        "PYTHONHASHSEED": "42",
        "CUDA_VISIBLE_DEVICES": "0"
    },
    deterministic_settings={
        "torch_deterministic": True,
        "torch_benchmark": False,
        "numpy_seed": 42
    }
)

report = check(
    cmd="python train.py",
    compare=["loss", "accuracy"],
    replicas=3,
    config=config
)

# Analyze results
if not report.passed:
    print("Determinism check failed!")
    for mismatch in report.mismatches:
        print(f"Metric: {mismatch.metric}")
        print(f"Step: {mismatch.step}")
        print(f"Expected: {mismatch.expected}")
        print(f"Actual: {mismatch.actual}")
        print(f"Difference: {mismatch.difference}")
```

## How It Works

### 1. Environment Setup
The determinism checker sets up a controlled environment:
- **Python**: Sets `PYTHONHASHSEED=42`
- **NumPy**: Fixed random seed
- **PyTorch**: Deterministic algorithms, fixed seeds
- **CUDA**: Deterministic operations when available
- **Threading**: Single-threaded execution

### 2. Multiple Replicas
Runs the same command multiple times with identical settings:
- Same random seeds
- Same environment variables
- Same deterministic settings
- Same input data

### 3. Metric Comparison
Compares metrics across replicas:
- **Step-by-step comparison** of specified metrics
- **Tolerance-based checking** for floating-point precision
- **Statistical analysis** of differences
- **Mismatch identification** with detailed reporting

### 4. Report Generation
Generates comprehensive reports:
- **Pass/fail status** for overall determinism
- **Detailed mismatches** with step and metric information
- **Recommended fixes** for common issues
- **Environment diagnostics** for troubleshooting

## Training Script Requirements

For determinism checking to work, your training script should:

1. **Accept `--seed` argument** for setting random seeds
2. **Output metrics in JSONL format** to stdout or a file
3. **Use deterministic operations** where possible
4. **Handle environment variables** for deterministic settings

### Example Training Script

```python
import argparse
import json
import os
import random
import numpy as np
import torch

def set_deterministic_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Enable deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    # Set deterministic environment
    set_deterministic_seeds(args.seed)
    
    # Training loop
    for epoch in range(args.epochs):
        # ... training code ...
        
        # Output metrics in JSONL format
        metrics = {
            "step": epoch,
            "seed": args.seed,
            "loss": loss_value,
            "accuracy": accuracy_value,
            "reward_mean": reward_mean
        }
        
        print(json.dumps(metrics))

if __name__ == "__main__":
    main()
```

## Common Issues and Fixes

### Non-Deterministic Operations

**Issue**: Some operations are inherently non-deterministic
**Fix**: Use deterministic alternatives or disable non-deterministic features

```python
# PyTorch deterministic settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# Avoid non-deterministic operations
# Bad: torch.nn.functional.interpolate(..., mode='bilinear')
# Good: torch.nn.functional.interpolate(..., mode='bilinear', align_corners=True)
```

### Hardware Differences

**Issue**: Different hardware produces different results
**Fix**: Use CPU-only mode or ensure identical hardware

```bash
# Force CPU execution
rldk check-determinism --cmd "python train.py" --device cpu --compare loss

# Or ensure identical GPU setup
export CUDA_VISIBLE_DEVICES=0
```

### Framework Versions

**Issue**: Different framework versions produce different results
**Fix**: Use identical environments or version pinning

```bash
# Check framework versions
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"

# Pin versions in requirements.txt
torch==1.12.0
numpy==1.21.0
```

### Random State Issues

**Issue**: Random state not properly initialized
**Fix**: Ensure all random number generators are seeded

```python
# Comprehensive seed setting
def set_all_seeds(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## Tolerance Configuration

Choose appropriate tolerance based on your use case:

- **0.0**: Exact match (only for integer metrics)
- **1e-6**: Very strict (for highly controlled environments)
- **1e-4**: Strict (good for most deterministic systems)
- **1e-3**: Standard (recommended for most use cases)
- **1e-2**: Relaxed (for systems with some variability)

## Integration with Other Tools

### With Replay
```bash
# First check determinism
rldk check-determinism --cmd "python train.py" --compare loss --replicas 3

# Then run replay if deterministic
rldk replay --run training_run.jsonl --command "python train.py" --metrics loss
```

### With Forensics
```bash
# Check determinism and run forensics
rldk check-determinism --cmd "python train.py" --compare loss --replicas 3
rldk forensics doctor ./training_output
```

### With Tracking
```python
from rldk.tracking import ExperimentTracker
from rldk.determinism import check

# Track experiment with determinism verification
tracker = ExperimentTracker(config)
tracker.start_experiment()

# Run determinism check
report = check(
    cmd="python train.py",
    compare=["loss"],
    replicas=3
)

# Add determinism results to tracking
tracker.add_metadata("determinism_check", {
    "passed": report.passed,
    "mismatches": len(report.mismatches),
    "tolerance": report.tolerance
})

tracker.finish_experiment()
```

## Best Practices

1. **Test Early**: Check determinism before long training runs
2. **Use Appropriate Tolerance**: Balance strictness with practicality
3. **Check Key Metrics**: Focus on metrics that matter for your use case
4. **Document Environment**: Record hardware and software configurations
5. **Version Control**: Track training script versions for reproducibility

## Troubleshooting

### Performance Tips
- Use `--max-steps` to limit checking scope for long training runs
- Start with fewer replicas (3) and increase if needed
- Use CPU mode for faster checking when GPU determinism is problematic

### Debugging Non-Determinism
1. **Start Simple**: Check with minimal training loop
2. **Isolate Components**: Test individual parts of your training pipeline
3. **Check Dependencies**: Verify all frameworks support deterministic operations
4. **Review Code**: Look for sources of randomness in your code

For more examples and advanced usage, see the [Examples](../examples/basic-ppo-cartpole.md) section.
