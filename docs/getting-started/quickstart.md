# Quick Start

Get up and running with RLDK in 5 minutes! This guide will walk you through the essential features and get you started with reproducible RL experiments.

## Prerequisites

- RLDK installed (see [Installation](installation.md))
- Python 3.8+
- Basic familiarity with reinforcement learning concepts

## 1. Set Up Reproducible Environment

Start by setting up a reproducible environment with a global seed:

```python
import rldk

# Set global seed for reproducibility
seed = rldk.set_global_seed(42)
print(f"Set seed to: {seed}")
```

Or using the CLI:

```bash
rldk seed --seed 42 --env
```

## 2. Track Your First Experiment

Let's create a simple experiment tracking example:

```python
from rldk.tracking import ExperimentTracker, TrackingConfig
import numpy as np

# Configure tracking
config = TrackingConfig(
    experiment_name="quickstart_demo",
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True,
    output_dir="./runs"
)

# Create tracker
tracker = ExperimentTracker(config)

# Start experiment
tracking_data = tracker.start_experiment()
print(f"Started experiment: {tracking_data['experiment_id']}")

# Simulate some training data
training_data = np.random.randn(100, 10)
eval_data = np.random.randn(20, 10)

# Track datasets
tracker.track_dataset(training_data, "training_data", {"samples": 100})
tracker.track_dataset(eval_data, "eval_data", {"samples": 20})

# Simulate a simple model
class SimpleModel:
    def __init__(self):
        self.weights = np.random.randn(10, 1)
    
    def forward(self, x):
        return x @ self.weights

model = SimpleModel()

# Track model
tracker.track_model(model, "simple_model", {
    "architecture": "linear",
    "input_dim": 10,
    "output_dim": 1
})

# Add custom metadata
tracker.add_metadata("learning_rate", 0.001)
tracker.add_metadata("batch_size", 32)
tracker.add_metadata("epochs", 100)

# Set seeds for reproducibility
tracker.set_seeds(42)

# Finish experiment
summary = tracker.finish_experiment()
print(f"Experiment completed: {summary['experiment_id']}")
```

## 3½. Validate the fullscale pipeline

Before digging into ad-hoc experiments, run `scripts/fullscale_acceptance.sh` to confirm the
tooling works end-to-end on your machine. The script provisions a temporary virtual
environment, installs RLDK in editable mode, and chains together:

1. A **primary** and **baseline** invocation of `scripts/fullscale_train_rl.py` so you can
   diff an updating policy against a frozen control run.
2. The **monitor once** workflow with `rules/fullscale_rules.yaml`, producing alerts and a
   detailed report.
3. **Metrics ingestion**, **reward health analysis**, and **reward card** generation to gate
   on the default detectors.
4. A **diff** against the baseline and a **determinism** replay to ensure reproducibility.

The helper exits with a non-zero status when those monitor rules raise warning-or-higher
alerts beyond their thresholds or when the reward card reports issues. Check the generated
`artifacts/fullscale/ACCEPTANCE_SUMMARY.md` for the same verdicts the CI gate expects.

### Tuned defaults and anomaly simulation

The acceptance trainer uses CPU-friendly defaults that still exercise every gate:

- `--learning-rate`: `8e-5`
- `--batch-size`: `4`
- `--temperature`: `0.95`
- `--max-grad-norm`: `2.5`

Keep the defaults for a healthy baseline run. Pass `--simulate-anomalies` to
`scripts/fullscale_train_rl.py` if you want the deterministic KL spikes and reward collapse
scenario that force alert paths. The remediation advice in
[`monitor_rules_cookbook.md`](monitor_rules_cookbook.md#fullscale-remediation-hints) mirrors the
suggestions emitted by `rules/fullscale_rules.yaml` when those guards activate.

## 3. Run Forensics Analysis

Now let's analyze some training data for anomalies:

```python
from rldk.forensics import ComprehensivePPOForensics
import pandas as pd

# Create sample training data with some anomalies
steps = list(range(1, 101))
losses = [0.5 * np.exp(-step/50) + 0.1 * np.random.randn() for step in steps]
rewards = [0.8 + 0.2 * np.sin(step/10) + 0.1 * np.random.randn() for step in steps]
kl_divs = [0.1 + 0.05 * np.random.randn() for step in steps]

# Add some anomalies
kl_divs[50] = 0.8  # KL spike
losses[75] = 2.0   # Loss spike

# Create DataFrame
df = pd.DataFrame({
    'step': steps,
    'loss': losses,
    'reward_mean': rewards,
    'kl': kl_divs
})

# Initialize forensics
forensics = ComprehensivePPOForensics(
    kl_target=0.1,
    enable_kl_schedule_tracking=True,
    enable_gradient_norms_analysis=True,
    enable_advantage_statistics=True,
    enable_length_bias_detection=True,
)

# Process each step
for _, row in df.iterrows():
    metrics = forensics.update(
        step=int(row['step']),
        kl=row['kl'],
        kl_coef=0.2,
        entropy=2.5,
        reward_mean=row['reward_mean'],
        reward_std=0.3,
        policy_grad_norm=1.2,
        value_grad_norm=0.8,
        advantage_mean=0.1,
        advantage_std=0.5,
        response_data=[{"response": "step" + str(row['step']), "reward": row['reward_mean']}],
    )

# Get analysis results
analysis = forensics.get_comprehensive_analysis()
anomalies = forensics.get_anomalies()
health_summary = forensics.get_health_summary()

print("🔍 Forensics Analysis Results:")
print(f"Total anomalies detected: {len(anomalies)}")
print(f"Overall health score: {health_summary.get('overall_health', 'N/A')}")

# Show specific anomalies
for anomaly in anomalies[:3]:  # Show first 3
    print(f"- {anomaly['rule']}: {anomaly['description']}")
```

## 4. Check Determinism

Let's verify that our training is deterministic:

```python
from rldk.determinism import check

# Define a simple training command
def simple_training():
    """Simple training simulation."""
    import random
    import numpy as np
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    
    # Simulate training
    losses = []
    for step in range(10):
        loss = random.random() * np.exp(-step/5)
        losses.append(loss)
        print(f"Step {step}: loss = {loss:.4f}")
    
    return losses

# Save the training function to a temporary file
import tempfile
import os

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write("""
import random
import numpy as np

def main():
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    
    # Simulate training
    for step in range(10):
        loss = random.random() * np.exp(-step/5)
        print(f"loss: {loss:.6f}")

if __name__ == "__main__":
    main()
""")
    temp_file = f.name

try:
    # Check determinism
    report = check(
        cmd=f"python {temp_file}",
        compare=["loss"],
        replicas=3,
        device="cpu"
    )
    
    print(f"\n🎯 Determinism Check Results:")
    print(f"Passed: {report.passed}")
    print(f"Replicas tested: {len(report.mismatches) + 1}")
    
    if not report.passed:
        print(f"Issues found: {len(report.mismatches)}")
        for mismatch in report.mismatches[:3]:
            print(f"- {mismatch}")
    
    if report.fixes:
        print(f"Recommended fixes: {report.fixes[:3]}")

finally:
    # Clean up
    os.unlink(temp_file)
```

## 5. CLI Usage

RLDK provides a comprehensive CLI for all operations:

```bash
# Show help
rldk --help

# Track an experiment
rldk track --name "my_experiment" --interactive

# Run forensics analysis
rldk forensics log-scan ./my_training_run

# Check determinism
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean --replicas 5

# Compare two runs
rldk compare-runs run_a run_b

# Set seed
rldk seed --seed 42 --show

# Run evaluation
rldk evals evaluate data.jsonl --suite quick --output results.json
```

## 6. Complete Example

Here's a complete example that demonstrates the full RLDK workflow:

```python
import rldk
import numpy as np
import pandas as pd
from rldk.tracking import ExperimentTracker, TrackingConfig
from rldk.forensics import ComprehensivePPOForensics
from rldk.determinism import check
from rldk.reward import health

def main():
    print("🚀 RLDK Quick Start Demo")
    print("=" * 50)
    
    # 1. Set up reproducible environment
    print("\n1. Setting up reproducible environment...")
    seed = rldk.set_global_seed(42)
    print(f"   Set seed to: {seed}")
    
    # 2. Start experiment tracking
    print("\n2. Starting experiment tracking...")
    config = TrackingConfig(
        experiment_name="quickstart_demo",
        enable_dataset_tracking=True,
        enable_model_tracking=True,
        enable_environment_tracking=True,
        enable_seed_tracking=True,
        enable_git_tracking=True
    )
    
    tracker = ExperimentTracker(config)
    tracking_data = tracker.start_experiment()
    print(f"   Started experiment: {tracking_data['experiment_id']}")
    
    # 3. Simulate training with forensics
    print("\n3. Simulating training with forensics...")
    forensics = ComprehensivePPOForensics(kl_target=0.1)
    
    training_data = []
    for step in range(1, 51):
        # Simulate training metrics
        loss = 0.5 * np.exp(-step/20) + 0.05 * np.random.randn()
        reward = 0.8 + 0.2 * np.sin(step/10) + 0.1 * np.random.randn()
        kl = 0.1 + 0.02 * np.random.randn()
        
        # Add some anomalies
        if step == 25:
            kl = 0.8  # KL spike
        
        # Update forensics
        forensics.update(
            step=step,
            kl=kl,
            kl_coef=0.2,
            entropy=2.5,
            reward_mean=reward,
            reward_std=0.3,
            policy_grad_norm=1.2,
            value_grad_norm=0.8
        )
        
        training_data.append({
            'step': step,
            'loss': loss,
            'reward_mean': reward,
            'kl': kl
        })
    
    # 4. Track training data
    print("\n4. Tracking training data...")
    df = pd.DataFrame(training_data)
    tracker.track_dataset(df, "training_metrics")
    
    # 5. Analyze forensics results
    print("\n5. Analyzing forensics results...")
    anomalies = forensics.get_anomalies()
    health_summary = forensics.get_health_summary()
    
    print(f"   Anomalies detected: {len(anomalies)}")
    print(f"   Health score: {health_summary.get('overall_health', 'N/A')}")
    
    # 6. Run reward health analysis
    print("\n6. Running reward health analysis...")
    health_report = health(
        run_data=df,
        reward_col="reward_mean",
        step_col="step"
    )
    
    print(f"   Reward health passed: {health_report.passed}")
    print(f"   Calibration score: {health_report.calibration_score:.3f}")
    
    # 7. Finish experiment
    print("\n7. Finishing experiment...")
    summary = tracker.finish_experiment()
    print(f"   Experiment completed: {summary['experiment_id']}")
    
    print("\n✅ Quick start demo completed!")
    print(f"   Check the runs/ directory for experiment data")

if __name__ == "__main__":
    main()
```

## Next Steps

Now that you've completed the quick start:

1. **[User Guide](../user-guide/tracking.md)** - Deep dive into experiment tracking
2. **[Forensics Analysis](../user-guide/forensics.md)** - Advanced debugging techniques
3. **[Examples](../examples/basic-ppo-cartpole.md)** - Real-world example notebooks
4. **[CLI Reference](../reference/commands.md)** - Complete command reference
5. **[API Reference](../reference/api.md)** - Detailed API documentation

## Tips for Success

- **Always set seeds** at the beginning of your experiments
- **Track everything** - datasets, models, hyperparameters, and environment
- **Run forensics regularly** to catch training issues early
- **Use the CLI** for quick analysis and debugging
- **Check determinism** before relying on results
- **Document your experiments** with meaningful names and metadata

Happy experimenting! 🎉
