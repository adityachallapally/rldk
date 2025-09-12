# RLDK Documentation

Welcome to the **RL Debug Kit (RLDK)** documentation! RLDK is a comprehensive debugging and analysis toolkit for reinforcement learning training runs. It provides experiment tracking, forensics analysis, reproducibility tools, and evaluation suites - all working offline with minimal dependencies.

## 🚀 What is RLDK?

RLDK is designed to be the missing piece for RL experiment reproducibility. It helps researchers and engineers:

- **Track everything** - Complete experiment tracking with dataset versioning, model fingerprinting, and environment capture
- **Debug anything** - Advanced forensics analysis with 30+ comprehensive anomaly detection rules
- **Reproduce everything** - Determinism checking, seeded replay, and comprehensive reproducibility verification

## 🎯 Key Features

### Experiment Tracking System
Complete experiment tracking with dataset versioning, model fingerprinting, and environment capture:

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Configure tracking
config = TrackingConfig(
    experiment_name="my_ppo_experiment",
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True
)
tracker = ExperimentTracker(config)

# Start experiment and capture state
tracker.start_experiment()
```

### Comprehensive PPO Forensics
Advanced PPO training analysis with comprehensive anomaly detection:

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
```

### Determinism Checking & Verification
Verify training reproducibility across multiple runs:

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

## 📦 Installation

### Core Package (Recommended)
```bash
pip install rldk
```

### Development Package
```bash
pip install rldk[dev]  # Includes testing and development tools
```

### Optional Dependencies
```bash
pip install rldk[parquet]  # For Parquet file support
pip install rldk[openrlhf]  # For OpenRLHF integration
```

## 🚀 Quick Start

### 1. Track Your Experiment
```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Configure tracking
config = TrackingConfig(
    experiment_name="ppo_training",
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True
)
tracker = ExperimentTracker(config)

# Start experiment and capture state
tracker.start_experiment()

# Track datasets with checksums
tracker.track_dataset(training_data, "training_data")
tracker.track_dataset(eval_data, "eval_data")

# Track models with architecture fingerprinting
tracker.track_model(model, "gpt2_policy")
tracker.track_tokenizer(tokenizer, "gpt2_tokenizer")

# Set reproducible seeds
tracker.set_seeds(42)

# Add custom metadata
tracker.add_metadata("learning_rate", 1e-5)
tracker.add_metadata("batch_size", 32)

# Finish and save
tracker.finish_experiment()
```

### 2. Debug Training Issues
```bash
# Environment audit - detect non-determinism
rldk forensics env-audit ./my_training_run

# Log scan - find PPO anomalies
rldk forensics log-scan ./my_training_run

# Checkpoint comparison - track model changes
rldk forensics diff-ckpt checkpoint_100.pt checkpoint_200.pt

# Comprehensive diagnostics
rldk forensics doctor ./my_training_run

# Reward drift detection
rldk reward reward-drift model_a model_b --prompts prompts.jsonl
```

### 3. Reproduce Experiments
```bash
# Check determinism across multiple runs
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean --replicas 5

# Replay with exact same seed
rldk replay ./my_training_run --command "python train.py --seed {seed}" --metrics loss,reward_mean

# Run evaluation suite
rldk evals evaluate data.jsonl --suite quick --output results.json

# Find regression with git bisect
rldk bisect --good abc123 --bad def456 --cmd "python train.py"
```

## 📊 What You Get

### Complete Reproducibility
- **Dataset versioning** - SHA-256 checksums with intelligent sampling for large datasets
- **Model fingerprinting** - Architecture fingerprinting and parameter tracking (up to 365M parameters)
- **Environment capture** - Complete system state snapshots including conda/pip environments
- **Seed management** - Comprehensive RNG state tracking (Python, NumPy, PyTorch, CUDA)
- **Git integration** - Repository state tracking with commit hashes and diff capture

### Advanced Debugging
- **PPO anomaly detection** - 182+ rules for training issues including KL spikes, gradient anomalies, advantage statistics
- **Determinism checking** - Multi-replica verification with detailed mismatch analysis
- **Reward drift detection** - Statistical analysis of reward model changes with correlation metrics
- **Checkpoint analysis** - Parameter-level comparison with L2 norms and cosine similarity
- **Run comparison** - Rolling z-score divergence detection with configurable thresholds

### Comprehensive Analysis
- **Health scoring** - Overall training health, stability, and convergence quality metrics
- **Statistical evaluation** - Multiple evaluation suites (quick, comprehensive, safety) with confidence intervals
- **Card generation** - Visual trust cards for determinism, drift, and reward analysis
- **Data ingestion** - Support for TRL, OpenRLHF, WandB, and custom JSONL formats
- **Seeded replay** - Exact reproduction of training runs with tolerance-based verification

## 🎯 Use Cases

### For Researchers
- **Complete reproducibility** - Every experiment can be exactly reproduced
- **Debugging tools** - Find training issues quickly with comprehensive analysis
- **Experiment management** - Track and compare experiments with detailed metadata
- **Collaboration** - Share reproducible experiments with team members

### For Teams
- **Experiment tracking** - Centralized experiment management with version control
- **Issue debugging** - Quick identification of training problems with automated analysis
- **Model comparison** - Track model evolution and changes with detailed metrics
- **Compliance** - Complete audit trail for experiments with full reproducibility

### For Production
- **Model deployment** - Verified model architectures and data with integrity checks
- **Rollback capability** - Revert to previous experiment states with exact reproduction
- **Debugging** - Full context for troubleshooting issues with comprehensive diagnostics
- **Audit trail** - Complete history for regulatory requirements with detailed tracking

## 📚 Documentation Structure

This documentation is organized into several sections:

- **[Getting Started](getting-started/installation.md)** - Installation and basic setup
- **[User Guide](user-guide/tracking.md)** - Comprehensive usage guides for all features
- **[CLI Reference](reference/commands.md)** - Complete command-line interface documentation
- **[API Reference](reference/api.md)** - Detailed API documentation with examples
- **[Examples](examples/basic-ppo-cartpole.md)** - CPU-friendly example notebooks
- **[Research Use Cases](research/failure-patterns.md)** - Common RL training failure patterns and troubleshooting
- **[Development](development/contributing.md)** - Contributing and development guidelines

## 🤝 Contributing

We welcome contributions! See [Contributing](development/contributing.md) for details.

### Development Setup
```bash
git clone https://github.com/your-org/rldk.git
cd rldk
pip install -e .[dev]
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rldk --cov-report=html

# Run acceptance tests
./scripts/dev/acceptance.sh
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

RLDK builds on the work of the open-source RL community, particularly:
- [TRL](https://github.com/huggingface/trl) - RL training library
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - RLHF framework
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://github.com/huggingface/transformers) - Model architectures
- [Datasets](https://github.com/huggingface/datasets) - Dataset handling

---

**Ready to ship reproducible RL experiments?** Get started with RLDK today!

```bash
pip install rldk
rldk track "my_first_experiment"
```