# RL Debug Kit (RLDK)

> **The missing piece for RL experiment reproducibility. Track everything, debug anything, reproduce everything.**

RLDK is a comprehensive debugging and analysis toolkit for reinforcement learning training runs. It provides experiment tracking, forensics analysis, reproducibility tools, and evaluation suites - all working offline with minimal dependencies.

## 🚀 **Core Features**

### **Experiment Tracking System**
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

# Track datasets with checksums
tracker.track_dataset(data, "training_data")
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

### **Comprehensive PPO Forensics**
Advanced PPO training analysis with 30+ comprehensive anomaly detection rules:

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

# Get comprehensive analysis
analysis = forensics.get_comprehensive_analysis()
anomalies = forensics.get_anomalies()
health_summary = forensics.get_health_summary()
```

### **Determinism Checking & Verification**
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

### **Reward Model Health Analysis**
Comprehensive reward model analysis and drift detection:

```python
from rldk.reward import health, compare_models

# Analyze reward model health
health_report = health(
    run_data=training_data,
    reference_data=baseline_data,
    reward_col="reward_mean",
    threshold_drift=0.1,
    threshold_saturation=0.8,
    threshold_calibration=0.7
)

# Compare two reward models
drift_report = compare_models(
    model_a="path/to/model_a",
    model_b="path/to/model_b", 
    prompts=prompt_texts
)
```

### **Evaluation Suites**
Statistical evaluation with multiple test suites:

```python
from rldk.evals import run
from rldk.evals.suites import QUICK_SUITE, COMPREHENSIVE_SUITE

# Run evaluation suite
eval_result = run(
    run_data=training_data,
    suite="comprehensive",
    seed=42,
    sample_size=200
)

print(f"Overall score: {eval_result.overall_score}")
print(f"Confidence intervals: {eval_result.confidence_intervals}")
```

### **Run Comparison & Divergence Detection**
Find when and why training runs diverge:

```python
from rldk.diff import first_divergence

# Compare two training runs
divergence_report = first_divergence(
    df_a=run_a_data,
    df_b=run_b_data,
    signals=["loss", "reward_mean", "kl"],
    k_consecutive=3,
    window=50,
    tolerance=2.0
)

print(f"Diverged: {divergence_report.diverged}")
print(f"First divergence at step: {divergence_report.first_step}")
print(f"Tripped signals: {divergence_report.tripped_signals}")
```

### **Seeded Replay System**
Reproduce training runs with exact same seeds:

```python
from rldk.replay import replay

# Replay a training run
replay_report = replay(
    run_path="./original_run",
    training_command="python train.py --seed {seed}",
    metrics_to_compare=["loss", "reward_mean"],
    tolerance=0.01,
    max_steps=1000
)

print(f"Replay passed: {replay_report.passed}")
print(f"Original seed: {replay_report.original_seed}")
print(f"Replay seed: {replay_report.replay_seed}")
```

### **Data Ingestion & Adapters**
Support for multiple training frameworks and data sources:

```python
from rldk.ingest import ingest_runs
from rldk.adapters import TRLAdapter, OpenRLHFAdapter, WandBAdapter

# Ingest from various sources
df = ingest_runs("path/to/logs", adapter_hint="trl")
df = ingest_runs("wandb://project/run_id", adapter_hint="wandb")
df = ingest_runs("path/to/openrlhf_logs", adapter_hint="openrlhf")
```

### **CLI Commands**
Comprehensive command-line interface for all functionality:

```bash
# Experiment tracking
rldk track "my_experiment" --interactive

# Forensics analysis
rldk forensics env-audit ./my_training_run
rldk forensics log-scan ./my_training_run
rldk forensics diff-ckpt model_a.pt model_b.pt
rldk forensics compare-runs run_a run_b
rldk forensics doctor ./my_training_run

# Reward analysis
rldk reward reward-drift model_a model_b --prompts prompts.jsonl
rldk reward reward-health run --scores scores.jsonl --out ./reports

# Evaluation
rldk evals evaluate data.jsonl --suite comprehensive --output results.json
rldk evals list-suites
rldk evals validate-data data.jsonl

# Determinism & reproducibility
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean
rldk replay ./run --command "python train.py --seed {seed}" --metrics loss,reward_mean
rldk bisect --good abc123 --bad def456 --cmd "python train.py"

# Data ingestion
rldk ingest ./logs --adapter trl --output metrics.jsonl
rldk diff --a run_a --b run_b --signals loss,reward_mean

# Demo data generation
rldk demo --out ./demo_data --seed 1337 --steps 200

# Card generation
rldk card determinism run_a
rldk card drift run_a run_b
rldk card reward run_a
```

## 📦 **Installation**

### **Core Package (Recommended)**
```bash
pip install rldk
```

### **Development Package**
```bash
pip install rldk[dev]  # Includes testing and development tools
```

### **Optional Dependencies**
```bash
pip install rldk[parquet]  # For Parquet file support
pip install rldk[openrlhf]  # For OpenRLHF integration
```

## 🚀 **Quick Start**

### **1. Generate Demo Data**
```bash
# Generate synthetic training data for testing
rldk demo --out ./demo_data --seed 1337 --steps 200
```

### **2. Track Your Experiment**
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

### **3. Debug Training Issues**
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

### **4. Reproduce Experiments**
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

## 📊 **What You Get**

### **Experiment Tracking System**
- ✅ **Dataset versioning** - SHA-256 checksums with intelligent sampling for large datasets
- ✅ **Model fingerprinting** - Architecture fingerprinting and parameter tracking (up to 100M parameters)
- ✅ **Environment capture** - Complete system state snapshots including conda/pip environments
- ✅ **Seed management** - Comprehensive RNG state tracking (Python, NumPy, PyTorch, CUDA)
- ✅ **Git integration** - Repository state tracking with commit hashes and diff capture
- ✅ **Metadata tracking** - Custom experiment metadata and tags
- ✅ **WandB integration** - Optional cloud logging with Weights & Biases

### **Comprehensive Forensics Analysis**
- ✅ **Environment audit** - Detect non-determinism issues with detailed diagnostics
- ✅ **PPO anomaly detection** - 30+ comprehensive rules for training problems including KL spikes, gradient anomalies
- ✅ **Checkpoint comparison** - Parameter-level comparison with L2 norms and cosine similarity
- ✅ **Reward drift detection** - Statistical analysis with correlation metrics and scatter plots
- ✅ **Run comparison** - Rolling z-score divergence detection with configurable thresholds
- ✅ **Comprehensive diagnostics** - Combined analysis with health scoring and anomaly detection
- ✅ **Advantage statistics** - Advanced tracking of advantage distribution and quality metrics

### **Reproducibility & Verification Tools**
- ✅ **Determinism checking** - Multi-replica verification with detailed mismatch analysis
- ✅ **Seeded replay** - Exact reproduction of training runs with tolerance-based verification
- ✅ **Evaluation suites** - Multiple test suites (quick, comprehensive, safety) with statistical analysis
- ✅ **Regression detection** - Git bisect integration for finding problematic commits
- ✅ **Health analysis** - Overall training health, stability, and convergence quality metrics
- ✅ **Card generation** - Visual trust cards for determinism, drift, and reward analysis

### **Data Ingestion & Integration**
- ✅ **Multi-framework support** - TRL, OpenRLHF, WandB, and custom JSONL adapters
- ✅ **Flexible data sources** - Support for local files, directories, and cloud URIs
- ✅ **Schema validation** - Automatic data validation and standardization
- ✅ **Event processing** - Normalized event schema for consistent analysis

## 🎯 **Use Cases**

### **For Researchers**
- **Complete reproducibility** - Every experiment can be exactly reproduced
- **Debugging tools** - Find training issues quickly with comprehensive analysis
- **Experiment management** - Track and compare experiments with detailed metadata
- **Collaboration** - Share reproducible experiments with team members

### **For Teams**
- **Experiment tracking** - Centralized experiment management with version control
- **Issue debugging** - Quick identification of training problems with automated analysis
- **Model comparison** - Track model evolution and changes with detailed metrics
- **Compliance** - Complete audit trail for experiments with full reproducibility

### **For Production**
- **Model deployment** - Verified model architectures and data with integrity checks
- **Rollback capability** - Revert to previous experiment states with exact reproduction
- **Debugging** - Full context for troubleshooting issues with comprehensive diagnostics
- **Audit trail** - Complete history for regulatory requirements with detailed tracking

## 🚀 **Current Status**

### **v0.1.0 - Core Release (Available Now)**
- ✅ **Complete experiment tracking** - Dataset versioning, model fingerprinting, environment capture
- ✅ **Comprehensive PPO forensics** - 30+ comprehensive anomaly detection rules with advanced tracking
- ✅ **Determinism verification** - Multi-replica checking with detailed analysis
- ✅ **Reward model analysis** - Health checking, drift detection, and calibration analysis
- ✅ **Evaluation suites** - Quick, comprehensive, and safety evaluation suites
- ✅ **Run comparison** - Rolling z-score divergence detection
- ✅ **Seeded replay** - Exact reproduction of training runs
- ✅ **Data ingestion** - Support for TRL, OpenRLHF, WandB, and custom formats
- ✅ **CLI interface** - Comprehensive command-line tools
- ✅ **Card generation** - Visual trust cards for analysis results
- ✅ **Git bisect integration** - Regression detection and debugging
- ✅ **Demo data generation** - Synthetic data for testing and examples

### **Integration Support (Available Now)**
- ✅ **TRL integration** - Seamless TRL training monitoring with callbacks
- ✅ **OpenRLHF integration** - Distributed training monitoring
- ✅ **WandB integration** - Cloud experiment tracking and logging
- ✅ **Custom adapters** - Extensible adapter system for new frameworks

### **Future Enhancements**
- 🔄 **Real-time monitoring** - Live training monitoring dashboard
- 🔄 **Advanced visualizations** - Interactive web dashboard
- 🔄 **Automated debugging** - AI-powered issue detection and recommendations
- 🔄 **Cloud deployment** - Managed experiment tracking service
- 🔄 **API access** - RESTful API for programmatic access

## 🤝 **Contributing**

We welcome contributions! See [Contributing](development/contributing.md) for details.

### **Development Setup**
```bash
git clone https://github.com/your-org/rldk.git
cd rldk
pip install -e .[dev]
```

### **Running Tests**
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/e2e/          # End-to-end tests

# Run with coverage
pytest --cov=rldk --cov-report=html

# Run specific test files
pytest tests/unit/test_tracking.py
pytest tests/unit/test_forensics.py
pytest tests/unit/test_determinism.py
```

### **Code Quality**
```bash
# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

### **Examples and Demos**
```bash
# Run example scripts
python examples/tracking_demo.py
python examples/comprehensive_ppo_forensics_example.py
python examples/replay_demo.py

# Run Jupyter notebook
jupyter notebook examples/rldk_demo.ipynb
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 **Acknowledgments**

RLDK builds on the work of the open-source RL community, particularly:
- [TRL](https://github.com/huggingface/trl) - RL training library
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - RLHF framework
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://github.com/huggingface/transformers) - Model architectures
- [Datasets](https://github.com/huggingface/datasets) - Dataset handling

## 📞 **Support**

- **Documentation**: [GitHub Wiki](https://github.com/your-org/rldk/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-org/rldk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rldk/discussions)
- **Examples**: [Examples Directory](examples/)

---

**Ready to ship reproducible RL experiments?** Get started with RLDK today!

```bash
pip install rldk
rldk demo --out ./demo_data --seed 1337
```