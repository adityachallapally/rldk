# RL Debug Kit (RLDK)

> **The missing piece for RL experiment reproducibility. Track everything, debug anything, reproduce everything.**

RLDK is a **super sharp but minimal knife** that solves critical pain points with RL frameworks today. It provides comprehensive experiment tracking, debugging tools, and reproducibility guarantees - all working offline with no cloud dependencies.

## 🚀 **What's Ready Now (Phase 1)**

### **Standalone Experiment Tracking**
Track every aspect of your RL experiments with zero dependencies:

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Complete experiment tracking
config = TrackingConfig(experiment_name="my_ppo_experiment")
tracker = ExperimentTracker(config)

tracker.start_experiment()
tracker.track_dataset(data, "training_data")
tracker.track_model(model, "gpt2_policy")
tracker.set_seeds(42)
tracker.finish_experiment()

# Reproduce exactly: same data, model, seeds, environment
```

### **Forensics & Debugging Tools**
Debug RL training issues with comprehensive analysis:

```bash
# Environment audit - detect non-determinism
rldk env-audit ./my_training_run
# → "❌ Non-deterministic training detected"

# Log scanning - find PPO anomalies  
rldk log-scan ./my_training_run
# → "🚨 KL spike detected at step 800"

# Checkpoint comparison - track model changes
rldk diff-ckpt model_a.pt model_b.pt
# → "⚠️ Value head parameters changed significantly"

# Reward drift detection
rldk reward-drift model_a model_b --prompts prompts.jsonl
# → "📉 Reward correlation dropped to 0.12"

# Run comparison - find divergences
rldk diff --a run_a --b run_b --signals loss,reward_mean
# → "🚨 Divergence detected at step 150"
```

### **Core CLI Commands**
All commands work offline with no external dependencies:

```bash
# Experiment tracking
rldk track "my_experiment"

# Forensics analysis  
rldk env-audit <run_or_repo>
rldk log-scan <run_or_export>
rldk diff-ckpt <ckpt_a> <ckpt_b>
rldk reward-drift <model_a> <model_b>
rldk compare-runs <run_a> <run_b>
rldk doctor <run_or_repo>

# Evaluation & reproducibility
rldk check-determinism --cmd "python train.py"
rldk replay <run_path> --command "python train.py"
rldk eval <run_path> --suite quick
```

## 🎯 **Value for Serious Researchers**

### **Complete Reproducibility**
- **Dataset versioning** - SHA-256 checksums for all data
- **Model fingerprinting** - Architecture and parameter tracking  
- **Environment capture** - Complete system state snapshots
- **Seed management** - Reproducible random state
- **Git integration** - Repository state tracking

### **Debugging Superpowers**
- **PPO anomaly detection** - 182+ rules for training issues
- **Determinism checking** - Find non-reproducible training
- **Drift detection** - Catch reward model changes
- **Checkpoint analysis** - Track model evolution
- **Run comparison** - Find divergences between experiments

### **Production Ready**
- **Works offline** - No cloud dependencies
- **CPU-only** - No GPU required for analysis
- **Lightweight** - Minimal dependencies
- **Extensible** - Plugin architecture for custom analysis

## 📦 **Installation**

### **Core Package (Recommended)**
```bash
pip install rldk
```

### **Full Package (Optional)**
```bash
pip install rldk[full]  # Includes all integrations
```

## 🚀 **Quick Start**

### **1. Track Your Experiment**
```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Start tracking
config = TrackingConfig(experiment_name="ppo_training")
tracker = ExperimentTracker(config)

tracker.start_experiment()

# Track your data
tracker.track_dataset(training_data, "training_data")
tracker.track_dataset(eval_data, "eval_data")

# Track your model
tracker.track_model(model, "gpt2_policy")

# Set reproducible seeds
tracker.set_seeds(42)

# Add metadata
tracker.add_metadata({
    "learning_rate": 1e-5,
    "batch_size": 32,
    "epochs": 10
})

tracker.finish_experiment()
```

### **2. Debug Training Issues**
```bash
# Check for non-determinism
rldk env-audit ./my_training_run

# Scan for PPO anomalies
rldk log-scan ./my_training_run

# Compare checkpoints
rldk diff-ckpt checkpoint_100.pt checkpoint_200.pt

# Detect reward drift
rldk reward-drift reward_model_v1 reward_model_v2 --prompts prompts.jsonl
```

### **3. Reproduce Experiments**
```bash
# Check determinism
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean

# Replay with same seed
rldk replay ./my_training_run --command "python train.py" --metrics loss,reward_mean

# Run evaluation
rldk eval ./my_training_run --suite quick
```

## 📊 **What You Get**

### **Experiment Tracking**
- ✅ **Dataset versioning** - Track data changes and integrity
- ✅ **Model fingerprinting** - Architecture and parameter checksums
- ✅ **Environment capture** - Complete system state snapshots
- ✅ **Seed management** - Reproducible random state
- ✅ **Git integration** - Repository state tracking
- ✅ **Metadata tracking** - Custom experiment metadata

### **Forensics Analysis**
- ✅ **Environment audit** - Detect non-determinism issues
- ✅ **PPO anomaly detection** - 182+ rules for training problems
- ✅ **Checkpoint comparison** - Track model parameter changes
- ✅ **Reward drift detection** - Catch reward model changes
- ✅ **Run comparison** - Find divergences between experiments
- ✅ **Comprehensive diagnostics** - Combined analysis tools

### **Reproducibility Tools**
- ✅ **Determinism checking** - Verify reproducible training
- ✅ **Experiment replay** - Reproduce with same seeds
- ✅ **Evaluation suites** - Statistical analysis of results
- ✅ **Regression detection** - Git bisect for finding issues

## 🔧 **Architecture**

### **Core Components**
```
rldk/
├── tracking/          # Standalone experiment tracking
├── forensics/         # PPO anomaly detection & analysis
├── ingest/           # Data ingestion from various sources
├── diff/             # Run comparison & divergence detection
├── determinism/       # Determinism checking & verification
├── reward/           # Reward model analysis & drift detection
├── evals/            # Evaluation suites & statistical analysis
└── cli.py            # Command-line interface
```

### **Integration Components** (Coming in v0.2)
```
rldk/
├── integrations/     # Framework integrations (optional)
│   ├── trl/         # TRL integration
│   ├── openrlhf/    # OpenRLHF integration
│   └── wandb/       # Weights & Biases integration
└── examples/        # Integration examples
```

## 📈 **Performance**

### **Large Model Support**
- ✅ **Models up to 365M parameters** - Efficient architecture fingerprinting
- ✅ **Datasets with 1M+ samples** - Intelligent sampling for checksums
- ✅ **Fast checksum computation** - 1M elements in 0.040 seconds

### **Memory Efficient**
- ✅ **No model weight storage** - Architecture fingerprinting only
- ✅ **Intelligent sampling** - Large dataset handling
- ✅ **Streaming support** - Process data without loading everything

## 🎯 **Use Cases**

### **For Researchers**
- **Complete reproducibility** - Every experiment can be exactly reproduced
- **Debugging tools** - Find training issues quickly
- **Experiment management** - Track and compare experiments
- **Collaboration** - Share reproducible experiments with team

### **For Teams**
- **Experiment tracking** - Centralized experiment management
- **Issue debugging** - Quick identification of training problems
- **Model comparison** - Track model evolution and changes
- **Compliance** - Complete audit trail for experiments

### **For Production**
- **Model deployment** - Verified model architectures and data
- **Rollback capability** - Revert to previous experiment states
- **Debugging** - Full context for troubleshooting issues
- **Audit trail** - Complete history for regulatory requirements

## 🚀 **Roadmap**

### **v0.1 (Current) - Core Release**
- ✅ **Standalone tracking system** - Complete experiment tracking
- ✅ **Phase A forensics** - PPO anomaly detection & analysis
- ✅ **Core CLI commands** - All debugging tools
- ✅ **Offline operation** - No cloud dependencies

### **v0.2 (Next) - Integration Release**
- 🔄 **TRL integration** - Seamless TRL training monitoring
- 🔄 **OpenRLHF integration** - Distributed training monitoring
- 🔄 **WandB integration** - Cloud experiment tracking
- 🔄 **Advanced visualizations** - Web dashboard

### **v0.3 (Future) - Advanced Features**
- 🔄 **Real-time monitoring** - Live training monitoring
- 🔄 **Automated debugging** - AI-powered issue detection
- 🔄 **Cloud deployment** - Managed experiment tracking
- 🔄 **API access** - Programmatic access to tracking data

## 🤝 **Contributing**

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
git clone https://github.com/your-org/rldk.git
cd rldk
pip install -e .[dev]
```

### **Running Tests**
```bash
# Core functionality tests
python test_tracking_standalone.py
python test_deterministic_standalone.py

# Integration tests (requires dependencies)
python test_acceptance.py
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 **Acknowledgments**

RLDK builds on the work of the open-source RL community, particularly:
- [TRL](https://github.com/huggingface/trl) - RL training library
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - RLHF framework
- [Weights & Biases](https://wandb.ai/) - Experiment tracking

---

**Ready to ship reproducible RL experiments?** Get started with RLDK today!
