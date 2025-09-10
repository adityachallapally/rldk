# RL Debug Kit (RLDK)

> **Debugging and analysis tools for reinforcement learning training runs.**

RLDK is a toolkit for debugging and analyzing RL training runs. It provides experiment tracking, forensics analysis, and reproducibility tools.

## 🚀 **What's Available Now**

### **Experiment Tracking**
Track your RL experiments with comprehensive metadata:

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Start tracking
config = TrackingConfig(experiment_name="my_ppo_experiment")
tracker = ExperimentTracker(config)

tracker.start_experiment()
tracker.track_dataset(data, "training_data")
tracker.track_model(model, "gpt2_policy")
tracker.set_seeds(42)
tracker.finish_experiment()
```

### **Forensics Analysis**
Debug training issues with built-in analysis tools:

```bash
# Environment audit - detect non-determinism
rldk env-audit ./my_training_run

# Log scanning - find PPO anomalies  
rldk log-scan ./my_training_run

# Checkpoint comparison - track model changes
rldk diff-ckpt model_a.pt model_b.pt

# Run comparison - find divergences
rldk forensics compare-runs run_a run_b

# Comprehensive diagnostics
rldk doctor ./my_training_run
```

### **Reward Model Analysis**
Analyze reward model health and detect issues:

```bash
# Compare reward models
rldk reward reward-drift model_a model_b --prompts prompts.jsonl

# Health analysis
rldk reward reward-health run --scores scores.jsonl --out analysis/
```

### **Determinism Checking**
Verify reproducible training:

```bash
# Check if training is deterministic
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean

# Replay with same seed
rldk replay ./my_training_run --command "python train.py" --metrics loss,reward_mean
```

### **Evaluation Suites**
Run evaluation on your training data:

```bash
# Quick evaluation
rldk evals evaluate data.jsonl --suite quick

# Comprehensive evaluation
rldk evals evaluate data.jsonl --suite comprehensive --output results.json
```

## 📦 **Installation**

```bash
pip install rldk
```

## 🚀 **Quick Start**

### **1. Track Your Experiment**
```python
from rldk.tracking import ExperimentTracker, TrackingConfig

config = TrackingConfig(experiment_name="ppo_training")
tracker = ExperimentTracker(config)

tracker.start_experiment()
tracker.track_dataset(training_data, "training_data")
tracker.track_model(model, "gpt2_policy")
tracker.set_seeds(42)
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
```

### **3. Analyze Reward Models**
```bash
# Compare reward models
rldk reward reward-drift reward_model_v1 reward_model_v2 --prompts prompts.jsonl

# Health analysis
rldk reward reward-health run --scores scores.jsonl --out analysis/
```

## 📊 **Core Features**

### **Experiment Tracking**
- ✅ **Dataset versioning** - Track data changes and integrity
- ✅ **Model fingerprinting** - Architecture and parameter checksums
- ✅ **Environment capture** - System state snapshots
- ✅ **Seed management** - Reproducible random state
- ✅ **Git integration** - Repository state tracking
- ✅ **Metadata tracking** - Custom experiment metadata

### **Forensics Analysis**
- ✅ **Environment audit** - Detect non-determinism issues
- ✅ **PPO anomaly detection** - Rules for training problems
- ✅ **Checkpoint comparison** - Track model parameter changes
- ✅ **Run comparison** - Find divergences between experiments
- ✅ **Comprehensive diagnostics** - Combined analysis tools

### **Reward Analysis**
- ✅ **Reward drift detection** - Compare reward models
- ✅ **Health analysis** - Detect reward model pathologies
- ✅ **Calibration analysis** - Check reward model calibration

### **Reproducibility Tools**
- ✅ **Determinism checking** - Verify reproducible training
- ✅ **Experiment replay** - Reproduce with same seeds
- ✅ **Evaluation suites** - Statistical analysis of results

## 🔧 **Architecture**

```
rldk/
├── tracking/          # Experiment tracking
├── forensics/         # PPO anomaly detection & analysis
├── reward/            # Reward model analysis
├── evals/             # Evaluation suites
├── determinism/       # Determinism checking
├── diff/              # Run comparison
├── replay/            # Experiment replay
└── cli.py             # Command-line interface
```

## 🚀 **Roadmap**

### **Phase 1 (Current) - Core Features**
- ✅ **Experiment tracking** - Basic tracking functionality
- ✅ **Forensics analysis** - PPO anomaly detection
- ✅ **Reward analysis** - Drift and health detection
- ✅ **Determinism checking** - Reproducibility verification
- ✅ **Evaluation suites** - Basic evaluation capabilities

### **Phase 2 (Future) - Integrations**
- 🔄 **TRL integration** - Seamless TRL training monitoring
- 🔄 **OpenRLHF integration** - Distributed training monitoring
- 🔄 **WandB integration** - Cloud experiment tracking
- 🔄 **Advanced visualizations** - Web dashboard

### **Phase 3 (Future) - Advanced Features**
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
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

---

**Ready to debug your RL experiments?** Get started with RLDK today!