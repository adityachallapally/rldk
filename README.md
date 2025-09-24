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
    enable_advantage_statistics=True,
    enable_kl_drift_tracking=True,
    kl_drift_threshold=0.15,
    kl_drift_window_size=100,
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
kl_drift = forensics.get_kl_drift_analysis()

if kl_drift.get("detected"):
    print(f"⚠️  KL drift detected with score {kl_drift['score']:.3f}")
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

### **Centralized Seed Management**
Comprehensive seed handling for reproducible experiments:

```python
from rldk.utils.seed import set_global_seed, get_current_seed, seed_context

# Set global seed for reproducibility
seed = set_global_seed(42, deterministic=True)
print(f"Set seed: {seed}")

# Get current seed state
current_seed = get_current_seed()
print(f"Current seed: {current_seed}")

# Use seed context manager for temporary seed changes
with seed_context(123):
    # Code here uses seed 123
    pass
# Seed automatically restored to previous value

# CLI seed management
# rldk seed --seed 42 --deterministic
# rldk seed --show
# rldk seed --env --validate
```

### **Live JSONL Monitoring (Framework-Agnostic)**
Observe any trainer in real time by streaming JSONL metrics into the `rldk monitor` CLI. Built-in rule presets and field-map presets
make it a zero-code add-on for PPO/GRPO, TRL, Accelerate, and OpenRLHF workflows.

```bash
# Stream the demo loop directly into the monitor (no files required)
python examples/minimal_streaming_loop.py \
  | rldk monitor --rules ppo_safe --preset trl --alerts artifacts/alerts.jsonl

# Stream the GRPO minimal loop with GRPO guardrails
python examples/grpo_minimal_loop.py \
  | rldk monitor --rules grpo_safe --preset grpo --alerts artifacts/grpo_alerts.jsonl

# Or point the monitor at a directory of JSONL files (tails the newest)
rldk monitor --stream artifacts --rules ppo_strict --preset accelerate --pid <trainer_pid>

# Batch analysis of a finished run with a custom rules file
rldk monitor --once artifacts/run.jsonl --rules my_rules.yaml --report artifacts/report.json

# Stream hosted runs without touching trainer code
rldk monitor --from-wandb my-entity/my-project/my-run --rules ppo_safe --alerts artifacts/wandb_alerts.jsonl
rldk monitor --from-mlflow 0123456789abcdef --rules dpo_basic --alerts artifacts/mlflow_alerts.jsonl

# Parse TRL stdout directly with regex sniffing
python train_trl.py | rldk monitor --rules ppo_safe --regex trl --alerts artifacts/stdout_alerts.jsonl
```

#### Observe any trainer in 2 lines

```python
from rldk.emit import EventWriter

w = EventWriter("artifacts/run.jsonl")
w.log(step=i, name="kl", value=kl_value)
```

```python
f = open("artifacts/run.jsonl", "a", buffering=1)
f.write(__import__("json").dumps({
    "time": __import__("datetime").datetime.utcnow().isoformat()+"Z",
    "step": i,
    "name": "kl",
    "value": float(kl_value),
}) + "\n")
f.flush()
```

Want a ready-made GRPO stream? `examples/grpo_minimal_loop.py` emits `kl`, `kl_coef`, `entropy`, `advantage_std`, and more via
`EventWriter`. Pair it with `rldk monitor --stream artifacts/grpo_run.jsonl --rules grpo_safe --preset grpo` or simply run
`make monitor-grpo` to launch the script, attach the monitor, and capture alerts plus a report automatically.

Pair either snippet with the ready-to-run `rules.yaml` in the repository root:

```bash
rldk monitor --stream artifacts/run.jsonl --rules rules.yaml --pid <trainer_pid>
# Dedicated KL drift monitoring preset
rldk monitor --stream artifacts/run.jsonl --rules kl_drift
```

Want the full walkthrough with alerts and auto-stop baked in? Run `make monitor-demo` to launch
`examples/minimal_streaming_loop.py`, attach the monitor, and review the resulting `artifacts/alerts.jsonl` and
`artifacts/report.json` files. Prefer GRPO metrics? `make monitor-grpo` runs the GRPO minimal loop, wires up `--preset grpo`
with `--rules grpo_safe`, and records the alerts plus report under `artifacts/` for inspection.

Key conveniences:

- **Rule presets** – `ppo_safe`, `ppo_strict`, `grpo_safe`, `grpo_strict`, `kl_drift`, and `dpo_basic` cover common KL, drift, reward, and gradient gates so you can start without writing YAML.
  - The GRPO presets now watch `grad_norm_policy` / `grad_norm_value` directly, warning once policy gradients rise above ~8.0 (6.5 in strict mode) or when the policy/value ratio derived from the latest gradient pair exceeds roughly 4.5 (3.5 strict). Tune these ceilings with `--set-rule grpo_safe_policy_grad_spike.condition="value > <new_threshold>"` or adjust the ratio guard with `--set-rule grpo_safe_grad_ratio_imbalance.condition="..."` if your trainer uses a different normalization scheme. Ensure both gradients are logged so the monitor can attach `policy_over_value` / `value_over_policy` metadata and compute the ratio.
- **Field-map presets** – `--preset grpo|trl|accelerate|openrlhf` normalizes popular logging key conventions; combine with `--field-map` for custom tweaks.
- **Directory tailing** – Passing a directory to `--stream` automatically follows the newest `*.jsonl` file and handles rotation.
- **Environment hook** – Set `RLDK_METRICS_PATH` to auto-tail a metrics file without specifying `--stream`; piping JSONL to stdin also just works.
- **Ready-made emitter** – `examples/minimal_streaming_loop.py` shows how to emit canonical events and responds to monitor stop actions out of the box.
- **Hosted run bridges** – `--from-wandb` and `--from-mlflow` follow remote projects with clear error messages and retry guards.
- **Regex sniffing** – `--regex` (for example `--regex trl`) extracts metrics from stdout/stderr without changing your training loop.

**Choosing a GRPO preset**

- `grpo_safe` keeps wider guard bands and primarily warns unless the KL spike or variance collapse persists. Use it for established
  runs where you want to observe drift but avoid premature stops.
- `grpo_strict` lowers KL and entropy ceilings, adds stop actions to most rules, and reacts faster to unstable acceptance rates.
  Reach for it when onboarding fresh policies or enforcing production guardrails.

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
rldk forensics kl-drift ./my_training_run
rldk forensics diff-ckpt model_a.pt model_b.pt
rldk forensics compare-runs run_a run_b
rldk forensics doctor ./my_training_run

# Reward analysis
rldk reward reward-drift model_a model_b --prompts prompts.jsonl
rldk reward reward-health run --scores scores.jsonl --out ./reports
rldk reward length-bias --run-path run.jsonl --response-col response_text --reward-col reward_mean --output-dir ./reports

# Evaluation
rldk evals evaluate data.jsonl --suite comprehensive --output results.json
rldk evals list-suites
rldk evals validate-data data.jsonl

# Determinism & reproducibility
rldk check-determinism --cmd "python train.py" --compare loss,reward_mean
rldk replay ./run --command "python train.py --seed {seed}" --metrics loss,reward_mean
rldk bisect --good abc123 --bad def456 --cmd "python train.py"

# Seed management
rldk seed --seed 42 --deterministic
rldk seed --show
rldk seed --env --validate

# Data ingestion
rldk ingest ./logs --adapter trl --output metrics.jsonl
rldk diff --a run_a --b run_b --signals loss,reward_mean

# Card generation
rldk card determinism run_a
rldk card drift run_a run_b
rldk card reward run_a
```

## 🎯 **Key Capabilities**

### **Complete Reproducibility**
- **Dataset versioning** - SHA-256 checksums with intelligent sampling for large datasets
- **Model fingerprinting** - Architecture fingerprinting and parameter tracking (up to 365M parameters)
- **Environment capture** - Complete system state snapshots including conda/pip environments
- **Seed management** - Comprehensive RNG state tracking (Python, NumPy, PyTorch, CUDA)
- **Git integration** - Repository state tracking with commit hashes and diff capture

### **Advanced Debugging**
- **PPO anomaly detection** - 182+ rules for training issues including KL spikes, gradient anomalies, advantage statistics
- **Determinism checking** - Multi-replica verification with detailed mismatch analysis
- **Reward drift detection** - Statistical analysis of reward model changes with correlation metrics
- **Checkpoint analysis** - Parameter-level comparison with L2 norms and cosine similarity
- **Run comparison** - Rolling z-score divergence detection with configurable thresholds

### **Comprehensive Analysis**
- **Health scoring** - Overall training health, stability, and convergence quality metrics
- **Statistical evaluation** - Multiple evaluation suites (quick, comprehensive, safety) with confidence intervals
- **Card generation** - Visual trust cards for determinism, drift, and reward analysis
- **Data ingestion** - Support for TRL, OpenRLHF, WandB, and custom JSONL formats
- **Seeded replay** - Exact reproduction of training runs with tolerance-based verification

### **Production Ready**
- **Offline operation** - No cloud dependencies required
- **Memory efficient** - Intelligent sampling for large datasets and models
- **Extensible** - Plugin architecture with custom adapters and evaluation metrics
- **CI/CD integration** - Gate mode with configurable exit codes for automated testing

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

### **1. Track Your Experiment**
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

### **2. Debug Training Issues**
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

### **3. Reproduce Experiments**
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

## 📚 **Examples & Tutorials**

RLDK includes comprehensive examples and tutorials to get you started:

### **CPU-Friendly Examples**
- **`basic_ppo_cartpole.py`** - Basic PPO implementation with RLDK integration
- **`custom_environment_tutorial.py`** - How to create custom environments with RLDK tracking
- **`distributed_training_guide.py`** - Multi-GPU and federated learning with RLDK
- **`hyperparameter_tuning.py`** - Systematic hyperparameter optimization
- **`benchmark_comparison.py`** - Algorithm comparison and benchmarking
- **`research_reproducibility_workflow.py`** - Complete research workflow example
- **`multi_run_analysis.py`** - Analyzing multiple training runs
- **`production_deployment_checklist.py`** - Production deployment preparation

### **Getting Started with Examples**
```bash
# Run a basic PPO example
python examples/basic_ppo_cartpole.py

# Try hyperparameter tuning
python examples/hyperparameter_tuning.py

# Explore distributed training
python examples/distributed_training_guide.py
```

All examples are designed to run on CPU with minimal dependencies and include comprehensive documentation.

## 🔬 **Research Use Cases**

RLDK addresses common RL training failure patterns and research challenges:

### **Training Instability & Non-Determinism**
- **KL Divergence Explosions** - Detect when policy updates become too large
- **Gradient Vanishing/Exploding** - Monitor gradient norms and identify problematic updates
- **Reward Collapse** - Track reward distribution changes and detect saturation
- **Seed Sensitivity** - Identify when results depend heavily on random initialization
- **Environment Non-Determinism** - Detect when environment behavior varies across runs

### **Hyperparameter Sensitivity**
- **Learning Rate Sensitivity** - Identify when small LR changes cause large performance differences
- **Batch Size Effects** - Monitor how batch size affects training stability
- **Entropy Coefficient Tuning** - Track exploration vs exploitation balance
- **Value Function Overfitting** - Detect when value estimates become unreliable

### **Model Architecture Issues**
- **Policy Collapse** - Detect when policy becomes too deterministic
- **Value Function Divergence** - Monitor value function training quality
- **Advantage Estimation Errors** - Track advantage distribution and bias
- **Actor-Critic Imbalance** - Identify when policy and value updates are misaligned

### **Data Quality Problems**
- **Reward Hacking** - Detect when model exploits reward function flaws
- **Distribution Shift** - Monitor training vs evaluation data differences
- **Label Noise** - Identify when human feedback contains errors
- **Data Imbalance** - Detect when certain types of examples are underrepresented

### **Research Reproducibility**
- **Code Version Drift** - Track when code changes affect results
- **Dependency Changes** - Monitor when library updates break reproducibility
- **Hardware Differences** - Detect when results vary across different machines
- **Random State Management** - Ensure proper seed handling across experiments

### **Performance Analysis**
- **Convergence Speed** - Compare how quickly different algorithms converge
- **Sample Efficiency** - Measure data efficiency across different approaches
- **Computational Cost** - Track training time and resource usage
- **Memory Usage** - Monitor memory consumption patterns

### **Debugging Workflows**
```python
# 1. Detect training anomalies
forensics = ComprehensivePPOForensics()
anomalies = forensics.get_anomalies()

# 2. Compare different runs
divergence = first_divergence(run_a, run_b, signals=["loss", "reward"])

# 3. Verify reproducibility
determinism_check = check(cmd="python train.py", replicas=5)

# 4. Analyze reward model health
health_result = reward_health(
    training_data,  # DataFrame, list of dicts, or path to JSONL/table logs
    reference_data,
    response_col="response_text",
    length_col="tokens_out",
)
print(health_result.report.passed)
print(health_result.report.length_bias_metrics.bias_severity)
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

## 🔧 **Architecture**

### **Core Components**
```
rldk/
├── tracking/              # Experiment tracking system
│   ├── tracker.py        # Main ExperimentTracker class
│   ├── config.py         # TrackingConfig and settings
│   ├── dataset_tracker.py # Dataset versioning and checksums
│   ├── model_tracker.py  # Model fingerprinting and architecture tracking
│   ├── environment_tracker.py # Environment state capture
│   ├── seed_tracker.py   # RNG state management
│   └── git_tracker.py    # Git repository state tracking
├── forensics/            # PPO forensics and anomaly detection
│   ├── comprehensive_ppo_forensics.py # Advanced PPO analysis
│   ├── ppo_scan.py       # PPO anomaly detection rules
│   ├── env_audit.py      # Environment determinism audit
│   ├── log_scan.py       # Training log analysis
│   ├── ckpt_diff.py      # Checkpoint comparison
│   └── advantage_statistics_tracker.py # Advantage analysis
├── ingest/               # Data ingestion and adapters
│   └── ingest.py         # Main ingestion functions
├── adapters/             # Framework-specific adapters
│   ├── base.py           # Base adapter interface
│   ├── trl.py            # TRL integration
│   ├── openrlhf.py       # OpenRLHF integration
│   ├── wandb.py          # Weights & Biases integration
│   └── custom_jsonl.py   # Custom JSONL format support
├── diff/                 # Run comparison and divergence detection
│   └── diff.py           # Rolling z-score divergence analysis
├── determinism/          # Determinism checking and verification
│   └── check.py          # Multi-replica determinism verification
├── reward/               # Reward model analysis and health checking
│   ├── health_analysis.py # Reward model health analysis
│   ├── drift.py          # Reward drift detection
│   └── calibration.py    # Reward calibration analysis
├── evals/                # Evaluation suites and statistical analysis
│   ├── suites.py         # Evaluation suite definitions
│   ├── runner.py         # Evaluation execution
│   ├── metrics/          # Evaluation metrics (bias, toxicity, throughput)
│   └── probes.py         # Evaluation probes and tests
├── replay/               # Seeded replay system
│   └── replay.py         # Training run reproduction
├── bisect/               # Git bisect integration
│   └── bisect.py         # Regression detection
├── cards/                # Trust card generation
│   ├── determinism.py    # Determinism cards
│   ├── drift.py          # Drift cards
│   └── reward.py         # Reward health cards
├── io/                   # I/O utilities and schemas
│   ├── event_schema.py   # Normalized event schema
│   ├── writers.py        # Data writers
│   └── readers.py        # Data readers
├── config/               # Configuration management
│   └── settings.py       # Global settings and configuration
└── cli.py                # Command-line interface
```

### **Integration Components**
```
rldk/
├── integrations/         # Framework integrations
│   ├── trl/             # TRL monitoring and callbacks
│   └── openrlhf/        # OpenRLHF monitoring
└── examples/            # Usage examples and demos
    ├── comprehensive_ppo_forensics_example.py
    ├── tracking_demo.py
    ├── replay_demo.py
    └── trl_integration/  # TRL integration examples
```

## 📈 **Performance**

### **Large Model Support**
- ✅ **Models up to 100M parameters** - Efficient architecture fingerprinting
- ✅ **Datasets with 1M+ samples** - Intelligent sampling for checksums
- ✅ **Fast checksum computation** - Efficient sampling for large datasets
- ✅ **Memory efficient** - No model weight storage, architecture fingerprinting only
- ✅ **Streaming support** - Process data without loading everything into memory

### **Scalability**
- ✅ **Multi-replica determinism checks** - Parallel execution across replicas
- ✅ **Rolling window analysis** - Efficient z-score computation for large datasets
- ✅ **Intelligent sampling** - Representative sampling for large datasets
- ✅ **Batch processing** - Efficient processing of multiple training runs

## 📚 **API Documentation**

### **Core Tracking API**

#### `ExperimentTracker`
Main class for experiment tracking and state capture.

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Initialize tracker
config = TrackingConfig(
    experiment_name="my_experiment",
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True,
    save_to_wandb=True,
    wandb_project="my_project"
)
tracker = ExperimentTracker(config)

# Core methods
tracker.start_experiment()                    # Start experiment and capture state
tracker.track_dataset(data, name, metadata)   # Track dataset with checksums
tracker.track_model(model, name, metadata)    # Track model with fingerprinting
tracker.track_tokenizer(tokenizer, name)      # Track tokenizer
tracker.set_seeds(seed)                       # Set reproducible seeds
tracker.add_metadata(key, value)              # Add custom metadata
tracker.add_tag(tag)                          # Add experiment tag
tracker.finish_experiment()                   # Finish and save experiment
```

#### `TrackingConfig`
Configuration class for experiment tracking.

```python
config = TrackingConfig(
    experiment_name="required",
    experiment_id="optional_uuid",
    output_dir=Path("./runs"),
    
    # Component toggles
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True,
    
    # Dataset tracking options
    dataset_checksum_algorithm="sha256",
    
    # Model tracking options
    model_fingerprint_algorithm="sha256",
    save_model_architecture=True,
    save_model_weights=False,
    
    # Environment tracking options
    capture_conda_env=True,
    capture_pip_freeze=True,
    capture_system_info=True,
    
    # Seed tracking options
    track_numpy_seed=True,
    track_torch_seed=True,
    track_python_seed=True,
    track_cuda_seed=True,
    
    # Output options
    save_to_json=True,
    save_to_yaml=True,
    save_to_wandb=False,
    wandb_project="rldk-experiments",
    
    # Metadata
    tags=["experiment", "ppo"],
    notes="Optional experiment notes",
    metadata={"learning_rate": 1e-5}
)
```

### **Forensics API**

#### `ComprehensivePPOForensics`
Advanced PPO training analysis with comprehensive tracking.

```python
from rldk.forensics import ComprehensivePPOForensics

# Initialize forensics
forensics = ComprehensivePPOForensics(
    kl_target=0.1,
    kl_target_tolerance=0.05,
    window_size=100,
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
    total_grad_norm=2.0,
    advantage_mean=0.1,
    advantage_std=0.5,
    advantage_min=-0.5,
    advantage_max=1.0,
    advantage_median=0.05,
    advantage_samples=[0.1, 0.2, -0.1, 0.3]
)

# Analysis methods
analysis = forensics.get_comprehensive_analysis()
anomalies = forensics.get_anomalies()
health_summary = forensics.get_health_summary()
forensics.save_analysis("analysis.json")
```

#### PPO Anomaly Detection
```python
from rldk.forensics import scan_logs, diff_checkpoints, audit_environment

# Scan training logs for anomalies
scan_report = scan_logs("path/to/training_logs")

# Compare model checkpoints
diff_report = diff_checkpoints("model_a.pt", "model_b.pt")

# Audit environment for determinism
determinism_card, lock_content = audit_environment("path/to/repo")
```

### **Determinism API**

#### `check`
Multi-replica determinism verification.

```python
from rldk.determinism import check, DeterminismReport

report = check(
    cmd="python train.py --seed 42",
    compare=["loss", "reward_mean", "kl"],
    steps=[100, 200, 300],  # Optional specific steps
    replicas=5,
    device="cuda"  # Optional device specification
)

# Report attributes
print(f"Passed: {report.passed}")
print(f"Culprit: {report.culprit}")
print(f"Fixes: {report.fixes}")
print(f"Replica variance: {report.replica_variance}")
print(f"RNG map: {report.rng_map}")
print(f"Mismatches: {report.mismatches}")
print(f"Dataloader notes: {report.dataloader_notes}")
```

### **Reward Analysis API**

#### `health`
Comprehensive reward model health analysis.

```python
from rldk.reward import health, compare_models, RewardHealthReport

# Analyze reward model health
health_report = health(
    run_data=training_data,
    reference_data=baseline_data,  # Optional
    reward_col="reward_mean",
    step_col="step",
    threshold_drift=0.1,
    threshold_saturation=0.8,
    threshold_calibration=0.7,
    threshold_shortcut=0.6,
    threshold_leakage=0.3
)

# Health report attributes
print(f"Passed: {health_report.passed}")
print(f"Drift detected: {health_report.drift_detected}")
print(f"Saturation issues: {health_report.saturation_issues}")
print(f"Calibration score: {health_report.calibration_score}")
print(f"Shortcut signals: {health_report.shortcut_signals}")
print(f"Label leakage risk: {health_report.label_leakage_risk}")

# Compare two reward models
drift_report = compare_models(
    model_a="path/to/model_a",
    model_b="path/to/model_b",
    prompts=["prompt1", "prompt2", "prompt3"]
)
```

### **Evaluation API**

#### `run`
Statistical evaluation with multiple test suites.

```python
from rldk.evals import run
from rldk.evals.suites import QUICK_SUITE, COMPREHENSIVE_SUITE, SAFETY_SUITE

# Run evaluation suite
eval_result = run(
    run_data=training_data,
    suite="comprehensive",  # "quick", "comprehensive", or "safety"
    seed=42,
    sample_size=200,  # Optional
    output_dir="./eval_results"
)

# Evaluation result attributes
print(f"Overall score: {eval_result.overall_score}")
print(f"Scores: {eval_result.scores}")
print(f"Confidence intervals: {eval_result.confidence_intervals}")
print(f"Effect sizes: {eval_result.effect_sizes}")
print(f"Sample size: {eval_result.sample_size}")
```

### **Run Comparison API**

#### `first_divergence`
Find when and why training runs diverge.

```python
from rldk.diff import first_divergence, DivergenceReport

divergence_report = first_divergence(
    df_a=run_a_data,
    df_b=run_b_data,
    signals=["loss", "reward_mean", "kl"],
    k_consecutive=3,
    window=50,
    tolerance=2.0,
    output_dir="diff_analysis"
)

# Divergence report attributes
print(f"Diverged: {divergence_report.diverged}")
print(f"First step: {divergence_report.first_step}")
print(f"Tripped signals: {divergence_report.tripped_signals}")
print(f"Notes: {divergence_report.notes}")
print(f"Suspected causes: {divergence_report.suspected_causes}")
```

### **Replay API**

#### `replay`
Reproduce training runs with exact same seeds.

```python
from rldk.replay import replay, ReplayReport

replay_report = replay(
    run_path="./original_run",
    training_command="python train.py --seed {seed}",
    metrics_to_compare=["loss", "reward_mean"],
    tolerance=0.01,
    max_steps=1000,
    output_dir="./replay_results",
    device="cuda"
)

# Replay report attributes
print(f"Passed: {replay_report.passed}")
print(f"Original seed: {replay_report.original_seed}")
print(f"Replay seed: {replay_report.replay_seed}")
print(f"Metrics compared: {replay_report.metrics_compared}")
print(f"Mismatches: {replay_report.mismatches}")
print(f"Comparison stats: {replay_report.comparison_stats}")
print(f"Replay duration: {replay_report.replay_duration}")
```

### **Data Ingestion API**

#### `ingest_runs`
Load training data from various sources.

```python
from rldk.ingest import ingest_runs, ingest_runs_to_events
from rldk.adapters import TRLAdapter, OpenRLHFAdapter, WandBAdapter

# Ingest from various sources
df = ingest_runs("path/to/logs", adapter_hint="trl")
df = ingest_runs("wandb://project/run_id", adapter_hint="wandb")
df = ingest_runs("path/to/openrlhf_logs", adapter_hint="openrlhf")

# Convert to normalized events
events = ingest_runs_to_events("path/to/logs", adapter_hint="trl")

# Use specific adapters
trl_adapter = TRLAdapter("path/to/trl_logs")
df = trl_adapter.load()

wandb_adapter = WandBAdapter("wandb://project/run_id")
df = wandb_adapter.load()
```

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

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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
- **Reports & Summaries**: [docs/reports/](docs/reports/)
- **Issues**: [GitHub Issues](https://github.com/your-org/rldk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/rldk/discussions)
- **Examples**: [Examples Directory](examples/)

---

**Ready to ship reproducible RL experiments?** Get started with RLDK today!

```bash
pip install rldk
rldk track "my_first_experiment"
```
