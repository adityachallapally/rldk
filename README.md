# RL Debug Kit (rldk)

[![Reward Health Gate](https://github.com/your-org/rldk/workflows/Reward%20Health%20Gate/badge.svg)](https://github.com/your-org/rldk/actions/workflows/health-gate.yml)
<!-- Update the badge URL above to point to your repository -->

**The gold standard companion for LLM + RL** - a tiny, sharp kit that makes RL work reliable, explainable, and reproducible across any trainer or stack. No lock-in. Batteries included for determinism, drift, data, reward health, and evals.

## 🎯 North Star

A tiny, sharp kit that makes RL work reliable, explainable, and reproducible across any trainer or stack. No lock in. Batteries included for determinism, drift, data, reward health, and evals.

## 🚀 Quick Start (60 seconds)

### Option 1: One-Command Demo Experience

Experience the complete RLDK debugging workflow with a single command:

```bash
# Run the complete demo experience
./scripts/demo.sh
```

This interactive demo will:
- Install RLDK and generate test artifacts
- Run all RLDK commands with explanations
- Show real debugging value with KL spikes and checkpoint analysis
- Generate comprehensive reports and visualizations

### Option 2: Docker Demo

Run the demo in a containerized environment:

```bash
# Build and run the demo container
docker build -t rldk-demo .
docker run -it rldk-demo
```

### Option 3: Manual Step-by-Step

If you prefer to run commands individually:

```bash
# Install the package and all dependencies
pip install -e .

# For systems with package conflicts, use:
# pip install -e . --break-system-packages

# Generate test fixtures and training logs
python3 tests/_make_fixtures.py
python3 generate_logs.py

# Compare runs to detect divergence
rldk compare-runs test_artifacts/logs_clean test_artifacts/logs_doctored_kl_spike

# Diff checkpoints to find changes
rldk diff-ckpt test_artifacts/ckpt_identical/a.pt test_artifacts/ckpt_value_head_edit/b.pt

# Environment audit for determinism
rldk env-audit test_artifacts/logs_clean

# Scan logs for PPO anomalies
rldk log-scan test_artifacts/logs_doctored_kl_spike

# Detect reward model drift
rldk reward-drift test_artifacts/reward_drift_demo/rmA test_artifacts/reward_drift_demo/rmB --prompts test_artifacts/reward_drift_demo/prompts.jsonl

# Run reward health analysis (uses default thresholds)
rldk reward-health run --scores test_artifacts/tiny_scores.jsonl --out artifacts/health

# Run comprehensive diagnostics
rldk doctor test_artifacts/logs_doctored_kl_spike
```

### Expected Outputs

After running the quickstart commands, you should see:

- **Compare Runs**: `rldk_reports/divergence_report.json` with first divergence step
- **Checkpoint Diff**: `rldk_reports/ckpt_diff.json` and `rldk_reports/ckpt_diff.png` with top movers
- **Environment Audit**: `rldk_reports/determinism_card.json` and `rldk.lock`
- **Log Scan**: `rldk_reports/ppo_scan.json` with KL spike detection and gradient ratio analysis
- **Reward Drift**: `rldk_reports/reward_drift.json` and `rldk_reports/reward_drift.png`
- **Reward Health**: `artifacts/health/health.json` with pass/fail status and detector results
- **Doctor**: Comprehensive diagnostics combining all analyses

### Forensic Snippets

**KL Spike Detection**: The doctored logs will trigger a KL spike rule around step 800:
```
Anomalies detected:
  - kl_spike: KL spike detected: 5 consecutive updates with KL > 4x median
    Steps: 800 to 805
    Details: KL penalty stayed near 0.001 while KL spiked to 0.15
```

**Value Head Collapse**: Checkpoint comparison reveals value head changes:
```
Top parameter movers:
  - model.value_head.weight: cosine=0.87, L2=0.13
  - model.value_head.bias: cosine=0.92, L2=0.08
Gradient norm ratio fell from 1.2 to 0.03 - potential value head collapse
```

## 🎯 Core Principles

1. **Attach not replace**: Sit beside TRL, OpenRLHF, vLLM, LlamaIndex, Ray
2. **One command to confidence**: Every feature proves or falsifies a claim in a single run
3. **Reproducibility first**: Seeds, environment capture, artifact pinning
4. **Notebook friendly plus CI friendly**: Identical outputs in both

## 🔬 Phase A: Forensics Core (Current Focus)

**Purpose**: Immediate value on any laptop. No keys, no GPU. Make the repo scream real RLHF debugging.

### New Commands

- `rldk compare-runs A B` - Detect first divergence between runs
- `rldk diff-ckpt ckptA ckptB` - Compare checkpoint parameters with top movers
- `rldk env-audit <repo_or_run>` - Audit environment for determinism risks
- `rldk log-scan <run_or_export>` - PPO forensics with KL and gradient health
- `rldk reward-drift modelA modelB --prompts test.jsonl` - Detect reward model drift
- `rldk doctor` - Comprehensive health check

### PPO Forensics Inside log-scan

- **KL schedule health**: Detect spikes and static controller while KL drifts
- **Policy versus value gradient ratio**: Flag collapse or explosion
- **Advantage sanity**: Mean, spread, sign rate and reward hacking pattern

### Tokenizer Parallelism and Determinism Audit

- Report `TOKENIZERS_PARALLELISM` and known torch and cudnn flags
- Scan logs for tokenizer fork warnings
- Emit `rldk.lock` and a Determinism Card JSON

### Reward Model Drift Detector

- Correlation, z scored distance, sign flip rate, slice deltas

## 🔬 Reference Suite

The reference suite provides end-to-end examples of rldk functionality on real datasets with CPU-only execution.

### Quick Start

```bash
# Install dependencies
pip install -e .

# Run CPU smoke tests (generates determinism, drift, and reward health cards)
make reference:cpu_smoke

# Run bisect demonstration (identifies bad commit)
make reference:bisect_demo

# Clean generated files
make clean

# Run tests
make test
```

### Expected Outputs

After running `make reference:cpu_smoke`, you should see these files in `reference/expected/`:

- `determinism_card.json` - Determinism analysis results
- `drift_card.json` - First divergence detection results  
- `reward_card.json` - Reward health analysis results
- `determinism_card.png` - Determinism visualization

After running `make reference:bisect_demo`, you should see:

- `bisect.json` - Bisect results identifying the bad commit

### Tasks

The reference suite includes three tasks:

1. **Summarization** - GPT-2 training on SAMSum dataset (50 steps)
2. **Safety Evaluation** - GPT-2 evaluation on Anthropic HH dataset
3. **Code Generation** - GPT-2 evaluation on MBPP dataset

All tasks use pinned dataset revisions and generate strict JSONL logs with the required schema.

## 🚀 Training Examples

### Basic Training with Profiler
```bash
# Train with simple model and profiler enabled
python train.py --profiler on

# Train with profiler disabled
python train.py --profiler off
```

### Hugging Face Model Training
```bash
# Train with DistilBERT (lightweight, fast)
python examples/train_hf_model.py --model distilbert-base-uncased --profiler on

# Train with BERT (larger, more accurate)
python examples/train_hf_model.py --model bert-base-uncased --profiler on

# Train with custom parameters
python examples/train_hf_model.py \
    --model distilbert-base-uncased \
    --epochs 5 \
    --batch-size 32 \
    --learning-rate 1e-5 \
    --profiler on
```

The Hugging Face example includes:
- **Real transformer models** (DistilBERT, BERT, etc.)
- **Proper tokenization** with attention masks
- **Mean pooling fallback** for models without pooler_output
- **Learning rate scheduling** with warmup
- **Full profiler integration** with all artifacts
- **Error handling** for missing dependencies

## 📚 API Examples

### Ingest Training Runs

```python
from rldk.ingest import ingest_runs

# Load from local files
df = ingest_runs("logs/trl_run/")
df = ingest_runs("logs/openrlhf_run/", adapter_hint="openrlhf")

# Load from Weights & Biases
df = ingest_runs("wandb://entity/project/run_id")

# Auto-detect adapter type
df = ingest_runs("logs/")  # Automatically detects TRL, OpenRLHF, etc.
```

### Detect Divergences

```python
from rldk.diff import first_divergence

# Compare two runs
report = first_divergence(
    df_a, df_b, 
    signals=['kl_mean', 'reward_mean'],
    k_consecutive=3,  # Require 3 consecutive violations
    window=50         # Rolling window for z-score calculation
)

if report.diverged:
    print(f"Divergence at step {report.first_step}")
    print(f"Tripped signals: {report.tripped_signals}")
```

### Check Determinism

```python
# Check if training is deterministic
from rldk.determinism import check_determinism

report = check_determinism(
    cmd="python train.py --cfg config.yml",
    compare=['kl_mean', 'entropy_mean'],
    replicas=3,  # Number of replicas to run
    steps=[50, 100, 150]  # Specific steps to compare
)

if not report.passed:
    print(f"Culprit operation: {report.culprit}")
    for fix in report.fixes:
        print(f"  - {fix}")
```

### Bisect Regressions

```python
from rldk.bisect import bisect_commits

# Find regression using metric predicate
result = bisect_commits(
    good_sha="abc123",
    bad_sha="HEAD",
    cmd="python train.py",
    metric="kl_mean",
    cmp="> 0.2",
    window=100
)

print(f"Regression at commit: {result.culprit_sha}")
print(f"Iterations: {result.iterations}")
```

### Experiment Tracking

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# W&B tracking (default)
config = TrackingConfig(
    experiment_name="my_experiment",
    wandb_project="my-project",
    tags=["ppo", "large-model"]
)
tracker = ExperimentTracker(config)

# Log metrics during training
tracker.log_metric("loss", 0.5)
tracker.log_metric("accuracy", 0.8)

# Finish experiment
summary = tracker.finish_experiment()

# File-only tracking (no W&B)
config = TrackingConfig(
    experiment_name="my_experiment",
    save_to_wandb=False  # Disable W&B
)
tracker = ExperimentTracker(config)
```

## 🖥️ CLI Cheatsheet

### Forensics Commands
```bash
# Compare runs to detect divergence
rldk compare-runs <run_a> <run_b>

# Diff checkpoints to find parameter changes
rldk diff-ckpt <ckpt_a> <ckpt_b>

# Environment audit for determinism
rldk env-audit <repo_or_run>

# Scan logs for PPO anomalies
rldk log-scan <run_or_export>

# Detect reward model drift
rldk reward-drift <model_a> <model_b> --prompts <prompts.jsonl>

# Comprehensive health check
rldk doctor <run_or_export>
```

### Tracking Commands
```bash
# Start experiment tracking with W&B (default)
rldk track my_experiment

# Start tracking with custom W&B project
rldk track my_experiment --wandb-project my-project

# Start tracking with tags and notes
rldk track my_experiment --tags "ppo,large-model" --notes "Testing new architecture"

# Disable W&B and use file logging only
rldk track my_experiment --no-wandb

# Custom output directory
rldk track my_experiment --output-dir ./my_runs
```

### Legacy Commands (Still Supported)
```bash
# Basic usage
rldk ingest <path_or_wandb_uri>

# Specify adapter type
rldk ingest logs/ --adapter trl

# Save to custom output file
rldk ingest logs/ --output my_metrics.jsonl

# Examples
rldk ingest runs_fixtures/clean_ppo.jsonl
rldk ingest wandb://username/project/abc123
rldk ingest logs/trl_run/ --adapter trl
```

### Diff Command
```bash
# Basic divergence detection
rldk diff --a <run_a> --b <run_b> --signals <metrics>

# Customize detection parameters
rldk diff --a clean.jsonl --b spike.jsonl \
    --signals kl_mean,reward_mean \
    --k 3 \
    --window 50 \
    --output-dir my_analysis

# Examples
rldk diff --a clean.jsonl --b spike.jsonl --signals kl_mean
rldk diff --a wandb://u/p/run1 --b wandb://u/p/run2 --signals reward_mean,entropy_mean
```

### Determinism Command
```bash
# Check determinism with metric comparison
rldk check-determinism --cmd "python train.py" --compare <metrics>

# Customize comparison
rldk check-determinism \
    --cmd "python train.py --cfg config.yml" \
    --compare kl_mean,entropy_mean \
    --stride 25 \
    --steps 50,100,150 \
    --device cuda

# Examples
rldk check-determinism --cmd "python train.py" --compare kl_mean --stride 50
rldk check-determinism --cmd "bash train.sh" --compare reward_mean,loss --stride 25
rldk check-determinism --cmd "python train.py" --compare kl_mean --steps 50,100,150
```

### Bisect Command
```bash
# Metric-based bisect
rldk bisect --good <sha> --bad <sha> --cmd <command> --metric <metric> --cmp <operator>

# Shell predicate bisect
rldk bisect --good <sha> --bad <sha> --shell-predicate <command>

# Examples
rldk bisect --good abc123 --bad HEAD --cmd "python train.py" --metric kl_mean --cmp "> 0.2"
rldk bisect --good v1.0 --bad HEAD --shell-predicate "python test.py"
```

## 🏗️ Architecture

### Package Structure
```
src/rldk/
├── __init__.py          # Main package exports
├── cli_forensics.py     # New forensics CLI commands
├── cli_reward.py        # Reward-specific CLI commands
├── artifacts/           # Artifact analysis
│   ├── ckpt_diff.py    # Checkpoint comparison
│   ├── env_audit.py    # Environment determinism audit
│   └── log_scan.py     # Generic log scanning
├── forensics/           # PPO-specific forensics
│   └── ppo_scan.py     # PPO anomaly detection
├── reward/              # Reward model analysis
│   └── drift.py        # Reward drift detection
├── adapters/           # Log format adapters
│   ├── base.py        # Base adapter class
│   ├── trl.py         # TRL logs adapter
│   ├── openrlhf.py    # OpenRLHF logs adapter
│   └── wandb.py       # Weights & Biases adapter
├── io/                 # I/O utilities
│   ├── schemas.py     # Pydantic schemas
│   ├── readers.py     # JSONL readers/writers
│   └── writers.py     # Report generators
├── ingest/            # Main ingest functionality
├── diff/              # Divergence detection
├── determinism/       # Determinism checking
└── bisect/            # Git bisect wrapper
```

### Data Flow
1. **Ingest**: Adapters convert various log formats → standardized DataFrame
2. **Diff**: Rolling z-score analysis with k-consecutive rule → DivergenceReport
3. **Determinism**: Run command twice with deterministic flags → DeterminismReport
4. **Bisect**: Git bisect with metric/shell predicates → BisectResult
5. **Forensics**: PPO-specific anomaly detection → PPOReport
6. **Reward**: Drift detection with correlation analysis → DriftReport

## 📦 Installation & Dependencies

### Core Dependencies
The package includes all necessary dependencies for the profiler system and monitoring dashboard:

- **Core ML**: `torch`, `transformers`, `datasets`, `scikit-learn`
- **Data Processing**: `numpy`, `pandas`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Monitoring**: `streamlit` (for dashboard), `wandb`
- **Utilities**: `typer`, `pydantic`, `rich`, `pyyaml`

### Installation Options

```bash
# Standard installation
pip install -e .

# For systems with package conflicts (e.g., some Linux distributions)
pip install -e . --break-system-packages

# Development installation with additional tools
pip install -e ".[dev]"
```

### Troubleshooting Installation Issues

If you encounter dependency conflicts:

1. **Use `--break-system-packages` flag** for systems with strict package management
2. **Install optional dependencies separately** if needed:
   ```bash
   pip install streamlit --break-system-packages
   pip install plotly --break-system-packages
   ```
3. **Use virtual environment** for isolated installation:
   ```bash
   python -m venv rldk_env
   source rldk_env/bin/activate  # On Windows: rldk_env\Scripts\activate
   pip install -e .
   ```

### Verify Installation

Test that all dependencies are properly installed:

```bash
# Check all optional dependencies
python test_dependencies.py

# Test basic functionality
python train.py --profiler on

# Test monitoring dashboard (requires streamlit/plotly)
python monitor/app.py
```

The dependency checker will provide helpful error messages if any optional dependencies are missing, along with specific installation commands.

## 🔧 Development

### Setup
```bash
# Clone and install in development mode
git clone <repo>
cd rldk
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Generate fixtures
python scripts/make_fixtures.py

# Test fixtures
python runs_fixtures/test_spike_detection.py
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=rldk --cov-report=html

# Run specific test file
pytest tests/test_diff.py -v
```

## 📊 Supported Formats

### Input Sources
- **TRL logs**: JSONL files with TRL-specific fields
- **OpenRLHF logs**: JSONL files with OpenRLHF-specific fields  
- **Weights & Biases**: `wandb://entity/project/run_id` URIs
- **Generic JSONL**: Standardized metrics format
- **Checkpoints**: PyTorch .pt files, SafeTensors, HuggingFace models

### Output Formats
- **JSONL**: Standardized training metrics
- **Markdown**: Human-readable divergence and determinism reports
- **CSV**: Detailed divergence events for further analysis
- **PNG**: Small plots for reports and cards
- **JSON**: Structured reports for CI/CD integration

## 🔗 Weights & Biases Integration

RLDK integrates seamlessly with Weights & Biases (W&B) for experiment tracking and logging. W&B is enabled by default for all tracking operations.

### Default W&B Configuration
- **Project**: `rldk-experiments` (default)
- **Logging**: Automatic experiment tracking with metadata
- **Fallback**: File logging when W&B is unavailable

### Using W&B (Default)
```bash
# W&B is enabled by default
rldk track my_experiment

# Custom W&B project
rldk track my_experiment --wandb-project my-custom-project

# With tags and notes
rldk track my_experiment --tags "ppo,large-model" --notes "Testing new architecture"
```

### Disabling W&B
Use the `--no-wandb` flag to disable W&B logging and use file logging only:

```bash
# Disable W&B, use file logging only
rldk track my_experiment --no-wandb

# Other commands also support --no-wandb
rldk replay --run my_run --command "python train.py" --no-wandb
rldk eval --run my_run --suite quick --no-wandb
```

### W&B Integration Features
- **Automatic project creation**: Uses `rldk-experiments` by default
- **Experiment metadata**: Tracks environment, git state, seeds, and datasets
- **Graceful fallback**: Automatically falls back to file logging if W&B is not available
- **No breaking changes**: Existing functionality continues to work

### W&B Requirements
- Install W&B: `pip install wandb`
- Login: `wandb login` (first time only)
- No additional configuration required

## 🎯 Use Cases

### Training Debugging
- Detect when runs diverge unexpectedly
- Identify the exact step of divergence
- Compare multiple training configurations
- Find parameter changes in checkpoints

### Reproducibility
- Ensure training is deterministic
- Get specific PyTorch fixes for non-deterministic operations
- Validate random seed handling
- Capture environment state for replay

### Regression Detection
- Find which commit introduced a bug
- Use metric-based or shell-based predicates
- Automate regression testing in CI/CD

### CI/CD Integration
- **Reward Health Gate**: Automatically fail CI when reward health checks fail
- Copy `.github/workflows/health-gate.yml` to your repository
- Update the badge URL in README.md to point to your repository
- The workflow tests both Python 3.10 and 3.11 with synthetic health data
- Exit codes: 0 (passed), 3 (failed) - compatible with CI systems

### PPO Forensics
- Detect KL spikes and controller failures
- Monitor gradient ratio health
- Track advantage statistics
- Identify reward hacking patterns

### Reward Model Health
- Detect drift between model versions
- Analyze correlation changes
- Identify problematic slices
- Validate reward model consistency
- **Default thresholds**: Opinionated defaults for out-of-the-box pass/fail behavior - see [Health Thresholds Guide](docs/health_thresholds.md)

## 🔮 Roadmap

### Phase A: Forensics Core (Current)
- [x] PPO forensics with KL and gradient health
- [x] Checkpoint comparison with top movers
- [x] Environment determinism audit
- [x] Reward drift detection
- [x] Comprehensive health diagnostics

### Phase B: Trust Cards and Normalized Events
- [ ] Normalized event schema
- [ ] Cards as first-class artifacts
- [ ] CLI for generating trust cards
- [ ] Stable filenames and field reference

### Phase C: Dataset Lineage and Split Integrity
- [ ] Lineage auditor for content addressing
- [ ] Simple leak and dup detector
- [ ] CLI for data audit
- [ ] Stable hashes across machines

### Phase D: Minimal Repro Bundle and Bisect Polish
- [ ] Repro packer for shareable proofs
- [ ] Bisect upgrades with caching
- [ ] One-line repro scripts
- [ ] Tolerance-based verification

### Phase E: Quick Eval Suite
- [ ] Seedable micro tasks
- [ ] KL to reference tracking
- [ ] Confidence intervals
- [ ] CPU-friendly evaluation

### Phase F: Adapter Protocol and Support Matrix
- [ ] Adapter protocol for cross-stack support
- [ ] TRL and OpenRLHF adapters
- [ ] Support matrix autogen
- [ ] Contributor guide for new adapters

### Phase G: Safety and Reward Side Effects
- [ ] Side effect scanners
- [ ] Toxicity and jailbreak detection
- [ ] Regurgitation proxy
- [ ] Configurable thresholds

### Phase H: Visibility and Trust Flywheel
- [ ] PyPI publish with artifacts
- [ ] Real bug demonstration
- [ ] Determinism Checklist PDF
- [ ] Weekly bug posts

### Optional Phase I: Test Time Autopilot
- [ ] Best of N strategies
- [ ] Self-consistency methods
- [ ] Budget-aware inference
- [ ] Accuracy vs cost optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 🔧 Troubleshooting

### Common Issues

**PyTorch Import Error**: If you see `ImportError: No module named 'torch'`:
```bash
# Install PyTorch first
pip install torch
# Then install RLDK
pip install -e .
```

**Permission Denied on Demo Script**: If `./scripts/demo.sh` fails:
```bash
chmod +x scripts/demo.sh
```

**Missing Test Artifacts**: If demo commands fail with missing files:
```bash
# Regenerate all test artifacts
python3 tests/_make_fixtures.py
python3 generate_logs.py
```

**Docker Build Fails**: If Docker build fails due to missing files:
```bash
# Ensure all required files exist
ls -la pyproject.toml tests/_make_fixtures.py generate_logs.py
```

**RLDK Command Not Found**: If the demo can't find the `rldk` command:
```bash
# Check if RLDK is installed
pip list | grep rldk

# Reinstall if needed
pip install -e . --break-system-packages

# Check common installation locations
which rldk || find /usr/local/bin /usr/bin ~/.local/bin -name rldk 2>/dev/null
```

### Demo Troubleshooting

**Demo Script Hangs**: The demo script waits for user input. Press Enter to continue.

**No Reports Generated**: Check that RLDK CLI commands are working:
```bash
rldk --help
```

**KL Spike Not Detected**: Ensure the doctored logs contain the spike:
```bash
# Check the KL values around step 800
tail -n 50 test_artifacts/logs_doctored_kl_spike/training.jsonl | grep -E '"step": 8[0-9][0-9]'
```

### Getting Help

- Check the [Issues](https://github.com/your-org/rldk/issues) page for known problems
- Review the [ANOMALY_DETECTION_GUIDE.md](ANOMALY_DETECTION_GUIDE.md) for detailed usage
- Open a new issue with reproduction steps if you encounter bugs

## 📄 License

MIT License - see LICENSE file for details.

---

**Note**: This is a placeholder for a demo GIF showing the CLI in action. The GIF would demonstrate the workflow from ingesting logs to detecting divergences and generating reports.
