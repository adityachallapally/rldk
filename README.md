# RL Debug Kit (rldk)

A library-first package with a thin CLI for debugging reinforcement learning training runs. Detect divergences, check determinism, and bisect regressions with ease.

## 🚀 Quick Start (60 seconds)

```bash
# Install the package
pip install -e .

# Generate test fixtures
python3 tests/_make_fixtures.py

# Environment audit
rldk env-audit test_artifacts/logs_clean

# Scan logs for PPO anomalies
rldk log-scan test_artifacts/logs_doctored_kl_spike

# Compare checkpoints
rldk diff-ckpt test_artifacts/ckpt_identical/a.pt test_artifacts/ckpt_identical/b.pt

# Compare reward models
rldk reward-drift test_artifacts/reward_drift_demo/rmA test_artifacts/reward_drift_demo/rmB --prompts test_artifacts/reward_drift_demo/prompts.jsonl

# Run comprehensive diagnostics
rldk doctor test_artifacts/logs_clean
```

### Expected Outputs

After running the quickstart commands, you should see:

- **Environment Audit**: `rldk_reports/determinism_card.json` and `rldk.lock`
- **Log Scan**: `rldk_reports/ppo_scan.json` with KL spike detection
- **Checkpoint Diff**: `rldk_reports/ckpt_diff.json` and `rldk_reports/ckpt_diff.png`
- **Reward Drift**: `rldk_reports/reward_drift.json` and `rldk_reports/reward_drift.png`
- **Doctor**: Comprehensive diagnostics combining all analyses

### Forensic Snippets

**KL Spike Detection**: The doctored logs will trigger a KL spike rule around step 800:
```
Anomalies detected:
  - kl_spike: KL spike detected: 5 consecutive updates with KL > 4x median
    Steps: 800 to 805
```

**Checkpoint Comparison**: Identical checkpoints should show high similarity:
```
Parameters compared: 4
Average cosine similarity: 1.0000
```

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
```

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

## 🖥️ CLI Cheatsheet

### Ingest Command
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
├── cli.py              # Typer CLI implementation
├── adapters/           # Log format adapters
│   ├── base.py        # Base adapter class
│   ├── trl.py         # TRL logs adapter
│   ├── openrlhf.py    # OpenRLHF logs adapter
│   └── wandb.py       # Weights & Biases adapter
├── io/                 # I/O utilities
│   ├── schema.py      # Pydantic schemas
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

### Output Formats
- **JSONL**: Standardized training metrics
- **Markdown**: Human-readable divergence and determinism reports
- **CSV**: Detailed divergence events for further analysis

## 🎯 Use Cases

### Training Debugging
- Detect when runs diverge unexpectedly
- Identify the exact step of divergence
- Compare multiple training configurations

### Reproducibility
- Ensure training is deterministic
- Get specific PyTorch fixes for non-deterministic operations
- Validate random seed handling

### Regression Detection
- Find which commit introduced a bug
- Use metric-based or shell-based predicates
- Automate regression testing in CI/CD

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🔮 Roadmap

- [ ] Dashboard for interactive analysis
- [ ] Support for more log formats (RLlib, Stable Baselines3)
- [ ] Advanced divergence detection algorithms
- [ ] Integration with experiment tracking platforms
- [ ] Performance profiling and optimization

---

**Note**: This is a placeholder for a demo GIF showing the CLI in action. The GIF would demonstrate the workflow from ingesting logs to detecting divergences and generating reports.
