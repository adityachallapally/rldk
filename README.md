# RL Debug Kit (rldk)

A library-first package with a thin CLI for debugging reinforcement learning training runs. Detect divergences, check determinism, and bisect regressions with ease.

## 🚀 Quick Start

```bash
# Install the package
pip install -e .

# Test the CLI
rldk version

# Ingest training logs
rldk ingest runs_fixtures/clean_ppo.jsonl

# Compare two runs for divergence
rldk diff --a runs_fixtures/clean_ppo.jsonl --b runs_fixtures/kl_spike.jsonl --signals kl_mean,reward_mean

# Check if a training command is deterministic
rldk check-determinism --cmd "python train.py" --compare kl_mean,entropy_mean

# Find regression using git bisect
rldk bisect --good abc123 --bad HEAD --cmd "python train.py" --metric kl_mean --cmp "> 0.2"
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
from rldk.determinism import check_determinism

# Check if training is deterministic
report = check_determinism(
    cmd="python train.py --cfg config.yml",
    compare=['kl_mean', 'entropy_mean'],
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
    --steps 50,100,150 \
    --device cuda

# Examples
rldk check-determinism --cmd "python train.py" --compare kl_mean
rldk check-determinism --cmd "bash train.sh" --compare reward_mean,loss --stride 25
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
