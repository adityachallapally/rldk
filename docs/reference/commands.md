# CLI Commands Reference

This page provides comprehensive documentation for all RLDK command-line interface commands.

## Main Commands

### `rldk --help`

Show the main help message with all available commands.

```bash
rldk --help
```

### `rldk version`

Show the current RLDK version.

```bash
rldk version
```

## Experiment Tracking

### `rldk track`

Start tracking an experiment with W&B (default) or file logging.

```bash
rldk track "my_experiment" [OPTIONS]
```

**Arguments:**
- `experiment_name`: Name of the experiment to track

**Options:**
- `--output-dir`, `-o`: Output directory for tracking data (default: `./runs`)
- `--no-wandb`: Disable W&B logging and use file logging only
- `--wandb-project`: W&B project name (default: `rldk-experiments`)
- `--tags`: Comma-separated list of tags
- `--notes`: Additional notes for the experiment
- `--interactive`, `-i`: Keep tracker running in interactive mode

**Examples:**
```bash
# Basic experiment tracking
rldk track "ppo_experiment_v1"

# With custom output directory and tags
rldk track "my_experiment" --output-dir ./experiments --tags "ppo,debug"

# Interactive mode for manual logging
rldk track "interactive_experiment" --interactive

# File logging only (no W&B)
rldk track "local_experiment" --no-wandb
```

## Forensics Commands

### `rldk forensics compare-runs`

Compare two training runs and identify divergences.

```bash
rldk forensics compare-runs RUN_A RUN_B
```

**Arguments:**
- `RUN_A`: Path to first run directory
- `RUN_B`: Path to second run directory

**Examples:**
```bash
rldk forensics compare-runs ./runs/experiment_1 ./runs/experiment_2
```

### `rldk forensics diff-ckpt`

Compare two model checkpoints and identify parameter differences.

```bash
rldk forensics diff-ckpt CKPT_A CKPT_B
```

**Arguments:**
- `CKPT_A`: Path to first checkpoint
- `CKPT_B`: Path to second checkpoint

**Examples:**
```bash
rldk forensics diff-ckpt checkpoint_100.pt checkpoint_200.pt
```

### `rldk forensics env-audit`

Audit environment for determinism and reproducibility.

```bash
rldk forensics env-audit REPO_OR_RUN
```

**Arguments:**
- `REPO_OR_RUN`: Path to repository or run directory

**Examples:**
```bash
rldk forensics env-audit ./my_project
rldk forensics env-audit ./runs/experiment_1
```

### `rldk forensics log-scan`

Scan training logs for PPO anomalies and issues.

```bash
rldk forensics log-scan RUN_OR_EXPORT
```

**Arguments:**
- `RUN_OR_EXPORT`: Path to run or export directory

**Examples:**
```bash
rldk forensics log-scan ./runs/experiment_1
rldk forensics log-scan ./exports/training_logs
```

### `rldk forensics doctor`

Run comprehensive diagnostics on a training run or repository.

```bash
rldk forensics doctor RUN_OR_REPO
```

**Arguments:**
- `RUN_OR_REPO`: Path to run or repository directory

**Examples:**
```bash
rldk forensics doctor ./runs/experiment_1
rldk forensics doctor ./my_project
```

## Reward Analysis Commands

### `rldk reward reward-drift`

Compare two reward models and detect drift.

```bash
rldk reward reward-drift MODEL_A MODEL_B --prompts PROMPTS_FILE
rldk reward reward-drift --scores-a scores_a.jsonl --scores-b scores_b.jsonl
```

**Arguments:**
- `MODEL_A`: Path to first reward model directory (required when using prompt mode)
- `MODEL_B`: Path to second reward model directory (required when using prompt mode)

**Options:**
- `--prompts`, `-p`: Path to prompts JSONL file (required when comparing model directories)
- `--scores-a`: Path to the first JSONL score file (each record must contain `prompt` and `score`)
- `--scores-b`: Path to the second JSONL score file

**Examples:**
```bash
rldk reward reward-drift ./models/reward_v1 ./models/reward_v2 --prompts prompts.jsonl
rldk reward reward-drift --scores-a outputs/baseline_scores.jsonl --scores-b outputs/new_model_scores.jsonl
```

### `rldk reward-health`

Normalize reward metrics and generate a health report.

```bash
rldk reward-health --run RUN_SOURCE [OPTIONS]
```

**Options:**
- `--run`, `-r`: Path to training run data (JSONL stream, metrics table, or logs directory) (required)
- `--reference`, `-ref`: Optional reference run for drift comparison
- `--output-dir`, `-o`: Output directory for reports (default: `reward_analysis`)
- `--preset`: Field map preset for common trainer schemas
- `--field-map`: JSON object mapping source columns to canonical training metrics
- `--reward-col`: Column name for reward values (default: `reward_mean`)
- `--step-col`: Column name for training steps (default: `step`)
- `--threshold-drift`: P-value threshold for drift detection (default: `0.1`)
- `--threshold-saturation`: Threshold for saturation detection (default: `0.8`)
- `--threshold-calibration`: Threshold for calibration quality (default: `0.7`)
- `--threshold-shortcut`: Threshold for shortcut signal detection (default: `0.6`)
- `--threshold-leakage`: Threshold for label leakage risk (default: `0.3`)
- `--gold`: Optional path to trusted gold metrics for overoptimization checks
- `--gold-col`: Column name containing trusted gold scores (in `--run` or `--gold` dataset)
- `--overopt-window`: Window size for early/late proxy vs gold comparison (default: `100`)
- `--overopt-delta-threshold`: Minimum proxy-minus-gold delta to raise an alert (default: `0.2`)
- `--overopt-min-samples`: Minimum paired samples required for overoptimization detector (default: `100`)
- `--gate`: Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)

**Examples:**
```bash
# Analyze a JSONL event stream with custom field mappings
rldk reward-health --run metrics.jsonl --field-map '{"reward": "reward_mean", "kl": "kl_mean"}'

# Use a CSV table that already contains normalized columns
rldk reward-health --run ./reports/training_metrics.csv

# Run with a reference directory and CI gating
rldk reward-health --run ./runs/current --reference ./runs/baseline --gate
```

### `rldk reward reward-health gate`

Gate CI based on health.json results (exit codes: 0=pass, 3=fail).

```bash
rldk reward reward-health gate --from HEALTH_JSON_FILE
```

**Options:**
- `--from`: Path to health.json file (required)

**Examples:**
```bash
rldk reward reward-health gate --from ./reports/health.json
```

## Evaluation Commands

### `rldk evals evaluate`

Run evaluation suite on normalized training runs or evaluation datasets.

```bash
rldk evals evaluate INPUT_PATH [OPTIONS]
```

**Arguments:**
- `INPUT_PATH`: Path to a training run directory, metrics table, or evaluation dataset

**Options:**
- `--suite`, `-s`: Evaluation suite to run (`quick`/`comprehensive`/`safety`/`training_metrics`) (default: `quick`)
- `--output`, `-o`: Path to output JSON file
- `--output-column`: Column name containing model outputs (default: `output`)
- `--events-column`: Column name containing event logs (default: `events`)
- `--min-samples`: Minimum samples required for evaluation (default: `10`)
- `--timeout`: Timeout in seconds for evaluation (default: `300`)
- `--verbose`, `-v`: Enable verbose logging
- `--preset`: Field map preset to normalize training metrics before evaluation
- `--field-map`: JSON object mapping source columns to canonical training metrics

**Examples:**
```bash
# Quick evaluation on a run directory
rldk evals evaluate /path/to/run --suite quick --preset trl

# Training metrics suite with a metrics table
rldk evals evaluate metrics.csv --suite training_metrics --field-map '{"progress":"step"}'

# Comprehensive evaluation with custom output
rldk evals evaluate data.jsonl --suite comprehensive --output results.json

# With custom columns and timeout
rldk evals evaluate data.jsonl --output-column "model_output" --events-column "training_events" --timeout 600
```

### `rldk evals list-suites`

List available evaluation suites.

```bash
rldk evals list-suites
```

### `rldk evals validate-data`

Validate JSONL file structure and data.

```bash
rldk evals validate-data INPUT_FILE [OPTIONS]
```

**Arguments:**
- `INPUT_FILE`: Path to JSONL input file to validate

**Options:**
- `--output-column`: Column name containing model outputs (default: `output`)
- `--events-column`: Column name containing event logs (default: `events`)

**Examples:**
```bash
rldk evals validate-data data.jsonl
rldk evals validate-data data.jsonl --output-column "model_output"
```

## Data Ingestion Commands

### `rldk ingest`

Ingest training runs from various sources.

```bash
rldk ingest RUNS [OPTIONS]
```

**Arguments:**
- `RUNS`: Path to runs directory, file, or wandb:// URI

**Options:**
- `--adapter`, `-a`: Adapter type (`trl`, `openrlhf`, `wandb`, `custom_jsonl`)
- `--output`, `-o`: Output file path (default: `metrics.jsonl`)
- `--validate/--no-validate`: Validate input data before processing (default: `True`)
- `--verbose`, `-v`: Enable verbose output

**Examples:**
```bash
# Ingest from local directory
rldk ingest ./logs --adapter trl

# Ingest from WandB
rldk ingest wandb://entity/project/run_id --adapter wandb

# Ingest JSONL file
rldk ingest data.jsonl --adapter custom_jsonl --output results.jsonl
```

## Run Comparison Commands

### `rldk diff`

Find first divergence between two training runs.

```bash
rldk diff --a RUN_A --b RUN_B --signals SIGNALS [OPTIONS]
```

**Options:**
- `--a`, `-a`: Path or wandb:// URI for run A (required)
- `--b`, `-b`: Path or wandb:// URI for run B (required)
- `--signals`, `-s`: Metrics to monitor for divergence (required)
- `--tolerance`, `-t`: Z-score threshold for violation detection (default: `2.0`)
- `--k`, `-k`: Number of consecutive violations required (default: `3`)
- `--window`, `-w`: Rolling window size for z-score calculation (default: `50`)
- `--output-dir`, `-o`: Output directory for reports (default: `diff_analysis`)

**Examples:**
```bash
# Basic divergence detection
rldk diff --a ./runs/run_1 --b ./runs/run_2 --signals loss,reward_mean

# With custom parameters
rldk diff --a ./runs/run_1 --b ./runs/run_2 --signals loss,reward_mean,kl --tolerance 1.5 --k 5 --window 100
```

## Determinism Commands

### `rldk check-determinism`

Check if a training command is deterministic.

```bash
rldk check-determinism [OPTIONS]
```

**Options:**
- `--cmd`, `-c`: Command to run for testing
- `--compare`, `-m`: Metrics to compare (comma-separated)
- `--steps`, `-s`: Specific steps to compare (comma-separated)
- `--stride`: Step interval for comparison if steps not specified (default: `50`)
- `--replicas`, `-r`: Number of replicas to run (default: `5`)
- `--runs`: Number of runs for determinism check (alias for replicas)
- `--tolerance`, `-t`: Tolerance for metric differences (default: `0.01`)
- `--device`, `-d`: Device to use (auto-detected if None)
- `--output-dir`, `-o`: Output directory for reports (default: `determinism_analysis`)
- `--gate`: Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)

**Examples:**
```bash
# Basic determinism check
rldk check-determinism --cmd "python train.py --seed 42" --compare loss,reward_mean

# With multiple replicas and specific steps
rldk check-determinism --cmd "python train.py --seed 42" --compare loss --replicas 10 --steps 100,200,300

# CI gate mode
rldk check-determinism --gate
```

## Seed Management Commands

### `rldk seed`

Manage global seed for reproducible experiments.

```bash
rldk seed [OPTIONS]
```

**Options:**
- `--seed`, `-s`: Seed value to set
- `--show`: Show current seed state
- `--deterministic/--non-deterministic`: Enable deterministic behavior (default: `True`)
- `--env`: Set environment variables for reproducibility
- `--validate`: Validate seed consistency

**Examples:**
```bash
# Set seed to 42
rldk seed --seed 42

# Show current seed state
rldk seed --show

# Set seed with environment variables
rldk seed --seed 1337 --env

# Validate seed consistency
rldk seed --validate

# Set non-deterministic seed
rldk seed --seed 42 --non-deterministic
```

## Replay Commands

### `rldk replay`

Replay a training run with the original seed and verify reproducibility.

```bash
rldk replay --run RUN_PATH --command COMMAND --metrics METRICS [OPTIONS]
```

**Options:**
- `--run`, `-r`: Path to original training run data (required)
- `--command`, `-c`: Training command to replay (should accept --seed) (required)
- `--metrics`, `-m`: Metrics to compare (comma-separated) (required)
- `--tolerance`, `-t`: Tolerance for metric differences (relative) (default: `0.01`)
- `--max-steps`, `-s`: Maximum steps to replay
- `--output-dir`, `-o`: Output directory for results (default: `replay_results`)
- `--device`, `-d`: Device to use (auto-detected if None)
- `--no-wandb`: Disable W&B logging and use file logging only

**Examples:**
```bash
# Basic replay
rldk replay --run ./runs/experiment_1 --command "python train.py --seed {seed}" --metrics loss,reward_mean

# With custom tolerance and max steps
rldk replay --run ./runs/experiment_1 --command "python train.py --seed {seed}" --metrics loss --tolerance 0.001 --max-steps 500

# File logging only
rldk replay --run ./runs/experiment_1 --command "python train.py --seed {seed}" --metrics loss --no-wandb
```

## Evaluation Commands

### `rldk eval`

Run evaluation suite with statistical analysis.

```bash
rldk eval --run RUN_PATH [OPTIONS]
```

**Options:**
- `--run`, `-r`: Path to training run data (required)
- `--suite`, `-s`: Evaluation suite to run (default: `quick`)
- `--output-dir`, `-o`: Output directory for results (default: `eval_results`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--sample-size`: Number of samples to evaluate
- `--no-wandb`: Disable W&B logging and use file logging only

**Examples:**
```bash
# Quick evaluation
rldk eval --run ./runs/experiment_1

# Comprehensive evaluation with custom sample size
rldk eval --run ./runs/experiment_1 --suite comprehensive --sample-size 1000

# With custom seed and output directory
rldk eval --run ./runs/experiment_1 --seed 123 --output-dir ./custom_eval
```

## Git Bisect Commands

### `rldk bisect`

Find regression using git bisect.

```bash
rldk bisect --good GOOD_COMMIT --bad BAD_COMMIT [OPTIONS]
```

**Options:**
- `--good`, `-g`: Known good commit SHA (required)
- `--bad`, `-b`: Known bad commit SHA (default: `HEAD`)
- `--cmd`, `-c`: Command to run for testing
- `--metric`, `-m`: Metric name to monitor
- `--cmp`: Comparison operator (e.g., '> 0.2')
- `--window`, `-w`: Window size for metric statistics (default: `100`)
- `--shell-predicate`: Shell command that returns non-zero on failure

**Examples:**
```bash
# Basic bisect with command and metric
rldk bisect --good abc123 --bad def456 --cmd "python train.py" --metric "reward_mean" --cmp "> 0.8"

# With shell predicate
rldk bisect --good abc123 --bad def456 --shell-predicate "python -c 'import sys; sys.exit(0 if check_condition() else 1)'"
```

## Card Generation Commands

### `rldk card`

Generate trust cards for RL training runs.

```bash
rldk card CARD_TYPE RUN_A [RUN_B] [OPTIONS]
```

**Arguments:**
- `CARD_TYPE`: Type of card to generate (`determinism`, `drift`, `reward`)
- `RUN_A`: Path to the primary run directory or metrics file (required)
- `RUN_B`: Path to the comparison run directory or metrics file (drift cards only)

**Options:**
- `--output-dir`, `-o`: Output directory for cards
- `--preset`: Field map preset for common trainer outputs (e.g. `trl`)
- `--field-map`: JSON object mapping source columns to canonical training metrics

**Examples:**
```bash
# Generate determinism card
rldk card determinism ./runs/experiment_1

# Generate drift card comparing two runs
rldk card drift ./runs/experiment_1 ./runs/experiment_2

# Generate reward card from a JSONL metrics stream using the TRL preset
rldk card reward ./logs/trl_stream.jsonl --preset trl

# Generate reward card
rldk card reward ./runs/experiment_1
```

## Format Information Commands

### `rldk format-info`

Show data format information for adapters.

```bash
rldk format-info [OPTIONS]
```

**Options:**
- `--adapter`, `-a`: Show format info for specific adapter
- `--examples`: Show example data

**Examples:**
```bash
# Show info for all adapters
rldk format-info

# Show info for specific adapter
rldk format-info --adapter trl

# Show examples
rldk format-info --adapter trl --examples
```

### `rldk validate-format`

Validate data format and suggest appropriate adapter.

```bash
rldk validate-format SOURCE [OPTIONS]
```

**Arguments:**
- `SOURCE`: Path to data source to validate

**Options:**
- `--adapter`, `-a`: Adapter type to test
- `--verbose`, `-v`: Show detailed analysis

**Examples:**
```bash
# Validate data format
rldk validate-format ./data/logs

# Test with specific adapter
rldk validate-format ./data/logs --adapter trl

# Verbose analysis
rldk validate-format ./data/logs --verbose
```

## Legacy Commands

These commands are maintained for backward compatibility but may be deprecated in future versions.

### `rldk compare-runs`

Alias for `rldk forensics compare-runs`.

### `rldk diff-ckpt`

Alias for `rldk forensics diff-ckpt`.

### `rldk env-audit`

Alias for `rldk forensics env-audit`.

### `rldk log-scan`

Alias for `rldk forensics log-scan`.

### `rldk doctor`

Alias for `rldk forensics doctor`.

### `rldk reward-drift`

Alias for `rldk reward reward-drift`.

## Exit Codes

RLDK commands use standardized exit codes:

- `0`: Success
- `1`: General error
- `2`: Critical error (CI gate mode: fail)
- `3`: Configuration error

In CI gate mode:
- `0`: Pass
- `1`: Warning
- `2`: Fail
- `3`: Critical failure

## Environment Variables

RLDK respects the following environment variables:

- `RLDK_OUTPUT_DIR`: Default output directory
- `RLDK_LOG_LEVEL`: Logging level
- `RLDK_CACHE_DIR`: Cache directory
- `WANDB_PROJECT`: Default W&B project name
- `WANDB_ENTITY`: Default W&B entity name
- `WANDB_API_KEY`: W&B API key for authentication

## Configuration Files

RLDK looks for configuration in the following order:

1. Command-line options
2. Environment variables
3. `~/.rldk/config.yaml`
4. `./rldk.yaml`
5. Default values

Example configuration file (`~/.rldk/config.yaml`):

```yaml
defaults:
  output_dir: "./runs"
  log_level: "INFO"
  wandb_project: "my-rldk-experiments"

forensics:
  kl_target: 0.1
  enable_advantage_statistics: true

determinism:
  default_replicas: 5
  default_tolerance: 0.01

evaluation:
  default_suite: "quick"
  default_sample_size: 200
```