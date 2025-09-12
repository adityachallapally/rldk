# CLI Commands Reference

This page provides comprehensive documentation for all RLDK CLI commands.

## Main Commands

### `rldk demo`

Generate synthetic PPO and GRPO training runs for offline demos and testing.

```bash
rldk demo [OPTIONS]
```

**Options:**
- `--out, -o TEXT`: Output directory for demo data (default: ./tracking_demo_output)
- `--seed, -s INTEGER`: Random seed for reproducible demo data (default: 1337)
- `--steps INTEGER`: Number of training steps to simulate (default: 200)
- `--variants INTEGER`: Number of seeded variants to generate (default: 3)

**Examples:**
```bash
# Generate basic demo data
rldk demo

# Generate demo data with custom parameters
rldk demo --out ./my_demo --seed 42 --steps 100 --variants 5

# Generate demo data for testing
rldk demo --steps 50 --variants 2
```

**Output:**
- `ppo_run_*.jsonl`: PPO training logs
- `grpo_run_*.jsonl`: GRPO training logs  
- `tests/fixtures/minirun/run.jsonl`: Test fixture data

### `rldk ingest`

Ingest training runs from various sources.

```bash
rldk ingest RUNS [OPTIONS]
```

**Arguments:**
- `RUNS`: Path to runs directory, file, or wandb:// URI

**Options:**
- `--adapter, -a TEXT`: Adapter type (trl, openrlhf, wandb, custom_jsonl, demo_jsonl)
- `--output, -o TEXT`: Output file path (default: metrics.jsonl)
- `--validate/--no-validate`: Validate input data before processing (default: True)
- `--verbose, -v`: Enable verbose output

**Examples:**
```bash
# Ingest TRL logs
rldk ingest /path/to/logs --adapter trl

# Ingest WandB run
rldk ingest wandb://entity/project/run_id --adapter wandb

# Ingest demo data
rldk ingest ./demo_data --adapter demo_jsonl --output results.jsonl
```

### `rldk diff`

Find first divergence between two training runs.

```bash
rldk diff --a A --b B --signals SIGNALS [OPTIONS]
```

**Options:**
- `--a, -a TEXT`: Path or wandb:// URI for run A
- `--b, -b TEXT`: Path or wandb:// URI for run B
- `--signals, -s TEXT`: Metrics to monitor for divergence
- `--tolerance, -t FLOAT`: Z-score threshold for violation detection (default: 2.0)
- `--k, -k INTEGER`: Number of consecutive violations required (default: 3)
- `--window, -w INTEGER`: Rolling window size for z-score calculation (default: 50)
- `--output-dir, -o TEXT`: Output directory for reports (default: diff_analysis)

**Examples:**
```bash
# Compare two runs
rldk diff --a run_a --b run_b --signals loss,reward_mean

# Compare with custom parameters
rldk diff --a run_a --b run_b --signals loss,reward_mean,kl --tolerance 1.5 --k 5
```

### `rldk check-determinism`

Check if a training command is deterministic.

```bash
rldk check-determinism [OPTIONS]
```

**Options:**
- `--cmd, -c TEXT`: Command to run for testing
- `--compare, -m TEXT`: Metrics to compare (comma-separated)
- `--steps, -s TEXT`: Specific steps to compare (comma-separated)
- `--stride INTEGER`: Step interval for comparison if steps not specified (default: 50)
- `--replicas, -r INTEGER`: Number of replicas to run (default: 5)
- `--runs INTEGER`: Number of runs for determinism check (alias for replicas)
- `--tolerance, -t FLOAT`: Tolerance for metric differences (default: 0.01)
- `--device, -d TEXT`: Device to use (auto-detected if None)
- `--output-dir, -o TEXT`: Output directory for reports (default: determinism_analysis)
- `--gate`: Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)

**Examples:**
```bash
# Check determinism of training script
rldk check-determinism --cmd "python train.py --seed 42" --compare loss,reward_mean

# Check with specific steps
rldk check-determinism --cmd "python train.py" --compare loss --steps 100,200,300

# Gate mode for CI
rldk check-determinism --gate
```

### `rldk replay`

Replay a training run with the original seed and verify reproducibility.

```bash
rldk replay --run RUN --command COMMAND --metrics METRICS [OPTIONS]
```

**Options:**
- `--run, -r TEXT`: Path to original training run data
- `--command, -c TEXT`: Training command to replay (should accept --seed)
- `--metrics, -m TEXT`: Metrics to compare (comma-separated)
- `--tolerance, -t FLOAT`: Tolerance for metric differences (relative) (default: 0.01)
- `--max-steps, -s INTEGER`: Maximum steps to replay
- `--output-dir, -o TEXT`: Output directory for results (default: replay_results)
- `--device, -d TEXT`: Device to use (auto-detected if None)
- `--no-wandb`: Disable W&B logging and use file logging only

**Examples:**
```bash
# Replay training run
rldk replay --run ./original_run --command "python train.py --seed {seed}" --metrics loss,reward_mean

# Replay with custom tolerance
rldk replay --run ./original_run --command "python train.py --seed {seed}" --metrics loss --tolerance 0.005
```

### `rldk track`

Start tracking an experiment with W&B (default) or file logging.

```bash
rldk track EXPERIMENT_NAME [OPTIONS]
```

**Arguments:**
- `EXPERIMENT_NAME`: Name of the experiment to track

**Options:**
- `--output-dir, -o TEXT`: Output directory for tracking data (default: ./runs)
- `--no-wandb`: Disable W&B logging and use file logging only
- `--wandb-project TEXT`: W&B project name (default: rldk-experiments)
- `--tags TEXT`: Comma-separated list of tags
- `--notes TEXT`: Additional notes for the experiment
- `--interactive, -i`: Keep tracker running in interactive mode

**Examples:**
```bash
# Start experiment tracking
rldk track "my_experiment"

# Start with custom settings
rldk track "my_experiment" --tags "ppo,experiment" --notes "Testing new hyperparameters"

# Interactive mode
rldk track "my_experiment" --interactive
```

## Forensics Commands

### `rldk forensics log-scan`

Scan training logs for PPO anomalies and issues.

```bash
rldk forensics log-scan RUN_OR_EXPORT
```

**Arguments:**
- `RUN_OR_EXPORT`: Path to run or export directory

**Examples:**
```bash
# Scan training logs
rldk forensics log-scan ./my_training_run

# Scan demo data
rldk forensics log-scan ./demo_data
```

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
# Compare two runs
rldk forensics compare-runs ./run_a ./run_b
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
# Compare checkpoints
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
# Audit environment
rldk forensics env-audit ./my_repo
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
# Run diagnostics
rldk forensics doctor ./my_training_run
```

## Reward Commands

### `rldk reward reward-drift`

Compare two reward models and detect drift.

```bash
rldk reward reward-drift MODEL_A MODEL_B --prompts PROMPTS
```

**Arguments:**
- `MODEL_A`: Path to first reward model directory
- `MODEL_B`: Path to second reward model directory

**Options:**
- `--prompts, -p TEXT`: Path to prompts JSONL file

**Examples:**
```bash
# Compare reward models
rldk reward reward-drift model_a model_b --prompts prompts.jsonl
```

### `rldk reward reward-health run`

Run reward health analysis on scores data.

```bash
rldk reward reward-health run --scores SCORES --out OUT [OPTIONS]
```

**Options:**
- `--scores TEXT`: Path to scores JSONL file
- `--config TEXT`: Path to health configuration YAML file
- `--out TEXT`: Output directory for reports
- `--adapter TEXT`: Adapter type for data ingestion (custom_jsonl, trl, openrlhf, wandb)
- `--gate`: Enable CI gate mode with exit codes (0=pass, 1=warn, 2=fail)

**Examples:**
```bash
# Run reward health analysis
rldk reward reward-health run --scores scores.jsonl --out ./reports

# Run with custom config
rldk reward reward-health run --scores scores.jsonl --config my_config.yaml --out ./reports
```

### `rldk reward reward-health gate`

Gate CI based on health.json results (exit codes: 0=pass, 3=fail).

```bash
rldk reward reward-health gate --from FROM_PATH
```

**Options:**
- `--from TEXT`: Path to health.json file

**Examples:**
```bash
# Gate CI based on health results
rldk reward reward-health gate --from ./reports/health.json
```

## Evaluation Commands

### `rldk evals evaluate`

Run evaluation suite on JSONL data.

```bash
rldk evals evaluate INPUT_FILE [OPTIONS]
```

**Arguments:**
- `INPUT_FILE`: Path to JSONL input file

**Options:**
- `--suite, -s TEXT`: Evaluation suite to run (quick/comprehensive/safety) (default: quick)
- `--output, -o TEXT`: Path to output JSON file
- `--output-column TEXT`: Column name containing model outputs (default: output)
- `--events-column TEXT`: Column name containing event logs (default: events)
- `--min-samples INTEGER`: Minimum samples required for evaluation (default: 10)
- `--timeout INTEGER`: Timeout in seconds for evaluation (default: 300)
- `--verbose, -v`: Enable verbose logging

**Examples:**
```bash
# Run quick evaluation
rldk evals evaluate data.jsonl --suite quick

# Run comprehensive evaluation
rldk evals evaluate data.jsonl --suite comprehensive --output results.json

# Run with custom parameters
rldk evals evaluate data.jsonl --suite safety --min-samples 50 --timeout 600
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
- `--output-column TEXT`: Column name containing model outputs (default: output)
- `--events-column TEXT`: Column name containing event logs (default: events)

**Examples:**
```bash
# Validate data file
rldk evals validate-data data.jsonl
```

## Utility Commands

### `rldk version`

Show version information.

```bash
rldk version
```

### `rldk format-info`

Show data format information for adapters.

```bash
rldk format-info [OPTIONS]
```

**Options:**
- `--adapter, -a TEXT`: Show format info for specific adapter
- `--examples`: Show example data

**Examples:**
```bash
# Show info for all adapters
rldk format-info

# Show info for specific adapter
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
- `--adapter, -a TEXT`: Adapter type to test
- `--verbose, -v`: Show detailed analysis

**Examples:**
```bash
# Validate data format
rldk validate-format ./my_data

# Test specific adapter
rldk validate-format ./my_data --adapter trl --verbose
```

## Card Generation Commands

### `rldk card`

Generate trust cards for RL training runs.

```bash
rldk card CARD_TYPE RUN_A [RUN_B] [OPTIONS]
```

**Arguments:**
- `CARD_TYPE`: Type of card to generate (determinism, drift, reward)
- `RUN_A`: Path to first run directory
- `RUN_B`: Path to second run directory (for drift cards)

**Options:**
- `--output-dir, -o TEXT`: Output directory for cards

**Examples:**
```bash
# Generate determinism card
rldk card determinism run_a

# Generate drift card
rldk card drift run_a run_b

# Generate reward card
rldk card reward run_a
```

## Common Options

Most commands support these common options:

- `--help`: Show help message and exit
- `--version`: Show version and exit

## Exit Codes

- `0`: Success
- `1`: General error
- `2`: Critical error (for gate mode)

## Examples

### Complete Workflow

```bash
# 1. Generate demo data
rldk demo --out ./demo_data --seed 1337 --steps 200

# 2. Ingest the data
rldk ingest ./demo_data --adapter demo_jsonl --output metrics.jsonl

# 3. Run forensics analysis
rldk forensics log-scan ./demo_data
rldk forensics compare-runs ./demo_data ./demo_data

# 4. Check determinism
rldk check-determinism --cmd "python train.py --seed 42" --compare loss,reward_mean

# 5. Run evaluation
rldk evals evaluate metrics.jsonl --suite comprehensive --output results.json

# 6. Generate cards
rldk card determinism ./demo_data
```

### CI/CD Integration

```bash
# Check determinism in CI
rldk check-determinism --gate

# Run reward health check
rldk reward reward-health run --scores scores.jsonl --out ./reports --gate

# Gate based on health results
rldk reward reward-health gate --from ./reports/health.json
```