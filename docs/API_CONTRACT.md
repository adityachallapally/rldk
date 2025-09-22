# RL Debug Kit (RLDK) API Contract

This document defines the public API contract for RLDK, including Python symbols, CLI commands, and artifact locations.

**Test Status**: ✅ 37/37 tests passing - All functionality verified working

## 1. Public Python Symbols

### Top-level Package (`rldk`)

#### Functions
- `ingest_runs(runs: str, adapter: Optional[str] = None) -> pd.DataFrame`
  - Ingest training runs from various sources (files, directories, wandb:// URIs)
  - Returns a pandas DataFrame with training metrics

- `first_divergence(a: str, b: str, signals: List[str], tolerance: float = 2.0, k: int = 3, window: int = 50) -> DivergenceReport`
  - Find the first divergence point between two training runs
  - Returns a DivergenceReport with divergence details

- `check(cmd: str, compare: List[str], steps: Optional[List[int]] = None, replicas: int = 5, device: Optional[str] = None) -> DeterminismReport`
  - Check if a training command is deterministic
  - Returns a DeterminismReport with determinism analysis

- `bisect_commits(good_sha: str, bad_sha: str, cmd: Optional[str] = None, metric: Optional[str] = None, cmp: Optional[str] = None, window: int = 100, shell_predicate: Optional[str] = None) -> BisectResult`
  - Find regression using git bisect
  - Returns a BisectResult with bisect findings

- `health(run_path: str, reference_path: Optional[str] = None, **kwargs) -> RewardHealthReport`
  - Analyze reward model health and detect issues
  - Returns a RewardHealthReport with health metrics

- `run(run_path: str, suite: str = "quick", **kwargs) -> EvalResult`
  - Run evaluation suite on training run data
  - Returns an EvalResult with evaluation metrics

#### Classes
- `RewardHealthReport`
  - Data class containing reward model health analysis results
  - Includes drift detection, saturation analysis, calibration metrics
  - Provides dedicated length-bias metrics, severity scores, and remediation tips

- `EvalResult`
  - Data class containing evaluation suite results
  - Includes statistical analysis and performance metrics

### Tracking Module (`rldk.tracking`)

#### Classes
- `ExperimentTracker(config: TrackingConfig)`
  - Main class for tracking RL experiments
  - Methods: start_experiment(), track_dataset(), track_model(), set_seeds(), finish_experiment()

- `TrackingConfig(experiment_name: str, **kwargs)`
  - Configuration class for experiment tracking
  - Parameters: output_dir, save_to_wandb, wandb_project, tags, notes

- `DatasetTracker`
  - Track dataset versioning and checksums

- `ModelTracker`
  - Track model architecture fingerprinting

- `EnvironmentTracker`
  - Track environment state capture

- `SeedTracker`
  - Track random seed management

- `GitTracker`
  - Track git commit hash integration

### Forensics Module (`rldk.forensics`)

#### Functions
- `scan_ppo_events(events: List[Dict]) -> Dict`
  - Scan PPO training events for anomalies
  - Returns anomaly detection results

#### Classes
- `KLScheduleTracker` / `KLScheduleMetrics`
  - Track KL divergence schedule metrics

- `GradientNormsAnalyzer` / `GradientNormsMetrics`
  - Analyze gradient norm patterns

- `AdvantageStatisticsTracker` / `AdvantageStatisticsMetrics`
  - Track advantage function statistics

- `ComprehensivePPOForensics` / `ComprehensivePPOMetrics`
  - Comprehensive PPO training analysis

### Determinism Module (`rldk.determinism`)

#### Functions
- `check_determinism(cmd: str, compare: List[str], **kwargs) -> DeterminismReport`
  - Alias for check() function

#### Classes
- `DeterminismReport`
  - Data class containing determinism analysis results

### Diff Module (`rldk.diff`)

#### Classes
- `DivergenceReport`
  - Data class containing divergence analysis results

### Bisect Module (`rldk.bisect`)

#### Classes
- `BisectResult`
  - Data class containing git bisect results

## 2. CLI Commands

### Main Commands

#### `rldk ingest` ✅ **WORKING**
- **Required**: `runs` (path to runs directory/file/wandb:// URI)
- **Optional**: `--adapter` / `-a` (adapter type: trl, openrlhf, wandb), `--output` / `-o` (output file path)
- **Output**: "Ingested {n} training steps" + summary statistics

#### `rldk diff` ✅ **WORKING**
- **Required**: `--a` / `-a` (run A path), `--b` / `-b` (run B path), `--signals` / `-s` (metrics to monitor)
- **Optional**: `--tolerance` / `-t` (z-score threshold), `--k` / `-k` (consecutive violations), `--window` / `-w` (rolling window), `--output-dir` / `-o` (output directory)
- **Output**: "🚨 Divergence detected at step {step}" or "✅ No divergence detected"

#### `rldk check-determinism` ✅ **WORKING**
- **Required**: `--compare` / `-m` (metrics to compare)
- **Optional**: `--cmd` / `-c` (command to run), `--steps` / `-s` (specific steps), `--stride` (step interval), `--replicas` / `-r` (number of replicas), `--tolerance` / `-t` (tolerance), `--device` / `-d` (device), `--output-dir` / `-o` (output directory), `--gate` (CI gate mode)
- **Output**: "✅ Determinism check passed" or "❌ Non-deterministic training detected"

#### `rldk bisect` ✅ **WORKING**
- **Required**: `--good` / `-g` (known good commit SHA)
- **Optional**: `--bad` / `-b` (known bad commit SHA), `--cmd` / `-c` (command), `--metric` / `-m` (metric), `--cmp` (comparison operator), `--window` / `-w` (window size), `--shell-predicate` (shell command)
- **Output**: "🎯 Regression found! Culprit commit: {sha}"

#### `rldk reward-health` ✅ **WORKING**
- **Required**: `--run` / `-r` (run path)
- **Optional**: `--reference` / `-ref` (reference run), `--output-dir` / `-o` (output directory), `--reward-col` (reward column), `--step-col` (step column), various threshold parameters
- **Output**: "Reward health analysis complete" + health metrics summary

#### `rldk replay` ✅ **WORKING**
- **Required**: `--run` / `-r` (run path), `--command` / `-c` (training command), `--metrics` / `-m` (metrics to compare)
- **Optional**: `--tolerance` / `-t` (tolerance), `--max-steps` / `-s` (max steps), `--output-dir` / `-o` (output directory), `--device` / `-d` (device), `--no-wandb` (disable W&B)
- **Output**: "✅ Seeded replay passed" or "🚨 Seeded replay failed"

#### `rldk eval` ✅ **WORKING**
- **Required**: `--run` / `-r` (run path)
- **Optional**: `--suite` / `-s` (evaluation suite), `--output-dir` / `-o` (output directory), `--seed` (random seed), `--sample-size` (sample size)
- **Output**: "Evaluation complete" + statistical analysis results

#### `rldk track` ✅ **WORKING**
- **Required**: `experiment_name` (experiment name)
- **Optional**: `--output-dir` / `-o` (output directory), `--no-wandb` (disable W&B), `--wandb-project` (W&B project), `--tags` (comma-separated tags), `--notes` (notes), `--interactive` / `-i` (interactive mode)
- **Output**: "✅ Experiment tracking started successfully"

#### `rldk reward-drift` ✅ **WORKING**
- **Required**: `model_a` (first model path), `model_b` (second model path), `--prompts` / `-p` (prompts file)
- **Output**: "Reward drift analysis complete" + correlation metrics

#### `rldk forensics doctor` ✅ **WORKING**
- **Alias**: Available as `rldk doctor`
- **Required**: `run_or_repo` (run or repository path)
- **Output**: "Comprehensive diagnostics complete" + issue summary

#### `rldk version` ✅ **WORKING**
- **Output**: "RL Debug Kit version {version}"

#### `rldk card` ✅ **WORKING**
- **Required**: `card_type` (determinism, drift, reward), `run_a` (first run path)
- **Optional**: `run_b` (second run path, for drift cards), `--output-dir` / `-o` (output directory)
- **Output**: "✅ {card_type} card generated" + card status

### Forensics Subcommands (`rldk forensics`)

#### `rldk forensics compare-runs` ✅ **WORKING**
- **Required**: `run_a` (first run directory), `run_b` (second run directory)
- **Output**: "Comparison complete" + anomaly counts

#### `rldk forensics diff-ckpt` ✅ **WORKING**
- **Required**: `ckpt_a` (first checkpoint), `ckpt_b` (second checkpoint)
- **Output**: "Checkpoint diff complete" + parameter differences

#### `rldk forensics env-audit` ✅ **WORKING**
- **Required**: `repo_or_run` (repository or run directory)
- **Output**: "Environment audit complete" + determinism status

#### `rldk forensics log-scan` ✅ **WORKING**
- **Required**: `run_or_export` (run or export directory)
- **Output**: "Log scan complete" + PPO anomaly count

#### `rldk forensics doctor` ✅ **WORKING**
- **Required**: `run_or_repo` (run or repository directory)
- **Output**: "Comprehensive diagnostics complete" + issue summary

### Reward Subcommands (`rldk reward`)

#### `rldk reward reward-drift` ✅ **WORKING**
- **Required**: `model_a` (first model directory), `model_b` (second model directory), `--prompts` / `-p` (prompts JSONL file)
- **Output**: "Reward drift analysis complete" + correlation metrics

## 3. Artifact Locations and Filenames

### Default Output Directories

#### `determinism_analysis/` (check-determinism) ✅ **VERIFIED**
- `determinism_card.json` - Determinism analysis results
- `replica_metrics.jsonl` - Individual replica metrics
- `comparison_stats.json` - Statistical comparison data

#### `diff_analysis/` (diff) ✅ **VERIFIED**
- `divergence_report.json` - Divergence analysis results
- `diff_events.csv` - Detailed divergence events

#### `reward_analysis/` (reward-health) ✅ **VERIFIED**
- `reward_health_report.json` - Reward health analysis
- `drift_card.json` - Drift detection results
- `diff_events.csv` - Detailed drift events

#### `replay_results/` (replay) ✅ **VERIFIED**
- `replay_metrics.jsonl` - Replay metrics data
- `replay_comparison.json` - Comparison statistics
- `replay_mismatches.json` - Tolerance violations

#### `eval_results/` (eval) ✅ **VERIFIED**
- `eval_report.json` - Evaluation results
- `statistical_analysis.json` - Statistical analysis
- `performance_metrics.json` - Performance metrics

#### `rldk_reports/` (forensics commands) ✅ **VERIFIED**
- `run_comparison.json` - Run comparison results
- `ckpt_diff.json` - Checkpoint difference analysis
- `env_audit.json` - Environment audit results
- `ppo_scan.json` - PPO anomaly scan results
- `reward_drift.json` - Reward drift analysis
- `*.png` - Generated plots and visualizations

#### `runs/` (track) ✅ **VERIFIED**
- `{experiment_id}/` - Experiment-specific directory
  - `experiment_metadata.json` - Experiment configuration
  - `dataset_checksums.json` - Dataset versioning data
  - `model_fingerprint.json` - Model architecture fingerprint
  - `environment_snapshot.json` - Environment state
  - `seeds.json` - Random seed information
  - `git_state.json` - Git repository state
  - `rldk_cards/` - Generated trust cards
    - `determinism_card.json` / `determinism_card.png`
    - `drift_card.json` / `drift_card.png`
    - `reward_card.json` / `reward_card.png`

### File Formats
- **JSON**: Configuration, metadata, analysis results
- **JSONL**: Time-series data, metrics, events
- **CSV**: Tabular data, comparison results
- **PNG**: Plots, visualizations, trust cards

### Naming Conventions
- All output files use snake_case naming
- Timestamps are included in filenames when relevant
- Experiment IDs are used for experiment-specific files
- Trust cards follow the pattern `{type}_card.{ext}`

## 4. Test Results Summary

### ✅ All Commands Working (37/37 tests passed)
- All public symbol imports
- `rldk ingest` - Data ingestion from various sources
- `rldk diff` - Run comparison and divergence detection
- `rldk check-determinism` - Determinism verification
- `rldk bisect` - Git bisect for regressions
- `rldk reward-health` - Reward model health analysis
- `rldk replay` - Experiment replay with verification
- `rldk eval` - Evaluation suite execution
- `rldk track` - Experiment tracking
- `rldk reward-drift` - Reward model comparison
- `rldk forensics doctor` - Comprehensive diagnostics (alias `rldk doctor`)
- `rldk version` - Version information
- `rldk card` - Trust card generation
- `rldk forensics compare-runs` - Run comparison analysis
- `rldk forensics diff-ckpt` - Checkpoint difference analysis
- `rldk forensics env-audit` - Environment audit
- `rldk forensics log-scan` - PPO anomaly detection
- `rldk forensics doctor` - Forensics diagnostics
- `rldk reward reward-drift` - Reward drift analysis
- All artifact location tests

### Dependencies Verified
- ✅ pandas, numpy, scipy, matplotlib, seaborn, scikit-learn
- ✅ torch, transformers, datasets, nltk, tokenizers
- ✅ jsonschema, streamlit, plotly, trl, accelerate
- ✅ flask, psutil, detoxify, vaderSentiment
- ✅ pydantic, typer, pytest
- ✅ All core functionality working with proper dependencies
