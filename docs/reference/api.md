# API Reference

This page provides comprehensive API documentation for RLDK's core modules and functions.

## Core Modules

::: rldk.tracking
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.forensics
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.determinism
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.reward
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.evals
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.diff
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.replay
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.ingest
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.adapters
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.utils
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.cards
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.config
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

::: rldk.io
    options:
      show_root_heading: true
      show_root_toc_entry: true
      show_object_full_path: false
      show_source: false

## Key Classes and Functions

### Experiment Tracking

#### `ExperimentTracker`

The main class for experiment tracking and state capture.

```python
from rldk.tracking import ExperimentTracker, TrackingConfig

# Initialize tracker
config = TrackingConfig(
    experiment_name="my_experiment",
    enable_dataset_tracking=True,
    enable_model_tracking=True,
    enable_environment_tracking=True,
    enable_seed_tracking=True,
    enable_git_tracking=True
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

### Forensics Analysis

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
    enable_advantage_statistics=True,
    enable_length_bias_detection=True,
    length_bias_threshold=0.35,
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
length_bias = forensics.get_length_bias_analysis()
forensics.save_analysis("analysis.json")
```

### Determinism Checking

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

### Reward Analysis

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

#### `reward_health`

High-level helper that normalizes inputs and returns a `HealthAnalysisResult`.

```python
from rldk import HealthAnalysisResult, reward_health

# Accepts DataFrames, lists of dictionaries, or JSONL/table paths
analysis = reward_health(
    run_data=[{"step": 1, "reward_mean": 0.5}, {"step": 2, "reward_mean": 0.6}],
    reward_col="reward_mean",
    response_col="response_text",  # Optional column with completions
    length_col="tokens_out",       # Optional column with token counts
    threshold_length_bias=0.3,
)

assert isinstance(analysis, HealthAnalysisResult)
print(analysis.report.passed)
print(analysis.metrics.head())
summary = analysis.to_dict()
# Dedicated length bias detector metrics
print(analysis.report.length_bias_metrics.bias_severity)
print(analysis.report.length_bias_recommendations)
```

#### `HealthAnalysisResult`

The object returned by `reward_health` combines the underlying
`RewardHealthReport` with the normalized metrics used for analysis.

- `report`: Raw `RewardHealthReport` dataclass with detailed findings
- `metrics`: Normalized training metrics DataFrame for the run
- `reference_metrics`: Optional normalized reference DataFrame
- `to_dict()`: JSON-ready summary for serialization or logging
- Length bias fields:
  - `length_bias_detected`: Boolean flag indicating severity crossed the configured threshold
  - `length_bias_metrics`: Structured metrics (correlations, ODIN heuristics, quartiles)
  - `length_bias_recommendations`: Human-readable remediation tips from the detector
- Overoptimization fields:
  - `overoptimization.flagged`: True when proxy reward rises while gold metrics stagnate and KL is elevated
  - `overoptimization.delta`: Difference between proxy and gold improvements across early/late windows
  - `overoptimization.correlation_trend`: Pearson/Spearman deltas to monitor alignment drift
  - `overoptimization.kl_summary`: Recent KL statistics pulled from PPO forensics hooks

### Evaluation Suites

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

### Run Comparison

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

### Seeded Replay

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

### Data Ingestion

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

### Seed Management

#### `set_global_seed`

Set a global seed for all random number generators.

```python
import rldk

# Set global seed
seed = rldk.set_global_seed(42)
print(f"Set seed to: {seed}")

# Get current seed
current_seed = rldk.get_current_seed()
print(f"Current seed: {current_seed}")

# Set up reproducible environment
reproducible_seed = rldk.set_reproducible_environment(1337)
print(f"Reproducible environment set with seed: {reproducible_seed}")

# Validate seed consistency
is_consistent = rldk.validate_seed_consistency(42)
print(f"Seed consistency: {is_consistent}")
```

### Utility Functions

#### Validation

```python
from rldk.utils import (
    validate_file_exists,
    validate_dataframe,
    validate_numeric_range,
    validate_string_not_empty
)

# File validation
path = validate_file_exists("data.jsonl")

# DataFrame validation
df = validate_dataframe(data, required_columns=["step", "loss"])

# Numeric validation
value = validate_numeric_range(0.5, min_val=0.0, max_val=1.0)

# String validation
text = validate_string_not_empty("hello world")
```

#### Error Handling

```python
from rldk.utils import (
    RLDKError,
    ValidationError,
    format_error_message,
    with_retry,
    with_timeout
)

# Custom error handling
try:
    # Some operation
    pass
except ValidationError as e:
    print(format_error_message(e))

# Retry decorator
@with_retry(max_retries=3, delay=1.0)
def unreliable_function():
    pass

# Timeout decorator
@with_timeout(30.0)
def slow_function():
    pass
```

#### Progress Indication

```python
from rldk.utils import (
    progress_bar,
    spinner,
    timed_operation,
    print_operation_status
)

# Progress bar
with progress_bar(100, "Processing") as bar:
    for i in range(100):
        # Do work
        bar.update(1)

# Spinner
with spinner("Processing") as sp:
    # Do work
    pass

# Timed operation
@timed_operation("Data loading")
def load_data():
    pass

# Status printing
print_operation_status("Data processing", "success", "100 items processed")
```

## Data Types and Schemas

### Common Data Types

#### `DeterminismReport`
```python
@dataclass
class DeterminismReport:
    passed: bool
    culprit: Optional[str]
    fixes: List[str]
    replica_variance: Dict[str, float]
    rng_map: Dict[str, Any]
    mismatches: List[Dict[str, Any]]
    dataloader_notes: List[str]
```

#### `RewardHealthReport`
```python
@dataclass
class RewardHealthReport:
    passed: bool
    drift_detected: bool
    saturation_issues: List[str]
    calibration_score: float
    shortcut_signals: List[str]
    label_leakage_risk: float
    fixes: List[str]
    drift_metrics: pd.DataFrame
    calibration_details: Dict[str, Any]
    shortcut_analysis: Dict[str, float]
    saturation_analysis: Dict[str, Any]
    length_bias_detected: bool
    length_bias_metrics: LengthBiasMetrics
    length_bias_recommendations: List[str]
    overoptimization: OveroptimizationAnalysis
```

#### `OveroptimizationAnalysis`
```python
@dataclass
class OveroptimizationAnalysis:
    proxy_improvement: float
    gold_improvement: float
    delta: float
    correlation_trend: Dict[str, Optional[float]]
    kl_summary: Dict[str, Any]
    flagged: bool
    gold_metrics_available: bool
    gold_regressed: bool
    gold_stagnant: bool
    kl_elevated: bool
    correlation_declined: bool
    warning: Optional[str]
    notes: List[str]
    window_size: int
    delta_threshold: float
    min_samples: int
    sample_size: int
```

#### `DivergenceReport`
```python
@dataclass
class DivergenceReport:
    diverged: bool
    first_step: Optional[int]
    tripped_signals: List[Dict[str, Any]]
    notes: List[str]
    suspected_causes: List[str]
    details: pd.DataFrame
```

#### `ReplayReport`
```python
@dataclass
class ReplayReport:
    passed: bool
    original_seed: Optional[int]
    replay_seed: Optional[int]
    metrics_compared: List[str]
    mismatches: List[Dict[str, Any]]
    comparison_stats: Dict[str, Dict[str, Any]]
    replay_duration: float
```

## Configuration

### Global Settings

```python
from rldk.config import settings

# Access global settings
print(f"Default output directory: {settings.default_output_dir}")
print(f"Logging level: {settings.logging_level}")
print(f"Cache directory: {settings.cache_dir}")

# Modify settings
settings.default_output_dir = Path("./custom_output")
settings.logging_level = "DEBUG"
```

### Environment Variables

RLDK respects the following environment variables:

- `RLDK_OUTPUT_DIR`: Default output directory
- `RLDK_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `RLDK_CACHE_DIR`: Cache directory for temporary files
- `PYTHONHASHSEED`: Python hash seed (set automatically by RLDK)
- `TOKENIZERS_PARALLELISM`: Tokenizers parallelism (set to 'false' by RLDK)
- `OMP_NUM_THREADS`: OpenMP threads (set to '1' by RLDK)
- `MKL_NUM_THREADS`: MKL threads (set to '1' by RLDK)

## Error Codes

RLDK uses standardized error codes for better debugging:

### Validation Errors
- `INVALID_PATH`: Invalid file or directory path
- `FILE_NOT_FOUND`: File does not exist
- `PERMISSION_DENIED`: Insufficient permissions
- `INVALID_DATA_TYPE`: Wrong data type provided
- `MISSING_REQUIRED_FIELDS`: Required fields missing

### Adapter Errors
- `ADAPTER_NOT_FOUND`: No suitable adapter found
- `INVALID_FORMAT`: Data format not supported
- `CONNECTION_FAILED`: Network connection failed
- `AUTHENTICATION_FAILED`: Authentication failed

### Evaluation Errors
- `INSUFFICIENT_DATA`: Not enough data for evaluation
- `INVALID_METRIC`: Unsupported metric
- `TIMEOUT`: Evaluation timed out
- `DEPENDENCY_ERROR`: Missing dependency

### System Errors
- `MEMORY_ERROR`: Insufficient memory
- `DISK_FULL`: Disk space exhausted
- `CONFIGURATION_ERROR`: Invalid configuration
- `VERSION_MISMATCH`: Version compatibility issue