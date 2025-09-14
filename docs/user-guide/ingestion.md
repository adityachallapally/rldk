# Data Ingestion

RLDK provides a unified data ingestion system that can import and normalize training data from various RL frameworks and formats.

## Overview

The ingestion system supports:
- **Multiple frameworks**: TRL, OpenRLHF, WandB, custom formats
- **Various file formats**: JSONL, CSV, Parquet, log files
- **Automatic detection**: Smart adapter selection based on data format
- **Schema validation**: Ensure data consistency and quality
- **Normalization**: Convert all data to standardized Event schema

## Supported Data Sources

### TRL (Transformer Reinforcement Learning)
- Training logs from TRL PPO, DPO, and other algorithms
- Both `.jsonl` and `.log` file formats
- Automatic metric extraction and normalization

### OpenRLHF
- Distributed training logs
- Network monitoring data
- Performance metrics

### Weights & Biases (WandB)
- Exported run data
- Metric histories
- Hyperparameter configurations

### Custom JSONL
- User-defined JSONL formats
- Flexible field mapping
- Custom validation rules

## Command Line Usage

### Basic Ingestion

```bash
# Auto-detect adapter and ingest
rldk ingest ./training_logs

# Specify adapter explicitly
rldk ingest ./trl_logs --adapter trl

# Custom output location
rldk ingest ./training_data --output normalized_data.jsonl

# Specific output format
rldk ingest ./logs --format parquet --output data.parquet
```

### Advanced Options

```bash
# Validate data schema
rldk ingest ./logs --adapter trl --validate --output clean_data.jsonl

# Sample data for testing
rldk ingest ./large_dataset --sample 1000 --output sample.jsonl

# Custom adapter configuration
rldk ingest ./custom_logs --adapter custom --config adapter_config.yaml

# Multiple sources
rldk ingest ./logs1 ./logs2 ./logs3 --output combined_data.jsonl
```

## Python API

### Basic Usage

```python
from rldk.ingest import ingest_runs

# Auto-detect and ingest
df = ingest_runs("./training_logs")

# Specify adapter
df = ingest_runs(
    source="./trl_logs",
    adapter_hint="trl",
    validate=True
)

# Save to file
df.to_json("normalized_data.jsonl", orient="records", lines=True)
```

### Advanced Configuration

```python
from rldk.ingest import ingest_runs, IngestionConfig

# Custom configuration
config = IngestionConfig(
    adapter="trl",
    validate_schema=True,
    sample_size=10000,
    output_format="parquet",
    field_mapping={
        "custom_reward": "reward_mean",
        "custom_kl": "kl_divergence"
    }
)

# Ingest with config
df = ingest_runs(
    source="./custom_logs",
    config=config
)
```

### Multiple Sources

```python
import pandas as pd
from rldk.ingest import ingest_runs

# Ingest multiple sources
sources = ["./run1", "./run2", "./run3"]
dataframes = []

for source in sources:
    df = ingest_runs(source, adapter_hint="trl")
    df["source"] = source  # Add source identifier
    dataframes.append(df)

# Combine all data
combined_df = pd.concat(dataframes, ignore_index=True)
combined_df.to_json("all_runs.jsonl", orient="records", lines=True)
```

## Data Adapters

### TRL Adapter

Handles TRL framework training logs:

```python
from rldk.adapters import TRLAdapter

adapter = TRLAdapter()

# Check if adapter can handle the data
if adapter.can_handle("./trl_logs"):
    data = adapter.ingest("./trl_logs")
    
# Validate data
if adapter.validate(data):
    print("Data validation passed")
```

**Supported TRL formats:**
- PPO training logs
- DPO training logs  
- Reward model training logs
- Custom TRL configurations

### OpenRLHF Adapter

Handles OpenRLHF distributed training data:

```python
from rldk.adapters import OpenRLHFAdapter

adapter = OpenRLHFAdapter()
data = adapter.ingest("./openrlhf_logs")
```

**Supported OpenRLHF formats:**
- Distributed training logs
- Network monitoring data
- Performance analytics
- Multi-node training metrics

### WandB Adapter

Handles Weights & Biases exported data:

```python
from rldk.adapters import WandBAdapter

adapter = WandBAdapter()
data = adapter.ingest("./wandb_export.csv")
```

**Supported WandB formats:**
- CSV exports from WandB
- JSON run data
- Metric histories
- Hyperparameter sweeps

### Custom Adapter

For custom data formats:

```python
from rldk.adapters import CustomJSONLAdapter

# Configure field mapping
adapter = CustomJSONLAdapter(
    field_mapping={
        "step_num": "step",
        "loss_value": "loss",
        "reward_avg": "reward_mean",
        "kl_div": "kl_divergence"
    }
)

data = adapter.ingest("./custom_format.jsonl")
```

## Data Schema

### Normalized Event Schema

All ingested data is converted to the standardized Event schema:

```python
@dataclass
class Event:
    step: int
    timestamp: Optional[float] = None
    seed: Optional[int] = None
    
    # Training metrics
    loss: Optional[float] = None
    reward_mean: Optional[float] = None
    reward_std: Optional[float] = None
    kl_divergence: Optional[float] = None
    entropy: Optional[float] = None
    
    # Gradient metrics
    policy_grad_norm: Optional[float] = None
    value_grad_norm: Optional[float] = None
    
    # PPO-specific metrics
    advantage_mean: Optional[float] = None
    advantage_std: Optional[float] = None
    clip_fraction: Optional[float] = None
    kl_coefficient: Optional[float] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Field Mapping

Configure how source fields map to the normalized schema:

```yaml
# field_mapping.yaml
field_mapping:
  # Source field -> Target field
  "episode_reward": "reward_mean"
  "kl": "kl_divergence"
  "entropy_loss": "entropy"
  "grad_norm": "policy_grad_norm"
  "step_count": "step"
  
validation:
  required_fields: ["step", "loss"]
  numeric_fields: ["loss", "reward_mean", "kl_divergence"]
  
preprocessing:
  normalize_timestamps: true
  fill_missing_steps: true
  remove_duplicates: true
```

## Validation and Quality Checks

### Schema Validation

```python
from rldk.ingest import validate_schema

# Validate against Event schema
is_valid, errors = validate_schema(data)

if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### Data Quality Checks

```python
from rldk.ingest import run_quality_checks

# Run comprehensive quality checks
quality_report = run_quality_checks(data)

print(f"Missing values: {quality_report.missing_values}")
print(f"Duplicate rows: {quality_report.duplicates}")
print(f"Outliers detected: {quality_report.outliers}")
print(f"Data quality score: {quality_report.quality_score}")
```

### Custom Validation Rules

```python
from rldk.ingest import CustomValidator

# Define custom validation rules
validator = CustomValidator([
    lambda row: row.get('step', 0) >= 0,  # Step must be non-negative
    lambda row: 0 <= row.get('reward_mean', 0) <= 1,  # Reward in [0,1]
    lambda row: row.get('kl_divergence', 0) >= 0,  # KL must be non-negative
])

# Validate data
valid_rows, invalid_rows = validator.validate(data)
print(f"Valid rows: {len(valid_rows)}, Invalid rows: {len(invalid_rows)}")
```

## Configuration Files

### Adapter Configuration

```yaml
# adapter_config.yaml
adapter: "custom"
source_format: "jsonl"

field_mapping:
  "iteration": "step"
  "total_loss": "loss"
  "mean_reward": "reward_mean"
  "kl_penalty": "kl_divergence"
  "policy_entropy": "entropy"

validation:
  required_fields: ["step", "loss", "reward_mean"]
  numeric_fields: ["loss", "reward_mean", "kl_divergence", "entropy"]
  range_checks:
    step: [0, null]
    reward_mean: [-10, 10]
    kl_divergence: [0, 1]

preprocessing:
  remove_duplicates: true
  fill_missing_values: true
  normalize_timestamps: true
  sort_by_step: true

output:
  format: "jsonl"
  include_metadata: true
  compress: false
```

### Global Ingestion Settings

```yaml
# ~/.rldk/ingest_config.yaml
default_adapter: "auto"
default_validation: true
default_sample_size: null

adapters:
  trl:
    log_formats: [".jsonl", ".log"]
    metric_patterns:
      loss: ["loss", "total_loss", "training_loss"]
      reward: ["reward", "reward_mean", "episode_reward"]
      kl: ["kl", "kl_divergence", "kl_penalty"]
  
  openrlhf:
    distributed_support: true
    network_metrics: true
    
  wandb:
    api_key: "${WANDB_API_KEY}"
    project_filter: null

quality_checks:
  detect_outliers: true
  outlier_threshold: 3.0
  missing_value_threshold: 0.1
  duplicate_threshold: 0.05

output:
  default_format: "jsonl"
  compression: "gzip"
  chunk_size: 10000
```

## Integration Examples

### With Tracking System

```python
from rldk.tracking import ExperimentTracker
from rldk.ingest import ingest_runs

# Ingest previous run data
previous_data = ingest_runs("./previous_run")

# Start new experiment with reference to previous data
tracker = ExperimentTracker(config)
tracker.start_experiment()

# Add previous run as reference
tracker.add_metadata("previous_run_data", {
    "source": "./previous_run",
    "num_steps": len(previous_data),
    "final_reward": previous_data.iloc[-1]["reward_mean"]
})

# Continue with current experiment...
```

### With Forensics Analysis

```python
from rldk.ingest import ingest_runs
from rldk.forensics import ComprehensivePPOForensics

# Ingest training data
data = ingest_runs("./training_logs", adapter_hint="trl")

# Run forensics analysis on ingested data
forensics = ComprehensivePPOForensics()

for _, row in data.iterrows():
    metrics = forensics.update(
        step=row["step"],
        kl=row["kl_divergence"],
        kl_coef=row.get("kl_coefficient", 0.1),
        entropy=row["entropy"],
        reward_mean=row["reward_mean"],
        reward_std=row.get("reward_std", 0.0),
        policy_grad_norm=row.get("policy_grad_norm", 0.0),
        value_grad_norm=row.get("value_grad_norm", 0.0),
        advantage_mean=row.get("advantage_mean", 0.0),
        advantage_std=row.get("advantage_std", 0.0)
    )

# Get forensics report
report = forensics.get_report()
print(f"Anomalies detected: {len(report.anomalies)}")
```

### With Evaluation System

```python
from rldk.ingest import ingest_runs
from rldk.evals import run_evaluation, get_eval_suite

# Ingest training data
training_data = ingest_runs("./training_logs")

# Prepare evaluation data from training metrics
eval_data = training_data[["step", "reward_mean", "loss"]].to_dict("records")

# Run evaluation
eval_suite = get_eval_suite("quick")
result = run_evaluation(
    data=eval_data,
    suite=eval_suite
)

print(f"Training quality score: {result.overall_score}")
```

## Best Practices

1. **Use Validation**: Always validate ingested data for quality assurance
2. **Sample First**: Test with small samples before processing large datasets
3. **Check Adapters**: Verify adapter compatibility before bulk ingestion
4. **Monitor Quality**: Track data quality metrics over time
5. **Backup Original**: Keep original data files before normalization
6. **Document Mapping**: Maintain clear field mapping documentation
7. **Version Control**: Track adapter configurations and field mappings

## Troubleshooting

### Common Issues

1. **Adapter Detection Fails**: Specify adapter explicitly with `--adapter`
2. **Schema Validation Errors**: Check field mapping and data types
3. **Memory Issues**: Use sampling for large datasets
4. **Missing Fields**: Configure field mapping for custom formats

### Performance Tips

- Use `--sample` for testing with large datasets
- Enable compression for large output files
- Process data in chunks for memory efficiency
- Use parallel processing when available

For more examples and advanced usage, see the [Examples](../examples/basic-ppo-cartpole.md) section.
