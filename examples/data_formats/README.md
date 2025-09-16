# RL Debug Kit Data Format Examples

This directory contains examples of the data formats supported by RL Debug Kit's flexible ingestion system.

## Quick Start

The flexible data adapters automatically resolve field names and work with multiple formats:

```python
from rldk.adapters.flexible import FlexibleDataAdapter

# Zero-config ingestion (works for common field names)
adapter = FlexibleDataAdapter("your_data.jsonl")
df = adapter.load()

# With explicit field mapping for custom schemas
field_map = {"step": "global_step", "reward": "reward_scalar", "kl": "kl_to_ref"}
adapter = FlexibleDataAdapter("your_data.jsonl", field_map=field_map)
df = adapter.load()
```

## Supported Formats

### 1. Flexible Data Adapter (Recommended)

**Description**: Universal adapter that works with multiple formats and automatically resolves field names

**File Types**: `.jsonl`, `.json`, `.csv`, `.parquet`

**Canonical Fields** (automatically resolved from synonyms):
- `step`: Training step number
- `reward`: Reward value  
- `kl`: KL divergence
- `entropy`: Policy entropy
- `loss`: Training loss
- `phase`: Training phase
- `wall_time`: Wall clock time
- `seed`: Random seed
- `run_id`: Run identifier
- `git_sha`: Git commit hash
- `lr`: Learning rate
- `grad_norm`: Gradient norm
- `clip_frac`: Clipping fraction
- `tokens_in`: Input tokens count
- `tokens_out`: Output tokens count

**Field Synonyms** (automatically resolved):
- `step`: global_step, step, iteration, iter, timestep, step_id, epoch, batch, update, training_step
- `reward`: reward_scalar, reward, score, return, r, reward_mean, avg_reward, mean_reward, total_reward, cumulative_reward
- `kl`: kl_to_ref, kl, kl_divergence, kl_ref, kl_value, kl_mean, kl_div, kl_loss, kl_penalty, kl_regularization
- `entropy`: entropy, entropy_mean, avg_entropy, mean_entropy, policy_entropy, action_entropy
- `loss`: loss, total_loss, policy_loss, value_loss, actor_loss, critic_loss, combined_loss, training_loss

**Examples**:

Zero-config (automatic field resolution):
```json
{"step": 0, "reward": 0.5, "kl": 0.1, "entropy": 0.8}
{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8}
{"iteration": 0, "score": 0.5, "kl_divergence": 0.1, "policy_entropy": 0.8}
```

With explicit field mapping:
```python
field_map = {
    "step": "iteration",
    "reward": "score", 
    "kl": "kl_divergence",
    "entropy": "policy_entropy"
}
```

With nested fields:
```python
field_map = {
    "reward": "metrics.reward",
    "kl": "metrics.kl_divergence",
    "entropy": "data.entropy_value"
}
```

**Usage**:
```python
# Zero-config
adapter = FlexibleDataAdapter("data.jsonl")
df = adapter.load()

# With field mapping
adapter = FlexibleDataAdapter("data.jsonl", field_map={"step": "global_step"})
df = adapter.load()

# With YAML config
adapter = FlexibleDataAdapter("data.jsonl", config_file="field_mapping.yaml")
df = adapter.load()
```

### 2. Legacy Adapters

#### TRL Adapter (`--adapter trl`)

**Description**: For TRL (Transformer Reinforcement Learning) training logs

**File Types**: `.jsonl`, `.log`

**Required Fields**:
- `step`: Training step number
- `phase`: Training phase (usually "train")
- `reward_mean`: Mean reward value
- `kl_mean`: Mean KL divergence

**Example**:
```json
{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8, "loss": 0.5, "lr": 0.001}
```

#### OpenRLHF Adapter (`--adapter openrlhf`)

**Description**: For OpenRLHF training logs

**File Types**: `.jsonl`, `.log`

**Required Fields**: Same as TRL adapter

#### Custom JSONL Adapter (`--adapter custom_jsonl`)

**Description**: For custom JSONL formats with specific field names

**File Types**: `.jsonl` only

**Required Fields**:
- `global_step`: Global training step number
- `reward_scalar`: Reward value
- `kl_to_ref`: KL divergence to reference model

**Example**:
```json
{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8, "loss": 0.5, "lr": 0.001}
```

#### WandB Adapter (`--adapter wandb`)

**Description**: For WandB run data

**Format**: `wandb://entity/project/run_id`

**Example**:
```
wandb://my-entity/my-project/abc123
```

## Cookbook

### Zero-Config Ingestion

For common field names, no configuration is needed:

```python
from rldk.adapters.flexible import FlexibleDataAdapter

# Works automatically with common field names
adapter = FlexibleDataAdapter("training_logs.jsonl")
df = adapter.load()
```

### Mapping via Code

For custom field names, provide explicit mapping:

```python
field_map = {
    "step": "global_step",
    "reward": "reward_scalar", 
    "kl": "kl_to_ref",
    "entropy": "entropy_value"
}

adapter = FlexibleDataAdapter("custom_logs.jsonl", field_map=field_map)
df = adapter.load()
```

### Mapping via YAML

Create a reusable configuration file:

```yaml
# field_mapping.yaml
field_map:
  step: global_step
  reward: reward_scalar
  kl: kl_to_ref
  entropy: entropy_value
  loss: total_loss
  lr: learning_rate
```

```python
adapter = FlexibleDataAdapter("data.jsonl", config_file="field_mapping.yaml")
df = adapter.load()
```

### Nested Keys

Access nested fields using dot notation:

```python
field_map = {
    "reward": "metrics.reward",
    "kl": "metrics.kl_divergence", 
    "entropy": "data.entropy_value"
}

adapter = FlexibleDataAdapter("nested_data.jsonl", field_map=field_map)
df = adapter.load()
```

### Helpful Errors

When fields are missing, you get helpful suggestions:

```python
try:
    adapter = FlexibleDataAdapter("incomplete_data.jsonl")
    df = adapter.load()
except SchemaError as e:
    print(e)  # Shows suggestions and ready-to-paste field_map
```

### Multiple Formats

Load from directories with mixed file types:

```python
# Loads all supported files from directory
adapter = FlexibleDataAdapter("/path/to/logs/")
df = adapter.load()
```

### Streaming Large Files

For large JSONL files, use streaming:

```python
from rldk.adapters.flexible import FlexibleJSONLAdapter

adapter = FlexibleJSONLAdapter("large_file.jsonl", stream_large_files=True)
df = adapter.load()
```

## Performance Tips

- **Parquet**: Fastest for large datasets, good compression
- **JSONL**: Good for streaming, human-readable
- **CSV**: Human-readable but slower for large files
- **JSON**: Good for small datasets, supports nested structures

## Common Issues and Solutions

### Issue: "Missing required fields"
**Solution**: The adapter provides suggestions for similar field names and ready-to-paste field_map:

```python
# Error message includes:
# Found similar fields:
#   step: step_count, step_id
#   reward: reward_value, score
# Try this field_map: {"step": "step_count", "reward": "reward_value"}
```

### Issue: "Cannot handle source"
**Solution**: Check file extension and format:
- Supported: `.jsonl`, `.json`, `.csv`, `.parquet`
- Ensure data is valid for the format
- Try explicit field mapping

### Issue: "Invalid JSON"
**Solution**: Validate JSONL files:
```bash
python -m json.tool your_file.jsonl
```

## Examples

See the example files in this directory:
- `jsonl_flexible_adapter_demo.py` - Comprehensive flexible adapter demo
- `csv_parquet_adapter_demo.py` - CSV and Parquet format demo
- `trl_example.jsonl` - TRL format example
- `openrlhf_example.jsonl` - OpenRLHF format example  
- `custom_jsonl_example.jsonl` - Custom JSONL format example

## Getting Help

1. Check error messages for specific suggestions
2. Try the flexible adapter with zero-config first
3. Use field mapping for custom schemas
4. Check example files for reference
5. Use `--verbose` flag for detailed output

```bash
rldk ingest your_data.jsonl --adapter flexible --verbose
```