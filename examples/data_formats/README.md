# RL Debug Kit Data Format Examples

This directory contains examples of the data formats supported by RL Debug Kit's ingestion system.

## Supported Adapters

### 1. TRL Adapter (`--adapter trl`)

**Description**: For TRL (Transformer Reinforcement Learning) training logs

**File Types**: `.jsonl`, `.log`

**Required Fields**:
- `step`: Training step number
- `phase`: Training phase (usually "train")
- `reward_mean`: Mean reward value
- `kl_mean`: Mean KL divergence

**Optional Fields**:
- `reward_std`: Standard deviation of rewards
- `entropy_mean`: Mean entropy
- `clip_frac`: Clipping fraction
- `grad_norm`: Gradient norm
- `lr`: Learning rate
- `loss`: Training loss
- `tokens_in`: Input tokens count
- `tokens_out`: Output tokens count
- `wall_time`: Wall clock time
- `seed`: Random seed
- `run_id`: Run identifier
- `git_sha`: Git commit hash

**Example**:
```json
{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8, "loss": 0.5, "lr": 0.001}
```

**Usage**:
```bash
rldk ingest /path/to/trl_logs --adapter trl
rldk ingest trl_example.jsonl --adapter trl
```

### 2. OpenRLHF Adapter (`--adapter openrlhf`)

**Description**: For OpenRLHF training logs

**File Types**: `.jsonl`, `.log`

**Required Fields**: Same as TRL adapter

**Example**:
```json
{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 0.8, "loss": 0.5, "lr": 0.001}
```

**Usage**:
```bash
rldk ingest /path/to/openrlhf_logs --adapter openrlhf
rldk ingest openrlhf_example.jsonl --adapter openrlhf
```

### 3. Custom JSONL Adapter (`--adapter custom_jsonl`)

**Description**: For custom JSONL formats with specific field names

**File Types**: `.jsonl` only

**Required Fields**:
- `global_step`: Global training step number
- `reward_scalar`: Reward value
- `kl_to_ref`: KL divergence to reference model

**Optional Fields**:
- `entropy`: Entropy value
- `clip_frac`: Clipping fraction
- `grad_norm`: Gradient norm
- `lr`: Learning rate
- `loss`: Training loss
- `tokens_in`: Input tokens count
- `tokens_out`: Output tokens count
- `wall_time`: Wall clock time
- `seed`: Random seed
- `run_id`: Run identifier
- `git_sha`: Git commit hash

**Example**:
```json
{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 0.8, "loss": 0.5, "lr": 0.001}
```

**Usage**:
```bash
rldk ingest /path/to/custom_logs --adapter custom_jsonl
rldk ingest custom_jsonl_example.jsonl --adapter custom_jsonl
```

### 4. WandB Adapter (`--adapter wandb`)

**Description**: For WandB run data

**Format**: `wandb://entity/project/run_id`

**Example**:
```
wandb://my-entity/my-project/abc123
wandb://team/project/run-2024-01-01-12-00-00
```

**Usage**:
```bash
rldk ingest wandb://my-entity/my-project/abc123 --adapter wandb
```

## Auto-Detection

If you don't specify an adapter, RL Debug Kit will try to auto-detect the format:

```bash
rldk ingest /path/to/logs  # Auto-detects adapter
```

## Common Issues and Solutions

### Issue: "Cannot handle source"
**Solution**: The adapter can't process your data format. Check:
1. File extension is supported (`.jsonl` or `.log`)
2. Required fields are present
3. Data is valid JSON (for JSONL files)
4. Try a different adapter type

### Issue: "Missing required fields"
**Solution**: Ensure your data contains the required fields for the adapter:
- TRL/OpenRLHF: `step`, `phase`, `reward_mean`, `kl_mean`
- Custom JSONL: `global_step`, `reward_scalar`, `kl_to_ref`

### Issue: "Invalid JSON"
**Solution**: Check that each line in your JSONL file is valid JSON:
```bash
# Test JSONL file
python -m json.tool your_file.jsonl
```

### Issue: "No log files found"
**Solution**: Ensure your directory contains `.jsonl` or `.log` files:
```bash
# Check directory contents
ls -la /path/to/logs/
```

## Data Validation

You can validate your data format before ingestion:

```bash
# Validate a JSONL file
rldk evals validate-data your_data.jsonl

# Validate with specific columns
rldk evals validate-data your_data.jsonl --output-column output --events-column events
```

## Examples

See the example files in this directory:
- `trl_example.jsonl` - TRL format example
- `openrlhf_example.jsonl` - OpenRLHF format example  
- `custom_jsonl_example.jsonl` - Custom JSONL format example
- `trl_log_example.log` - TRL log format example

## Getting Help

If you're still having issues:

1. Check the error message for specific suggestions
2. Try different adapter types
3. Validate your data format
4. Check the example files for reference
5. Use `--verbose` flag for detailed output

```bash
rldk ingest your_data.jsonl --adapter trl --verbose
```