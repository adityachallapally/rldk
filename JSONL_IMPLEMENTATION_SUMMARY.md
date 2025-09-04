# TRL Callback JSONL Event Emission Implementation

## Overview

This implementation addresses the issue where the TRL callback only wrote CSV/JSON summaries but lacked per-step JSONL logs compatible with TRLAdapter or the Event schema. The solution adds standardized JSONL event emission that follows proper research standards and enables seamless downstream analysis.

## Problem Statement

**Original Issue**: `src/rldk/integrations/trl/callbacks.py` only wrote CSV/JSON summaries; it lacked per-step JSONL logs compatible with TRLAdapter or the Event schema, making downstream analyses cumbersome.

**Solution**: Implement standardized JSONL event emission from TRL callback that:
- Follows the Event schema for consistency
- Is compatible with TRLAdapter for data loading
- Provides per-step granularity for detailed analysis
- Maintains research-grade data quality

## Implementation Details

### 1. Enhanced RLDKCallback Class

**New Parameters**:
- `enable_jsonl_logging: bool = True` - Enable/disable JSONL event emission
- `jsonl_log_interval: int = 1` - Control logging frequency (default: every step)

**New Methods**:
- `_setup_jsonl_logging()` - Initialize JSONL file and logging
- `_emit_jsonl_event()` - Emit standardized JSONL events
- `_close_jsonl_file()` - Proper file cleanup

### 2. Event Schema Integration

The implementation leverages the existing Event schema (`src/rldk/io/event_schema.py`) to ensure:
- **Consistency**: All events follow the same structure
- **Compatibility**: Events work with existing analysis tools
- **Extensibility**: Easy to add new metrics or metadata

**Event Structure**:
```json
{
  "step": 10,
  "wall_time": 100.0,
  "metrics": {
    "loss": 0.5,
    "reward_mean": 0.8,
    "kl_mean": 0.05,
    "entropy_mean": 0.9,
    "clip_frac": 0.1,
    "grad_norm": 0.1,
    "lr": 0.001,
    "value_loss": 0.3,
    "policy_loss": 0.2
  },
  "rng": {
    "seed": 42
  },
  "data_slice": {
    "tokens_in": 1000,
    "tokens_out": 500
  },
  "model_info": {
    "run_id": "rldk_run_1234567890",
    "git_sha": "abc123def456",
    "phase": "train"
  },
  "notes": [
    "High clipping fraction detected",
    "Large gradient norm detected"
  ]
}
```

### 3. TRLAdapter Compatibility

The emitted JSONL events are fully compatible with the TRLAdapter:
- **Automatic Detection**: TRLAdapter can automatically detect and load the files
- **Standardized Format**: Events contain all required fields for TRLAdapter
- **Seamless Integration**: No additional processing needed for downstream analysis

### 4. Research-Grade Features

**Data Quality**:
- **Immediate Flushing**: Events are written immediately to prevent data loss
- **Error Handling**: Graceful handling of missing metrics or schema issues
- **Validation**: Events are validated against the Event schema

**Performance**:
- **Configurable Intervals**: Control logging frequency to balance detail vs. performance
- **Efficient I/O**: Minimal overhead with direct JSONL writing
- **Memory Management**: Proper file cleanup and resource management

**Monitoring**:
- **Health Indicators**: Automatic generation of training health notes
- **Alert Integration**: JSONL events include notes for training issues
- **Metadata Preservation**: Complete run information preserved

## Usage Examples

### Basic Usage
```python
from rldk.integrations.trl.callbacks import RLDKCallback

# Enable JSONL logging (default)
callback = RLDKCallback(
    output_dir="./rldk_logs",
    enable_jsonl_logging=True,
    jsonl_log_interval=1  # Log every step
)
```

### Advanced Configuration
```python
callback = RLDKCallback(
    output_dir="./rldk_logs",
    enable_jsonl_logging=True,
    jsonl_log_interval=5,  # Log every 5 steps
    log_interval=10,       # Detailed logging every 10 steps
    run_id="my_experiment_001"
)
```

### Disable JSONL Logging
```python
callback = RLDKCallback(
    output_dir="./rldk_logs",
    enable_jsonl_logging=False  # Disable JSONL logging
)
```

## File Output

The callback generates the following files:
- `{run_id}_events.jsonl` - Per-step JSONL events (new)
- `{run_id}_metrics.csv` - Summary metrics (existing)
- `{run_id}_metrics.json` - Summary metrics (existing)
- `{run_id}_alerts.json` - Training alerts (existing)
- `{run_id}_final_report.json` - Final analysis (existing)

## Downstream Analysis

### Using TRLAdapter
```python
from rldk.adapters.trl import TRLAdapter

# Load JSONL events
adapter = TRLAdapter("rldk_logs/my_experiment_001_events.jsonl")
df = adapter.load()

# Analyze training progression
print(f"Steps: {df['step'].min()} - {df['step'].max()}")
print(f"Reward progression: {df['reward_mean'].tolist()}")
```

### Using Event Schema
```python
from rldk.io.event_schema import Event, dataframe_to_events

# Convert to Event objects
events = dataframe_to_events(df, run_id="my_experiment_001")

# Process events
for event in events:
    print(f"Step {event.step}: {event.metrics}")
```

### Direct JSONL Processing
```python
import json

# Read JSONL directly
with open("rldk_logs/my_experiment_001_events.jsonl", "r") as f:
    for line in f:
        event = json.loads(line)
        print(f"Step {event['step']}: Loss={event['metrics']['loss']}")
```

## Testing

Comprehensive tests are included in `tests/test_trl_callbacks.py`:

- **JSONL Logging**: Basic functionality and configuration
- **Event Structure**: Validation of Event schema compliance
- **TRLAdapter Compatibility**: End-to-end compatibility testing
- **Interval Control**: Verification of logging frequency
- **Error Handling**: Graceful handling of edge cases
- **File Management**: Proper file creation and cleanup

## Benefits for Researchers

1. **Standardized Data**: All events follow the same schema for consistency
2. **Granular Analysis**: Per-step data enables detailed training analysis
3. **Tool Compatibility**: Works seamlessly with existing RLDK tools
4. **Performance Monitoring**: Built-in health indicators and alerts
5. **Reproducibility**: Complete metadata preservation for experiments
6. **Flexibility**: Configurable logging to balance detail and performance

## Migration Guide

### For Existing Users
- **Backward Compatible**: Existing functionality unchanged
- **Opt-in Feature**: JSONL logging enabled by default but can be disabled
- **No Breaking Changes**: All existing APIs remain the same

### For New Users
- **Default Behavior**: JSONL logging enabled automatically
- **Immediate Benefits**: Get detailed training logs without configuration
- **Future-Proof**: Compatible with all RLDK analysis tools

## Conclusion

This implementation provides a robust, research-grade solution for TRL training log emission. The standardized JSONL events enable seamless downstream analysis while maintaining compatibility with existing tools and workflows. The implementation follows best practices for data quality, performance, and extensibility, making it suitable for production research environments.