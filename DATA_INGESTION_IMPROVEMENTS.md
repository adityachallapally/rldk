# Data Ingestion Format Requirements - Improvements

## Overview

This document outlines the comprehensive improvements made to the RL Debug Kit (RLDK) data ingestion system to address format requirements and provide better error handling.

## Problem Statement

The original data ingestion system had several issues:

1. **Poor Error Messages**: Generic "Cannot handle source" errors without context
2. **No Format Documentation**: Users didn't know what formats were supported
3. **Rigid Adapter Detection**: Limited flexibility in detecting different data formats
4. **Missing Validation**: No input validation before processing
5. **No Examples**: No sample data or format examples provided

## Solutions Implemented

### 1. Enhanced Error Messages ✅

**Before:**
```
ERROR:root:Failed to ingest /workspace/forensics_test_output: Cannot handle source: /workspace/forensics_test_output
```

**After:**
```
ValueError: Cannot handle trl format for directory: /workspace/forensics_test_output
Expected directory structure for trl:
TRL directory structure:
  training_logs/
    ├── trainer_log.jsonl
    ├── training.log
    └── *_events.jsonl

Supported extensions: .jsonl, .log
```

### 2. Comprehensive Format Examples ✅

Added detailed format examples for each adapter:

#### TRL Format
```json
{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}
{"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.09, "entropy_mean": 1.9, "loss": 0.25}
```

#### OpenRLHF Format
```json
{"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}
{"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.09, "entropy_mean": 1.9, "loss": 0.25}
```

#### Custom JSONL Format
```json
{"global_step": 0, "reward_scalar": 0.5, "kl_to_ref": 0.1, "entropy": 2.0, "loss": 0.3}
{"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.09, "entropy": 1.9, "loss": 0.25}
```

#### WandB Format
```
wandb://project_name/run_id
wandb://username/project_name/run_id
./wandb/run-20240101_120000-abc123/
```

### 3. Input Data Validation ✅

Added comprehensive validation:

- **File Existence Check**: Validates source path exists before processing
- **Adapter Capability Check**: Verifies adapter can handle the source format
- **Format Detection**: Improved auto-detection with fallback strategies
- **Error Context**: Provides specific error messages based on file type and adapter

### 4. Improved Adapter Detection ✅

Enhanced detection logic:

1. **WandB URI Detection**: Checks for `wandb://` prefix first (before file existence check)
2. **WandB Directory Detection**: Looks for wandb directory structure
3. **Custom JSONL Detection**: Most specific format detection
4. **TRL/OpenRLHF Detection**: Standard format detection
5. **Fallback Strategy**: Defaults to appropriate adapter based on file extension

**Bug Fix**: WandB URI detection now happens before file existence check, preventing valid URIs like `wandb://project_name/run_id` from being incorrectly rejected as non-existent files.

### 5. Sample Data Files ✅

Created comprehensive sample data for testing:

```
sample_data/
├── trl_training_output.jsonl          # TRL format example
├── openrlhf_training_output.jsonl     # OpenRLHF format example
├── custom_training_output.jsonl       # Custom JSONL format example
├── sample_eval_data.jsonl             # Evaluation data example
├── forensics_test_output/
│   └── trainer_log.jsonl              # TRL directory structure
└── rl_training_output/
    └── training.log                   # Log format example
```

### 6. Directory Structure Examples ✅

Provided clear directory structure expectations:

#### TRL Directory Structure
```
training_logs/
├── trainer_log.jsonl
├── training.log
└── *_events.jsonl
```

#### OpenRLHF Directory Structure
```
training_logs/
├── training.log
├── metrics.jsonl
└── logs/
```

#### Custom JSONL Directory Structure
```
data/
├── metrics.jsonl
├── training_data.jsonl
└── *.jsonl files
```

#### WandB Directory Structure
```
wandb/
└── run-20240101_120000-abc123/
    ├── files/
    ├── logs/
    └── config.yaml
```

## Code Changes

### 1. Enhanced `ingest_runs()` Function

**File**: `src/rldk/ingest/ingest.py`

- Added file existence validation (with WandB URI exception)
- Improved error messages with format examples
- Better adapter creation error handling
- Enhanced validation before processing
- **Bug Fix**: WandB URI detection before file existence check

### 2. New Helper Functions

**File**: `src/rldk/ingest/ingest.py`

- `_get_format_examples(adapter_type)`: Returns format examples for each adapter
- `_get_supported_extensions(adapter_type)`: Returns supported file extensions
- `_get_directory_structure_examples(adapter_type)`: Returns directory structure examples

### 3. Improved Adapter Detection

**File**: `src/rldk/ingest/ingest.py`

- Enhanced `_detect_adapter_type()` function
- Better WandB detection
- Improved fallback strategies
- More robust JSONL file handling

## Testing

### Test Results ✅

All tests pass successfully:

```
📊 Test Results: 4/4 tests passed
🎉 All tests passed! The improved ingest system components are working correctly.

📋 Summary of improvements:
✅ Better error messages with format examples
✅ Input data validation
✅ Sample data files created
✅ Directory structure examples
✅ Adapter detection improvements
```

### Test Coverage

1. **Format Examples Test**: Validates all format examples are properly formatted
2. **Sample Data Creation Test**: Ensures all sample files are created and valid
3. **Error Message Structure Test**: Verifies error messages include helpful context
4. **Directory Structure Test**: Confirms directory structure examples are complete

## Usage Examples

### Before (Original Commands)
```bash
# These would fail with generic errors
rldk diff --a /workspace/forensics_test_output --b /workspace/rl_training_output --signals "loss,reward_mean,kl"
rldk ingest /workspace/sample_eval_data.jsonl --adapter trl --output /workspace/ingested_metrics.jsonl
rldk card determinism /workspace/rl_training_output
```

### After (Improved Commands)
```bash
# These now work with proper error messages and format detection
rldk diff --a /workspace/sample_data/forensics_test_output --b /workspace/sample_data/rl_training_output --signals "loss,reward_mean,kl"
rldk ingest /workspace/sample_data/sample_eval_data.jsonl --adapter custom_jsonl --output /workspace/ingested_metrics.jsonl
rldk card determinism /workspace/sample_data/forensics_test_output
```

## Error Message Examples

### File Not Found
```
FileNotFoundError: Source path does not exist: /workspace/nonexistent_file.jsonl
Please check the path and ensure the file or directory exists.
```

### Wrong Adapter Type
```
ValueError: Cannot handle trl format for file: /workspace/sample_data/custom_training_output.jsonl
Expected format for trl:
TRL format examples:
  JSONL format:
    {"step": 0, "phase": "train", "reward_mean": 0.5, "kl_mean": 0.1, "entropy_mean": 2.0, "loss": 0.3}
    {"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.09, "entropy_mean": 1.9, "loss": 0.25}
  
  Log format:
    step: 0, reward: 0.5, kl: 0.1, entropy: 2.0, loss: 0.3
    step: 1, reward: 0.6, kl: 0.09, entropy: 1.9, loss: 0.25

Try using --adapter custom_jsonl for generic JSONL files.
```

### Unsupported File Extension
```
ValueError: Cannot handle trl format for file: /workspace/data.txt
File extension '.txt' is not supported by trl adapter.
Supported extensions: .jsonl, .log
```

## Bug Fixes

### WandB URI Detection Bug ✅

**Problem**: The file existence check was running before WandB URI detection, causing valid WandB URIs like `wandb://project_name/run_id` to fail with `FileNotFoundError` because they're not local filesystem paths.

**Solution**: Moved WandB URI detection before the file existence check, so WandB URIs are properly identified and skip the filesystem validation.

**Before**:
```python
# File existence check first
if not source_path.exists():
    raise FileNotFoundError(...)

# WandB detection after (too late!)
if source_str.startswith("wandb://"):
    adapter_hint = "wandb"
```

**After**:
```python
# WandB detection first
if source_str.startswith("wandb://"):
    adapter_hint = "wandb"

# File existence check only for non-WandB URIs
if not source_str.startswith("wandb://"):
    if not source_path.exists():
        raise FileNotFoundError(...)
```

## Benefits

1. **Better User Experience**: Clear, actionable error messages
2. **Reduced Support Burden**: Users can self-diagnose format issues
3. **Improved Debugging**: Detailed context for troubleshooting
4. **Format Documentation**: Built-in examples and documentation
5. **Flexible Detection**: Better handling of various input formats
6. **Comprehensive Testing**: Sample data for all supported formats
7. **WandB URI Support**: Fixed bug preventing WandB URI usage

## Future Improvements

1. **Interactive Format Detection**: CLI tool to help users identify their data format
2. **Format Conversion**: Tools to convert between different formats
3. **Schema Validation**: More detailed schema validation with specific field requirements
4. **Performance Optimization**: Faster format detection for large files
5. **Extended Format Support**: Support for additional training frameworks

## Conclusion

The data ingestion system has been significantly improved with:

- ✅ **Better Error Messages**: Clear, actionable error messages with format examples
- ✅ **Input Validation**: Comprehensive validation before processing
- ✅ **Format Examples**: Detailed examples for each supported format
- ✅ **Flexible Adapters**: More robust adapter detection and handling
- ✅ **Sample Data**: Complete set of sample data files for testing
- ✅ **Documentation**: Comprehensive directory structure examples

These improvements address all the original issues and provide a much better user experience for data ingestion in the RL Debug Kit.