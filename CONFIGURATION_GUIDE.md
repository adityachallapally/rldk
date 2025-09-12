# Configuration Guide for RL Debug Kit Bug Fixes

## Overview
This guide explains how to configure the bug fixes and new features in RL Debug Kit. All settings use Pydantic validation to ensure type safety and reasonable bounds.

## Memory Management Settings

### Parameter Size Limits
```python
from src.rldk.config.settings import ExtendedConfigSchema

settings = ExtendedConfigSchema()
settings.memory.max_parameter_size_mb = 100  # Increase from default 50MB
```

**Purpose**: Controls the maximum size of individual model parameters that will be stored for drift calculation.

**Default**: 50MB
**Range**: 1-1000MB
**Impact**: Larger values allow monitoring of bigger models but increase memory usage.

### Total Memory Limits
```python
settings.memory.max_total_memory_mb = 1000  # Increase from default 500MB
```

**Purpose**: Controls the total memory usage for parameter storage across all parameters.

**Default**: 500MB
**Range**: 1-10000MB
**Impact**: Higher values allow monitoring of larger models but risk OOM errors.

### Cleanup Threshold
```python
settings.memory.cleanup_threshold_mb = 200  # Increase from default 100MB
```

**Purpose**: Memory threshold at which cleanup operations are triggered.

**Default**: 100MB
**Range**: 1-1000MB
**Impact**: Lower values trigger cleanup more frequently, reducing memory usage but increasing CPU overhead.

## File Operation Settings

### File Size Limits
```python
settings.file.max_file_size_mb = 10  # Increase from default 5MB
```

**Purpose**: Maximum size of log files that will be processed for determinism checks.

**Default**: 5MB
**Range**: 1-100MB
**Impact**: Larger values allow processing of bigger log files but increase memory usage and processing time.

**Note**: Files larger than this limit will be skipped with a warning message.

### Read Size Limits
```python
settings.file.max_read_size_kb = 1024  # Increase from default 512KB
```

**Purpose**: Maximum number of bytes to read from each log file.

**Default**: 512KB
**Range**: 1-10240KB
**Impact**: Larger values allow more thorough log analysis but increase memory usage.

**Note**: Only the first N bytes of each file are read to avoid memory issues.

### File Count Limits
```python
settings.file.max_files_to_process = 100  # Increase from default 50
```

**Purpose**: Maximum number of log files to process in a single operation.

**Default**: 50
**Range**: 1-1000
**Impact**: Higher values allow processing more files but increase processing time.

**Note**: If more files are found, only the first N will be processed.

### File Encoding
```python
settings.file.encoding = "utf-8"  # Default encoding
```

**Purpose**: Text encoding for reading log files.

**Default**: "utf-8"
**Options**: Any valid Python encoding (utf-8, latin-1, cp1252, etc.)
**Impact**: Incorrect encoding may cause read errors, but the system handles this gracefully.

## Statistical Settings

### Confidence Level
```python
settings.statistical.default_confidence_level = 0.99  # Increase from default 0.95
```

**Purpose**: Default confidence level for statistical calculations.

**Default**: 0.95
**Range**: 0.5-0.99
**Impact**: Higher values give wider confidence intervals but higher confidence in results.

### Conservative Standard Deviation
```python
settings.statistical.conservative_std_estimate = 0.2  # Decrease from default 0.3
```

**Purpose**: Conservative estimate for standard deviation when actual data is not available.

**Default**: 0.3
**Range**: 0.0-1.0
**Impact**: Lower values give narrower confidence intervals but may be less conservative.

### Binomial Threshold
```python
settings.statistical.binomial_threshold = 0.05  # Decrease from default 0.1
```

**Purpose**: Threshold for applying binomial distribution assumptions.

**Default**: 0.1
**Range**: 0.0-1.0
**Impact**: Lower values apply binomial assumptions more broadly.

## Feature Flags

### Enable New Statistical Methods
```python
settings.bug_fixes.use_new_confidence_intervals = True
settings.bug_fixes.use_new_effect_sizes = True
settings.bug_fixes.use_new_calibration = True
```

**Purpose**: Enable new statistical methods (disabled by default for safety).

**Default**: False
**Impact**: Enables more accurate but potentially less tested methods.

### Enable Bug Fixes
```python
settings.bug_fixes.enable_data_driven_stats = True
settings.bug_fixes.enable_memory_management = True
settings.bug_fixes.enable_file_safety = True
settings.bug_fixes.enable_legacy_methods = True
```

**Purpose**: Enable/disable specific bug fixes.

**Default**: True
**Impact**: Allows fine-grained control over which fixes are active.

## Environment Variables

You can override settings using environment variables:

```bash
export RLDK_MAX_PARAMETER_SIZE_MB=100
export RLDK_MAX_FILE_SIZE_MB=10
export RLDK_USE_NEW_STATS=true
```

Then load settings:
```python
from src.rldk.config.settings import load_settings_from_env
settings = load_settings_from_env()
```

## Configuration Validation

All settings are validated using Pydantic:

```python
# This will raise a ValueError
settings.memory.max_parameter_size_mb = -1  # Must be > 0

# This will raise a ValueError
settings.file.encoding = "invalid-encoding"  # Must be valid encoding

# This will raise a ValueError
settings.statistical.default_confidence_level = 1.5  # Must be < 1
```

## Best Practices

### For Large Models
```python
settings.memory.max_parameter_size_mb = 200
settings.memory.max_total_memory_mb = 2000
settings.memory.cleanup_threshold_mb = 500
```

### For Large Log Files
```python
settings.file.max_file_size_mb = 20
settings.file.max_read_size_kb = 2048
settings.file.max_files_to_process = 200
```

### For High Precision
```python
settings.statistical.default_confidence_level = 0.99
settings.statistical.conservative_std_estimate = 0.1
```

### For Conservative Operation
```python
settings.statistical.default_confidence_level = 0.90
settings.statistical.conservative_std_estimate = 0.5
settings.bug_fixes.use_new_confidence_intervals = False
```

## Troubleshooting

### Memory Issues
- Reduce `max_parameter_size_mb` and `max_total_memory_mb`
- Increase `cleanup_threshold_mb` to trigger cleanup more frequently
- Enable `enable_memory_management`

### File Processing Issues
- Reduce `max_file_size_mb` and `max_read_size_kb`
- Reduce `max_files_to_process`
- Check file permissions and encoding

### Statistical Issues
- Use `enable_legacy_methods` for backward compatibility
- Adjust `conservative_std_estimate` based on your data
- Enable `use_new_confidence_intervals` for better accuracy

## Migration from Old Settings

If you were using hardcoded values before:

```python
# Old hardcoded approach
max_file_size = 5 * 1024 * 1024  # 5MB

# New configurable approach
settings = ExtendedConfigSchema()
max_file_size = settings.file.max_file_size_mb * 1024 * 1024
```

This provides the same functionality but with validation and flexibility.