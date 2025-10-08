# IO Module Consolidation Migration Guide

This guide explains the changes made to consolidate the IO module and how to migrate existing code.

## Overview

The IO module has been consolidated to address the following issues:
- Multiple writer functions scattered across modules
- Duplicated `write_json` functions in different files
- Schema validation spread across multiple files
- Inconsistent file naming conventions

## New Structure

### Consolidated Modules

1. **`consolidated_schemas.py`** - All schema definitions in one place
2. **`consolidated_readers.py`** - All reading functionality
3. **`consolidated_writers.py`** - All writing functionality
4. **`unified_writer.py`** - Unified writer interface with consistent error handling
5. **`naming_conventions.py`** - Standardized file naming conventions
6. **`validator.py`** - Schema validation utilities (unchanged)

### Deprecated Modules

The following modules are now deprecated and will be removed in a future version:
- `schema.py` (moved to `consolidated_schemas.py`)
- `event_schema.py` (moved to `consolidated_schemas.py`)
- `writers.py` (moved to `consolidated_writers.py`)
- `readers.py` (moved to `consolidated_readers.py`)
- `reward_writers.py` (moved to `consolidated_writers.py`)

## Migration Guide

### 1. Import Changes

**Old imports:**
```python
from rldk.io.schema import TrainingMetrics, MetricsSchema
from rldk.io.writers import write_json, write_png, mkdir_reports
from rldk.io.readers import read_metrics_jsonl, write_metrics_jsonl
from rldk.io.reward_writers import generate_reward_health_report
from rldk.io.schemas import validate, DeterminismCardV1
```

**New imports:**
```python
from rldk.io import (
    TrainingMetrics, MetricsSchema,
    write_json, write_png, mkdir_reports,
    read_metrics_jsonl, write_metrics_jsonl,
    generate_reward_health_report,
    validate, DeterminismCardV1
)
```

### 2. Using the Unified Writer

**Old approach:**
```python
from rldk.io.writers import write_json

write_json(data, "output/report.json")
```

**New approach (recommended):**
```python
from rldk.io import UnifiedWriter

writer = UnifiedWriter("output")
writer.write_json(data, "report.json")
```

**Benefits of UnifiedWriter:**
- Consistent error handling
- Automatic directory creation
- Schema validation support
- Better logging
- Standardized file naming

### 3. File Naming Conventions

**Old approach:**
```python
# Inconsistent naming
write_json(data, "drift_card.json")
write_json(data, "determinism_report.json")
write_json(data, "reward_analysis.json")
```

**New approach:**
```python
from rldk.io import FileNamingConventions

# Standardized naming
filename = FileNamingConventions.get_filename("drift_card", "json")
writer.write_json(data, filename)

# Or use convenience functions
from rldk.io import get_drift_card_filename
writer.write_json(data, get_drift_card_filename())
```

### 4. Schema Validation

**Old approach:**
```python
from rldk.io.schemas import validate, DeterminismCardV1

validate(DeterminismCardV1, data)
```

**New approach:**
```python
from rldk.io import validate, DeterminismCardV1

validate(DeterminismCardV1, data)

# Or with UnifiedWriter
writer = UnifiedWriter("output")
writer.write_json(data, "determinism_card.json", validate_schema=lambda x: validate(DeterminismCardV1, x))
```

### 5. Error Handling

**Old approach:**
```python
try:
    write_json(data, "output.json")
except Exception as e:
    print(f"Error: {e}")
```

**New approach:**
```python
from rldk.io import UnifiedWriter, FileWriteError, SchemaValidationError

writer = UnifiedWriter("output")
try:
    writer.write_json(data, "output.json")
except FileWriteError as e:
    print(f"File write error: {e}")
except SchemaValidationError as e:
    print(f"Schema validation error: {e}")
```

## Backward Compatibility

All existing functions are still available through the main `rldk.io` module for backward compatibility. However, we recommend migrating to the new unified approach for better error handling and consistency.

## New Features

### 1. Unified Writer Class

The `UnifiedWriter` class provides:
- Consistent error handling across all file types
- Automatic directory creation
- Schema validation support
- Better logging and error messages
- Support for multiple file formats (JSON, JSONL, CSV, Markdown, PNG)

### 2. Standardized File Naming

The `FileNamingConventions` class provides:
- Consistent naming across all report types
- Support for timestamps and run IDs
- Validation of filename formats
- Convenience functions for common use cases

### 3. Consolidated Schemas

All schemas are now in one place:
- Pydantic schemas for data validation
- JSON schemas for report validation
- Artifact schemas for all output types
- Event schemas for normalized data

### 4. Enhanced Error Handling

New exception types:
- `RLDebugKitIOError` - Base exception for RL Debug Kit IO operations
- `FileWriteError` - File writing failures
- `SchemaValidationError` - Schema validation failures

**Note:** We use `RLDebugKitIOError` instead of `IOError` to avoid conflicts with Python's built-in `IOError` (which is an alias for `OSError`).

## Testing

To test the migration:

```python
# Test basic functionality
from rldk.io import UnifiedWriter, TrainingMetrics

writer = UnifiedWriter("test_output")
data = {"test": "data"}
writer.write_json(data, "test.json")

# Test schema validation
metrics = TrainingMetrics(step=1, reward_mean=0.5)
print(metrics.model_dump())
```

## Performance Considerations

The consolidated modules may have slightly higher import overhead due to the larger number of imports. However, this is offset by:
- Reduced code duplication
- Better error handling
- Consistent behavior across all IO operations
- Easier maintenance and testing

## Future Plans

1. Remove deprecated modules in next major version
2. Add more file format support (YAML, TOML, etc.)
3. Add compression support for large files
4. Add streaming support for very large datasets
5. Add async IO support for better performance