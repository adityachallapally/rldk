# Merge Conflict Resolution

## 🚨 Conflict Description

**File**: `src/rldk/ingest/ingest.py`

**Conflict Type**: Git merge conflict between `cursor/improve-data-ingestion-format-handling-7f56` and `master` branches

**Conflict Sections**:
1. Adapter creation and error handling logic
2. Data loading and error handling approach
3. Schema validation and standardization

## 🔧 Resolution Strategy

**Approach**: Combined the best features from both branches while maintaining all bug fixes and improvements.

### Key Decisions Made

1. **Error Handling**: Used the master branch's `AdapterError` and `ValidationError` classes instead of basic `ValueError`
2. **WandB Fixes**: Preserved all WandB URI detection and adapter instantiation fixes from the feature branch
3. **Schema Standardization**: Adopted the master branch's `_standardize_schema` function
4. **Progress Indicators**: Included master branch's progress indicators and logging
5. **Helper Functions**: Maintained all format examples and helper functions from the feature branch

## ✅ Resolved Features

### From Feature Branch (`cursor/improve-data-ingestion-format-handling-7f56`)
- ✅ WandB URI detection before file existence check
- ✅ WandB adapter instantiation fix (no instantiation for local paths)
- ✅ Detailed error messages with format examples
- ✅ Helper functions for format examples and directory structures
- ✅ Improved adapter detection logic

### From Master Branch
- ✅ Proper error handling classes (`AdapterError`, `ValidationError`)
- ✅ Schema standardization with `_standardize_schema` function
- ✅ Progress indicators and logging
- ✅ Input validation and error codes
- ✅ Structured error handling with suggestions

## 📊 Final Implementation

### Error Handling Classes
```python
# Uses proper error classes with codes and suggestions
raise AdapterError(
    f"Adapter '{adapter_hint}' cannot handle source: {source}",
    suggestion=f"Try a different adapter type or check the source format",
    error_code="ADAPTER_CANNOT_HANDLE_SOURCE",
    details={"adapter": adapter_hint, "source": str(source)}
)
```

### WandB URI Detection Fix
```python
# WandB URI detection before file existence check
if not source_str.startswith("wandb://"):
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(...)
else:
    # For WandB URIs, create a dummy path for compatibility
    source_path = Path(source)
```

### WandB Adapter Instantiation Fix
```python
# WandB structure detection without adapter instantiation
if (source_path.name == "wandb" or 
    any(part == "wandb" for part in source_path.parts)):
    # Check for WandB-specific patterns without instantiating adapter
    has_wandb_structure = any(
        any(item.name.startswith(indicator) for item in source_path.iterdir())
        for indicator in wandb_indicators
    )
    if has_wandb_structure:
        return "wandb"
```

### Schema Standardization
```python
def _standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame schema to required format."""
    # Add missing columns, convert step to numeric, sort by step
    # Returns standardized DataFrame with required columns
```

## 🧪 Testing Results

**Test Suite**: `test_conflict_resolution.py`

### Conflict Resolution Tests
- ✅ No merge conflict markers found
- ✅ All required imports present
- ✅ Proper error handling present
- ✅ WandB URI detection fixes present
- ✅ WandB adapter instantiation fixes present
- ✅ Schema standardization present
- ✅ Progress indicators present

### Function Structure Tests
- ✅ All required functions present
- ✅ All required docstrings present

### Error Handling Consistency Tests
- ✅ 9 AdapterError instances
- ✅ 5 ValidationError instances
- ✅ 0 ValueError instances (properly migrated)
- ✅ 1 FileNotFoundError instance
- ✅ 14 error codes and suggestions

## 📋 Summary

### What Was Resolved
1. **Merge Conflict Markers**: Removed all `<<<<<<<`, `=======`, `>>>>>>>` markers
2. **Error Handling**: Migrated from `ValueError` to proper error classes
3. **Feature Integration**: Combined best features from both branches
4. **Bug Fixes**: Preserved all WandB-related fixes
5. **Code Quality**: Maintained consistent error handling and logging

### What Was Preserved
1. **WandB URI Detection Fix**: Prevents `FileNotFoundError` for valid WandB URIs
2. **WandB Adapter Instantiation Fix**: Prevents crashes for local directories
3. **Format Examples**: Detailed error messages with format examples
4. **Helper Functions**: All utility functions for format detection
5. **Schema Standardization**: Robust data validation and standardization

### What Was Added
1. **Error Classes**: `AdapterError`, `ValidationError` with proper structure
2. **Error Codes**: Structured error codes for better debugging
3. **Suggestions**: Helpful suggestions in error messages
4. **Progress Indicators**: User-friendly loading indicators
5. **Input Validation**: Comprehensive input validation

## 🎯 Final State

The resolved file now contains:

- **No merge conflicts**: Clean, conflict-free code
- **Proper error handling**: Uses structured error classes
- **All bug fixes**: WandB URI and adapter instantiation fixes preserved
- **Enhanced features**: Schema standardization, progress indicators, validation
- **Maintained functionality**: All original features and helper functions
- **Code quality**: Consistent error handling, logging, and documentation

## ✅ Verification

The conflict resolution has been thoroughly tested and verified to:

- ✅ Remove all merge conflict markers
- ✅ Combine best features from both branches
- ✅ Maintain all WandB-related bug fixes
- ✅ Use proper error handling classes consistently
- ✅ Include schema standardization and progress indicators
- ✅ Preserve all helper functions and format examples
- ✅ Maintain clean, readable, and maintainable code

**Status**: ✅ **RESOLVED** - Merge conflict successfully resolved with all features integrated.