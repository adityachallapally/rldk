# Determinism Dependency Warnings Implementation Summary

## Overview
Successfully implemented deterministic dependency warnings in `rldk/determinism.py` to replace silent ImportError handling with clear, single-line user warnings that list exactly which checks were skipped.

## Changes Made

### 1. Updated DeterminismReport Dataclass
- Added `skipped_checks: List[str]` field to track which determinism checks were skipped due to missing dependencies

### 2. Added Warning Logging Function
- Implemented `_log_determinism_warning(message: str)` that respects the `RLDK_SILENCE_DETERMINISM_WARN` environment variable
- When `RLDK_SILENCE_DETERMINISM_WARN=1`, warnings are suppressed
- When `RLDK_SILENCE_DETERMINISM_WARN=0` (default), warnings are displayed

### 3. Implemented Dependency Checks
- **PyTorch CUDA Kernels Check** (`_check_pytorch_cuda_kernels()`)
  - Checks if PyTorch is available and CUDA kernels work
  - Warning: "Determinism: Skipped PyTorch CUDA kernels check. Install torch>=2.0.0 to enable. Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress."
  - Returns `"pytorch_cuda_kernels"` in skipped_checks when missing

- **TensorFlow Determinism Check** (`_check_tensorflow_determinism()`)
  - Checks if TensorFlow has required determinism features (read-only check)
  - Warning: "Determinism: Skipped TensorFlow determinism check. Install tensorflow>=2.8.0 to enable. Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress."
  - Returns `"tensorflow_determinism"` in skipped_checks when missing
  - **Fixed**: No longer modifies global TensorFlow configuration during availability check

- **JAX Determinism Check** (`_check_jax_determinism()`)
  - Checks if JAX is available and has required config options (read-only check)
  - Warning: "Determinism: Skipped JAX determinism check. Install jax>=0.4.0 to enable. Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress."
  - Returns `"jax_determinism"` in skipped_checks when missing
  - **Fixed**: No longer modifies global JAX configuration during availability check

### 4. Updated Check Function
- Modified the `check()` function to run all dependency checks at the beginning
- Populates the `skipped_checks` field in the returned `DeterminismReport`
- Warnings are logged once per missing dependency

### 5. Created Comprehensive Unit Tests
- Created `tests/test_determinism_warnings.py` with full test coverage
- Tests warning suppression and display behavior
- Tests all three dependency checks with missing modules
- Tests multiple missing dependencies scenario
- Tests DeterminismReport structure

## Acceptance Criteria Met

✅ **When torch is absent, a warning logs once and report.skipped_checks contains "pytorch_cuda_kernels"**
- PyTorch check shows warning when module is missing
- `skipped_checks` field contains `"pytorch_cuda_kernels"`

✅ **With RLDK_SILENCE_DETERMINISM_WARN=1, no warning is printed, but skipped_checks still populated**
- Environment variable check implemented
- Warnings suppressed when flag is set
- `skipped_checks` still populated for reporting

✅ **Unit tests cover both paths**
- Comprehensive test suite created
- Tests both warning display and suppression paths
- Tests all three dependency checks

## Files Modified

1. **`src/rldk/determinism/check.py`**
   - Added `skipped_checks` field to `DeterminismReport`
   - Added `_log_determinism_warning()` function
   - Added `_check_pytorch_cuda_kernels()` function
   - Added `_check_tensorflow_determinism()` function
   - Added `_check_jax_determinism()` function
   - Updated `check()` function to run dependency checks

2. **`tests/test_determinism_warnings.py`**
   - Created comprehensive unit tests
   - Tests warning behavior and silence flag
   - Tests all dependency checks

## Usage Examples

### Basic Usage
```python
from rldk.determinism.check import check

# This will show warnings for missing dependencies
report = check("python train.py", ["loss"], replicas=2)

# Check which dependencies were missing
print("Skipped checks:", report.skipped_checks)
# Output: ['pytorch_cuda_kernels', 'tensorflow_determinism']
```

### Silent Mode
```bash
# Suppress warnings
export RLDK_SILENCE_DETERMINISM_WARN=1
python -c "from rldk.determinism.check import check; check('python train.py', ['loss'])"
```

### Warning Messages
When dependencies are missing, users will see clear warnings like:
```
UserWarning: Determinism: Skipped PyTorch CUDA kernels check. Install torch>=2.0.0 to enable. Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress.
UserWarning: Determinism: Skipped TensorFlow determinism check. Install tensorflow>=2.8.0 to enable. Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress.
UserWarning: Determinism: Skipped JAX determinism check. Install jax>=0.4.0 to enable. Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress.
```

## Bug Fixes

### Fixed: Configuration Modification During Availability Checks
- **Issue**: `_check_jax_determinism()` was calling `jax.config.update()` during availability check, modifying global JAX configuration
- **Issue**: `_check_tensorflow_determinism()` was calling `tf.config.experimental.enable_op_determinism()` during availability check, modifying global TensorFlow configuration
- **Fix**: Changed both functions to use read-only checks (`hasattr()`) instead of modifying global configuration
- **Impact**: Availability checks are now truly read-only and won't affect user code or subsequent operations

### Fixed: Determinism Checks Always Return True
- **Issue**: `_check_tensorflow_determinism()` and `_check_jax_determinism()` were calling `hasattr()` but ignoring the result, always returning `True`
- **Impact**: Functions incorrectly reported determinism support even when features were missing, leading to inaccurate `skipped_checks` in `DeterminismReport`
- **Fix**: Changed both functions to `return hasattr(...)` instead of just calling `hasattr(...)` and returning `True`
- **Result**: Functions now correctly return `False` when determinism features are missing and `True` when they exist

### Fixed: Import Mocking in Tests
- **Issue**: Tests were using global import patches that made ALL imports fail, not just the targeted module
- **Impact**: Tests incorrectly showed multiple warnings when only one dependency was supposed to be missing
- **Fix**: Changed tests to use conditional import mocking that only fails for specific module names
- **Result**: Tests now correctly verify individual dependency warnings without affecting other imports

## Implementation Notes

- All warnings are single-line and informative
- Environment variable `RLDK_SILENCE_DETERMINISM_WARN` controls warning display
- `skipped_checks` field is always populated for reporting purposes
- Dependency checks are run once at the beginning of the `check()` function
- Warning messages include version requirements and silence instructions
- Implementation is backward compatible with existing code
- **All availability checks are read-only and don't modify global configuration**