# Bug Fix: Inconsistent Negative Denominator Handling

## Problem Description

The `try_divide` function was handling negative denominators differently than `safe_percentage` and `safe_rate_calculation`. While `try_divide` was designed to skip negative denominators (returning the fallback value), the existing `safe_percentage` and `safe_rate_calculation` functions were only checking for zero denominators and proceeding with normal division for negative values.

This inconsistency created a bug where:
- `try_divide(10, -2)` would return `0.0` (fallback)
- `safe_percentage(10, -2)` would return `-500.0` (actual division result)
- `safe_rate_calculation(10, -2)` would return `-5.0` (actual division result)

## Root Cause

The `safe_divide` function in `src/rldk/utils/error_handling.py` was only checking for zero denominators (`denominator == 0`) but not negative denominators. Since `safe_percentage` and `safe_rate_calculation` both depend on `safe_divide`, they inherited this inconsistent behavior.

## Solution

### 1. Updated `safe_divide` function
Changed the condition from `denominator == 0` to `denominator <= 0` to skip both zero and negative denominators.

```python
# Before
if denominator == 0:
    return fallback

# After  
if denominator <= 0:
    return fallback
```

### 2. Updated function docstrings
Updated the docstrings for `safe_divide`, `safe_rate_calculation`, `safe_percentage`, and `safe_ratio` to reflect that they now avoid both zero and negative denominators.

### 3. Created `try_divide` function
Created a new `try_divide` function in `src/rldk/utils/math_utils.py` that is consistent with the updated behavior.

### 4. Updated test cases
Updated existing test cases in `tests/unit/test_division_by_zero_fixes.py` to expect fallback values for negative denominators instead of division results.

### 5. Added comprehensive tests
Created `tests/test_division_by_zero_handling.py` with comprehensive tests to verify consistent behavior across all division functions.

## Files Modified

1. **`src/rldk/utils/error_handling.py`**
   - Updated `safe_divide` to skip negative denominators
   - Updated docstrings for all division functions

2. **`src/rldk/utils/math_utils.py`** (new file)
   - Added `try_divide` function with consistent behavior
   - Added additional utility functions

3. **`tests/test_division_by_zero_handling.py`** (new file)
   - Comprehensive tests for division function consistency
   - Tests specifically for negative denominator handling

4. **`tests/unit/test_division_by_zero_fixes.py`**
   - Updated existing test cases to reflect new behavior

## Verification

All division functions now behave consistently:

```python
# All functions now return 0.0 for negative denominators
try_divide(10, -2)           # 0.0
safe_percentage(10, -2)      # 0.0  
safe_rate_calculation(10, -2) # 0.0

# All functions still return 0.0 for zero denominators
try_divide(10, 0)            # 0.0
safe_percentage(10, 0)       # 0.0
safe_rate_calculation(10, 0)  # 0.0

# All functions perform normal division for positive denominators
try_divide(10, 2)            # 5.0
safe_percentage(10, 2)       # 500.0
safe_rate_calculation(10, 2)  # 5.0
```

## Breaking Changes

This is a **breaking change** for any code that was relying on the previous behavior where `safe_percentage` and `safe_rate_calculation` would perform division with negative denominators. However, this change makes the behavior consistent and more predictable across all division functions.

## Alternative Implementation

For cases where negative denominators should be allowed, an alternative function `safe_divide_with_negative_support` is provided in `math_utils.py` that only checks for zero denominators.