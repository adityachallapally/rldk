# Import Fix Summary - ingest_runs_to_events

## 🐛 Bug Identified

**Issue**: The `ingest_runs_to_events` function was missing from the `rldk.ingest` module exports, causing an ImportError when CLI card generation commands were run.

**Root Cause**: During the package restructuring, I updated the `src/rldk/ingest/__init__.py` file to only export `ingest_runs` but forgot to include `ingest_runs_to_events`, even though the CLI was importing and using it.

**Impact**: 
- CLI card commands (`rldk card determinism`, `rldk card drift`, `rldk card reward`) would fail with ImportError
- Function exists in source code but not accessible through module imports

## ✅ Fix Applied

### 1. Updated `src/rldk/ingest/__init__.py`

**Before:**
```python
from .ingest import ingest_runs

__all__ = [
    "ingest_runs",
    # ... other exports
]
```

**After:**
```python
from .ingest import ingest_runs, ingest_runs_to_events

__all__ = [
    "ingest_runs",
    "ingest_runs_to_events",
    # ... other exports
]
```

### 2. Updated `src/rldk/__init__.py`

**Before:**
```python
from .ingest import ingest_runs

__all__ = [
    "ingest_runs",
    # ... other exports
]
```

**After:**
```python
from .ingest import ingest_runs, ingest_runs_to_events

__all__ = [
    "ingest_runs",
    "ingest_runs_to_events",
    # ... other exports
]
```

## 🧪 Verification

All tests pass (4/4):

1. **Ingest module exports** ✅
   - `ingest_runs_to_events` imported from `.ingest`
   - Function included in `__all__` list

2. **Main package exports** ✅
   - `ingest_runs_to_events` imported from `.ingest`
   - Function included in `__all__` list

3. **CLI imports** ✅
   - CLI imports `ingest_runs_to_events` correctly
   - Function used in all card generation commands

4. **Function exists** ✅
   - `ingest_runs_to_events` function exists in `ingest.py`

## 📍 Usage in CLI

The function is used in all three card generation commands:

```python
# Determinism card
events = ingest_runs_to_events(run_a)

# Drift card  
events_a = ingest_runs_to_events(run_a)
events_b = ingest_runs_to_events(run_b)

# Reward card
events = ingest_runs_to_events(run_a)
```

## ✅ Status

**FIXED**: The missing `ingest_runs_to_events` export has been resolved. All CLI card generation commands should now work correctly without ImportError.

**Files Modified:**
- `src/rldk/ingest/__init__.py` - Added missing export
- `src/rldk/__init__.py` - Added to public API

**No Breaking Changes**: This is purely an additive fix that restores missing functionality.