# CLI Refactoring Summary

## Overview
Successfully refactored the large CLI file (1700+ lines) into a modular structure organized by domain.

## New Structure

```
src/rldk/
├── cli_main.py          # Main CLI entry point (renamed from cli.py)
└── cli/
    ├── __init__.py      # Exports main app
    ├── main.py          # Main app and sub-app registration
    ├── forensics.py     # Forensics commands (compare-runs, diff-ckpt, env-audit, log-scan, doctor)
    ├── reward.py        # Reward commands (reward-drift, reward-health)
    ├── evals.py         # Evaluation commands (evaluate, list-suites, validate-data)
    ├── tracking.py      # Tracking commands (track)
    └── main_commands.py # Main CLI commands (ingest, diff, check-determinism, bisect, etc.)
```

## Key Changes

### 1. Modular Organization
- **Forensics commands**: Moved to `forensics.py` with sub-app `forensics_app`
- **Reward commands**: Moved to `reward.py` with sub-app `reward_app` and `reward_health_app`
- **Evaluation commands**: Moved to `evals.py` with sub-app `evals_app`
- **Tracking commands**: Moved to `tracking.py` with sub-app `tracking_app`
- **Main commands**: Kept in `main_commands.py` for top-level commands

### 2. Sub-App Structure
- Each domain has its own Typer sub-app
- Sub-apps are registered in `main.py`
- Commands are organized logically by functionality

### 3. Backward Compatibility
- All original commands remain available at the top level
- Legacy command aliases preserved
- No breaking changes to command interfaces

### 4. Clean Separation
- Each module focuses on a specific domain
- Reduced file size from 1700+ lines to manageable modules
- Easier maintenance and testing

## Command Organization

### Forensics Commands (`rldk forensics`)
- `compare-runs` - Compare two training runs
- `diff-ckpt` - Compare model checkpoints
- `env-audit` - Audit environment for determinism
- `log-scan` - Scan training logs for anomalies
- `doctor` - Run comprehensive diagnostics

### Reward Commands (`rldk reward`)
- `reward-drift` - Compare reward models for drift
- `reward-health run` - Run reward health analysis
- `reward-health gate` - Gate CI based on health results

### Evaluation Commands (`rldk evals`)
- `evaluate` - Run evaluation suite on data
- `list-suites` - List available evaluation suites
- `validate-data` - Validate JSONL file structure

### Tracking Commands (`rldk tracking`)
- `track` - Start experiment tracking

### Main Commands (top-level)
- `ingest` - Ingest training runs
- `diff` - Find divergence between runs
- `check-determinism` - Check command determinism
- `bisect` - Find regression using git bisect
- `reward-health` - Analyze reward model health
- `replay` - Replay training run
- `eval` - Run evaluation suite
- `track` - Start experiment tracking
- `version` - Show version information
- `card` - Generate trust cards

## Bug Fixes Applied

### 1. Circular Import Issue
- **Problem**: Naming conflict between `cli.py` and `cli/` directory caused circular imports
- **Solution**: Renamed main CLI file to `cli_main.py` and updated imports
- **Result**: Clean import structure without circular dependencies

### 2. Missing Configuration Initialization
- **Problem**: Forensics commands were missing `ensure_config_initialized()` calls
- **Solution**: Added configuration initialization to all forensics commands
- **Result**: Proper RLDK configuration setup for all forensics operations

## Benefits

1. **Maintainability**: Each module is focused and manageable
2. **Organization**: Commands grouped by logical domain
3. **Scalability**: Easy to add new commands to appropriate modules
4. **Testing**: Individual modules can be tested independently
5. **Documentation**: Clear separation makes documentation easier
6. **Backward Compatibility**: No breaking changes for existing users
7. **Robustness**: Proper configuration initialization and clean imports

## Usage

The CLI can be used exactly as before:

```bash
# Top-level commands (unchanged)
rldk ingest runs/
rldk diff --a run1 --b run2 --signals loss,reward
rldk check-determinism --cmd "python train.py"

# Sub-app commands (new organization)
rldk forensics compare-runs run1/ run2/
rldk reward reward-drift model1/ model2/ --prompts prompts.jsonl
rldk evals evaluate data.jsonl --suite comprehensive
rldk tracking track my_experiment
```

## File Size Reduction

- **Before**: Single file with 1700+ lines
- **After**: 7 focused modules with clear responsibilities
  - `main.py`: 50 lines (app setup)
  - `forensics.py`: 280 lines (forensics commands)
  - `reward.py`: 200 lines (reward commands)
  - `evals.py`: 300 lines (evaluation commands)
  - `tracking.py`: 80 lines (tracking commands)
  - `main_commands.py`: 900 lines (main commands)
  - `__init__.py`: 5 lines (exports)

Total: ~1815 lines across 7 files (better organized and maintainable)