# PR #15 Test Plan: write_drift_card Functionality

## What Changed in PR #15

PR #15 added the `write_drift_card` function to the RL Debug Kit with the following changes:

1. **New Function Added**: `write_drift_card()` was added to `src/rldk/io/writers.py`
2. **Import Path**: The function is re-exported in `src/rldk/io/__init__.py` for easy importing
3. **Dual Output**: The function writes both:
   - `drift_card.json` - Machine-readable JSON format
   - `drift_card.md` - Human-readable markdown format

## Function Signature

```python
def write_drift_card(drift_data: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """Write drift card to both JSON and markdown formats.
    
    Args:
        drift_data: Dictionary containing drift analysis data with keys like:
            - diverged: bool indicating if divergence was detected
            - first_step: int step where divergence first occurred
            - tripped_signals: list of signals that triggered
            - signals_monitored: list of all signals monitored
            - k_consecutive: number of consecutive violations required
            - window_size: rolling window size for analysis
            - tolerance: z-score threshold used
        output_dir: Directory to write the drift card files
    """
```

## Test Scripts

### 1. Comprehensive Test (`test_pr15_drift_card.sh`)

This script provides a complete test of the PR #15 functionality following the original test plan structure:

- **Environment Setup**: Creates virtual environment and installs dependencies
- **Fixture Generation**: Generates test artifacts using `tests/_make_fixtures.py`
- **Code Quality**: Runs linting, typechecking, and unit tests
- **CLI Testing**: Exercises all RL Debug Kit CLI commands
- **Artifact Verification**: Checks that all expected JSON and image files are created
- **PR #15 Specific Testing**: Explicitly tests the `write_drift_card` function
- **Negative Testing**: Verifies clean logs don't trigger false positives
- **Packaging Test**: Builds and installs the package to verify console entry points
- **Content Validation**: Verifies both JSON and markdown output have correct content

### 2. Minimal Test (`test_pr15_minimal.sh`)

A quick test focused specifically on the PR #15 changes:

- **Import Test**: Verifies `from rldk.io import write_drift_card` works
- **Function Test**: Calls `write_drift_card` with sample data
- **File Creation**: Verifies both `drift_card.json` and `drift_card.md` are created
- **Content Validation**: Checks that both files contain expected content

## Expected Output Files

After running the tests, the following files should exist in `./rldk_reports/`:

### Standard Artifacts
- `determinism_card.json` - Environment audit results
- `ppo_scan.json` - PPO log analysis results
- `ppo_kl.png` - KL divergence visualization
- `ppo_grad_ratio.png` - Gradient ratio visualization
- `ckpt_diff.json` - Checkpoint difference analysis
- `ckpt_top_movers.png` - Top parameter changes visualization
- `reward_drift.json` - Reward model drift analysis
- `reward_drift_scatter.png` - Reward drift scatter plot

### PR #15 Specific Artifacts
- `drift_card_test/drift_card.json` - Machine-readable drift card
- `drift_card_test/drift_card.md` - Human-readable drift card

## Content Validation

### drift_card.json Structure
```json
{
  "diverged": true,
  "first_step": 847,
  "tripped_signals": ["kl_spike", "controller_stuck"],
  "signals_monitored": ["kl", "kl_coef", "grad_ratio"],
  "tolerance": 3.0,
  "k_consecutive": 5,
  "window_size": 50,
  "output_path": "rldk_reports/drift_card_test"
}
```

### drift_card.md Content
The markdown file should contain:
- Title: "Drift Detection Card"
- Status: "🚨 Drift Detected" (when diverged)
- Tripped signals list
- Analysis parameters
- Report location information

## Running the Tests

### Quick Test (Recommended)
```bash
./test_pr15_minimal.sh
```

### Full Test Suite
```bash
./test_pr15_drift_card.sh
```

## Success Criteria

The tests are considered successful if:

1. ✅ `pytest` passes
2. ✅ All CLI commands exit with code 0
3. ✅ All expected artifacts are created in `./rldk_reports/`
4. ✅ `write_drift_card` imports cleanly from `rldk.io`
5. ✅ Both `drift_card.json` and `drift_card.md` are created
6. ✅ Both files contain the expected content structure
7. ✅ Clean logs don't trigger false positive spike/controller rules
8. ✅ Doctored logs correctly contain spike and controller rules
9. ✅ Identical checkpoints have `avg_cosine >= 0.9999`
10. ✅ `rldk --help` works from wheel install
11. ✅ Package imports work correctly after installation

## Manual Testing

You can also test the functionality manually:

```python
from pathlib import Path
from rldk.io import write_drift_card

# Test the function
write_drift_card(
    {
        "diverged": True,
        "first_step": 847,
        "tripped_signals": ["kl_spike", "controller_stuck"],
        "signals_monitored": ["kl", "kl_coef", "grad_ratio"],
        "tolerance": 3.0,
        "k_consecutive": 5,
        "window_size": 50,
        "output_path": "test_output"
    },
    "test_output"
)

# Verify files exist
assert Path("test_output/drift_card.json").exists()
assert Path("test_output/drift_card.md").exists()
```

## Troubleshooting

If tests fail:

1. **Import Errors**: Check that `write_drift_card` is properly exported in `src/rldk/io/__init__.py`
2. **Missing Files**: Ensure the function creates both JSON and markdown files
3. **Content Issues**: Verify the markdown generation logic in `_generate_drift_card_md()`
4. **Permission Issues**: Make sure test scripts are executable (`chmod +x test_*.sh`)
5. **Dependency Issues**: Ensure all dev dependencies are installed (`pip install -e ".[dev]"`)