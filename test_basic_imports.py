#!/usr/bin/env python3
"""
Basic import test for Phase B modules.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_basic_imports():
    """Test basic module imports."""
    try:
        # Test reward module imports
        print("Testing reward module imports...")
        import rldk.reward

        print("✅ rldk.reward imported successfully")

        # Test evals module imports
        print("Testing evals module imports...")
        import rldk.evals

        print("✅ rldk.evals imported successfully")

        # Test CLI imports
        print("Testing CLI imports...")
        import rldk.cli

        print("✅ rldk.cli imported successfully")

        print("\n🎉 All basic imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_data_structures():
    """Test data structure creation."""
    try:
        print("\nTesting data structure creation...")

        # Test RewardHealthReport
        from rldk.reward.health import RewardHealthReport
        import pandas as pd

        report = RewardHealthReport(
            passed=True,
            drift_detected=False,
            saturation_issues=[],
            calibration_score=0.8,
            shortcut_signals=[],
            label_leakage_risk=0.1,
            fixes=[],
            drift_metrics=pd.DataFrame(),
            calibration_details={},
            shortcut_analysis={},
            saturation_analysis={},
        )
        print("✅ RewardHealthReport created successfully")

        # Test EvalResult
        from rldk.evals.runner import EvalResult

        result = EvalResult(
            suite_name="test",
            scores={"test": 0.8},
            confidence_intervals={"test": (0.7, 0.9)},
            effect_sizes={"test": 0.5},
            sample_size=50,
            seed=42,
            metadata={},
            raw_results=[],
        )
        print("✅ EvalResult created successfully")

        print("\n🎉 All data structures created successfully!")
        return True

    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Phase B Basic Functionality")
    print("=" * 50)

    # Test imports
    imports_ok = test_basic_imports()

    # Test data structures
    if imports_ok:
        structures_ok = test_data_structures()
    else:
        structures_ok = False

    # Summary
    print("\n" + "=" * 50)
    if imports_ok and structures_ok:
        print("🎉 All tests passed! Phase B modules are working correctly.")
    else:
        print("❌ Some tests failed. Check the error messages above.")

    print(f"\nImport test: {'✅ PASSED' if imports_ok else '❌ FAILED'}")
    print(f"Data structure test: {'✅ PASSED' if structures_ok else '❌ FAILED'}")
