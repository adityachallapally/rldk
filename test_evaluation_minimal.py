#!/usr/bin/env python3
"""Minimal test script to verify the RLDK evaluation system fixes without external dependencies."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_evalresult_class():
    """Test that EvalResult class has the overall_score attribute."""
    print("Testing EvalResult class...")
    
    try:
        from rldk.evals.runner import EvalResult
        
        # Create a mock EvalResult instance
        result = EvalResult(
            suite_name="test",
            scores={"test_metric": 0.5, "another_metric": 0.7},
            confidence_intervals={"test_metric": (0.4, 0.6)},
            effect_sizes={"test_metric": 0.2},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )
        
        # Test that overall_score property exists and works
        overall_score = result.overall_score
        print(f"✓ overall_score property works: {overall_score}")
        
        # Test that it calculates the mean correctly
        expected_score = (0.5 + 0.7) / 2  # Mean of 0.5 and 0.7
        if abs(overall_score - expected_score) < 1e-10:
            print("✓ overall_score calculates mean correctly")
        else:
            print(f"✗ overall_score calculation incorrect: expected {expected_score}, got {overall_score}")
            return False
        
        # Test with NaN values
        result_with_nan = EvalResult(
            suite_name="test",
            scores={"test_metric": 0.5, "nan_metric": float('nan')},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )
        
        nan_score = result_with_nan.overall_score
        if nan_score == 0.5:  # Should ignore NaN values
            print("✓ overall_score handles NaN values correctly")
        else:
            print(f"✗ overall_score NaN handling incorrect: expected 0.5, got {nan_score}")
            return False
        
        # Test with all NaN values
        result_all_nan = EvalResult(
            suite_name="test",
            scores={"nan_metric1": float('nan'), "nan_metric2": float('nan')},
            confidence_intervals={},
            effect_sizes={},
            sample_size=100,
            seed=42,
            metadata={},
            raw_results=[]
        )
        
        all_nan_score = result_all_nan.overall_score
        if all_nan_score == 0.0:  # Should return 0.0 when all values are NaN
            print("✓ overall_score handles all NaN values correctly")
        else:
            print(f"✗ overall_score all NaN handling incorrect: expected 0.0, got {all_nan_score}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ EvalResult class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test that all required modules can be imported."""
    print("\nTesting imports...")
    
    try:
        from rldk.evals import run, EvalResult
        print("✓ Main evaluation imports work")
        
        from rldk.evals.runner import run as run_func
        print("✓ Runner module imports work")
        
        from rldk.evals.suites import get_eval_suite
        print("✓ Suites module imports work")
        
        from rldk.evals.probes import evaluate_alignment, evaluate_helpfulness
        print("✓ Probes module imports work")
        
        from rldk.evals.metrics import calculate_confidence_intervals, calculate_effect_sizes
        print("✓ Metrics module imports work")
        
        return True
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_suite_configuration():
    """Test that the evaluation suites are properly configured."""
    print("\nTesting suite configuration...")
    
    try:
        from rldk.evals.suites import get_eval_suite, QUICK_SUITE
        
        # Test that quick suite exists
        quick_suite = get_eval_suite("quick")
        if quick_suite is None:
            print("✗ Quick suite not found")
            return False
        print("✓ Quick suite found")
        
        # Test that it has the expected structure
        required_keys = ["name", "description", "evaluations", "baseline_scores"]
        for key in required_keys:
            if key not in quick_suite:
                print(f"✗ Quick suite missing key: {key}")
                return False
        print("✓ Quick suite has required keys")
        
        # Test that it has the new rl_training_quality evaluation
        if "rl_training_quality" not in quick_suite["evaluations"]:
            print("✗ Quick suite missing rl_training_quality evaluation")
            return False
        print("✓ Quick suite has rl_training_quality evaluation")
        
        # Test that baseline scores include the new evaluation
        if "rl_training_quality" not in quick_suite["baseline_scores"]:
            print("✗ Quick suite missing rl_training_quality baseline score")
            return False
        print("✓ Quick suite has rl_training_quality baseline score")
        
        return True
        
    except Exception as e:
        print(f"✗ Suite configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running minimal RLDK evaluation system tests...")
    
    tests = [
        test_imports,
        test_evalresult_class,
        test_suite_configuration,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! The evaluation system structure is correct.")
        return True
    else:
        print("❌ Some tests failed! The evaluation system needs more fixes.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)