#!/usr/bin/env python3
"""Direct test of the evaluation system without importing the main package."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_evalresult_class_direct():
    """Test EvalResult class directly without pandas dependency."""
    print("Testing EvalResult class directly...")
    
    try:
        # Import only the specific module we need
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "runner", 
            os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "runner.py")
        )
        runner_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner_module)
        
        # Test the EvalResult class
        EvalResult = runner_module.EvalResult
        
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
        import math
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

def test_suites_direct():
    """Test suites module directly."""
    print("\nTesting suites module directly...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "suites", 
            os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "suites.py")
        )
        suites_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(suites_module)
        
        # Test that the quick suite exists and has the expected structure
        quick_suite = suites_module.QUICK_SUITE
        
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
        
        # Test get_eval_suite function
        get_eval_suite = suites_module.get_eval_suite
        retrieved_suite = get_eval_suite("quick")
        if retrieved_suite is None:
            print("✗ get_eval_suite function not working")
            return False
        print("✓ get_eval_suite function works")
        
        return True
        
    except Exception as e:
        print(f"✗ Suites module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_probes_direct():
    """Test probes module directly."""
    print("\nTesting probes module directly...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "probes", 
            os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "probes.py")
        )
        probes_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probes_module)
        
        # Test that the new function exists
        if not hasattr(probes_module, 'evaluate_rl_training_quality'):
            print("✗ evaluate_rl_training_quality function not found")
            return False
        print("✓ evaluate_rl_training_quality function exists")
        
        # Test that other functions exist
        required_functions = [
            'evaluate_alignment', 'evaluate_helpfulness', 'evaluate_harmlessness',
            'evaluate_hallucination', 'evaluate_reward_alignment', 'evaluate_kl_divergence'
        ]
        
        for func_name in required_functions:
            if not hasattr(probes_module, func_name):
                print(f"✗ {func_name} function not found")
                return False
        print("✓ All required probe functions exist")
        
        return True
        
    except Exception as e:
        print(f"✗ Probes module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Running direct RLDK evaluation system tests...")
    
    tests = [
        test_evalresult_class_direct,
        test_suites_direct,
        test_probes_direct,
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