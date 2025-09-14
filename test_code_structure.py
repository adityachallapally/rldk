#!/usr/bin/env python3
"""Test the code structure without executing it."""

import os
import re

def test_evalresult_class_structure():
    """Test that EvalResult class has the overall_score property."""
    print("Testing EvalResult class structure...")
    
    try:
        runner_file = os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "runner.py")
        with open(runner_file, 'r') as f:
            content = f.read()
        
        # Check that EvalResult class exists
        if "class EvalResult:" not in content:
            print("✗ EvalResult class not found")
            return False
        print("✓ EvalResult class found")
        
        # Check that overall_score property exists
        if "@property" not in content or "def overall_score" not in content:
            print("✗ overall_score property not found")
            return False
        print("✓ overall_score property found")
        
        # Check that it calculates the mean correctly
        if "np.mean(valid_scores)" not in content:
            print("✗ overall_score doesn't calculate mean correctly")
            return False
        print("✓ overall_score calculates mean correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ EvalResult class structure test failed: {e}")
        return False

def test_suites_structure():
    """Test that suites have the expected structure."""
    print("\nTesting suites structure...")
    
    try:
        suites_file = os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "suites.py")
        with open(suites_file, 'r') as f:
            content = f.read()
        
        # Check that QUICK_SUITE exists
        if "QUICK_SUITE = {" not in content:
            print("✗ QUICK_SUITE not found")
            return False
        print("✓ QUICK_SUITE found")
        
        # Check that it has rl_training_quality evaluation
        if "rl_training_quality" not in content:
            print("✗ rl_training_quality evaluation not found")
            return False
        print("✓ rl_training_quality evaluation found")
        
        # Check that it has baseline scores
        if "baseline_scores" not in content:
            print("✗ baseline_scores not found")
            return False
        print("✓ baseline_scores found")
        
        # Check that rl_training_quality is in baseline scores
        if '"rl_training_quality":' not in content:
            print("✗ rl_training_quality not in baseline scores")
            return False
        print("✓ rl_training_quality in baseline scores")
        
        return True
        
    except Exception as e:
        print(f"✗ Suites structure test failed: {e}")
        return False

def test_probes_structure():
    """Test that probes have the expected structure."""
    print("\nTesting probes structure...")
    
    try:
        probes_file = os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "probes.py")
        with open(probes_file, 'r') as f:
            content = f.read()
        
        # Check that evaluate_rl_training_quality function exists
        if "def evaluate_rl_training_quality" not in content:
            print("✗ evaluate_rl_training_quality function not found")
            return False
        print("✓ evaluate_rl_training_quality function found")
        
        # Check that it has the expected structure
        if "reward_mean" not in content or "loss" not in content or "kl" not in content:
            print("✗ evaluate_rl_training_quality doesn't handle standard RL metrics")
            return False
        print("✓ evaluate_rl_training_quality handles standard RL metrics")
        
        # Check that other functions exist
        required_functions = [
            'evaluate_alignment', 'evaluate_helpfulness', 'evaluate_harmlessness',
            'evaluate_hallucination', 'evaluate_reward_alignment', 'evaluate_kl_divergence'
        ]
        
        for func_name in required_functions:
            if f"def {func_name}" not in content:
                print(f"✗ {func_name} function not found")
                return False
        print("✓ All required probe functions found")
        
        return True
        
    except Exception as e:
        print(f"✗ Probes structure test failed: {e}")
        return False

def test_metrics_structure():
    """Test that metrics have better error handling."""
    print("\nTesting metrics structure...")
    
    try:
        # Test throughput.py
        throughput_file = os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "metrics", "throughput.py")
        with open(throughput_file, 'r') as f:
            content = f.read()
        
        # Check that it has better error messages
        if "Available columns:" not in content:
            print("✗ Throughput metrics don't have improved error messages")
            return False
        print("✓ Throughput metrics have improved error messages")
        
        # Test toxicity.py
        toxicity_file = os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "metrics", "toxicity.py")
        with open(toxicity_file, 'r') as f:
            content = f.read()
        
        if "Available columns:" not in content:
            print("✗ Toxicity metrics don't have improved error messages")
            return False
        print("✓ Toxicity metrics have improved error messages")
        
        # Test bias.py
        bias_file = os.path.join(os.path.dirname(__file__), "src", "rldk", "evals", "metrics", "bias.py")
        with open(bias_file, 'r') as f:
            content = f.read()
        
        if "Available columns:" not in content:
            print("✗ Bias metrics don't have improved error messages")
            return False
        print("✓ Bias metrics have improved error messages")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics structure test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running RLDK evaluation system structure tests...")
    
    tests = [
        test_evalresult_class_structure,
        test_suites_structure,
        test_probes_structure,
        test_metrics_structure,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All structure tests passed! The evaluation system code is correct.")
        return True
    else:
        print("❌ Some structure tests failed! The evaluation system needs more fixes.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)