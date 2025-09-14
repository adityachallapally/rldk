#!/usr/bin/env python3
"""Test script to verify the RLDK evaluation system fixes."""

import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_evaluation_system():
    """Test the evaluation system with the provided test case."""
    print("Testing RLDK evaluation system fixes...")
    
    try:
        from rldk.evals import run
        
        # Create test data as specified in the user's test case
        print("Creating test data...")
        data = pd.DataFrame({
            'step': range(100),
            'reward_mean': np.random.normal(0.5, 0.2, 100),
            'loss': np.random.normal(0.3, 0.1, 100),
            'kl': np.random.normal(0.1, 0.05, 100)
        })
        
        print(f"Test data shape: {data.shape}")
        print(f"Test data columns: {list(data.columns)}")
        print(f"Test data head:\n{data.head()}")
        
        # Test the evaluation system
        print("\nRunning evaluation...")
        result = run(data, suite='quick', seed=42)
        
        # Test the overall_score attribute
        print(f"\nOverall score: {result.overall_score}")
        print(f"Overall score type: {type(result.overall_score)}")
        
        # Test other attributes
        print(f"Scores: {result.scores}")
        print(f"Confidence intervals: {result.confidence_intervals}")
        print(f"Effect sizes: {result.effect_sizes}")
        print(f"Sample size: {result.sample_size}")
        
        # Test that the result object has all expected attributes
        expected_attrs = ['overall_score', 'scores', 'confidence_intervals', 'effect_sizes', 'sample_size']
        for attr in expected_attrs:
            if hasattr(result, attr):
                print(f"✓ {attr} attribute exists")
            else:
                print(f"✗ {attr} attribute missing")
                return False
        
        # Test that scores are not -1 (indicating failures)
        failed_metrics = [name for name, score in result.scores.items() if score == -1]
        if failed_metrics:
            print(f"✗ Some metrics returned -1 (failure): {failed_metrics}")
            return False
        else:
            print("✓ No metrics returned -1 (failure)")
        
        # Test that we have reasonable scores
        nan_scores = [name for name, score in result.scores.items() if np.isnan(score)]
        if nan_scores:
            print(f"⚠ Some metrics returned NaN: {nan_scores}")
        else:
            print("✓ No metrics returned NaN")
        
        print(f"\nEvaluation completed successfully!")
        print(f"Number of metrics evaluated: {len(result.scores)}")
        print(f"Overall score: {result.overall_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation_system()
    if success:
        print("\n✅ All tests passed! The evaluation system is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! The evaluation system needs more fixes.")
        sys.exit(1)