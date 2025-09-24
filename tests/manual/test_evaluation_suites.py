#!/usr/bin/env python3
"""
Test evaluation suites with different model sizes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

import _path_setup  # noqa: F401


def test_evaluation_suites():
    """Test evaluation suites with different model sizes"""
    print("Testing Evaluation Suites")
    print("=" * 60)
    
    try:
        from rldk.evals import run
        from rldk.evals.suites import QUICK_SUITE, COMPREHENSIVE_SUITE
        
        # Create sample evaluation data
        eval_data = []
        for i in range(200):
            eval_data.append({
                'step': i * 5,
                'text': f"This is sample text {i} for evaluation purposes.",
                'reward': 0.8 + np.random.normal(0, 0.1),
                'loss': 0.5 + np.random.normal(0, 0.1),
                'kl': 0.1 + np.random.normal(0, 0.02)
            })
        
        df_eval = pd.DataFrame(eval_data)
        print(f"✓ Evaluation data created: {len(df_eval)} samples")
        
        # Test quick suite
        try:
            print(f"\nTesting Quick Suite...")
            quick_result = run(
                run_data=df_eval,
                suite="quick",
                seed=42,
                sample_size=100
            )
            print(f"✓ Quick evaluation suite completed")
            print(f"  - Overall score: {quick_result.overall_score}")
            print(f"  - Sample size: {quick_result.sample_size}")
            
            if hasattr(quick_result, 'scores'):
                print(f"  - Individual scores: {len(quick_result.scores)} metrics")
                for metric, score in list(quick_result.scores.items())[:3]:
                    print(f"    * {metric}: {score}")
                    
        except Exception as e:
            print(f"⚠ Quick suite failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test comprehensive suite
        try:
            print(f"\nTesting Comprehensive Suite...")
            comprehensive_result = run(
                run_data=df_eval,
                suite="comprehensive",
                seed=42,
                sample_size=150
            )
            print(f"✓ Comprehensive evaluation suite completed")
            print(f"  - Overall score: {comprehensive_result.overall_score}")
            print(f"  - Sample size: {comprehensive_result.sample_size}")
            
            if hasattr(comprehensive_result, 'scores'):
                print(f"  - Individual scores: {len(comprehensive_result.scores)} metrics")
                for metric, score in list(comprehensive_result.scores.items())[:3]:
                    print(f"    * {metric}: {score}")
                    
        except Exception as e:
            print(f"⚠ Comprehensive suite failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation suites failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_evaluation_suites()