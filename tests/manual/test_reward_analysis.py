#!/usr/bin/env python3
"""
Test reward model health analysis and drift detection
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import _path_setup  # noqa: F401


def test_reward_analysis():
    """Test reward model health analysis and drift detection"""
    print("Testing Reward Analysis")
    print("=" * 60)
    
    try:
        from rldk.reward import health, compare_models
        
        # Create sample training data
        np.random.seed(42)
        steps = list(range(0, 1000, 10))
        
        # Simulate training data with reward drift
        training_data = []
        for i, step in enumerate(steps):
            # Simulate reward drift over time
            base_reward = 0.8
            if step > 500:  # Drift starts after step 500
                drift = (step - 500) * 0.001
                reward_mean = base_reward + drift
            else:
                reward_mean = base_reward
            
            training_data.append({
                'step': step,
                'reward_mean': reward_mean + np.random.normal(0, 0.1),
                'reward_std': 0.3 + np.random.normal(0, 0.05),
                'loss': 0.5 + np.random.normal(0, 0.1)
            })
        
        df_training = pd.DataFrame(training_data)
        print(f"✓ Training data created: {len(df_training)} samples")
        
        # Create reference data (baseline)
        reference_data = df_training[df_training['step'] <= 500].copy()
        print(f"✓ Reference data created: {len(reference_data)} samples")
        
        # Analyze reward model health
        health_report = health(
            run_data=df_training,
            reference_data=reference_data,
            reward_col="reward_mean",
            step_col="step",
            threshold_drift=0.1,
            threshold_saturation=0.8,
            threshold_calibration=0.7
        )
        
        print(f"✓ Reward health analysis completed")
        print(f"  - Passed: {health_report.passed}")
        print(f"  - Drift detected: {health_report.drift_detected}")
        print(f"  - Saturation issues: {health_report.saturation_issues}")
        print(f"  - Calibration score: {health_report.calibration_score}")
        
        # Test model comparison (simulated)
        print(f"\nTesting model comparison...")
        try:
            # Create sample prompts for model comparison
            prompts = [
                "The quick brown fox jumps over the lazy dog.",
                "Machine learning is revolutionizing artificial intelligence.",
                "Reinforcement learning requires careful debugging and analysis."
            ]
            
            # This would normally compare two actual models
            # For testing, we'll just verify the function exists
            print(f"✓ Model comparison function available")
            print(f"  - Sample prompts: {len(prompts)}")
            
        except Exception as e:
            print(f"⚠ Model comparison test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reward analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_reward_analysis()