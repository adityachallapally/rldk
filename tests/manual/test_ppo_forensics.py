#!/usr/bin/env python3
"""
Test PPO forensics with simulated training data
"""

import sys
import numpy as np
from pathlib import Path

import _path_setup  # noqa: F401


def test_ppo_forensics():
    """Test PPO forensics with simulated training data"""
    print("Testing PPO Forensics with Simulated Data")
    print("=" * 60)
    
    try:
        from rldk.forensics import ComprehensivePPOForensics
        
        # Initialize forensics
        forensics = ComprehensivePPOForensics(
            kl_target=0.1,
            kl_target_tolerance=0.05,
            window_size=100,
            enable_kl_schedule_tracking=True,
            enable_gradient_norms_analysis=True,
            enable_advantage_statistics=True
        )
        print(f"✓ ComprehensivePPOForensics initialized successfully")
        
        # Simulate training data with some anomalies
        np.random.seed(42)
        steps = list(range(0, 500, 10))  # Reduced for faster testing
        
        print(f"Simulating {len(steps)} training steps...")
        
        for i, step in enumerate(steps):
            # Simulate normal training with some anomalies
            if 200 <= step <= 250:  # KL spike anomaly
                kl = 0.3 + np.random.normal(0, 0.05)
                policy_grad_norm = 1.0 + np.random.normal(0, 0.1)
                value_grad_norm = 0.8 + np.random.normal(0, 0.1)
            elif 300 <= step <= 350:  # Gradient explosion
                kl = 0.08 + np.random.normal(0, 0.02)
                policy_grad_norm = 5.0 + np.random.normal(0, 0.5)
                value_grad_norm = 3.0 + np.random.normal(0, 0.3)
            else:  # Normal training
                kl = 0.08 + np.random.normal(0, 0.02)
                policy_grad_norm = 1.0 + np.random.normal(0, 0.1)
                value_grad_norm = 0.8 + np.random.normal(0, 0.1)
            
            # Update forensics
            metrics = forensics.update(
                step=step,
                kl=kl,
                kl_coef=0.2,
                entropy=2.5 + np.random.normal(0, 0.1),
                reward_mean=0.8 + np.random.normal(0, 0.1),
                reward_std=0.3 + np.random.normal(0, 0.05),
                policy_grad_norm=policy_grad_norm,
                value_grad_norm=value_grad_norm,
                total_grad_norm=policy_grad_norm + value_grad_norm,
                advantage_mean=0.1 + np.random.normal(0, 0.05),
                advantage_std=0.5 + np.random.normal(0, 0.1),
                advantage_min=-0.5 + np.random.normal(0, 0.1),
                advantage_max=1.0 + np.random.normal(0, 0.1),
                advantage_median=0.05 + np.random.normal(0, 0.05),
                advantage_samples=[np.random.normal(0, 0.5) for _ in range(10)]
            )
            
            if i % 25 == 0:  # Print progress every 25 steps
                print(f"  Processed step {step}")
        
        print(f"✓ Training data processed successfully")
        
        # Get comprehensive analysis
        analysis = forensics.get_comprehensive_analysis()
        print(f"✓ Comprehensive analysis generated")
        print(f"  - Total metrics tracked: {len(analysis.get('metrics', {}))}")
        
        # Get anomalies
        anomalies = forensics.get_anomalies()
        print(f"✓ Anomalies detected: {len(anomalies)}")
        for i, anomaly in enumerate(anomalies[:5]):  # Show first 5 anomalies
            print(f"  {i+1}. {anomaly.get('type', 'Unknown')}: {anomaly.get('description', 'No description')}")
        
        # Get health summary
        health_summary = forensics.get_health_summary()
        print(f"✓ Health summary generated")
        print(f"  - Overall health: {health_summary.get('overall_health', 'Unknown')}")
        print(f"  - Health score: {health_summary.get('health_score', 'N/A')}")
        
        # Test saving analysis
        try:
            forensics.save_analysis("ppo_analysis_test.json")
            print(f"✓ Analysis saved to file")
        except Exception as e:
            print(f"⚠ Could not save analysis: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ PPO forensics failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_ppo_forensics()