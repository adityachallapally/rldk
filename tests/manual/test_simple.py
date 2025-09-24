#!/usr/bin/env python3
"""
Simple test script to identify issues with RLDK components
"""

import sys

import _path_setup  # noqa: F401


def test_imports():
    """Test basic imports"""
    print("Testing imports...")
    
    try:
        from rldk.tracking import ExperimentTracker, TrackingConfig
        print("✓ Tracking imports successful")
    except Exception as e:
        print(f"✗ Tracking imports failed: {e}")
        return False
    
    try:
        from rldk.forensics import ComprehensivePPOForensics
        print("✓ Forensics imports successful")
    except Exception as e:
        print(f"✗ Forensics imports failed: {e}")
        return False
    
    try:
        from rldk.determinism import check
        print("✓ Determinism imports successful")
    except Exception as e:
        print(f"✗ Determinism imports failed: {e}")
        return False
    
    try:
        from rldk.reward import health
        print("✓ Reward imports successful")
    except Exception as e:
        print(f"✗ Reward imports failed: {e}")
        return False
    
    try:
        from rldk.evals import run
        print("✓ Evals imports successful")
    except Exception as e:
        print(f"✗ Evals imports failed: {e}")
        return False
    
    try:
        from rldk.ingest import ingest_runs
        print("✓ Ingest imports successful")
    except Exception as e:
        print(f"✗ Ingest imports failed: {e}")
        return False
    
    return True

def test_tracking_config():
    """Test TrackingConfig creation"""
    print("\nTesting TrackingConfig...")
    
    try:
        from rldk.tracking import TrackingConfig
        
        config = TrackingConfig(
            experiment_name="test_experiment",
            enable_dataset_tracking=True,
            enable_model_tracking=True,
            enable_environment_tracking=True,
            enable_seed_tracking=True,
            enable_git_tracking=True
        )
        print("✓ TrackingConfig created successfully")
        print(f"  - Experiment name: {config.experiment_name}")
        print(f"  - Dataset tracking: {config.enable_dataset_tracking}")
        print(f"  - Model tracking: {config.enable_model_tracking}")
        return True
        
    except Exception as e:
        print(f"✗ TrackingConfig creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ppo_forensics_init():
    """Test PPO forensics initialization"""
    print("\nTesting PPO Forensics initialization...")
    
    try:
        from rldk.forensics import ComprehensivePPOForensics
        
        forensics = ComprehensivePPOForensics(
            kl_target=0.1,
            kl_target_tolerance=0.05,
            window_size=100
        )
        print("✓ ComprehensivePPOForensics initialized successfully")
        print(f"  - KL target: {forensics.kl_target}")
        print(f"  - Window size: {forensics.window_size}")
        return True
        
    except Exception as e:
        print(f"✗ ComprehensivePPOForensics initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_determinism_check():
    """Test determinism check function"""
    print("\nTesting determinism check...")
    
    try:
        from rldk.determinism import check
        
        # Simple test command
        report = check(
            cmd="python -c 'import numpy as np; np.random.seed(42); print(f\"result:{np.random.rand():.6f}\")'",
            compare=["result"],
            replicas=2,
            device="cpu"
        )
        print("✓ Determinism check completed")
        print(f"  - Passed: {report.passed}")
        return True
        
    except Exception as e:
        print(f"✗ Determinism check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run simple tests"""
    print("RLDK SIMPLE TESTING")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("TrackingConfig", test_tracking_config),
        ("PPO Forensics Init", test_ppo_forensics_init),
        ("Determinism Check", test_determinism_check),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    return results

if __name__ == "__main__":
    main()