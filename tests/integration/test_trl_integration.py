#!/usr/bin/env python3
"""Comprehensive test suite for RLDK TRL integration."""

import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required packages can be imported."""
    print("🔍 Testing imports...")

    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import torch
        print("✅ Basic dependencies imported")

        # Test RLDK imports
        from rldk.integrations.trl import (
            CheckpointMonitor,
            PPOMonitor,
            RLDKCallback,
            RLDKDashboard,
        )
        print("✅ RLDK TRL integration imported")

        # Test TRL imports
        try:
            from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
            print("✅ TRL imported successfully")
            return True
        except ImportError as e:
            print(f"⚠️  TRL not available: {e}")
            print("   Install with: pip install trl")
            return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic RLDK callback functionality."""
    print("\n🧪 Testing basic functionality...")

    try:
        from rldk.integrations.trl import RLDKCallback, RLDKMetrics

        # Test metrics creation
        metrics = RLDKMetrics(
            step=1,
            loss=0.5,
            reward_mean=0.3,
            kl_mean=0.05,
            training_stability_score=0.8
        )

        metrics_dict = metrics.to_dict()
        assert 'step' in metrics_dict
        assert 'loss' in metrics_dict
        assert 'reward_mean' in metrics_dict
        print("✅ RLDKMetrics working")

        # Test callback initialization
        callback = RLDKCallback(
            output_dir="./test_output",
            log_interval=5,
            run_id="test_run"
        )

        assert callback.run_id == "test_run"
        assert callback.log_interval == 5
        print("✅ RLDKCallback initialization working")

        # Test alert system
        callback._add_alert("test_alert", "This is a test")
        assert len(callback.alerts) == 1
        print("✅ Alert system working")

        return True

    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_ppo_monitor():
    """Test PPO monitor functionality."""
    print("\n🎯 Testing PPO monitor...")

    try:
        from rldk.integrations.trl import PPOMetrics, PPOMonitor

        # Test PPO metrics
        ppo_metrics = PPOMetrics(
            rollout_reward_mean=0.5,
            policy_kl_mean=0.05,
            policy_entropy_mean=2.0,
            policy_clip_frac=0.1
        )

        ppo_dict = ppo_metrics.to_dict()
        assert 'rollout_reward_mean' in ppo_dict
        assert 'policy_kl_mean' in ppo_dict
        print("✅ PPOMetrics working")

        # Test PPO monitor initialization
        ppo_monitor = PPOMonitor(
            output_dir="./test_output",
            kl_threshold=0.1,
            reward_threshold=0.05,
            run_id="test_ppo"
        )

        assert ppo_monitor.kl_threshold == 0.1
        assert ppo_monitor.reward_threshold == 0.05
        print("✅ PPOMonitor initialization working")

        return True

    except Exception as e:
        print(f"❌ PPO monitor test failed: {e}")
        return False


def test_checkpoint_monitor():
    """Test checkpoint monitor functionality."""
    print("\n💾 Testing checkpoint monitor...")

    try:
        from rldk.integrations.trl import CheckpointMetrics, CheckpointMonitor

        # Test checkpoint metrics
        checkpoint_metrics = CheckpointMetrics(
            step=1,
            total_parameters=1000000,
            trainable_parameters=500000,
            model_size_mb=4.0
        )

        checkpoint_dict = checkpoint_metrics.to_dict()
        assert 'step' in checkpoint_dict
        assert 'total_parameters' in checkpoint_dict
        print("✅ CheckpointMetrics working")

        # Test checkpoint monitor initialization
        checkpoint_monitor = CheckpointMonitor(
            output_dir="./test_output",
            enable_parameter_analysis=True,
            run_id="test_checkpoint"
        )

        assert checkpoint_monitor.enable_parameter_analysis
        print("✅ CheckpointMonitor initialization working")

        return True

    except Exception as e:
        print(f"❌ Checkpoint monitor test failed: {e}")
        return False


def test_dashboard():
    """Test dashboard functionality."""
    print("\n📊 Testing dashboard...")

    try:
        from rldk.integrations.trl import RLDKDashboard

        # Test dashboard initialization
        dashboard = RLDKDashboard(
            output_dir="./test_output",
            port=8503,  # Use different port for testing
            run_id="test_dashboard"
        )

        assert dashboard.port == 8503
        assert dashboard.run_id == "test_dashboard"
        print("✅ RLDKDashboard initialization working")

        # Test dashboard app creation
        app_file = dashboard.output_dir / "test_dashboard_app.py"
        dashboard._create_dashboard_app(app_file)

        if app_file.exists():
            print("✅ Dashboard app creation working")
        else:
            print("❌ Dashboard app creation failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False


def test_integration_examples():
    """Test the integration examples."""
    print("\n📚 Testing integration examples...")

    try:
        # Test basic integration example
        example_file = Path("examples/trl_integration/basic_ppo_integration.py")
        if example_file.exists():
            print("✅ Basic integration example exists")
        else:
            print("❌ Basic integration example missing")
            return False

        # Test advanced monitoring example
        advanced_file = Path("examples/trl_integration/advanced_monitoring.py")
        if advanced_file.exists():
            print("✅ Advanced monitoring example exists")
        else:
            print("❌ Advanced monitoring example missing")
            return False

        # Test custom callbacks example
        custom_file = Path("examples/trl_integration/custom_callbacks.py")
        if custom_file.exists():
            print("✅ Custom callbacks example exists")
        else:
            print("❌ Custom callbacks example missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Integration examples test failed: {e}")
        return False


def test_file_structure():
    """Test that all required files are in place."""
    print("\n📁 Testing file structure...")

    required_files = [
        "src/rldk/integrations/__init__.py",
        "src/rldk/integrations/trl/__init__.py",
        "src/rldk/integrations/trl/callbacks.py",
        "src/rldk/integrations/trl/monitors.py",
        "src/rldk/integrations/trl/dashboard.py",
        "examples/trl_integration/basic_ppo_integration.py",
        "examples/trl_integration/advanced_monitoring.py",
        "examples/trl_integration/custom_callbacks.py",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All required files present")
    return True


def run_example_tests():
    """Run the example test scripts without downloading models."""
    print("\n🚀 Running example tests...")

    # Set environment variable to skip model loading
    env = os.environ.copy()
    env["SKIP_MODEL_LOADING"] = "true"

    try:
        # Run basic integration test
        print("Running basic integration test...")
        result = subprocess.run([
            sys.executable, "examples/trl_integration/basic_ppo_integration.py"
        ], capture_output=True, text=True, timeout=60, env=env)

        if result.returncode == 0:
            print("✅ Basic integration test passed")
        else:
            print(f"❌ Basic integration test failed: {result.stderr}")
            return False

        # Run advanced monitoring test
        print("Running advanced monitoring test...")
        result = subprocess.run([
            sys.executable, "examples/trl_integration/advanced_monitoring.py"
        ], capture_output=True, text=True, timeout=60, env=env)

        if result.returncode == 0:
            print("✅ Advanced monitoring test passed")
        else:
            print(f"❌ Advanced monitoring test failed: {result.stderr}")
            return False

        # Run custom callbacks test
        print("Running custom callbacks test...")
        result = subprocess.run([
            sys.executable, "examples/trl_integration/custom_callbacks.py"
        ], capture_output=True, text=True, timeout=60, env=env)

        if result.returncode == 0:
            print("✅ Custom callbacks test passed")
        else:
            print(f"❌ Custom callbacks test failed: {result.stderr}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print("❌ Example tests timed out")
        return False
    except Exception as e:
        print(f"❌ Example tests failed: {e}")
        return False


def cleanup_test_files():
    """Clean up test files."""
    print("\n🧹 Cleaning up test files...")

    test_dirs = [
        "./test_output",
        "./test_ppo_output",
        "./test_advanced_output",
        "./test_custom_output",
        "./test_callback_output",
    ]

    for test_dir in test_dirs:
        if Path(test_dir).exists():
            import shutil
            shutil.rmtree(test_dir)
            print(f"✅ Cleaned up {test_dir}")


def main():
    """Run the comprehensive test suite."""
    print("🎯 RLDK TRL Integration Test Suite")
    print("=" * 60)

    tests = [
        ("File Structure", test_file_structure),
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("PPO Monitor", test_ppo_monitor),
        ("Checkpoint Monitor", test_checkpoint_monitor),
        ("Dashboard", test_dashboard),
        ("Integration Examples", test_integration_examples),
        ("Example Tests", run_example_tests),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    passed = 0
    total = len(results)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! RLDK TRL integration is working correctly.")
        print("\n🚀 You can now use RLDK with TRL:")
        print("   from rldk.integrations.trl import RLDKCallback, PPOMonitor")
        print("   from trl import PPOTrainer")
        print("   ")
        print("   monitor = RLDKCallback()")
        print("   trainer = PPOTrainer(")
        print("       args=config,")
        print("       model=model,")
        print("       ref_model=ref_model,")
        print("       reward_model=reward_model,")
        print("       value_model=value_model,")
        print("       callbacks=[monitor]")
        print("   )")
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the output above.")

    # Cleanup
    cleanup_test_files()

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
