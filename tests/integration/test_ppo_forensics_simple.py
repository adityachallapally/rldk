#!/usr/bin/env python3
"""Simple test script for PPO forensics without external dependencies."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test individual modules without pandas/numpy dependencies
def test_kl_schedule_tracker():
    """Test KL schedule tracker basic functionality."""
    print("Testing KL Schedule Tracker...")

    # Import and test basic functionality
    from rldk.forensics.kl_schedule_tracker import KLScheduleMetrics, KLScheduleTracker

    # Test initialization
    tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)
    assert tracker.kl_target == 0.1
    assert tracker.kl_target_tolerance == 0.05

    # Test update
    metrics = tracker.update(0, 0.1, 1.0)
    assert metrics.current_kl == 0.1
    assert metrics.current_kl_coef == 1.0

    print("✅ KL Schedule Tracker basic functionality works")
    return True

def test_gradient_norms_analyzer():
    """Test gradient norms analyzer basic functionality."""
    print("Testing Gradient Norms Analyzer...")

    from rldk.forensics.gradient_norms_analyzer import (
        GradientNormsAnalyzer,
        GradientNormsMetrics,
    )

    # Test initialization
    analyzer = GradientNormsAnalyzer()
    assert analyzer.exploding_threshold == 10.0
    assert analyzer.vanishing_threshold == 0.001

    # Test update
    metrics = analyzer.update(0, 0.5, 0.3)
    assert metrics.policy_grad_norm == 0.5
    assert metrics.value_grad_norm == 0.3

    print("✅ Gradient Norms Analyzer basic functionality works")
    return True

def test_advantage_statistics_tracker():
    """Test advantage statistics tracker basic functionality."""
    print("Testing Advantage Statistics Tracker...")

    from rldk.forensics.advantage_statistics_tracker import (
        AdvantageStatisticsMetrics,
        AdvantageStatisticsTracker,
    )

    # Test initialization
    tracker = AdvantageStatisticsTracker()
    assert tracker.bias_threshold == 0.1
    assert tracker.scale_threshold == 2.0

    # Test update
    metrics = tracker.update(0, 0.0, 1.0)
    assert metrics.advantage_mean == 0.0
    assert metrics.advantage_std == 1.0

    print("✅ Advantage Statistics Tracker basic functionality works")
    return True

def test_comprehensive_ppo_forensics():
    """Test comprehensive PPO forensics basic functionality."""
    print("Testing Comprehensive PPO Forensics...")

    from rldk.forensics.comprehensive_ppo_forensics import (
        ComprehensivePPOForensics,
        ComprehensivePPOMetrics,
    )

    # Test initialization
    forensics = ComprehensivePPOForensics()
    assert forensics.kl_target == 0.1
    assert forensics.kl_schedule_tracker is not None
    assert forensics.gradient_norms_analyzer is not None
    assert forensics.advantage_statistics_tracker is not None

    # Test update
    metrics = forensics.update(
        step=0,
        kl=0.1,
        kl_coef=1.0,
        entropy=2.0,
        reward_mean=0.5,
        reward_std=0.2,
        policy_grad_norm=0.5,
        value_grad_norm=0.3,
        advantage_mean=0.0,
        advantage_std=1.0
    )
    assert metrics.step == 0
    assert metrics.kl == 0.1

    print("✅ Comprehensive PPO Forensics basic functionality works")
    return True

def test_ppo_scan_integration():
    """Test PPO scan integration."""
    print("Testing PPO Scan Integration...")

    from rldk.forensics.ppo_scan import scan_ppo_events

    # Test with empty events
    result = scan_ppo_events(iter([]))
    assert result["version"] == "1"
    assert len(result["rules_fired"]) == 0
    assert result["earliest_step"] is None

    # Test with sample events
    events = [
        {"step": 0, "kl": 0.1, "kl_coef": 1.0, "entropy": 2.0, "advantage_mean": 0.0, "advantage_std": 1.0, "grad_norm_policy": 0.5, "grad_norm_value": 0.3},
        {"step": 1, "kl": 0.1, "kl_coef": 1.0, "entropy": 2.0, "advantage_mean": 0.0, "advantage_std": 1.0, "grad_norm_policy": 0.5, "grad_norm_value": 0.3},
    ]

    result = scan_ppo_events(iter(events))
    assert result["version"] == "1"
    assert result["earliest_step"] == 0

    print("✅ PPO Scan Integration works")
    return True

def main():
    """Run all tests."""
    print("🧪 Testing Comprehensive PPO Forensics")
    print("=" * 50)

    tests = [
        test_kl_schedule_tracker,
        test_gradient_norms_analyzer,
        test_advantage_statistics_tracker,
        test_comprehensive_ppo_forensics,
        test_ppo_scan_integration,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
