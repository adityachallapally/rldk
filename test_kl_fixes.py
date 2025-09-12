#!/usr/bin/env python3
"""Simple test script to validate KL divergence fixes without pytest."""

import sys
import os
import warnings
import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rldk.evals.metrics import calculate_kl_divergence
from rldk.forensics.kl_schedule_tracker import (
    KLScheduleTracker, 
    _safe_kl_value, 
    _safe_coefficient_value,
    KLScheduleMetrics
)


def test_kl_divergence_basic():
    """Test basic KL divergence calculation."""
    print("Testing basic KL divergence calculation...")
    
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    
    kl_div = calculate_kl_divergence(p, q)
    assert isinstance(kl_div, float)
    assert kl_div >= 0.0
    assert not np.isnan(kl_div)
    assert not np.isinf(kl_div)
    print(f"✓ Basic KL divergence: {kl_div}")
    
    # Test identical distributions
    p_identical = np.array([0.5, 0.3, 0.2])
    q_identical = np.array([0.5, 0.3, 0.2])
    
    kl_div_identical = calculate_kl_divergence(p_identical, q_identical)
    assert abs(kl_div_identical) < 1e-6
    print(f"✓ Identical distributions KL: {kl_div_identical}")


def test_kl_divergence_edge_cases():
    """Test KL divergence edge cases."""
    print("\nTesting KL divergence edge cases...")
    
    # Test zero probabilities
    p_zero = np.array([0.0, 0.5, 0.5])
    q_zero = np.array([0.1, 0.4, 0.5])
    
    kl_div_zero = calculate_kl_divergence(p_zero, q_zero)
    assert isinstance(kl_div_zero, float)
    assert kl_div_zero >= 0.0
    assert not np.isnan(kl_div_zero)
    assert not np.isinf(kl_div_zero)
    print(f"✓ Zero probabilities KL: {kl_div_zero}")
    
    # Test very small probabilities
    p_small = np.array([1e-10, 0.5, 0.5])
    q_small = np.array([1e-12, 0.4, 0.6])
    
    kl_div_small = calculate_kl_divergence(p_small, q_small)
    assert isinstance(kl_div_small, float)
    assert kl_div_small >= 0.0
    assert not np.isnan(kl_div_small)
    assert not np.isinf(kl_div_small)
    print(f"✓ Small probabilities KL: {kl_div_small}")
    
    # Test error handling
    print("✓ Testing error handling...")
    
    # NaN input
    try:
        p_nan = np.array([0.5, np.nan, 0.3])
        q_nan = np.array([0.4, 0.4, 0.2])
        calculate_kl_divergence(p_nan, q_nan)
        assert False, "Should have raised ValueError for NaN"
    except ValueError as e:
        assert "NaN values" in str(e)
        print("✓ NaN input handling works")
    
    # Infinite input
    try:
        p_inf = np.array([0.5, np.inf, 0.3])
        q_inf = np.array([0.4, 0.4, 0.2])
        calculate_kl_divergence(p_inf, q_inf)
        assert False, "Should have raised ValueError for inf"
    except ValueError as e:
        assert "infinite values" in str(e)
        print("✓ Infinite input handling works")
    
    # Negative input
    try:
        p_neg = np.array([0.5, -0.1, 0.3])
        q_neg = np.array([0.4, 0.4, 0.2])
        calculate_kl_divergence(p_neg, q_neg)
        assert False, "Should have raised ValueError for negative"
    except ValueError as e:
        assert "non-negative" in str(e)
        print("✓ Negative input handling works")


def test_safe_value_processing():
    """Test safe value processing functions."""
    print("\nTesting safe value processing...")
    
    # Test valid inputs
    assert _safe_kl_value(0.5) == 0.5
    assert _safe_kl_value("0.5") == 0.5
    assert _safe_coefficient_value(1.0) == 1.0
    assert _safe_coefficient_value("1.0") == 1.0
    print("✓ Valid input processing works")
    
    # Test edge cases with warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # NaN handling
        result = _safe_kl_value(np.nan)
        assert result == 0.0
        assert len(w) == 1
        assert "NaN KL value detected" in str(w[0].message)
        print("✓ NaN KL value handling works")
        
        # Infinite handling
        result = _safe_kl_value(np.inf)
        assert result == 1e6
        assert len(w) == 1
        assert "Positive infinity KL value detected" in str(w[0].message)
        print("✓ Infinite KL value handling works")
        
        # Negative handling
        result = _safe_kl_value(-0.1)
        assert result == 0.0
        assert len(w) == 1
        assert "Negative KL value" in str(w[0].message)
        print("✓ Negative KL value handling works")
        
        # Large value handling
        result = _safe_kl_value(2e6)
        assert result == 1e6
        assert len(w) == 1
        assert "Extremely large KL value" in str(w[0].message)
        print("✓ Large KL value handling works")


def test_kl_schedule_tracker():
    """Test KL schedule tracker robustness."""
    print("\nTesting KL schedule tracker...")
    
    tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)
    assert tracker.kl_target == 0.1
    assert tracker.kl_target_tolerance == 0.05
    print("✓ Tracker initialization works")
    
    # Test normal updates
    metrics = tracker.update(step=1, kl_value=0.05, kl_coef=1.0)
    assert isinstance(metrics, KLScheduleMetrics)
    assert metrics.current_kl == 0.05
    assert metrics.current_kl_coef == 1.0
    print("✓ Normal tracker update works")
    
    # Test edge case updates
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        metrics = tracker.update(step=2, kl_value=np.nan, kl_coef=np.nan)
        assert metrics.current_kl == 0.0
        assert metrics.current_kl_coef == 1.0
        assert len(w) >= 2
        print("✓ NaN tracker update handling works")
        
        metrics = tracker.update(step=3, kl_value=np.inf, kl_coef=np.inf)
        assert metrics.current_kl == 1e6
        assert metrics.current_kl_coef == 1e6
        assert len(w) >= 2
        print("✓ Infinite tracker update handling works")
        
        metrics = tracker.update(step=4, kl_value=-0.1, kl_coef=-0.1)
        assert metrics.current_kl == 0.0
        assert metrics.current_kl_coef == 1.0
        assert len(w) >= 2
        print("✓ Negative tracker update handling works")
    
    # Test analysis functions
    for i in range(20):
        tracker.update(step=i+5, kl_value=0.1 + 0.01 * np.sin(i), kl_coef=1.0)
    
    summary = tracker.get_summary()
    assert isinstance(summary, dict)
    assert "current_kl" in summary
    assert "current_kl_coef" in summary
    assert "kl_health_score" in summary
    assert "schedule_health_score" in summary
    print("✓ Tracker analysis functions work")
    
    anomalies = tracker.get_anomalies()
    assert isinstance(anomalies, list)
    print("✓ Tracker anomaly detection works")


def test_numerical_stability():
    """Test numerical stability."""
    print("\nTesting numerical stability...")
    
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    
    # Test repeated calculations
    results = []
    for _ in range(100):
        kl_div = calculate_kl_divergence(p, q)
        results.append(kl_div)
    
    # All results should be identical (within numerical precision)
    assert all(abs(r - results[0]) < 1e-12 for r in results)
    print("✓ Repeated calculations are consistent")
    
    # Test tracker consistency
    tracker1 = KLScheduleTracker()
    tracker2 = KLScheduleTracker()
    
    for i in range(10):
        tracker1.update(step=i, kl_value=0.1 + 0.01 * i, kl_coef=1.0 + 0.1 * i)
        tracker2.update(step=i, kl_value=0.1 + 0.01 * i, kl_coef=1.0 + 0.1 * i)
    
    summary1 = tracker1.get_summary()
    summary2 = tracker2.get_summary()
    
    assert abs(summary1["current_kl"] - summary2["current_kl"]) < 1e-12
    assert abs(summary1["current_kl_coef"] - summary2["current_kl_coef"]) < 1e-12
    print("✓ Tracker consistency works")


def main():
    """Run all tests."""
    print("🧪 Running KL Divergence Stability Tests")
    print("=" * 50)
    
    try:
        test_kl_divergence_basic()
        test_kl_divergence_edge_cases()
        test_safe_value_processing()
        test_kl_schedule_tracker()
        test_numerical_stability()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! KL divergence calculations are now numerically stable.")
        print("✅ Fixed issues:")
        print("   - Improved numerical stability in KL divergence calculation")
        print("   - Added comprehensive input validation and error handling")
        print("   - Enhanced KL schedule tracker robustness")
        print("   - Added edge case handling for extreme values")
        print("   - Implemented safe value processing functions")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())