#!/usr/bin/env python3
"""Simple test script for KL divergence functionality."""

import warnings

import numpy as np
import _path_setup  # noqa: F401

# Import the specific modules
from rldk.evals.metrics import (
    calculate_kl_divergence,
    calculate_kl_divergence_between_runs,
)
from rldk.forensics.kl_schedule_tracker import (
    KLScheduleMetrics,
    KLScheduleTracker,
    _safe_coefficient_value,
    _safe_kl_value,
)


def test_kl_divergence_basic():
    """Test basic KL divergence calculation."""
    print("Testing basic KL divergence calculation...")

    # Test basic calculation
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])

    kl_div = calculate_kl_divergence(p, q)
    print(f"KL divergence: {kl_div}")
    assert isinstance(kl_div, float)
    assert kl_div >= 0.0
    assert not np.isnan(kl_div)
    assert not np.isinf(kl_div)
    print("✓ Basic KL divergence test passed")

def test_kl_divergence_identical():
    """Test KL divergence between identical distributions."""
    print("Testing identical distributions...")

    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.5, 0.3, 0.2])

    kl_div = calculate_kl_divergence(p, q)
    print(f"KL divergence (identical): {kl_div}")
    assert abs(kl_div) < 1e-6
    print("✓ Identical distributions test passed")

def test_kl_divergence_zeros():
    """Test handling of zero distributions."""
    print("Testing zero distributions...")

    # Both distributions are zero
    p = np.array([0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.0])

    kl_div = calculate_kl_divergence(p, q)
    print(f"KL divergence (both zero): {kl_div}")
    assert kl_div == 0.0
    print("✓ Zero distributions test passed")

def test_kl_divergence_negative_rejection():
    """Test that negative inputs are rejected."""
    print("Testing negative input rejection...")

    p = np.array([0.5, -0.1, 0.3])
    q = np.array([0.4, 0.4, 0.2])

    try:
        calculate_kl_divergence(p, q)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Probability distributions must be non-negative" in str(e)
        print("✓ Negative input rejection test passed")

def test_kl_divergence_nan_rejection():
    """Test that NaN inputs are rejected."""
    print("Testing NaN input rejection...")

    p = np.array([0.5, np.nan, 0.3])
    q = np.array([0.4, 0.4, 0.2])

    try:
        calculate_kl_divergence(p, q)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Input distributions contain NaN values" in str(e)
        print("✓ NaN input rejection test passed")

def test_safe_kl_value():
    """Test safe KL value processing."""
    print("Testing safe KL value processing...")

    # Valid input
    assert _safe_kl_value(0.5) == 0.5

    # None input
    assert _safe_kl_value(None) == 0.0

    # NaN input
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _safe_kl_value(np.nan)
        assert result == 0.0
        assert len(w) == 1
        assert "NaN KL value detected" in str(w[0].message)

    print("✓ Safe KL value processing test passed")

def test_safe_coefficient_value():
    """Test safe coefficient value processing."""
    print("Testing safe coefficient value processing...")

    # Valid input
    assert _safe_coefficient_value(1.0) == 1.0

    # None input
    assert _safe_coefficient_value(None) == 1.0

    # Zero input
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _safe_coefficient_value(0.0)
        assert result == 1.0
        assert len(w) == 1
        assert "Non-positive coefficient value" in str(w[0].message)

    print("✓ Safe coefficient value processing test passed")

def test_kl_schedule_tracker():
    """Test KL schedule tracker."""
    print("Testing KL schedule tracker...")

    tracker = KLScheduleTracker(kl_target=0.1, kl_target_tolerance=0.05)
    assert tracker.kl_target == 0.1
    assert tracker.kl_target_tolerance == 0.05

    # Test normal update
    metrics = tracker.update(step=1, kl_value=0.05, kl_coef=1.0)
    assert isinstance(metrics, KLScheduleMetrics)
    assert metrics.current_kl == 0.05
    assert metrics.current_kl_coef == 1.0

    # Test edge case update
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        metrics = tracker.update(step=2, kl_value=np.nan, kl_coef=np.nan)
        assert metrics.current_kl == 0.0
        assert metrics.current_kl_coef == 1.0
        assert len(w) >= 2

    print("✓ KL schedule tracker test passed")

def test_kl_divergence_between_runs():
    """Test KL divergence between runs."""
    print("Testing KL divergence between runs...")

    run1_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    run2_data = np.array([1.1, 2.1, 2.9, 4.1, 4.9])

    result = calculate_kl_divergence_between_runs(run1_data, run2_data)

    assert isinstance(result, dict)
    assert "kl_divergence" in result
    assert isinstance(result["kl_divergence"], float)
    assert not np.isnan(result["kl_divergence"])
    assert not np.isinf(result["kl_divergence"])

    print("✓ KL divergence between runs test passed")

def main():
    """Run all tests."""
    print("Running KL divergence numeric stability tests...")
    print("=" * 60)

    try:
        test_kl_divergence_basic()
        test_kl_divergence_identical()
        test_kl_divergence_zeros()
        test_kl_divergence_negative_rejection()
        test_kl_divergence_nan_rejection()
        test_safe_kl_value()
        test_safe_coefficient_value()
        test_kl_schedule_tracker()
        test_kl_divergence_between_runs()

        print("=" * 60)
        print("🎉 All tests passed! KL divergence implementation is robust and fast.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
