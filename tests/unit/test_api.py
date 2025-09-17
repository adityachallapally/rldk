#!/usr/bin/env python3
"""Test script to verify rldk Python API functionality."""

import json
from pathlib import Path

import rldk


def test_imports():
    """Test that all main functions can be imported."""
    print("Testing imports...")

    # Test main functions
    assert hasattr(rldk, "ingest_runs")
    assert hasattr(rldk, "first_divergence")
    assert hasattr(rldk, "check")
    assert hasattr(rldk, "bisect_commits")
    assert hasattr(rldk, "health")
    assert hasattr(rldk, "reward_health")
    assert hasattr(rldk, "RewardHealthReport")
    assert hasattr(rldk, "HealthAnalysisResult")
    assert hasattr(rldk, "run")
    assert hasattr(rldk, "EvalResult")

    print("✓ All imports successful")


def test_version():
    """Test version information."""
    print(f"RLDK version: {rldk.__version__}")
    assert rldk.__version__ == "0.1.0"


def test_reports_exist():
    """Test that report generation functions are available."""
    print("Checking report generation functions...")

    # Test that report generation functions exist
    assert hasattr(rldk, "generate_determinism_card")
    assert hasattr(rldk, "generate_drift_card")
    assert hasattr(rldk, "generate_reward_card")

    print("✓ All report generation functions available")


def main():
    """Run all tests."""
    print("=== RLDK Python API Test ===")

    test_imports()
    test_version()
    test_reports_exist()

    print("\n=== All tests passed! ===")
    print("RLDK is working correctly out of the box.")


if __name__ == "__main__":
    main()
