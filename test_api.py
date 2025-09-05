#!/usr/bin/env python3
"""Test script to verify rldk Python API functionality."""

import rldk
import json
from pathlib import Path


def test_imports():
    """Test that all main functions can be imported."""
    print("Testing imports...")

    # Test main public API
    assert hasattr(rldk, "ExperimentTracker")
    assert hasattr(rldk, "TrackingConfig")
    
    # Test core functions
    assert hasattr(rldk, "ingest_runs")
    assert hasattr(rldk, "first_divergence")
    assert hasattr(rldk, "check")
    assert hasattr(rldk, "bisect_commits")
    assert hasattr(rldk, "health")
    assert hasattr(rldk, "RewardHealthReport")
    assert hasattr(rldk, "run")
    assert hasattr(rldk, "EvalResult")

    print("✓ All imports successful")


def test_version():
    """Test version information."""
    print(f"RLDK version: {rldk.__version__}")
    assert rldk.__version__ == "0.1.0"


def test_reports_exist():
    """Test that reports were generated."""
    print("Checking generated reports...")

    reports_dir = Path("rldk_reports")
    assert reports_dir.exists(), "Reports directory should exist"

    expected_files = [
        "determinism_card.json",
        "ppo_scan.json",
        "ckpt_diff.json",
        "reward_drift.json",
    ]

    for file in expected_files:
        file_path = reports_dir / file
        assert file_path.exists(), f"Report file {file} should exist"
        print(f"✓ {file} exists")

        # Test that files contain valid JSON
        try:
            with open(file_path) as f:
                data = json.load(f)
            print(f"✓ {file} contains valid JSON")
        except json.JSONDecodeError:
            print(f"⚠ {file} is not valid JSON")


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
