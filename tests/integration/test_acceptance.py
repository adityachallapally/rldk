#!/usr/bin/env python3
"""Acceptance test for Phase A forensics functionality."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Running: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Failed with return code {result.returncode}")
            if result.stderr.strip():
                print(f"Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def has_pytorch():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def main():
    """Run acceptance tests."""
    print("🚀 RL Debug Kit Phase A Acceptance Test")
    print("=" * 50)

    # Check if fixtures exist
    artifact_root = "data/fixtures/test_artifacts"
    if not Path(artifact_root).exists():
        print("❌ Test artifacts not found. Run: python3 tests/_make_fixtures.py")
        return False

    # Check PyTorch availability
    pytorch_available = has_pytorch()
    if not pytorch_available:
        print("⚠️  PyTorch not available, skipping checkpoint and reward drift tests")

    tests = [
        # Environment audit
        (f"rldk env-audit {artifact_root}/logs_clean", "Environment audit"),
        # Log scan
        (
            f"rldk log-scan {artifact_root}/logs_doctored_kl_spike",
            "Log scan with KL spike detection",
        ),
    ]

    # Only add PyTorch-dependent tests if PyTorch is available
    if pytorch_available:
        tests.extend([
            # Checkpoint diff
            (
                f"rldk diff-ckpt {artifact_root}/ckpt_identical/a.pt {artifact_root}/ckpt_identical/b.pt",
                "Checkpoint diff",
            ),
            # Reward drift
            (
                f"rldk reward-drift {artifact_root}/reward_drift_demo/rmA {artifact_root}/reward_drift_demo/rmB --prompts {artifact_root}/reward_drift_demo/prompts.jsonl",
                "Reward drift analysis",
            ),
        ])

    # Add non-PyTorch dependent tests
    tests.extend([
        # Doctor
        (f"rldk doctor {artifact_root}/logs_clean", "Comprehensive diagnostics"),
        # Compare runs
        (
            f"rldk compare-runs {artifact_root}/logs_clean {artifact_root}/logs_doctored_kl_spike",
            "Compare runs",
        ),
    ])

    passed = 0
    total = len(tests)

    for cmd, description in tests:
        if run_command(cmd, description):
            passed += 1

    print(f"\n📊 Results: {passed}/{total} tests passed")

    # Check for generated reports
    reports_dir = Path("rldk_reports")
    if reports_dir.exists():
        print(f"\n📁 Generated reports in {reports_dir}:")
        for report_file in reports_dir.glob("*"):
            print(f"  - {report_file.name}")

    if passed == total:
        print("\n🎉 All tests passed! Phase A implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
