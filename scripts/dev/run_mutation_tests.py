#!/usr/bin/env python3
"""
Script to run mutation testing with mutmut.
"""

import subprocess
import sys
from pathlib import Path


def run_mutation_tests():
    """Run mutation testing on core modules."""
    print("🧬 Running mutation testing with mutmut...")

    # Install mutmut if not available
    try:
        import mutmut
    except ImportError:
        print("Installing mutmut...")
        subprocess.run([sys.executable, "-m", "pip", "install", "mutmut"], check=True)

    # Run mutation testing on seed module
    print("\n🔬 Testing seed module...")
    result = subprocess.run([
        sys.executable, "-m", "mutmut", "run",
        "--paths-to-mutate=src/rldk/utils/seed.py",
        "--simple-output"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Seed module mutation testing passed")
    else:
        print(f"⚠️ Seed module mutation testing issues: {result.stdout}")

    # Run mutation testing on validation module
    print("\n🔬 Testing validation module...")
    result = subprocess.run([
        sys.executable, "-m", "mutmut", "run",
        "--paths-to-mutate=src/rldk/utils/validation.py",
        "--simple-output"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Validation module mutation testing passed")
    else:
        print(f"⚠️ Validation module mutation testing issues: {result.stdout}")

    # Run mutation testing on error handling module
    print("\n🔬 Testing error handling module...")
    result = subprocess.run([
        sys.executable, "-m", "mutmut", "run",
        "--paths-to-mutate=src/rldk/utils/error_handling.py",
        "--simple-output"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Error handling module mutation testing passed")
    else:
        print(f"⚠️ Error handling module mutation testing issues: {result.stdout}")

    # Run mutation testing on progress module
    print("\n🔬 Testing progress module...")
    result = subprocess.run([
        sys.executable, "-m", "mutmut", "run",
        "--paths-to-mutate=src/rldk/utils/progress.py",
        "--simple-output"
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print("✅ Progress module mutation testing passed")
    else:
        print(f"⚠️ Progress module mutation testing issues: {result.stdout}")

    print("\n🎯 Mutation testing completed!")
    print("Note: Some mutations may be expected (e.g., in error handling paths)")


if __name__ == "__main__":
    run_mutation_tests()
