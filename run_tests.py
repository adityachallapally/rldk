#!/usr/bin/env python3
"""
Test runner script for RL Debug Kit.

This script provides an easy way to run different types of tests with proper
environment setup.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def validate_project_layout() -> bool:
    """Ensure test and documentation files follow the expected layout."""
    project_root = Path(__file__).parent
    issues = []

    stray_tests = sorted(project_root.glob("test_*.py"))
    if stray_tests:
        issues.append(
            "Top-level test scripts detected: "
            + ", ".join(path.name for path in stray_tests)
        )

    stray_summaries = []
    for pattern in ("*_SUMMARY.md", "notes.md"):
        stray_summaries.extend(project_root.glob(pattern))
    if stray_summaries:
        issues.append(
            "Summary markdown files must live under docs/: "
            + ", ".join(path.name for path in sorted(stray_summaries))
        )

    if issues:
        print("Project layout validation failed:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease move the files into the appropriate tests/ or docs/reports/ folders.")
        return False

    return True


def setup_environment():
    """Set up the Python environment for testing."""
    # Add src to Python path
    project_root = Path(__file__).parent
    src_path = project_root / "src"
    tools_path = project_root / "tools"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    if str(tools_path) not in sys.path:
        sys.path.insert(0, str(tools_path))

    # Set environment variables
    os.environ["RLDK_TEST_MODE"] = "true"
    os.environ["WANDB_MODE"] = "disabled"


def run_tests(test_type="unit", verbose=True, specific_test=None):
    """Run tests with proper environment setup."""
    if not validate_project_layout():
        return 1

    setup_environment()

    # Build pytest command
    cmd = ["python3", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    elif test_type == "e2e":
        cmd.append("tests/e2e/")
    elif test_type == "all":
        cmd.append("tests/")
    else:
        cmd.append(f"tests/{test_type}/")

    # Add specific test if provided
    if specific_test:
        cmd.append(specific_test)

    # Add pytest options
    cmd.extend([
        "--tb=short",
        "--disable-warnings",
        "-x"  # Stop on first failure
    ])

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)

    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run RL Debug Kit tests")
    parser.add_argument(
        "test_type",
        nargs="?",
        default="unit",
        choices=["unit", "integration", "e2e", "all"],
        help="Type of tests to run (default: unit)"
    )
    parser.add_argument(
        "-t", "--test",
        help="Specific test to run (e.g., test_basic_imports.py::test_imports)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Run tests quietly (less verbose output)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run tests with coverage reporting"
    )

    args = parser.parse_args()

    # Add coverage if requested
    if args.coverage:
        print("Note: Coverage reporting requires pytest-cov to be installed")
        print("Install with: pip install pytest-cov")

    # Run the tests
    return run_tests(
        test_type=args.test_type,
        verbose=not args.quiet,
        specific_test=args.test
    )


if __name__ == "__main__":
    sys.exit(main())
