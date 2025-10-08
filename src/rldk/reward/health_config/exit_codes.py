"""Exit code mapping for reward health gates."""

import json
import sys
from pathlib import Path
from typing import Any, Dict


def get_exit_code(passed: bool) -> int:
    """
    Map health status to exit codes.

    Args:
        passed: Whether the health check passed

    Returns:
        Exit code: 0 for passed=True, 3 for passed=False
    """
    return 0 if passed else 3


def raise_on_failure(health_path: str) -> None:
    """
    Read health.json and exit with appropriate code based on passed field.

    Args:
        health_path: Path to health.json file

    Raises:
        SystemExit: With exit code 0 (passed) or 3 (failed)
    """
    health_file = Path(health_path)

    if not health_file.exists():
        print(f"Error: Health file not found at {health_path}", file=sys.stderr)
        sys.exit(1)
        return  # For test mocking scenarios

    health_data: Dict[str, Any] = {}
    try:
        with open(health_file) as f:
            health_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in health file: {e}", file=sys.stderr)
        sys.exit(1)
        return  # For test mocking scenarios
    except Exception as e:
        print(f"Error: Failed to read health file: {e}", file=sys.stderr)
        sys.exit(1)
        return  # For test mocking scenarios

    if 'passed' not in health_data:
        print("Error: 'passed' field missing from health data", file=sys.stderr)
        sys.exit(1)
        return  # For test mocking scenarios

    passed = health_data['passed']
    exit_code = get_exit_code(passed)

    # Print summary
    warnings = health_data.get('warnings', [])
    failures = health_data.get('failures', [])

    if passed:
        print("✅ Health check passed")
        if warnings:
            print(f"Warnings: {len(warnings)}")
            for warning in warnings:
                print(warning)
    else:
        print("🚨 Health check failed")
        if failures:
            print(f"Failures: {len(failures)}")
            for failure in failures:
                print(failure)
        if warnings:
            print(f"Warnings: {len(warnings)}")
            for warning in warnings:
                print(warning)

    sys.exit(exit_code)
