#!/usr/bin/env python3
"""Replay golden master for RL Debug Kit.

This script replays the same runs and compares:
- exit codes exact match
- stdout one liners exact match
- JSON Schemas validate
- artifact checksums exact match where deterministic, otherwise field level stable deltas only
"""

import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import Draft7Validator

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set deterministic execution
import random

import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from artifact_schemas import get_schema_for_artifact, validate_artifact
from capture_golden_master import (
    CommandResult,
    calculate_file_checksum,
    create_synthetic_data,
    run_cli_command,
    run_cli_tests,
    run_programmatic_tests,
)

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))
from normalize import get_normalized_checksum, normalize_json, normalize_text


@dataclass
class ComparisonResult:
    """Result of comparing current vs golden master."""
    command: str
    exit_code_match: bool
    stdout_match: bool
    stderr_match: bool
    artifacts_match: bool
    schema_valid: bool
    checksum_matches: Dict[str, bool]
    field_deltas: Dict[str, Any]
    passed: bool
    errors: List[str]


@dataclass
class ReplaySummary:
    """Summary of replay comparison."""
    version: str
    timestamp: str
    golden_master_path: str
    comparisons: List[ComparisonResult]
    total_commands: int
    passed_commands: int
    failed_commands: int
    all_passed: bool


def load_golden_master(golden_master_path: Path) -> Dict[str, Any]:
    """Load golden master data."""
    with open(golden_master_path / "golden_master_summary.json") as f:
        return json.load(f)


def compare_json_objects(current: Any, expected: Any, path: str = "") -> List[Tuple[str, Any, Any]]:
    """Compare JSON objects and return differences."""
    differences = []

    if type(current) != type(expected):
        differences.append((path, current, expected))
        return differences

    if isinstance(current, dict):
        # Check for missing keys
        for key in expected:
            if key not in current:
                differences.append((f"{path}.{key}", None, expected[key]))
        # Check for extra keys
        for key in current:
            if key not in expected:
                differences.append((f"{path}.{key}", current[key], None))
        # Check common keys
        for key in current:
            if key in expected:
                differences.extend(compare_json_objects(current[key], expected[key], f"{path}.{key}"))

    elif isinstance(current, list):
        if len(current) != len(expected):
            differences.append((path, len(current), len(expected)))
        else:
            for i, (curr_item, exp_item) in enumerate(zip(current, expected)):
                differences.extend(compare_json_objects(curr_item, exp_item, f"{path}[{i}]"))

    elif current != expected:
        differences.append((path, current, expected))

    return differences


def is_deterministic_artifact(artifact_name: str) -> bool:
    """Check if an artifact type is deterministic."""
    # Artifacts that should have exact checksum matches
    deterministic_artifacts = {
        "ingest_result",
        "diff_result",
        "determinism_result",
        "reward_health_result",
        "eval_result",
        "golden_master_summary",
        "determinism_card",
        "drift_card",
        "reward_card",
        "diff_report",
        "reward_health_summary",
        "eval_summary",
        "run_comparison",
        "reward_drift",
        "ckpt_diff",
        "ppo_scan",
        "env_audit",
        "replay_comparison",
        "tracking_data",
    }
    return artifact_name in deterministic_artifacts


def compare_command_results(
    current: CommandResult,
    expected: Dict[str, Any],
    golden_master_artifacts_dir: Path,
) -> ComparisonResult:
    """Compare current command result with golden master."""
    errors = []

    # Compare exit codes
    exit_code_match = current.exit_code == expected["exit_code"]
    if not exit_code_match:
        errors.append(f"Exit code mismatch: current={current.exit_code}, expected={expected['exit_code']}")

    # Compare stdout (both should already be normalized)
    stdout_match = current.stdout == expected["stdout"]
    if not stdout_match:
        errors.append("STDOUT mismatch")

    # Compare stderr (both should already be normalized)
    stderr_match = current.stderr == expected["stderr"]
    if not stderr_match:
        errors.append("STDERR mismatch")

    # Compare artifacts
    artifacts_match = True
    checksum_matches = {}
    field_deltas = {}

    # Check if artifacts exist in golden master
    expected_artifacts = expected.get("artifacts", [])
    current_artifacts = current.artifacts

    if len(expected_artifacts) != len(current_artifacts):
        artifacts_match = False
        errors.append(f"Artifact count mismatch: current={len(current_artifacts)}, expected={len(expected_artifacts)}")
    else:
        for i, (current_artifact, expected_artifact) in enumerate(zip(current_artifacts, expected_artifacts)):
            current_path = Path(current_artifact)
            expected_path = golden_master_artifacts_dir / Path(expected_artifact).name

            if not current_path.exists():
                errors.append(f"Current artifact missing: {current_artifact}")
                artifacts_match = False
                continue

            if not expected_path.exists():
                errors.append(f"Expected artifact missing: {expected_artifact}")
                artifacts_match = False
                continue

            # Calculate checksums
            current_checksum = calculate_file_checksum(current_path)
            expected_checksum = expected["artifact_checksums"].get(expected_artifact, "")

            checksum_match = current_checksum == expected_checksum
            checksum_matches[current_artifact] = checksum_match

            if not checksum_match:
                # Check if this is a deterministic artifact
                artifact_name = current_path.stem
                if is_deterministic_artifact(artifact_name):
                    errors.append(f"Deterministic artifact checksum mismatch: {current_artifact}")
                    artifacts_match = False
                else:
                    # For non-deterministic artifacts, compare field-level differences
                    try:
                        with open(current_path) as f:
                            current_data = json.load(f)
                        with open(expected_path) as f:
                            expected_data = json.load(f)

                        differences = compare_json_objects(current_data, expected_data)
                        if differences:
                            field_deltas[current_artifact] = differences
                            # Only flag as error if differences are significant
                            significant_diffs = [d for d in differences if not d[0].endswith(('.timestamp', '.duration'))]
                            if significant_diffs:
                                errors.append(f"Non-deterministic artifact has significant field differences: {current_artifact}")
                                artifacts_match = False
                    except Exception as e:
                        errors.append(f"Error comparing artifact {current_artifact}: {e}")
                        artifacts_match = False

    # Validate schemas
    schema_valid = True
    for artifact_path in current.artifacts:
        try:
            with open(artifact_path) as f:
                artifact_data = json.load(f)

            artifact_name = Path(artifact_path).stem
            if not validate_artifact(artifact_name, artifact_data):
                errors.append(f"Schema validation failed for {artifact_name}")
                schema_valid = False
        except Exception as e:
            errors.append(f"Error validating schema for {artifact_path}: {e}")
            schema_valid = False

    passed = exit_code_match and stdout_match and stderr_match and artifacts_match and schema_valid

    return ComparisonResult(
        command=current.command,
        exit_code_match=exit_code_match,
        stdout_match=stdout_match,
        stderr_match=stderr_match,
        artifacts_match=artifacts_match,
        schema_valid=schema_valid,
        checksum_matches=checksum_matches,
        field_deltas=field_deltas,
        passed=passed,
        errors=errors,
    )


def replay_golden_master(golden_master_path: Path) -> ReplaySummary:
    """Replay golden master and compare results."""
    print(f"Loading golden master from: {golden_master_path}")
    golden_master = load_golden_master(golden_master_path)

    print("Creating synthetic test data...")
    data_paths = create_synthetic_data()

    print("Running CLI command tests...")
    cli_results = run_cli_tests(data_paths)

    print("Running programmatic entry point tests...")
    prog_results = run_programmatic_tests(data_paths)

    all_results = cli_results + prog_results

    print("Comparing results with golden master...")
    comparisons = []

    golden_master_artifacts_dir = golden_master_path / "artifacts"

    for i, current_result in enumerate(all_results):
        if i < len(golden_master["commands"]):
            expected_result = golden_master["commands"][i]
            comparison = compare_command_results(current_result, expected_result, golden_master_artifacts_dir)
            comparisons.append(comparison)
        else:
            # New command not in golden master
            comparison = ComparisonResult(
                command=current_result.command,
                exit_code_match=False,
                stdout_match=False,
                stderr_match=False,
                artifacts_match=False,
                schema_valid=True,
                checksum_matches={},
                field_deltas={},
                passed=False,
                errors=["Command not present in golden master"],
            )
            comparisons.append(comparison)

    # Check for missing commands
    for i in range(len(all_results), len(golden_master["commands"])):
        expected_result = golden_master["commands"][i]
        comparison = ComparisonResult(
            command=expected_result["command"],
            exit_code_match=False,
            stdout_match=False,
            stderr_match=False,
            artifacts_match=False,
            schema_valid=True,
            checksum_matches={},
            field_deltas={},
            passed=False,
            errors=["Command missing from current run"],
        )
        comparisons.append(comparison)

    passed_commands = len([c for c in comparisons if c.passed])
    failed_commands = len(comparisons) - passed_commands
    all_passed = failed_commands == 0

    summary = ReplaySummary(
        version="1.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        golden_master_path=str(golden_master_path),
        comparisons=comparisons,
        total_commands=len(comparisons),
        passed_commands=passed_commands,
        failed_commands=failed_commands,
        all_passed=all_passed,
    )

    # Clean up temp data
    shutil.rmtree(data_paths["temp_dir"])

    return summary


def write_replay_report(summary: ReplaySummary, output_dir: Path):
    """Write replay comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write summary
    with open(output_dir / "replay_summary.json", "w") as f:
        json.dump(asdict(summary), f, indent=2, default=str)

    # Write detailed comparison report
    with open(output_dir / "replay_comparison_report.txt", "w") as f:
        f.write("RL Debug Kit Golden Master Replay Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Golden Master: {summary.golden_master_path}\n")
        f.write(f"Timestamp: {summary.timestamp}\n")
        f.write(f"Total Commands: {summary.total_commands}\n")
        f.write(f"Passed: {summary.passed_commands}\n")
        f.write(f"Failed: {summary.failed_commands}\n")
        f.write(f"All Passed: {summary.all_passed}\n\n")

        for comparison in summary.comparisons:
            f.write(f"Command: {comparison.command}\n")
            f.write(f"  Passed: {comparison.passed}\n")
            f.write(f"  Exit Code Match: {comparison.exit_code_match}\n")
            f.write(f"  STDOUT Match: {comparison.stdout_match}\n")
            f.write(f"  STDERR Match: {comparison.stderr_match}\n")
            f.write(f"  Artifacts Match: {comparison.artifacts_match}\n")
            f.write(f"  Schema Valid: {comparison.schema_valid}\n")

            if comparison.checksum_matches:
                f.write("  Checksum Matches:\n")
                for artifact, match in comparison.checksum_matches.items():
                    f.write(f"    {artifact}: {match}\n")

            if comparison.field_deltas:
                f.write("  Field Deltas:\n")
                for artifact, deltas in comparison.field_deltas.items():
                    f.write(f"    {artifact}:\n")
                    for path, current, expected in deltas:
                        f.write(f"      {path}: current={current}, expected={expected}\n")

            if comparison.errors:
                f.write("  Errors:\n")
                for error in comparison.errors:
                    f.write(f"    - {error}\n")

            f.write("\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay golden master and compare results")
    parser.add_argument(
        "--golden-master",
        type=Path,
        default=Path("golden_master_output"),
        help="Path to golden master directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("replay_output"),
        help="Output directory for replay results"
    )

    args = parser.parse_args()

    if not args.golden_master.exists():
        print(f"Error: Golden master directory not found: {args.golden_master}")
        sys.exit(1)

    print("Starting golden master replay...")
    summary = replay_golden_master(args.golden_master)

    print("Writing replay report...")
    write_replay_report(summary, args.output_dir)

    print("\nReplay comparison complete!")
    print(f"Total commands: {summary.total_commands}")
    print(f"Passed: {summary.passed_commands}")
    print(f"Failed: {summary.failed_commands}")
    print(f"All passed: {summary.all_passed}")
    print(f"Output directory: {args.output_dir}")

    # Exit with appropriate code
    if summary.all_passed:
        print("✅ All comparisons passed!")
        sys.exit(0)
    else:
        print("❌ Some comparisons failed!")
        print("\nFailed commands:")
        for comparison in summary.comparisons:
            if not comparison.passed:
                print(f"  - {comparison.command}")
                for error in comparison.errors:
                    print(f"    {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
