"""Lightweight validator utility for JSONL files."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .event_schema import Event, create_event_from_row


def validate_jsonl_schema(
    file_path: Path,
    required_fields: Optional[List[str]] = None,
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate JSONL file for schema conformance.

    Args:
        file_path: Path to JSONL file to validate
        required_fields: List of required fields (defaults to Event schema fields)
        strict: If True, require exact Event schema compliance

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if required_fields is None:
        required_fields = [
            "step", "phase", "reward_mean", "kl_mean", "entropy_mean",
            "clip_frac", "grad_norm", "lr", "loss"
        ]

    errors = []

    if not file_path.exists():
        return False, [f"File does not exist: {file_path}"]

    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    if strict:
                        # Try to create Event object - be more flexible for custom formats
                        try:
                            # First try direct Event creation
                            event = create_event_from_row(data, "validation", None)
                            # Verify Event object is valid
                            event_dict = event.to_dict()
                            if not isinstance(event_dict, dict):
                                errors.append(f"Line {line_num}: Invalid Event object structure")
                        except Exception as e:
                            # If direct creation fails, try to detect if it's a custom format
                            # and suggest using an adapter
                            custom_indicators = ["global_step", "reward_scalar", "kl_to_ref"]
                            if any(key in data for key in custom_indicators):
                                errors.append(f"Line {line_num}: Custom format detected - use CustomJSONLAdapter for proper mapping: {e}")
                            else:
                                errors.append(f"Line {line_num}: Event schema validation failed: {e}")
                    else:
                        # Check for required fields
                        missing_fields = [field for field in required_fields if field not in data]
                        if missing_fields:
                            errors.append(f"Line {line_num}: Missing required fields: {missing_fields}")

                        # Check for non-primitive types
                        non_primitive = _check_primitive_types(data)
                        if non_primitive:
                            errors.append(f"Line {line_num}: Non-primitive types found: {non_primitive}")

                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: JSON decode error: {e}")
                except Exception as e:
                    errors.append(f"Line {line_num}: Unexpected error: {e}")

    except Exception as e:
        errors.append(f"File read error: {e}")

    return len(errors) == 0, errors


def _check_primitive_types(obj: Any, path: str = "") -> List[str]:
    """Check for non-primitive types in object."""
    non_primitive = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            non_primitive.extend(_check_primitive_types(value, current_path))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{path}[{i}]"
            non_primitive.extend(_check_primitive_types(item, current_path))
    else:
        # Check if it's a primitive type
        if not isinstance(obj, (int, float, str, bool, type(None))):
            non_primitive.append(f"{path}: {type(obj).__name__}")

    return non_primitive


def validate_jsonl_file(
    file_path: Path,
    output_errors: bool = True,
    max_errors: int = 10
) -> bool:
    """
    Validate a JSONL file and optionally output errors.

    Args:
        file_path: Path to JSONL file
        output_errors: Whether to print errors to console
        max_errors: Maximum number of errors to report

    Returns:
        True if file is valid, False otherwise
    """
    is_valid, errors = validate_jsonl_schema(file_path)

    if output_errors and errors:
        print(f"Validation errors in {file_path}:")
        for error in errors[:max_errors]:
            print(f"  {error}")
        if len(errors) > max_errors:
            print(f"  ... and {len(errors) - max_errors} more errors")

    return is_valid


def validate_jsonl_directory(
    directory: Path,
    pattern: str = "*.jsonl",
    output_errors: bool = True
) -> Dict[str, bool]:
    """
    Validate all JSONL files in a directory.

    Args:
        directory: Directory containing JSONL files
        pattern: File pattern to match
        output_errors: Whether to print errors to console

    Returns:
        Dictionary mapping file paths to validation results
    """
    results = {}

    if not directory.exists():
        logging.warning(f"Directory does not exist: {directory}")
        return results

    for file_path in directory.glob(pattern):
        results[str(file_path)] = validate_jsonl_file(file_path, output_errors)

    return results


def create_jsonl_validator(
    required_fields: Optional[List[str]] = None,
    strict: bool = True
):
    """
    Create a JSONL validator function with specific requirements.

    Args:
        required_fields: List of required fields
        strict: Whether to use strict Event schema validation

    Returns:
        Validator function that takes a file path and returns (is_valid, errors)
    """
    def validator(file_path: Path) -> Tuple[bool, List[str]]:
        return validate_jsonl_schema(file_path, required_fields, strict)

    return validator


def validate_event_schema_compatibility(
    file_path: Path
) -> Tuple[bool, List[str]]:
    """
    Validate that JSONL file is compatible with Event schema.

    Args:
        file_path: Path to JSONL file

    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []

    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Check if we can create an Event object
                    try:
                        event = create_event_from_row(data, "validation", None)

                        # Verify the Event object can be serialized
                        event_dict = event.to_dict()
                        event_json = event.to_json()

                        # Verify we can recreate the Event from JSON
                        recreated_event = Event.from_json(event_json)

                        # Verify the recreated event matches the original
                        if event_dict != recreated_event.to_dict():
                            issues.append(f"Line {line_num}: Event serialization/deserialization mismatch")

                    except Exception as e:
                        issues.append(f"Line {line_num}: Event schema compatibility failed: {e}")

                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: JSON decode error: {e}")

    except Exception as e:
        issues.append(f"File read error: {e}")

    return len(issues) == 0, issues


def validate_custom_jsonl_with_adapter(
    file_path: Path
) -> Tuple[bool, List[str]]:
    """
    Validate custom JSONL file using the CustomJSONLAdapter.

    Args:
        file_path: Path to custom JSONL file

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    try:
        # Import here to avoid circular imports
        from ..adapters.custom_jsonl import CustomJSONLAdapter

        # Use the adapter to load and convert the data
        adapter = CustomJSONLAdapter(file_path)
        df = adapter.load()

        # Validate that we can create Event objects from the adapter output
        for idx, row in df.iterrows():
            try:
                event = create_event_from_row(row.to_dict(), "validation", None)

                # Verify the Event object can be serialized
                event_dict = event.to_dict()
                event_json = event.to_json()

                # Verify we can recreate the Event from JSON
                recreated_event = Event.from_json(event_json)

                # Verify the recreated event matches the original
                if event_dict != recreated_event.to_dict():
                    issues.append(f"Row {idx}: Event serialization/deserialization mismatch")

            except Exception as e:
                issues.append(f"Row {idx}: Event creation failed: {e}")

    except Exception as e:
        issues.append(f"Adapter validation failed: {e}")

    return len(issues) == 0, issues


def validate_jsonl_consistency(
    file_path: Path,
    check_sequential_steps: bool = True,
    check_monotonic_time: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate JSONL file for consistency issues.

    Args:
        file_path: Path to JSONL file
        check_sequential_steps: Whether to check that steps are sequential
        check_monotonic_time: Whether to check that wall_time is monotonic

    Returns:
        Tuple of (is_consistent, list_of_issues)
    """
    issues = []
    steps = []
    times = []

    try:
        with open(file_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Check step field
                    if "step" in data:
                        step = data["step"]
                        if not isinstance(step, (int, float)):
                            issues.append(f"Line {line_num}: Step must be numeric, got {type(step)}")
                        else:
                            steps.append((line_num, step))

                    # Check wall_time field
                    if "wall_time" in data:
                        wall_time = data["wall_time"]
                        if not isinstance(wall_time, (int, float)):
                            issues.append(f"Line {line_num}: Wall time must be numeric, got {type(wall_time)}")
                        else:
                            times.append((line_num, wall_time))

                except json.JSONDecodeError as e:
                    issues.append(f"Line {line_num}: JSON decode error: {e}")

    except Exception as e:
        issues.append(f"File read error: {e}")

    # Check sequential steps (in file order)
    if check_sequential_steps and steps:
        for i, (line_num, step) in enumerate(steps):
            if i > 0 and step <= steps[i-1][1]:
                issues.append(f"Line {line_num}: Non-sequential step {step} (previous: {steps[i-1][1]})")

    # Check monotonic time (in file order)
    if check_monotonic_time and times:
        for i, (line_num, time_val) in enumerate(times):
            if i > 0 and time_val < times[i-1][1]:
                issues.append(f"Line {line_num}: Non-monotonic time {time_val} (previous: {times[i-1][1]})")

    return len(issues) == 0, issues


def main():
    """Command-line interface for JSONL validation."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate JSONL files for schema conformance")
    parser.add_argument("file", help="JSONL file to validate")
    parser.add_argument("--strict", action="store_true", help="Use strict Event schema validation")
    parser.add_argument("--check-consistency", action="store_true", help="Check for consistency issues")
    parser.add_argument("--max-errors", type=int, default=10, help="Maximum number of errors to report")

    args = parser.parse_args()

    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File does not exist: {file_path}")
        sys.exit(1)

    # Basic schema validation
    is_valid, errors = validate_jsonl_schema(file_path, strict=args.strict)

    if errors:
        print(f"Schema validation errors in {file_path}:")
        for error in errors[:args.max_errors]:
            print(f"  {error}")
        if len(errors) > args.max_errors:
            print(f"  ... and {len(errors) - args.max_errors} more errors")

    # Consistency validation
    if args.check_consistency:
        is_consistent, consistency_errors = validate_jsonl_consistency(file_path)

        if consistency_errors:
            print(f"Consistency errors in {file_path}:")
            for error in consistency_errors[:args.max_errors]:
                print(f"  {error}")
            if len(consistency_errors) > args.max_errors:
                print(f"  ... and {len(consistency_errors) - args.max_errors} more errors")

        is_valid = is_valid and is_consistent

    if is_valid:
        print(f"✅ {file_path} is valid")
        sys.exit(0)
    else:
        print(f"❌ {file_path} has validation errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
