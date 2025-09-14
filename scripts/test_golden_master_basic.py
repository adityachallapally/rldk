#!/usr/bin/env python3
"""Basic test for golden master system components."""

import json
import sys
import tempfile
from pathlib import Path

# Add src and scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from artifact_schemas import ARTIFACT_SCHEMAS, validate_artifact


def test_schemas():
    """Test that all schemas are valid JSON."""
    print("Testing JSON schemas...")

    for artifact_name, schema in ARTIFACT_SCHEMAS.items():
        try:
            # Test that schema is valid JSON
            json.dumps(schema)
            print(f"✅ {artifact_name}: Valid JSON")
        except Exception as e:
            print(f"❌ {artifact_name}: Invalid JSON - {e}")
            return False

    return True


def test_schema_validation():
    """Test schema validation with sample data."""
    print("\nTesting schema validation...")

    # Test ingest_result schema
    sample_ingest = {
        "step": [1, 2, 3],
        "loss": [0.5, 0.4, 0.3],
        "reward_scalar": [0.1, 0.15, 0.2],
    }

    if validate_artifact("ingest_result", sample_ingest):
        print("✅ ingest_result validation: Passed")
    else:
        print("❌ ingest_result validation: Failed")
        return False

    # Test determinism_result schema
    sample_determinism = {
        "passed": True,
        "culprit": None,
        "fixes": ["Fix 1", "Fix 2"],
    }

    if validate_artifact("determinism_result", sample_determinism):
        print("✅ determinism_result validation: Passed")
    else:
        print("❌ determinism_result validation: Failed")
        return False

    # Test invalid data
    invalid_data = {"invalid": "data"}
    if not validate_artifact("ingest_result", invalid_data):
        print("✅ Invalid data correctly rejected")
    else:
        print("❌ Invalid data incorrectly accepted")
        return False

    return True


def test_synthetic_data_creation():
    """Test synthetic data creation."""
    print("\nTesting synthetic data creation...")

    try:
        from capture_golden_master import create_synthetic_data

        data_paths = create_synthetic_data()

        # Check that files were created
        required_files = ["run_a", "run_b", "prompts", "model_a", "model_b"]
        for file_key in required_files:
            if file_key in data_paths and data_paths[file_key].exists():
                print(f"✅ {file_key}: Created")
            else:
                print(f"❌ {file_key}: Missing")
                return False

        # Clean up
        import shutil
        shutil.rmtree(data_paths["temp_dir"])
        print("✅ Cleanup: Successful")

        return True

    except Exception as e:
        print(f"❌ Synthetic data creation failed: {e}")
        return False


def test_file_operations():
    """Test file operations used by the system."""
    print("\nTesting file operations...")

    try:
        from capture_golden_master import calculate_file_checksum

        # Create a test file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('{"test": "data"}')
            temp_file = f.name

        # Calculate checksum
        checksum = calculate_file_checksum(Path(temp_file))

        if checksum and len(checksum) == 64:  # SHA256 is 64 chars
            print("✅ Checksum calculation: Passed")
        else:
            print("❌ Checksum calculation: Failed")
            return False

        # Clean up
        Path(temp_file).unlink()
        print("✅ File cleanup: Successful")

        return True

    except Exception as e:
        print(f"❌ File operations failed: {e}")
        return False


def main():
    """Run all basic tests."""
    print("Golden Master System Basic Tests")
    print("=" * 40)

    tests = [
        ("JSON Schemas", test_schemas),
        ("Schema Validation", test_schema_validation),
        ("Synthetic Data Creation", test_synthetic_data_creation),
        ("File Operations", test_file_operations),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print(f"\n{'=' * 40}")
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
