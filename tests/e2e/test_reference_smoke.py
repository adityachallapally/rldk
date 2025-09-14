"""Test reference smoke tests produce expected outputs."""

import json
from pathlib import Path

import pytest


def test_reference_smoke_outputs_exist():
    """Test that reference:cpu_smoke creates the expected output files."""
    expected_dir = Path("reference/expected")

    # Check that expected directory exists
    assert expected_dir.exists(), f"Expected directory {expected_dir} does not exist"

    # Check that all required files exist
    required_files = [
        "determinism_card.json",
        "drift_card.json",
        "reward_card.json",
        "determinism_card.png",
    ]

    for filename in required_files:
        file_path = expected_dir / filename
        assert file_path.exists(), f"Required file {filename} does not exist"

        # For JSON files, check they contain valid JSON
        if filename.endswith(".json"):
            with open(file_path) as f:
                try:
                    data = json.load(f)
                    assert isinstance(
                        data, (dict, list)
                    ), f"File {filename} does not contain valid JSON object or array"
                except json.JSONDecodeError as e:
                    pytest.fail(f"File {filename} contains invalid JSON: {e}")


def test_determinism_card_structure():
    """Test that determinism card has required structure."""
    card_path = Path("reference/expected/determinism_card.json")

    if card_path.exists():
        with open(card_path) as f:
            data = json.load(f)

        # Check for required keys (adjust based on actual determinism card structure)
        required_keys = ["passed", "replicas", "metrics_compared"]
        for key in required_keys:
            assert key in data, f"Determinism card missing required key: {key}"


def test_drift_card_structure():
    """Test that drift card has required structure."""
    card_path = Path("reference/expected/drift_card.json")

    if card_path.exists():
        with open(card_path) as f:
            data = json.load(f)

        # Check for required keys (adjust based on actual drift card structure)
        required_keys = ["diverged", "first_step", "tripped_signals"]
        for key in required_keys:
            assert key in data, f"Drift card missing required key: {key}"


def test_reward_card_structure():
    """Test that reward health card has required structure."""
    card_path = Path("reference/expected/reward_card.json")

    if card_path.exists():
        with open(card_path) as f:
            data = json.load(f)

        # Check for required keys (adjust based on actual reward health card structure)
        required_keys = ["overall_status", "drift_detected", "calibration_score"]
        for key in required_keys:
            assert key in data, f"Reward health card missing required key: {key}"


def test_determinism_png_exists():
    """Test that determinism card PNG file exists and has content."""
    png_path = Path("reference/expected/determinism_card.png")

    assert png_path.exists(), "Determinism card PNG file does not exist"

    # Check file has content (not empty)
    file_size = png_path.stat().st_size
    assert file_size > 0, "Determinism card PNG file is empty"


def test_expected_directory_structure():
    """Test that expected directory contains all required subdirectories."""
    expected_dir = Path("reference/expected")

    # Check for analysis subdirectories
    analysis_dirs = ["determinism_analysis", "drift_analysis", "reward_analysis"]

    for dir_name in analysis_dirs:
        dir_path = expected_dir / dir_name
        assert dir_path.exists(), f"Analysis directory {dir_name} does not exist"


if __name__ == "__main__":
    pytest.main([__file__])
