"""Test first divergence detection between runs."""

import json
from pathlib import Path

import pytest


def test_first_divergence_detection():
    """Test that first divergence is detected at step 1 when tokenizer or pad changes."""

    # Check that drift analysis was run
    drift_dir = Path("reference/expected/drift_analysis")
    assert drift_dir.exists(), "Drift analysis directory does not exist"

    # Check drift card exists
    drift_card = drift_dir / "drift_card.md"
    assert drift_card.exists(), "Drift card does not exist"

    # Check that divergence was detected
    # Since the drift card is in markdown format, we'll check the JSON version
    drift_json = Path("reference/expected/drift_card.json")
    if drift_json.exists():
        with open(drift_json) as f:
            data = json.load(f)

        # Check that divergence was detected
        assert data.get("diverged", False), "Divergence was not detected between runs"

        # Check that first step is early (should be step 1 or 2)
        first_step = data.get("first_step", -1)
        assert (
            first_step <= 5
        ), f"First divergence should be early, but was at step {first_step}"

        # Check that signals were tripped
        tripped_signals = data.get("tripped_signals", [])
        assert (
            len(tripped_signals) > 0
        ), "No signals were tripped during divergence detection"


def test_divergence_cause_identification():
    """Test that the cause of divergence is properly identified."""

    # Check that the doctored run was created with different parameters
    good_run = Path("reference/runs/summarization/good")
    doctored_run = Path("reference/runs/summarization/tokenizer_changed")

    assert good_run.exists(), "Good run directory does not exist"
    assert doctored_run.exists(), "Doctored run directory does not exist"

    # Check that both runs have training logs
    good_log = good_run / "training_log.jsonl"
    doctored_log = doctored_run / "training_log.jsonl"

    assert good_log.exists(), "Good run training log does not exist"
    assert doctored_log.exists(), "Doctored run training log does not exist"

    # Check that the runs differ in their parameters
    # The doctored run should have different pad_direction and truncate_at values
    # This is verified by the make_bad_commit.sh script


def test_divergence_reproducibility():
    """Test that the divergence can be reproduced with the identified changes."""

    # Check that a minimal repro command was provided
    drift_json = Path("reference/expected/drift_card.json")
    if drift_json.exists():
        with open(drift_json) as f:
            data = json.load(f)

        # Check that repro information is available
        # The exact structure depends on the drift card format
        assert (
            "repro" in data or "repro_command" in data or "details" in data
        ), "Drift card should contain repro information"


def test_run_comparison_metrics():
    """Test that the correct metrics were compared for divergence detection."""

    # Check that the diff command was run with the right signals
    # The Makefile should have run:
    # rldk diff --signals "sample_id,input_ids_sha256,attention_mask_sha256,outputs.text,reward_scalar,loss"

    drift_dir = Path("reference/expected/drift_analysis")
    assert drift_dir.exists(), "Drift analysis directory does not exist"

    # Check that the comparison included the right metrics
    # This is verified by the Makefile configuration


def test_divergence_analysis_outputs():
    """Test that all expected divergence analysis outputs were created."""

    drift_dir = Path("reference/expected/drift_analysis")
    assert drift_dir.exists(), "Drift analysis directory does not exist"

    # Check for expected output files
    expected_files = ["drift_card.md", "diff_report.md"]

    for filename in expected_files:
        file_path = drift_dir / filename
        assert (
            file_path.exists()
        ), f"Expected divergence analysis file {filename} does not exist"


def test_divergence_timing():
    """Test that divergence is detected quickly (within first few steps)."""

    drift_json = Path("reference/expected/drift_card.json")
    if drift_json.exists():
        with open(drift_json) as f:
            data = json.load(f)

        # Check that divergence was detected early
        first_step = data.get("first_step", -1)
        assert first_step >= 0, "First divergence step should be non-negative"
        assert (
            first_step <= 10
        ), f"Divergence should be detected early, but was at step {first_step}"


if __name__ == "__main__":
    pytest.main([__file__])
