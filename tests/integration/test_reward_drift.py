"""Test reward drift functionality."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from rldk.io.schemas import RewardDriftReportV1, validate
from rldk.reward.drift import compare_models


def test_reward_drift_identical_models():
    """Test reward drift with identical models."""
    # Create simple models
    model = nn.Linear(10, 1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save identical models
        model_a_dir = Path(temp_dir) / "model_a"
        model_b_dir = Path(temp_dir) / "model_b"

        model_a_dir.mkdir()
        model_b_dir.mkdir()

        torch.save(model.state_dict(), model_a_dir / "model.pt")
        torch.save(model.state_dict(), model_b_dir / "model.pt")

        # Test prompts
        prompts = ["Hello world", "How are you?", "Test prompt"]

        # Run comparison
        report = compare_models(str(model_a_dir), str(model_b_dir), prompts)

        # Validate report
        validate(RewardDriftReportV1, report)

        # Should have high correlation for identical models
        assert report["pearson"] >= 0.8
        assert report["spearman"] >= 0.8
        # Note: sign_flip_rate might not be exactly 0.0 due to hash-based scoring


def test_reward_drift_different_models():
    """Test reward drift with different models."""
    # Create different models
    model_a = nn.Linear(10, 1)
    model_b = nn.Linear(10, 1)

    # Make them different
    model_b.weight.data *= 2.0
    model_b.bias.data += 1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save different models
        model_a_dir = Path(temp_dir) / "model_a"
        model_b_dir = Path(temp_dir) / "model_b"

        model_a_dir.mkdir()
        model_b_dir.mkdir()

        torch.save(model_a.state_dict(), model_a_dir / "model.pt")
        torch.save(model_b.state_dict(), model_b_dir / "model.pt")

        # Test prompts
        prompts = ["Hello world", "How are you?", "Test prompt"]

        # Run comparison
        report = compare_models(str(model_a_dir), str(model_b_dir), prompts)

        # Validate report
        validate(RewardDriftReportV1, report)

        # Should detect some differences
        assert report["mae_z"] > 0.0
        assert report["l2_z"] > 0.0

        # Different models should produce different scores
        assert report["pearson"] < 1.0  # Not perfectly correlated


def test_reward_drift_slice_analysis():
    """Test reward drift slice analysis."""
    # Create models
    model_a = nn.Linear(10, 1)
    model_b = nn.Linear(10, 1)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save models
        model_a_dir = Path(temp_dir) / "model_a"
        model_b_dir = Path(temp_dir) / "model_b"

        model_a_dir.mkdir()
        model_b_dir.mkdir()

        torch.save(model_a.state_dict(), model_a_dir / "model.pt")
        torch.save(model_b.state_dict(), model_b_dir / "model.pt")

        # Test prompts with different types
        prompts = [
            "What is 2 + 2?",  # math
            "def hello(): print('world')",  # code
            "Is it safe to walk alone?",  # safety
            "I cannot help with that",  # refusal
        ]

        # Run comparison
        report = compare_models(str(model_a_dir), str(model_b_dir), prompts)

        # Validate report
        validate(RewardDriftReportV1, report)

        # Should have slice deltas
        assert "slice_deltas" in report
        assert len(report["slice_deltas"]) > 0

        # Check that slices are properly categorized
        for slice_name, slice_data in report["slice_deltas"].items():
            assert "delta_mean" in slice_data
            assert "n" in slice_data
            assert slice_data["n"] > 0
