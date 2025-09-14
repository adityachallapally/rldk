"""Test checkpoint diff functionality."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from rldk.forensics.ckpt_diff import diff_checkpoints
from rldk.io.schemas import CkptDiffReportV1, validate


def test_diff_identical_checkpoints():
    """Test diff on identical checkpoints."""
    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save identical checkpoints
        ckpt_a = Path(temp_dir) / "a.pt"
        ckpt_b = Path(temp_dir) / "b.pt"

        torch.save(model.state_dict(), ckpt_a)
        torch.save(model.state_dict(), ckpt_b)

        # Run diff
        report = diff_checkpoints(str(ckpt_a), str(ckpt_b))

        # Validate report
        validate(CkptDiffReportV1, report)

        # Should have high cosine similarity for identical checkpoints
        assert report["summary"]["avg_cosine"] >= 0.9999
        assert report["summary"]["num_params"] > 0


def test_diff_modified_checkpoints():
    """Test diff on modified checkpoints."""
    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save base checkpoint
        ckpt_a = Path(temp_dir) / "a.pt"
        torch.save(model.state_dict(), ckpt_a)

        # Create modified checkpoint
        modified_state = model.state_dict().copy()
        modified_state["2.weight"] = (
            modified_state["2.weight"] * 1.5
        )  # Modify last layer
        modified_state["2.bias"] = modified_state["2.bias"] + 0.1

        ckpt_b = Path(temp_dir) / "b.pt"
        torch.save(modified_state, ckpt_b)

        # Run diff
        report = diff_checkpoints(str(ckpt_a), str(ckpt_b))

        # Validate report
        validate(CkptDiffReportV1, report)

        # Should detect differences
        assert report["summary"]["avg_cosine"] < 0.9999
        assert len(report["top_movers"]) > 0

        # Check that modified parameters are in top movers
        top_mover_names = [m["name"] for m in report["top_movers"]]
        assert "2.weight" in top_mover_names or "2.bias" in top_mover_names


def test_diff_checkpoint_formats():
    """Test diff with different checkpoint formats."""
    # Create a simple model
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save as state_dict
        ckpt_a = Path(temp_dir) / "a.pt"
        torch.save(model.state_dict(), ckpt_a)

        # Save as full checkpoint
        ckpt_b = Path(temp_dir) / "b.pt"
        torch.save(
            {"state_dict": model.state_dict(), "optimizer": None, "epoch": 0}, ckpt_b
        )

        # Run diff
        report = diff_checkpoints(str(ckpt_a), str(ckpt_b))

        # Should work and find identical parameters
        validate(CkptDiffReportV1, report)
        assert report["summary"]["avg_cosine"] >= 0.9999
