"""Test environment audit functionality."""

import tempfile

from rldk.forensics.env_audit import audit_environment
from rldk.io.schemas import DeterminismCardV1, validate


def test_env_audit():
    """Test environment audit on a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Run audit
        determinism_card, lock_content = audit_environment(temp_dir)

        # Validate determinism card
        validate(DeterminismCardV1, determinism_card)

        # Check required fields
        assert "version" in determinism_card
        assert "rng" in determinism_card
        assert "flags" in determinism_card
        assert "nondeterminism_hints" in determinism_card
        assert "pass" in determinism_card

        # Check flags
        flags = determinism_card["flags"]
        assert "cudnn_deterministic" in flags
        assert "cudnn_benchmark" in flags
        assert "tokenizers_parallelism" in flags

        # Check lock content
        assert "RL Debug Kit Environment Lock" in lock_content
        assert "Python:" in lock_content
        assert "PyTorch:" in lock_content


def test_env_audit_nondeterminism_detection():
    """Test that nondeterminism hints are properly detected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        determinism_card, _ = audit_environment(temp_dir)

        # Should have some nondeterminism hints (like missing seeds)
        assert isinstance(determinism_card["nondeterminism_hints"], list)

        # Check that pass is boolean
        assert isinstance(determinism_card["pass"], bool)


def test_env_audit_boolean_flags():
    """Test that boolean flags are properly converted."""
    with tempfile.TemporaryDirectory() as temp_dir:
        determinism_card, _ = audit_environment(temp_dir)

        # Check that flags are boolean
        flags = determinism_card["flags"]
        assert isinstance(flags["cudnn_deterministic"], bool)
        assert isinstance(flags["cudnn_benchmark"], bool)
        assert isinstance(flags["tokenizers_parallelism"], (str, type(None)))
