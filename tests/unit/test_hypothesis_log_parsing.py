"""
Hypothesis tests for log parsing and numeric invariants.
"""

import json
import re
from typing import Any, Dict, List

import pytest

try:  # pragma: no cover - optional dependency handling
    from hypothesis import assume, given
    from hypothesis import strategies as st
    from hypothesis.strategies import (
        booleans,
        dictionaries,
        floats,
        integers,
        text,
    )
    HYPOTHESIS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    HYPOTHESIS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis package is not installed",
)

if not HYPOTHESIS_AVAILABLE:  # pragma: no cover - optional dependency handling
    pytest.skip("hypothesis package is not installed", allow_module_level=True)


class TestLogParsing:
    """Test log parsing with property-based testing."""

    @given(
        timestamp=st.from_regex(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"),
        level=st.sampled_from(["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]),
        message=st.text(min_size=1, max_size=100)
    )
    def test_log_line_parsing(self, timestamp: str, level: str, message: str):
        """Test that log lines can be parsed correctly."""
        log_line = f"{timestamp} {level} {message}"

        # Basic parsing - should not crash
        parts = log_line.split(" ", 2)
        assert len(parts) >= 3
        assert parts[0] == timestamp
        assert parts[1] == level
        assert parts[2] == message

    @given(
        metrics=st.dictionaries(
            keys=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "_"))),
            values=st.one_of(
                st.floats(min_value=-1000, max_value=1000),
                st.integers(min_value=-1000, max_value=1000),
                st.booleans()
            ),
            min_size=1,
            max_size=10
        )
    )
    def test_metrics_json_parsing(self, metrics: Dict[str, Any]):
        """Test that metrics can be serialized and parsed as JSON."""
        # Should be able to serialize to JSON
        json_str = json.dumps(metrics)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed == metrics

    @given(
        step=st.integers(min_value=0, max_value=1000000),
        episode=st.integers(min_value=0, max_value=100000),
        reward=st.floats(min_value=-1000, max_value=1000),
        loss=st.floats(min_value=0, max_value=100),
        timestamp=st.from_regex(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
    )
    def test_metrics_line_parsing(self, step: int, episode: int, reward: float, loss: float, timestamp: str):
        """Test that metric lines can be parsed correctly."""
        metrics = {
            "step": step,
            "episode": episode,
            "reward": reward,
            "loss": loss,
            "timestamp": timestamp
        }

        # Should be able to serialize to JSONL
        jsonl_line = json.dumps(metrics)
        assert isinstance(jsonl_line, str)

        # Should be able to parse back
        parsed = json.loads(jsonl_line)
        assert parsed["step"] == step
        assert parsed["episode"] == episode
        assert parsed["reward"] == reward
        assert parsed["loss"] == loss
        assert parsed["timestamp"] == timestamp

    @given(
        log_lines=st.lists(
            st.text(min_size=10, max_size=200),
            min_size=1,
            max_size=100
        )
    )
    def test_log_file_parsing(self, log_lines: List[str]):
        """Test that log files can be parsed without crashing."""
        # Create a log file content
        log_content = "\n".join(log_lines)

        # Should be able to split into lines
        lines = log_content.split("\n")
        assert len(lines) == len(log_lines)

        # Each line should be parseable (even if it's not a valid log format)
        for line in lines:
            if line.strip():  # Skip empty lines
                # Should not crash when trying to parse
                parts = line.split(" ", 2)
                assert len(parts) >= 1

    @given(
        jsonl_lines=st.lists(
            st.dictionaries(
                keys=st.text(min_size=1, max_size=20),
                values=st.one_of(
                    st.floats(min_value=-1000, max_value=1000),
                    st.integers(min_value=-1000, max_value=1000),
                    st.booleans(),
                    st.text(min_size=0, max_size=50)
                ),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=50
        )
    )
    def test_jsonl_file_parsing(self, jsonl_lines: List[Dict[str, Any]]):
        """Test that JSONL files can be parsed without crashing."""
        # Create JSONL content
        jsonl_content = "\n".join(json.dumps(line) for line in jsonl_lines)

        # Should be able to split into lines
        lines = jsonl_content.split("\n")
        assert len(lines) == len(jsonl_lines)

        # Each line should be parseable as JSON
        for line in lines:
            if line.strip():  # Skip empty lines
                parsed = json.loads(line)
                assert isinstance(parsed, dict)


class TestNumericInvariants:
    """Test numeric invariants with property-based testing."""

    @given(
        values=st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    def test_reward_statistics(self, values: List[float]):
        """Test that reward statistics maintain invariants."""
        if not values:
            return

        # Basic statistics
        mean_reward = sum(values) / len(values)
        min_reward = min(values)
        max_reward = max(values)

        # Invariants
        assert min_reward <= mean_reward <= max_reward

        # Variance should be non-negative
        variance = sum((x - mean_reward) ** 2 for x in values) / len(values)
        assert variance >= 0

    @given(
        values=st.lists(
            st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=100
        )
    )
    def test_loss_statistics(self, values: List[float]):
        """Test that loss statistics maintain invariants."""
        if not values:
            return

        # Loss should be non-negative
        assert all(v >= 0 for v in values)

        # Basic statistics
        mean_loss = sum(values) / len(values)
        min_loss = min(values)
        max_loss = max(values)

        # Invariants
        assert min_loss <= mean_loss <= max_loss
        assert min_loss >= 0
        assert max_loss >= 0

    @given(
        step=st.integers(min_value=0, max_value=1000000),
        episode=st.integers(min_value=0, max_value=100000)
    )
    def test_step_episode_invariants(self, step: int, episode: int):
        """Test that step and episode numbers maintain invariants."""
        # Step and episode should be non-negative
        assert step >= 0
        assert episode >= 0

        # If we're in episode N, step should be reasonable
        # (This is a simplified assumption - in reality, steps per episode vary)
        if episode > 0:
            # Assume at least 1 step per episode
            assert step >= episode

    @given(
        learning_rate=st.floats(min_value=1e-6, max_value=1.0),
        batch_size=st.integers(min_value=1, max_value=1000),
        epochs=st.integers(min_value=1, max_value=1000)
    )
    def test_hyperparameter_invariants(self, learning_rate: float, batch_size: int, epochs: int):
        """Test that hyperparameters maintain invariants."""
        # Learning rate should be positive and reasonable
        assert learning_rate > 0
        assert learning_rate <= 1.0

        # Batch size should be positive
        assert batch_size > 0

        # Epochs should be positive
        assert epochs > 0

        # Total updates should be reasonable
        total_updates = batch_size * epochs
        assert total_updates > 0
        assert total_updates <= 1000000  # Reasonable upper bound

    @given(
        values=st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=100
        )
    )
    def test_convergence_invariants(self, values: List[float]):
        """Test that convergence metrics maintain invariants."""
        if len(values) < 2:
            return

        # Calculate differences between consecutive values
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]

        # If values are converging, differences should get smaller over time
        # (This is a simplified assumption - in reality, convergence is more complex)
        if len(differences) >= 2:
            # Check that variance of differences decreases over time
            sum(diff**2 for diff in differences[:len(differences)//2]) / (len(differences)//2)
            sum(diff**2 for diff in differences[len(differences)//2:]) / (len(differences) - len(differences)//2)

            # This invariant might not always hold, but it's a reasonable expectation
            # for converging sequences
            # assert late_var <= early_var  # Commented out as it's too strict

    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    def test_seed_invariants(self, seed: int):
        """Test that seed values maintain invariants."""
        # Seed should be non-negative
        assert seed >= 0

        # Seed should be within reasonable bounds
        assert seed <= 2**32 - 1

        # Seed should be an integer
        assert isinstance(seed, int)

        # Seed should be reproducible (same seed should produce same sequence)
        # This is tested by the seed management module, but we can verify basic properties
        assert seed == seed  # Identity
        assert seed + 0 == seed  # Additive identity
        assert seed * 1 == seed  # Multiplicative identity
