"""Tests for Phase B card functionality."""

import json
import tempfile
from pathlib import Path

import pytest

from rldk.cards.determinism import generate_determinism_card
from rldk.cards.drift import generate_drift_card
from rldk.cards.reward import generate_reward_card
from rldk.io.event_schema import (
    Event,
    create_event_from_row,
    dataframe_to_events,
    events_to_dataframe,
)
from rldk.io.schemas import DeterminismCardV2, DriftCardV1, RewardCardV1, validate


@pytest.fixture
def sample_events():
    """Create sample events for testing."""
    events = []
    for i in range(10):
        event = Event(
            step=i,
            wall_time=100.0 + i,
            metrics={
                "reward_mean": 0.5 + 0.1 * i,
                "reward_std": 0.1,
                "kl_mean": 0.1 + 0.01 * i,
                "entropy_mean": 0.8 - 0.02 * i,
                "clip_frac": 0.1,
                "grad_norm": 1.0 + 0.1 * i,
                "lr": 0.001,
                "loss": 0.5 - 0.05 * i,
            },
            rng={
                "seed": 42,
                "python_hash_seed": "42",
                "torch_seed": 42,
                "numpy_seed": 42,
                "random_seed": 42,
            },
            data_slice={
                "tokens_in": 512,
                "tokens_out": 128,
                "batch_size": 4,
                "sequence_length": 512,
            },
            model_info={
                "run_id": "test_run",
                "git_sha": "abc123",
                "phase": "train",
                "model_name": "test_model",
                "model_size": "7B",
                "optimizer": "adamw",
                "scheduler": "cosine",
            },
            notes=[],
        )
        events.append(event)
    return events


@pytest.fixture
def divergent_events():
    """Create events that will show divergence."""
    events = []
    for i in range(10):
        # Create divergent pattern after step 5
        if i < 5:
            reward = 0.5 + 0.1 * i
            kl = 0.1 + 0.01 * i
        else:
            reward = 0.5 + 0.1 * i + 0.5  # Add divergence
            kl = 0.1 + 0.01 * i + 0.2  # Add divergence

        event = Event(
            step=i,
            wall_time=100.0 + i,
            metrics={
                "reward_mean": reward,
                "reward_std": 0.1,
                "kl_mean": kl,
                "entropy_mean": 0.8 - 0.02 * i,
                "clip_frac": 0.1,
                "grad_norm": 1.0 + 0.1 * i,
                "lr": 0.001,
                "loss": 0.5 - 0.05 * i,
            },
            rng={
                "seed": 42,
                "python_hash_seed": "42",
                "torch_seed": 42,
                "numpy_seed": 42,
                "random_seed": 42,
            },
            data_slice={
                "tokens_in": 512,
                "tokens_out": 128,
                "batch_size": 4,
                "sequence_length": 512,
            },
            model_info={
                "run_id": "divergent_run",
                "git_sha": "def456",
                "phase": "train",
                "model_name": "test_model",
                "model_size": "7B",
                "optimizer": "adamw",
                "scheduler": "cosine",
            },
            notes=[],
        )
        events.append(event)
    return events


class TestEventSchema:
    """Test the normalized event schema."""

    def test_create_event_from_row(self):
        """Test creating an Event from a data row."""
        row = {
            "step": 1,
            "wall_time": 100.0,
            "reward_mean": 0.5,
            "kl_mean": 0.1,
            "entropy_mean": 0.8,
            "clip_frac": 0.1,
            "grad_norm": 1.0,
            "lr": 0.001,
            "loss": 0.5,
            "tokens_in": 512,
            "tokens_out": 128,
            "seed": 42,
            "phase": "train",
        }

        event = create_event_from_row(row, "test_run", "abc123")

        assert event.step == 1
        assert event.wall_time == 100.0
        assert event.metrics["reward_mean"] == 0.5
        assert event.metrics["kl_mean"] == 0.1
        assert event.rng["seed"] == 42
        assert event.model_info["run_id"] == "test_run"
        assert event.model_info["git_sha"] == "abc123"

    def test_event_serialization(self):
        """Test Event serialization to/from JSON."""
        event = Event(
            step=1,
            wall_time=100.0,
            metrics={"reward_mean": 0.5},
            rng={"seed": 42},
            data_slice={"tokens_in": 512},
            model_info={"run_id": "test"},
            notes=["test note"],
        )

        # Test to_dict
        event_dict = event.to_dict()
        assert event_dict["step"] == 1
        assert event_dict["metrics"]["reward_mean"] == 0.5
        assert event_dict["notes"] == ["test note"]

        # Test from_dict
        new_event = Event.from_dict(event_dict)
        assert new_event.step == 1
        assert new_event.metrics["reward_mean"] == 0.5
        assert new_event.notes == ["test note"]

        # Test JSON serialization
        json_str = event.to_json()
        parsed_event = Event.from_json(json_str)
        assert parsed_event.step == 1
        assert parsed_event.metrics["reward_mean"] == 0.5

    def test_events_dataframe_conversion(self, sample_events):
        """Test conversion between events and DataFrame."""
        # Convert events to DataFrame
        df = events_to_dataframe(sample_events)

        assert len(df) == 10
        assert "step" in df.columns
        assert "reward_mean" in df.columns
        assert "run_id" in df.columns
        assert df["run_id"].iloc[0] == "test_run"

        # Convert back to events
        new_events = dataframe_to_events(df, "test_run", "abc123")

        assert len(new_events) == 10
        assert new_events[0].step == 0
        assert new_events[0].metrics["reward_mean"] == 0.5
        assert new_events[0].model_info["run_id"] == "test_run"


class TestDeterminismCard:
    """Test determinism card generation."""

    def test_generate_determinism_card(self, sample_events, tmp_path):
        """Test determinism card generation."""
        # Mock the determinism check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write some dummy data
            for i in range(5):
                f.write(json.dumps({"step": i, "reward_mean": 0.5 + i * 0.1}) + "\n")
            run_path = f.name

        try:
            card_data = generate_determinism_card(
                sample_events, run_path, str(tmp_path)
            )

            # Validate schema
            validate(DeterminismCardV2, card_data)

            # Check required fields
            assert "version" in card_data
            assert "run_id" in card_data
            assert "passed" in card_data
            assert "replicas" in card_data
            assert "metrics_compared" in card_data
            assert "replica_variance" in card_data
            assert "rng_map" in card_data
            assert "mismatches" in card_data
            assert "fixes" in card_data
            assert "nondeterminism_hints" in card_data
            assert "flags" in card_data

            # Check file outputs
            json_file = tmp_path / "determinism_card.json"
            png_file = tmp_path / "determinism_card.png"

            assert json_file.exists()
            assert png_file.exists()

            # Verify JSON content
            with open(json_file) as f:
                saved_data = json.load(f)
            assert saved_data["run_id"] == "test_run"

        finally:
            # Cleanup
            Path(run_path).unlink(missing_ok=True)


class TestDriftCard:
    """Test drift card generation."""

    def test_generate_drift_card(self, sample_events, divergent_events, tmp_path):
        """Test drift card generation."""
        # Mock run paths
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f1:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f2:
                # Write dummy data
                for i in range(5):
                    f1.write(
                        json.dumps({"step": i, "reward_mean": 0.5 + i * 0.1}) + "\n"
                    )
                    f2.write(
                        json.dumps({"step": i, "reward_mean": 0.5 + i * 0.1 + 0.5})
                        + "\n"
                    )
                run_a_path = f1.name
                run_b_path = f2.name

        try:
            card_data = generate_drift_card(
                sample_events, divergent_events, run_a_path, run_b_path, str(tmp_path)
            )

            # Validate schema
            validate(DriftCardV1, card_data)

            # Check required fields
            assert "version" in card_data
            assert "run_a" in card_data
            assert "run_b" in card_data
            assert "diverged" in card_data
            assert "first_step" in card_data
            assert "tripped_signals" in card_data
            assert "suspected_causes" in card_data
            assert "repro" in card_data
            assert "details" in card_data
            assert "notes" in card_data

            # Check file outputs
            json_file = tmp_path / "drift_card.json"
            png_file = tmp_path / "drift_card.png"

            assert json_file.exists()
            assert png_file.exists()

            # Verify JSON content
            with open(json_file) as f:
                saved_data = json.load(f)
            assert saved_data["run_a"] == "test_run"
            assert saved_data["run_b"] == "divergent_run"

        finally:
            # Cleanup
            Path(run_a_path).unlink(missing_ok=True)
            Path(run_b_path).unlink(missing_ok=True)


class TestRewardCard:
    """Test reward card generation."""

    def test_generate_reward_card(self, sample_events, tmp_path):
        """Test reward card generation."""
        # Mock run path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write dummy data
            for i in range(5):
                f.write(json.dumps({"step": i, "reward_mean": 0.5 + i * 0.1}) + "\n")
            run_path = f.name

        try:
            card_data = generate_reward_card(sample_events, run_path, str(tmp_path))

            # Validate schema
            validate(RewardCardV1, card_data)

            # Check required fields
            assert "version" in card_data
            assert "run_id" in card_data
            assert "passed" in card_data
            assert "drift_detected" in card_data
            assert "calibration_score" in card_data
            assert "saturation_detected" in card_data
            assert "shortcut_signals" in card_data
            assert "label_noise" in card_data
            assert "metrics" in card_data
            assert "slice_analysis" in card_data
            assert "recommendations" in card_data

            # Check file outputs
            json_file = tmp_path / "reward_card.json"
            png_file = tmp_path / "reward_card.png"

            assert json_file.exists()
            assert png_file.exists()

            # Verify JSON content
            with open(json_file) as f:
                saved_data = json.load(f)
            assert saved_data["run_id"] == "test_run"
            assert isinstance(saved_data["calibration_score"], (int, float))
            assert isinstance(saved_data["label_noise"], (int, float))

        finally:
            # Cleanup
            Path(run_path).unlink(missing_ok=True)


class TestCardIntegration:
    """Test integration between different card components."""

    def test_card_consistency(self, sample_events, tmp_path):
        """Test that cards are consistent across runs."""
        # Mock run path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(5):
                f.write(json.dumps({"step": i, "reward_mean": 0.5 + i * 0.1}) + "\n")
            run_path = f.name

        try:
            # Generate determinism card
            det_card = generate_determinism_card(
                sample_events, run_path, str(tmp_path / "det")
            )

            # Generate reward card
            reward_card = generate_reward_card(
                sample_events, run_path, str(tmp_path / "reward")
            )

            # Check consistency
            assert det_card["run_id"] == reward_card["run_id"]
            assert det_card["version"] == reward_card["version"]

        finally:
            Path(run_path).unlink(missing_ok=True)

    def test_card_with_identical_runs(self, sample_events, tmp_path):
        """Test that identical runs produce consistent cards."""
        # Mock run paths
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f1:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f2:
                for i in range(5):
                    data = {"step": i, "reward_mean": 0.5 + i * 0.1}
                    f1.write(json.dumps(data) + "\n")
                    f2.write(json.dumps(data) + "\n")
                run_a_path = f1.name
                run_b_path = f2.name

        try:
            # Generate drift card for identical runs
            card_data = generate_drift_card(
                sample_events, sample_events, run_a_path, run_b_path, str(tmp_path)
            )

            # Should not detect divergence for identical runs
            assert not card_data["diverged"]
            assert card_data["first_step"] is None
            assert len(card_data["tripped_signals"]) == 0

        finally:
            Path(run_a_path).unlink(missing_ok=True)
            Path(run_b_path).unlink(missing_ok=True)


class TestCardEdgeCases:
    """Test edge cases for card generation."""

    def test_empty_events(self, tmp_path):
        """Test card generation with empty events."""
        empty_events = []

        # Mock run path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"step": 0, "reward_mean": 0.5}) + "\n")
            run_path = f.name

        try:
            # Should handle empty events gracefully
            det_card = generate_determinism_card(
                empty_events, run_path, str(tmp_path / "det")
            )
            reward_card = generate_reward_card(
                empty_events, run_path, str(tmp_path / "reward")
            )

            assert det_card["run_id"] == "unknown"
            assert reward_card["run_id"] == "unknown"

        finally:
            Path(run_path).unlink(missing_ok=True)

    def test_single_event(self, tmp_path):
        """Test card generation with single event."""
        single_event = Event(
            step=0,
            wall_time=100.0,
            metrics={"reward_mean": 0.5},
            rng={"seed": 42},
            data_slice={"tokens_in": 512},
            model_info={"run_id": "single_test"},
            notes=[],
        )

        # Mock run path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"step": 0, "reward_mean": 0.5}) + "\n")
            run_path = f.name

        try:
            # Should handle single event
            det_card = generate_determinism_card(
                [single_event], run_path, str(tmp_path / "det")
            )
            reward_card = generate_reward_card(
                [single_event], run_path, str(tmp_path / "reward")
            )

            assert det_card["run_id"] == "single_test"
            assert reward_card["run_id"] == "single_test"

        finally:
            Path(run_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__])
