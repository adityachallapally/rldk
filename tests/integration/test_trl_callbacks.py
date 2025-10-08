"""Tests for TRL callbacks JSONL event emission."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from rldk.integrations.trl.callbacks import RLDKCallback, RLDKMetrics
from rldk.io.event_schema import Event


class TestRLDKCallbackJSONL:
    """Test JSONL event emission functionality."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_jsonl_logging_enabled(self):
        """Test that JSONL logging is enabled by default."""
        callback = RLDKCallback(output_dir=self.output_dir)
        assert callback.enable_jsonl_logging is True
        assert callback.jsonl_log_interval == 1

    def test_jsonl_logging_disabled(self):
        """Test that JSONL logging can be disabled."""
        callback = RLDKCallback(
            output_dir=self.output_dir,
            enable_jsonl_logging=False
        )
        assert callback.enable_jsonl_logging is False
        assert callback.jsonl_file is None

    def test_jsonl_file_creation(self):
        """Test that JSONL file is created correctly."""
        callback = RLDKCallback(output_dir=self.output_dir)

        # Check that JSONL file was created
        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        assert len(jsonl_files) == 1

        jsonl_path = jsonl_files[0]
        assert callback.run_id in jsonl_path.name
        assert jsonl_path.exists()
        assert "None" not in jsonl_path.name  # Ensure run_id is properly set

    def test_jsonl_event_emission(self):
        """Test that JSONL events are emitted correctly."""
        callback = RLDKCallback(
            output_dir=self.output_dir,
            jsonl_log_interval=1
        )

        # Mock trainer state and logs
        state = Mock()
        state.global_step = 10
        state.epoch = 1.0

        logs = {
            'train_loss': 0.5,
            'learning_rate': 0.001,
            'grad_norm': 0.1,
            'ppo/rewards/mean': 0.8,
            'ppo/rewards/std': 0.2,
            'ppo/policy/kl_mean': 0.05,
            'ppo/policy/entropy': 0.9,
            'ppo/policy/clipfrac': 0.1,
            'ppo/val/value_loss': 0.3,
            'ppo/val/policy_loss': 0.2,
        }

        # Update current metrics
        callback.current_metrics.step = 10
        callback.current_metrics.epoch = 1.0
        callback.current_metrics.loss = 0.5
        callback.current_metrics.learning_rate = 0.001
        callback.current_metrics.grad_norm = 0.1
        callback.current_metrics.reward_mean = 0.8
        callback.current_metrics.reward_std = 0.2
        callback.current_metrics.kl_mean = 0.05
        callback.current_metrics.entropy_mean = 0.9
        callback.current_metrics.clip_frac = 0.1
        callback.current_metrics.value_loss = 0.3
        callback.current_metrics.policy_loss = 0.2
        callback.current_metrics.wall_time = 100.0
        callback.current_metrics.tokens_in = 1000
        callback.current_metrics.tokens_out = 500
        callback.current_metrics.seed = 42
        callback.current_metrics.git_sha = "abc123"

        # Emit JSONL event
        callback._emit_jsonl_event(state, logs)

        # Check that event was written
        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0]) as f:
            lines = f.readlines()
            assert len(lines) == 1

            # Parse the JSONL line
            event_data = json.loads(lines[0])

            # Verify Event schema structure
            assert "step" in event_data
            assert "wall_time" in event_data
            assert "metrics" in event_data
            assert "rng" in event_data
            assert "data_slice" in event_data
            assert "model_info" in event_data
            assert "notes" in event_data

            # Verify specific values
            assert event_data["step"] == 10
            assert event_data["wall_time"] == 100.0
            assert event_data["metrics"]["loss"] == 0.5
            assert event_data["metrics"]["reward_mean"] == 0.8
            assert event_data["metrics"]["kl_mean"] == 0.05
            assert event_data["model_info"]["run_id"] == callback.run_id
            assert event_data["model_info"]["phase"] == "train"

    def test_jsonl_event_compatibility_with_trl_adapter(self):
        """Test that emitted JSONL events are compatible with TRLAdapter."""
        callback = RLDKCallback(
            output_dir=self.output_dir,
            jsonl_log_interval=1
        )

        # Mock trainer state and logs
        state = Mock()
        state.global_step = 5
        state.epoch = 0.5

        logs = {
            'train_loss': 0.6,
            'learning_rate': 0.0005,
            'grad_norm': 0.15,
            'ppo/rewards/mean': 0.7,
            'ppo/rewards/std': 0.25,
            'ppo/policy/kl_mean': 0.08,
            'ppo/policy/entropy': 0.85,
            'ppo/policy/clipfrac': 0.12,
        }

        # Update current metrics
        callback.current_metrics.step = 5
        callback.current_metrics.epoch = 0.5
        callback.current_metrics.loss = 0.6
        callback.current_metrics.learning_rate = 0.0005
        callback.current_metrics.grad_norm = 0.15
        callback.current_metrics.reward_mean = 0.7
        callback.current_metrics.reward_std = 0.25
        callback.current_metrics.kl_mean = 0.08
        callback.current_metrics.entropy_mean = 0.85
        callback.current_metrics.clip_frac = 0.12
        callback.current_metrics.wall_time = 50.0
        callback.current_metrics.tokens_in = 1000
        callback.current_metrics.tokens_out = 500
        callback.current_metrics.seed = 42
        callback.current_metrics.git_sha = "abc123"

        # Emit JSONL event
        callback._emit_jsonl_event(state, logs)

        # Close the file to ensure it's written
        callback._close_jsonl_file()

        # Test that TRLAdapter can read the file
        from rldk.adapters.trl import TRLAdapter

        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        assert len(jsonl_files) == 1

        adapter = TRLAdapter(jsonl_files[0])
        assert adapter.can_handle()

        df = adapter.load()
        assert len(df) == 1
        assert df["step"].iloc[0] == 5
        assert df["reward_mean"].iloc[0] == 0.7
        assert df["kl_mean"].iloc[0] == 0.08
        assert df["wall_time"].iloc[0] == 50.0

    def test_jsonl_logging_interval(self):
        """Test that JSONL logging respects the interval setting."""
        callback = RLDKCallback(
            output_dir=self.output_dir,
            jsonl_log_interval=3  # Log every 3 steps
        )

        state = Mock()
        logs = {'train_loss': 0.5}

        # Update current metrics
        callback.current_metrics.loss = 0.5
        callback.current_metrics.wall_time = 10.0

        # Step 1 - should not log (step % 3 != 0)
        state.global_step = 1
        callback._emit_jsonl_event(state, logs)

        # Step 3 - should log (step % 3 == 0)
        state.global_step = 3
        callback._emit_jsonl_event(state, logs)

        # Step 6 - should log (step % 3 == 0)
        state.global_step = 6
        callback._emit_jsonl_event(state, logs)

        # Close file and check
        callback._close_jsonl_file()

        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        assert len(jsonl_files) == 1

        with open(jsonl_files[0]) as f:
            lines = f.readlines()
            assert len(lines) == 2  # Only steps 3 and 6 should be logged

    def test_jsonl_event_notes_generation(self):
        """Test that notes are generated based on training health indicators."""
        callback = RLDKCallback(output_dir=self.output_dir)

        state = Mock()
        state.global_step = 10

        # Set metrics that should trigger notes
        callback.current_metrics.clip_frac = 0.25  # > 0.2 threshold
        callback.current_metrics.grad_norm = 15.0  # > 10.0 threshold
        callback.current_metrics.kl_mean = 0.25  # > 0.2 threshold
        callback.current_metrics.wall_time = 100.0

        logs = {'train_loss': 0.5}

        # Emit JSONL event
        callback._emit_jsonl_event(state, logs)

        # Close file and check notes
        callback._close_jsonl_file()

        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        with open(jsonl_files[0]) as f:
            event_data = json.loads(f.readline())
            notes = event_data["notes"]

            assert "High clipping fraction detected" in notes
            assert "Large gradient norm detected" in notes
            assert "High KL divergence detected" in notes

    def test_jsonl_event_with_missing_metrics(self):
        """Test that JSONL events handle missing metrics gracefully."""
        callback = RLDKCallback(output_dir=self.output_dir)

        state = Mock()
        state.global_step = 1

        # Don't set any metrics
        callback.current_metrics.wall_time = 10.0

        logs = {}

        # Emit JSONL event
        callback._emit_jsonl_event(state, logs)

        # Close file and check
        callback._close_jsonl_file()

        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        with open(jsonl_files[0]) as f:
            event_data = json.loads(f.readline())

            # Should still have the basic structure
            assert "step" in event_data
            assert "wall_time" in event_data
            assert "metrics" in event_data
            assert "model_info" in event_data

            # Metrics should have default values (0.0) for missing fields
            metrics = event_data["metrics"]
            # The Event schema sets default values for missing metrics
            assert len(metrics) > 0  # Should have some metrics with default values

    @patch('rldk.integrations.trl.callbacks.EVENT_SCHEMA_AVAILABLE', False)
    def test_jsonl_logging_without_event_schema(self):
        """Test that JSONL logging is disabled when Event schema is not available."""
        callback = RLDKCallback(output_dir=self.output_dir)

        # Should be disabled when Event schema is not available
        assert callback.enable_jsonl_logging is False
        assert callback.jsonl_file is None

    def test_jsonl_file_cleanup(self):
        """Test that JSONL file is properly closed on training end."""
        callback = RLDKCallback(output_dir=self.output_dir)

        # Verify file is open
        assert callback.jsonl_file is not None

        # Mock training end
        args = Mock()
        state = Mock()
        control = Mock()

        callback.on_train_end(args, state, control)

        # Verify file is closed
        assert callback.jsonl_file is None

    def test_jsonl_initialization_order(self):
        """Test that JSONL setup happens after run_id initialization."""
        callback = RLDKCallback(output_dir=self.output_dir)

        # Verify run_id is set before JSONL file is created
        assert callback.run_id is not None
        assert callback.run_id != "None"

        # Verify JSONL file name contains the proper run_id
        jsonl_files = list(self.output_dir.glob("*_events.jsonl"))
        assert len(jsonl_files) == 1

        jsonl_path = jsonl_files[0]
        assert callback.run_id in jsonl_path.name
        assert "None" not in jsonl_path.name

    def test_jsonl_log_interval_validation(self):
        """Test that jsonl_log_interval validation works correctly."""
        # Test valid intervals
        callback = RLDKCallback(output_dir=self.output_dir, jsonl_log_interval=1)
        assert callback.jsonl_log_interval == 1

        callback = RLDKCallback(output_dir=self.output_dir, jsonl_log_interval=5)
        assert callback.jsonl_log_interval == 5

        # Test invalid intervals
        with pytest.raises(ValueError, match="jsonl_log_interval must be positive"):
            RLDKCallback(output_dir=self.output_dir, jsonl_log_interval=0)

        with pytest.raises(ValueError, match="jsonl_log_interval must be positive"):
            RLDKCallback(output_dir=self.output_dir, jsonl_log_interval=-1)

    def test_log_interval_validation(self):
        """Test that log_interval validation works correctly."""
        # Test valid intervals
        callback = RLDKCallback(output_dir=self.output_dir, log_interval=1)
        assert callback.log_interval == 1

        callback = RLDKCallback(output_dir=self.output_dir, log_interval=10)
        assert callback.log_interval == 10

        # Test invalid intervals
        with pytest.raises(ValueError, match="log_interval must be positive"):
            RLDKCallback(output_dir=self.output_dir, log_interval=0)

        with pytest.raises(ValueError, match="log_interval must be positive"):
            RLDKCallback(output_dir=self.output_dir, log_interval=-1)

    def test_jsonl_emission_with_zero_interval(self):
        """Test that JSONL emission handles zero interval gracefully."""
        # This test verifies the defensive check in on_log method
        # Even though the constructor should prevent this, we test the defensive check

        # Create a callback with a valid interval first
        callback = RLDKCallback(output_dir=self.output_dir, jsonl_log_interval=1)

        # Manually set jsonl_log_interval to 0 to test defensive check
        callback.jsonl_log_interval = 0

        # Mock training step
        from unittest.mock import Mock
        state = Mock()
        state.global_step = 1

        logs = {'train_loss': 0.5}

        # This should not raise ZeroDivisionError due to defensive check
        try:
            callback.on_log(Mock(), state, Mock(), logs)
            print("✅ JSONL emission with zero interval handled gracefully")
        except ZeroDivisionError:
            pytest.fail("ZeroDivisionError should not occur with defensive check")

    def test_malformed_jsonl_handling(self):
        """Test that malformed JSONL is handled gracefully by the adapter."""
        # Create a JSONL file with malformed JSON
        jsonl_path = self.output_dir / "malformed_events.jsonl"

        with open(jsonl_path, 'w') as f:
            # Write valid JSON line
            f.write('{"step": 0, "metrics": {"loss": 0.5}, "model_info": {"run_id": "test"}}\n')
            # Write malformed JSON line (missing closing brace)
            f.write('{"step": 1, "metrics": {"loss": 0.6}, "model_info": {"run_id": "test"\n')
            # Write another valid JSON line
            f.write('{"step": 2, "metrics": {"loss": 0.4}, "model_info": {"run_id": "test"}}\n')

        # Test that TRLAdapter can handle this
        from rldk.adapters.trl import TRLAdapter

        adapter = TRLAdapter(jsonl_path)
        assert adapter.can_handle()

        # Should only load the valid lines (steps 0 and 2)
        df = adapter.load()
        assert len(df) == 2
        assert df["step"].iloc[0] == 0
        assert df["step"].iloc[1] == 2
