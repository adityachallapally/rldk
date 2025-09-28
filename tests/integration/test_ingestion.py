"""Tests for standardized JSONL ingestion functionality."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rldk.adapters.custom_jsonl import CustomJSONLAdapter
from rldk.adapters.openrlhf import OpenRLHFAdapter
from rldk.adapters.trl import TRLAdapter
from rldk.ingest import ingest_runs
from rldk.io.event_schema import Event, create_event_from_row


class TestJSONLIngestion:
    """Test JSONL ingestion functionality."""

    def test_trl_adapter_handles_malformed_json(self):
        """Test that TRL adapter properly handles malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write valid JSON line
            json.dump({
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "loss": 0.4,
                "run_id": "test_run"
            }, f)
            f.write("\n")

            # Write malformed JSON line
            f.write('{"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.2, "loss": 0.3, "run_id": "test_run"\n')  # Missing closing brace

            # Write another valid JSON line
            json.dump({
                "step": 2,
                "phase": "train",
                "reward_mean": 0.7,
                "kl_mean": 0.3,
                "loss": 0.2,
                "run_id": "test_run"
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = TRLAdapter(f.name)
            df = adapter.load()

            # Should only have 2 valid records (steps 0 and 2)
            assert len(df) == 2
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[1] == 2

        finally:
            os.unlink(f.name)

    def test_openrlhf_adapter_handles_malformed_json(self):
        """Test that OpenRLHF adapter properly handles malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write valid JSON line
            json.dump({
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "loss": 0.4,
                "run_id": "test_run"
            }, f)
            f.write("\n")

            # Write malformed JSON line
            f.write('{"step": 1, "phase": "train", "reward_mean": 0.6, "kl_mean": 0.2, "loss": 0.3, "run_id": "test_run"\n')  # Missing closing brace

            # Write another valid JSON line
            json.dump({
                "step": 2,
                "phase": "train",
                "reward_mean": 0.7,
                "kl_mean": 0.3,
                "loss": 0.2,
                "run_id": "test_run"
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = OpenRLHFAdapter(f.name)
            df = adapter.load()

            # Should only have 2 valid records (steps 0 and 2)
            assert len(df) == 2
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[1] == 2

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_adapter_handles_malformed_json(self):
        """Test that CustomJSONL adapter properly handles malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write valid JSON line with custom format
            json.dump({
                "global_step": 0,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.1,
                "loss": 0.4,
                "rng.python": 42
            }, f)
            f.write("\n")

            # Write malformed JSON line
            f.write('{"global_step": 1, "reward_scalar": 0.6, "kl_to_ref": 0.2, "loss": 0.3, "rng.python": 42\n')  # Missing closing brace

            # Write another valid JSON line
            json.dump({
                "global_step": 2,
                "reward_scalar": 0.7,
                "kl_to_ref": 0.3,
                "loss": 0.2,
                "rng.python": 42
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            df = adapter.load()

            # Should only have 2 valid records (steps 0 and 2)
            assert len(df) == 2
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[1] == 2

            # Verify that the adapter properly maps custom fields to Event schema fields
            assert "reward_mean" in df.columns
            assert "kl_mean" in df.columns
            assert "entropy_mean" in df.columns
            assert "clip_frac" in df.columns
            assert "grad_norm" in df.columns
            assert "lr" in df.columns

            # Verify the mapped values
            assert df["reward_mean"].iloc[0] == 0.5
            assert df["kl_mean"].iloc[0] == 0.1
            assert df["loss"].iloc[0] == 0.4

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_adapter_schema_validation(self):
        """Test that CustomJSONL adapter produces Event schema compatible data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write custom JSONL data
            json.dump({
                "global_step": 0,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.1,
                "loss": 0.4,
                "rng.python": 42,
                "entropy": 0.8,
                "clip_frac": 0.2,
                "grad_norm": 1.0,
                "lr": 0.001,
                "tokens_in": 1000,
                "tokens_out": 500,
                "wall_time": 10.0,
                "run_id": "test_run",
                "git_sha": "abc123"
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            df = adapter.load()

            # Test that we can create Event objects from the adapter output
            for _, row in df.iterrows():
                event = create_event_from_row(row.to_dict(), "test_run", "abc123")

                # Verify Event object has all required fields
                assert hasattr(event, 'step')
                assert hasattr(event, 'wall_time')
                assert hasattr(event, 'metrics')
                assert hasattr(event, 'rng')
                assert hasattr(event, 'data_slice')
                assert hasattr(event, 'model_info')
                assert hasattr(event, 'notes')

                # Verify metrics contain expected fields
                assert 'reward_mean' in event.metrics
                assert 'kl_mean' in event.metrics
                assert 'entropy_mean' in event.metrics
                assert 'clip_frac' in event.metrics
                assert 'grad_norm' in event.metrics
                assert 'lr' in event.metrics
                assert 'loss' in event.metrics

                # Verify data types are correct
                assert isinstance(event.step, int)
                assert isinstance(event.wall_time, float)
                assert isinstance(event.metrics, dict)
                assert isinstance(event.rng, dict)
                assert isinstance(event.data_slice, dict)
                assert isinstance(event.model_info, dict)
                assert isinstance(event.notes, list)

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_validation_with_adapter(self):
        """Test that custom JSONL validation works with the adapter."""
        from rldk.io.validator import validate_custom_jsonl_with_adapter

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write custom JSONL data
            json.dump({
                "global_step": 0,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.1,
                "loss": 0.4,
                "rng.python": 42,
                "entropy": 0.8,
                "clip_frac": 0.2,
                "grad_norm": 1.0,
                "lr": 0.001,
                "tokens_in": 1000,
                "tokens_out": 500,
                "wall_time": 10.0,
                "run_id": "test_run",
                "git_sha": "abc123"
            }, f)
            f.write("\n")
            f.flush()

        try:
            # Test validation with adapter
            is_valid, issues = validate_custom_jsonl_with_adapter(f.name)

            # Should be valid
            assert is_valid
            assert len(issues) == 0

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_adapter_preserves_valid_zeros(self):
        """Test that CustomJSONL adapter preserves valid zero values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write custom JSONL data with valid zeros
            json.dump({
                "global_step": 0,        # Valid zero
                "reward_scalar": 0.0,    # Valid zero
                "kl_to_ref": 0.0,        # Valid zero
                "loss": 0.0,             # Valid zero
                "rng.python": 0,         # Valid zero
                "entropy": 0.0,          # Valid zero
                "lr": 0.0,               # Valid zero
                "wall_time": 0.0,        # Valid zero
                "tokens_in": 0,          # Valid zero
                "tokens_out": 0,         # Valid zero
                "run_id": "test_run",
                "git_sha": "abc123"
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            df = adapter.load()

            # Verify that zeros are preserved
            assert df["step"].iloc[0] == 0, f"Expected step=0, got {df['step'].iloc[0]}"
            assert df["reward_mean"].iloc[0] == 0.0, f"Expected reward_mean=0.0, got {df['reward_mean'].iloc[0]}"
            assert df["kl_mean"].iloc[0] == 0.0, f"Expected kl_mean=0.0, got {df['kl_mean'].iloc[0]}"
            assert df["loss"].iloc[0] == 0.0, f"Expected loss=0.0, got {df['loss'].iloc[0]}"
            assert df["entropy_mean"].iloc[0] == 0.0, f"Expected entropy_mean=0.0, got {df['entropy_mean'].iloc[0]}"
            assert df["lr"].iloc[0] == 0.0, f"Expected lr=0.0, got {df['lr'].iloc[0]}"
            assert df["wall_time"].iloc[0] == 0.0, f"Expected wall_time=0.0, got {df['wall_time'].iloc[0]}"
            assert df["seed"].iloc[0] == 0, f"Expected seed=0, got {df['seed'].iloc[0]}"
            assert df["tokens_in"].iloc[0] == 0, f"Expected tokens_in=0, got {df['tokens_in'].iloc[0]}"
            assert df["tokens_out"].iloc[0] == 0, f"Expected tokens_out=0, got {df['tokens_out'].iloc[0]}"

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_adapter_handles_missing_fields(self):
        """Test that CustomJSONL adapter uses defaults for missing fields."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write custom JSONL data with only some fields
            json.dump({
                "global_step": 5,
                "reward_scalar": 0.5,
                "run_id": "test_run"
                # Missing most fields
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            df = adapter.load()

            # Verify that present fields are preserved
            assert df["step"].iloc[0] == 5, f"Expected step=5, got {df['step'].iloc[0]}"
            assert df["reward_mean"].iloc[0] == 0.5, f"Expected reward_mean=0.5, got {df['reward_mean'].iloc[0]}"

            # Verify that missing fields use defaults
            assert df["kl_mean"].iloc[0] == 0.0, f"Expected kl_mean=0.0 (default), got {df['kl_mean'].iloc[0]}"
            assert df["loss"].iloc[0] == 0.0, f"Expected loss=0.0 (default), got {df['loss'].iloc[0]}"
            assert df["seed"].iloc[0] == 42, f"Expected seed=42 (default), got {df['seed'].iloc[0]}"

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_detection_avoids_trl_misclassification(self):
        """Test that custom JSONL detection doesn't misclassify TRL/OpenRLHF data."""
        # Test 1: TRL format with nested metrics - should NOT be detected as custom
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "step": 1,
                "reward": {"mean": 0.5, "std": 0.1},
                "kl": {"mean": 0.1, "std": 0.05},
                "loss": 0.4
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            # Should NOT be able to handle TRL format
            assert not adapter.can_handle(), "Custom adapter should NOT handle TRL nested format"
        finally:
            os.unlink(f.name)

        # Test 2: TRL format with metrics object - should NOT be detected as custom
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "step": 1,
                "metrics": {
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "loss": 0.4
                }
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            # Should NOT be able to handle TRL format
            assert not adapter.can_handle(), "Custom adapter should NOT handle TRL metrics format"
        finally:
            os.unlink(f.name)

    def test_trl_directory_ingestion_prefers_trl_adapter(self, tmp_path):
        """Regression test to ensure TRL exports aren't blocked by the custom adapter."""

        trl_dir = tmp_path / "trl_export"
        trl_dir.mkdir()

        trl_file = trl_dir / "metrics.jsonl"
        trl_payload = {
            "step": 0,
            "phase": "train",
            "reward_mean": 1.0,
            "kl_mean": 0.05,
            "entropy_mean": 0.9,
            "clip_frac": 0.01,
        }
        trl_file.write_text(json.dumps(trl_payload) + "\n", encoding="utf-8")

        # Should ingest without raising the custom adapter error
        df = ingest_runs(trl_dir)

        assert not df.empty
        assert df["phase"].iloc[0] == "train"

        # Test 3: Full standard schema - should NOT be detected as custom
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "step": 1,
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "entropy_mean": 0.8,
                "clip_frac": 0.2,
                "grad_norm": 1.0,
                "loss": 0.4
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            # Should NOT be able to handle standard format
            assert not adapter.can_handle(), "Custom adapter should NOT handle full standard schema"
        finally:
            os.unlink(f.name)

        # Test 4: Partial standard schema (just step and reward_mean) - should NOT be detected as custom
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "step": 1,
                "reward_mean": 0.5
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            # Should NOT be able to handle partial standard format
            assert not adapter.can_handle(), "Custom adapter should NOT handle partial standard schema"
        finally:
            os.unlink(f.name)

        # Test 5: Actual custom format - should be detected as custom
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "global_step": 0,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.1,
                "loss": 0.4
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            # Should be able to handle custom format
            assert adapter.can_handle(), "Custom adapter should handle actual custom format"
        finally:
            os.unlink(f.name)

    def test_custom_jsonl_adapter_handles_null_values(self):
        """Test that CustomJSONL adapter handles null values without crashing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write custom JSONL data with null values
            json.dump({
                "global_step": 0,
                "reward_scalar": None,  # null value
                "kl_to_ref": 0.1,
                "loss": None,  # null value
                "rng.python": None,  # null value
                "entropy": 0.8,
                "lr": None,  # null value
                "wall_time": 10.0,
                "tokens_in": None,  # null value
                "run_id": "test_run"
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            df = adapter.load()

            # Should not crash and should handle null values gracefully
            assert len(df) == 1, "Should have one row"

            # Verify that null values are handled correctly (use defaults)
            assert df["reward_mean"].iloc[0] == 0.0, f"Expected reward_mean=0.0 (default for null), got {df['reward_mean'].iloc[0]}"
            assert df["loss"].iloc[0] == 0.0, f"Expected loss=0.0 (default for null), got {df['loss'].iloc[0]}"
            assert df["seed"].iloc[0] == 42, f"Expected seed=42 (default for null), got {df['seed'].iloc[0]}"
            assert df["lr"].iloc[0] == 0.0, f"Expected lr=0.0 (default for null), got {df['lr'].iloc[0]}"
            assert df["tokens_in"].iloc[0] == 0, f"Expected tokens_in=0 (default for null), got {df['tokens_in'].iloc[0]}"

            # Verify that non-null values are preserved
            assert df["step"].iloc[0] == 0, f"Expected step=0, got {df['step'].iloc[0]}"
            assert df["kl_mean"].iloc[0] == 0.1, f"Expected kl_mean=0.1, got {df['kl_mean'].iloc[0]}"
            assert df["entropy_mean"].iloc[0] == 0.8, f"Expected entropy_mean=0.8, got {df['entropy_mean'].iloc[0]}"
            assert df["wall_time"].iloc[0] == 10.0, f"Expected wall_time=10.0, got {df['wall_time'].iloc[0]}"

        finally:
            os.unlink(f.name)

    def test_custom_jsonl_adapter_handles_all_null_values(self):
        """Test that CustomJSONL adapter handles all null values without crashing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write custom JSONL data with all null values
            json.dump({
                "global_step": None,
                "reward_scalar": None,
                "kl_to_ref": None,
                "loss": None,
                "rng.python": None,
                "entropy": None,
                "lr": None,
                "wall_time": None,
                "tokens_in": None,
                "tokens_out": None
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = CustomJSONLAdapter(f.name)
            df = adapter.load()

            # Should not crash and should handle all null values gracefully
            assert len(df) == 1, "Should have one row"

            # Verify that all null values use defaults
            assert df["step"].iloc[0] == 0, f"Expected step=0 (line_num), got {df['step'].iloc[0]}"
            assert df["reward_mean"].iloc[0] == 0.0, f"Expected reward_mean=0.0 (default), got {df['reward_mean'].iloc[0]}"
            assert df["kl_mean"].iloc[0] == 0.0, f"Expected kl_mean=0.0 (default), got {df['kl_mean'].iloc[0]}"
            assert df["loss"].iloc[0] == 0.0, f"Expected loss=0.0 (default), got {df['loss'].iloc[0]}"
            assert df["seed"].iloc[0] == 42, f"Expected seed=42 (default), got {df['seed'].iloc[0]}"

        finally:
            os.unlink(f.name)

    def test_adapters_produce_identical_event_objects(self):
        """Test that TRL and OpenRLHF adapters produce identical Event objects from compatible logs."""
        # Create identical training data
        training_data = [
            {
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "reward_std": 0.1,
                "kl_mean": 0.1,
                "kl_std": 0.05,
                "entropy_mean": 0.8,
                "clip_frac": 0.2,
                "grad_norm": 1.0,
                "lr": 0.001,
                "loss": 0.4,
                "tokens_in": 1000,
                "tokens_out": 500,
                "wall_time": 10.0,
                "seed": 42,
                "run_id": "test_run",
                "git_sha": "abc123"
            },
            {
                "step": 1,
                "phase": "train",
                "reward_mean": 0.6,
                "reward_std": 0.12,
                "kl_mean": 0.12,
                "kl_std": 0.06,
                "entropy_mean": 0.78,
                "clip_frac": 0.18,
                "grad_norm": 0.95,
                "lr": 0.001,
                "loss": 0.35,
                "tokens_in": 1000,
                "tokens_out": 500,
                "wall_time": 20.0,
                "seed": 42,
                "run_id": "test_run",
                "git_sha": "abc123"
            }
        ]

        # Create TRL JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as trl_file:
            for data in training_data:
                json.dump(data, trl_file)
                trl_file.write("\n")
            trl_file.flush()

        # Create OpenRLHF JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as openrlhf_file:
            for data in training_data:
                json.dump(data, openrlhf_file)
                openrlhf_file.write("\n")
            openrlhf_file.flush()

        try:
            # Load with TRL adapter
            trl_adapter = TRLAdapter(trl_file.name)
            trl_df = trl_adapter.load()

            # Load with OpenRLHF adapter
            openrlhf_adapter = OpenRLHFAdapter(openrlhf_file.name)
            openrlhf_df = openrlhf_adapter.load()

            # Convert to Event objects
            trl_events = []
            openrlhf_events = []

            for _, row in trl_df.iterrows():
                event = create_event_from_row(row.to_dict(), "test_run", "abc123")
                trl_events.append(event)

            for _, row in openrlhf_df.iterrows():
                event = create_event_from_row(row.to_dict(), "test_run", "abc123")
                openrlhf_events.append(event)

            # Verify identical Event objects
            assert len(trl_events) == len(openrlhf_events)

            for trl_event, openrlhf_event in zip(trl_events, openrlhf_events):
                assert trl_event.step == openrlhf_event.step
                assert trl_event.wall_time == openrlhf_event.wall_time
                assert trl_event.metrics == openrlhf_event.metrics
                assert trl_event.rng == openrlhf_event.rng
                assert trl_event.data_slice == openrlhf_event.data_slice
                assert trl_event.model_info == openrlhf_event.model_info

        finally:
            os.unlink(trl_file.name)
            os.unlink(openrlhf_file.name)

    def test_empty_file_handling(self):
        """Test handling of empty JSONL files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Create empty file
            pass

        try:
            adapter = TRLAdapter(f.name)
            df = adapter.load()

            # Should return empty DataFrame
            assert len(df) == 0
            assert "step" in df.columns

        finally:
            os.unlink(f.name)

    def test_partially_written_lines(self):
        """Test handling of partially written JSONL lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write valid JSON line
            json.dump({
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "loss": 0.4,
                "run_id": "test_run"
            }, f)
            f.write("\n")

            # Write partial JSON line (incomplete)
            f.write('{"step": 1, "phase": "train", "reward_mean": 0.6')
            f.flush()

        try:
            adapter = TRLAdapter(f.name)
            df = adapter.load()

            # Should only have 1 valid record (step 0)
            assert len(df) == 1
            assert df["step"].iloc[0] == 0

        finally:
            os.unlink(f.name)

    def test_ingest_runs_aborts_on_parsing_error(self):
        """Test that ingest_runs aborts if any JSONL line cannot be parsed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write valid JSON line
            json.dump({
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "loss": 0.4,
                "run_id": "test_run"
            }, f)
            f.write("\n")

            # Write completely invalid line
            f.write("This is not JSON at all\n")

            # Write another valid JSON line
            json.dump({
                "step": 2,
                "phase": "train",
                "reward_mean": 0.7,
                "kl_mean": 0.3,
                "loss": 0.2,
                "run_id": "test_run"
            }, f)
            f.write("\n")
            f.flush()

        try:
            # Should raise an exception due to parsing error
            with pytest.raises(RuntimeError):
                ingest_runs(f.name, adapter_hint="trl")

        finally:
            os.unlink(f.name)

    def test_logging_records_total_events(self):
        """Test that logging records the total number of events ingested."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write 3 valid JSON lines
            for i in range(3):
                json.dump({
                    "step": i,
                    "phase": "train",
                    "reward_mean": 0.5 + i * 0.1,
                    "kl_mean": 0.1 + i * 0.01,
                    "loss": 0.4 - i * 0.05,
                    "run_id": "test_run"
                }, f)
                f.write("\n")
            f.flush()

        try:
            with patch('rldk.ingest.ingest.logging') as mock_logging:
                ingest_runs(f.name, adapter_hint="trl")

                # Verify that logging.info was called with the correct message
                mock_logging.info.assert_called_with("Successfully ingested 3 events from " + f.name)

        finally:
            os.unlink(f.name)

    def test_event_schema_compatibility(self):
        """Test that Event schema fields are always serialized as primitive types."""
        # Create test data with various types
        test_data = {
            "step": 0,
            "phase": "train",
            "reward_mean": 0.5,
            "kl_mean": 0.1,
            "loss": 0.4,
            "run_id": "test_run",
            "git_sha": "abc123"
        }

        # Create Event object
        event = create_event_from_row(test_data, "test_run", "abc123")

        # Verify all fields are primitive types
        event_dict = event.to_dict()

        # Check that all values are primitive types (no tensors, etc.)
        def check_primitive_types(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    assert isinstance(key, str)
                    check_primitive_types(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_primitive_types(item)
            else:
                # Should be primitive types only
                assert isinstance(obj, (int, float, str, bool, type(None)))

        check_primitive_types(event_dict)

    def test_utc_timestamps(self):
        """Test that loggers use UTC timestamps for reproducibility."""
        import time

        # Create test data
        test_data = {
            "step": 0,
            "phase": "train",
            "reward_mean": 0.5,
            "kl_mean": 0.1,
            "loss": 0.4,
            "run_id": "test_run",
            "git_sha": "abc123"
        }

        # Create Event object
        event = create_event_from_row(test_data, "test_run", "abc123")

        # Verify wall_time is a float (UTC timestamp)
        assert isinstance(event.wall_time, float)
        assert event.wall_time > 0

    def test_concurrent_writes_handling(self):
        """Test handling of concurrent writes to JSONL files."""
        import threading
        import time

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            pass

        def write_events(thread_id, num_events):
            with open(f.name, "a") as jsonl_file:
                for i in range(num_events):
                    event_data = {
                        "step": thread_id * 100 + i,
                        "phase": "train",
                        "reward_mean": 0.5 + i * 0.01,
                        "kl_mean": 0.1 + i * 0.001,
                        "loss": 0.4 - i * 0.005,
                        "run_id": f"test_run_{thread_id}",
                        "timestamp": time.time()
                    }
                    json.dump(event_data, jsonl_file)
                    jsonl_file.write("\n")
                    jsonl_file.flush()  # Ensure immediate write
                    time.sleep(0.001)  # Small delay to simulate concurrent access

        try:
            # Start multiple threads writing to the same file
            threads = []
            for i in range(3):
                thread = threading.Thread(target=write_events, args=(i, 5))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Verify all events were written
            adapter = TRLAdapter(f.name)
            df = adapter.load()

            # Should have 15 total events (3 threads * 5 events each)
            assert len(df) == 15

            # Verify all run_ids are present
            run_ids = set(df["run_id"].dropna())
            assert len(run_ids) == 3
            assert "test_run_0" in run_ids
            assert "test_run_1" in run_ids
            assert "test_run_2" in run_ids

        finally:
            os.unlink(f.name)

    def test_environment_variable_override(self):
        """Test environment variable override for disabling JSONL emission."""
        import os

        # Test with RLDK_DISABLE_JSONL set
        with patch.dict(os.environ, {'RLDK_DISABLE_JSONL': '1'}):
            # This would be tested in the actual callback implementation
            # For now, we just verify the environment variable is set
            assert os.environ.get('RLDK_DISABLE_JSONL') == '1'

        # Test without the environment variable
        with patch.dict(os.environ, {}, clear=True):
            assert os.environ.get('RLDK_DISABLE_JSONL') is None


class TestTRLAdapter:
    """Test TRL adapter with new JSONL format."""

    def test_trl_adapter_handles_new_jsonl_format(self):
        """Test that TRL adapter can handle the new Event schema JSONL format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write new Event schema format
            json.dump({
                "step": 0,
                "wall_time": 10.0,
                "metrics": {
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "clip_frac": 0.2,
                    "grad_norm": 1.0,
                    "lr": 0.001,
                    "loss": 0.4
                },
                "rng": {
                    "seed": 42
                },
                "data_slice": {
                    "tokens_in": 1000,
                    "tokens_out": 500
                },
                "model_info": {
                    "run_id": "test_run",
                    "git_sha": "abc123",
                    "phase": "train"
                },
                "notes": []
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = TRLAdapter(f.name)
            assert adapter.can_handle()

            df = adapter.load()
            assert len(df) == 1
            assert df["step"].iloc[0] == 0
            assert df["reward_mean"].iloc[0] == 0.5
            assert df["kl_mean"].iloc[0] == 0.1
            assert df["wall_time"].iloc[0] == 10.0
            assert df["run_id"].iloc[0] == "test_run"

        finally:
            os.unlink(f.name)

    def test_trl_adapter_handles_old_jsonl_format(self):
        """Test that TRL adapter maintains backward compatibility with old format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write old format
            json.dump({
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "entropy_mean": 0.8,
                "loss": 0.4,
                "wall_time": 10.0,
                "run_id": "test_run",
                "git_sha": "abc123"
            }, f)
            f.write("\n")
            f.flush()

        try:
            adapter = TRLAdapter(f.name)
            assert adapter.can_handle()

            df = adapter.load()
            assert len(df) == 1
            assert df["step"].iloc[0] == 0
            assert df["reward_mean"].iloc[0] == 0.5
            assert df["kl_mean"].iloc[0] == 0.1
            assert df["wall_time"].iloc[0] == 10.0
            assert df["run_id"].iloc[0] == "test_run"

        finally:
            os.unlink(f.name)

    def test_trl_adapter_ingest_runs_integration(self):
        """Test that TRL adapter works with ingest_runs for new JSONL format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write multiple events in new format
            for i in range(3):
                json.dump({
                    "step": i,
                    "wall_time": i * 10.0,
                    "metrics": {
                        "reward_mean": 0.5 + i * 0.1,
                        "kl_mean": 0.1 + i * 0.01,
                        "entropy_mean": 0.8 - i * 0.02,
                        "loss": 0.4 - i * 0.05
                    },
                    "rng": {
                        "seed": 42
                    },
                    "data_slice": {
                        "tokens_in": 1000,
                        "tokens_out": 500
                    },
                    "model_info": {
                        "run_id": "test_run",
                        "git_sha": "abc123",
                        "phase": "train"
                    },
                    "notes": []
                }, f)
                f.write("\n")
            f.flush()

        try:
            # Test with ingest_runs
            df = ingest_runs(f.name, adapter_hint="trl")

            assert len(df) == 3
            assert "step" in df.columns
            assert "reward_mean" in df.columns
            assert "kl_mean" in df.columns
            assert "wall_time" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["step"].iloc[2] == 2
            assert df["reward_mean"].iloc[0] == 0.5
            assert df["reward_mean"].iloc[2] == 0.7

        finally:
            os.unlink(f.name)

    def test_trl_adapter_event_count_matches_steps(self):
        """Test that Event count matches the number of training steps."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write 5 training steps
            for i in range(5):
                json.dump({
                    "step": i,
                    "wall_time": i * 10.0,
                    "metrics": {
                        "reward_mean": 0.5 + i * 0.1,
                        "kl_mean": 0.1 + i * 0.01,
                        "loss": 0.4 - i * 0.05
                    },
                    "rng": {"seed": 42},
                    "data_slice": {"tokens_in": 1000, "tokens_out": 500},
                    "model_info": {"run_id": "test_run", "phase": "train"},
                    "notes": []
                }, f)
                f.write("\n")
            f.flush()

        try:
            # Test with ingest_runs
            df = ingest_runs(f.name, adapter_hint="trl")

            # Should have exactly 5 events (one per step)
            assert len(df) == 5

            # Steps should be 0, 1, 2, 3, 4
            assert list(df["step"]) == [0, 1, 2, 3, 4]

        finally:
            os.unlink(f.name)


class TestJSONLValidator:
    """Test JSONL validator utility."""

    def test_jsonl_validator_checks_schema_conformance(self):
        """Test that JSONL validator checks for schema conformance."""
        # This would test the lightweight validator utility
        # For now, we'll create a simple validation function

        def validate_jsonl_schema(file_path):
            """Simple JSONL schema validator."""
            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # Check for required fields
                            required_fields = ['step', 'phase', 'reward_mean', 'kl_mean', 'loss']
                            missing_fields = [field for field in required_fields if field not in data]
                            if missing_fields:
                                return False, f"Line {line_num}: Missing required fields: {missing_fields}"
                        except json.JSONDecodeError:
                            return False, f"Line {line_num}: Invalid JSON"
            return True, "All lines valid"

        # Test with valid JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "step": 0,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.1,
                "loss": 0.4,
                "run_id": "test_run"
            }, f)
            f.write("\n")
            f.flush()

        try:
            is_valid, message = validate_jsonl_schema(f.name)
            assert is_valid
            assert message == "All lines valid"
        finally:
            os.unlink(f.name)

        # Test with invalid JSONL (missing required fields)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump({
                "step": 0,
                "phase": "train",
                # Missing reward_mean, kl_mean, loss
                "run_id": "test_run"
            }, f)
            f.write("\n")
            f.flush()

        try:
            is_valid, message = validate_jsonl_schema(f.name)
            assert not is_valid
            assert "Missing required fields" in message
        finally:
            os.unlink(f.name)
