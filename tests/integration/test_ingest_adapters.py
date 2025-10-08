"""Tests for ingest adapters."""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from rldk.adapters.openrlhf import OpenRLHFAdapter
from rldk.adapters.trl import TRLAdapter
from rldk.adapters.wandb import WandBAdapter
from rldk.io.schema import MetricsSchema
from rldk.utils.error_handling import AdapterError


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


class TestTRLAdapter:
    """Test TRL adapter functionality."""

    def test_can_handle_jsonl(self):
        """Test that TRL adapter can handle JSONL files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001,
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            adapter = TRLAdapter(f.name)
            assert adapter.can_handle()

    def test_flat_and_nested_logs_normalize(self, tmp_path: Path):
        """TRL adapter normalizes flat and nested records to the same table."""

        flat_path = tmp_path / "flat.jsonl"
        nested_path = tmp_path / "nested.jsonl"

        _write_jsonl(
            flat_path,
            [
                {
                    "step": 1,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "lr": 0.001,
                    "grad_norm": 0.2,
                    "wall_time_ms": 2000,
                    "seed": 11,
                    "run_id": "trl-run",
                    "git_sha": "abc123",
                    "tokens_in": 256,
                    "tokens_out": 128,
                },
                {
                    "step": 2,
                    "phase": "train",
                    "reward_mean": 0.6,
                    "kl_mean": 0.12,
                    "entropy_mean": 0.75,
                    "lr": 0.001,
                    "grad_norm": 0.25,
                    "wall_time_ms": 4000,
                    "seed": 11,
                    "run_id": "trl-run",
                    "git_sha": "abc123",
                    "tokens_in": 300,
                    "tokens_out": 150,
                },
            ],
        )

        _write_jsonl(
            nested_path,
            [
                {
                    "step": 1,
                    "wall_time": 2.0,
                    "metrics": {
                        "reward_mean": 0.5,
                        "kl_mean": 0.1,
                        "entropy_mean": 0.8,
                        "lr": 0.001,
                        "grad_norm": 0.2,
                    },
                    "data_slice": {"tokens_in": 256, "tokens_out": 128},
                    "rng": {"seed": 11},
                    "model_info": {
                        "run_id": "trl-run",
                        "phase": "train",
                        "git_sha": "abc123",
                    },
                },
                {
                    "step": 2,
                    "wall_time": 4.0,
                    "metrics": {
                        "reward_mean": 0.6,
                        "kl_mean": 0.12,
                        "entropy_mean": 0.75,
                        "lr": 0.001,
                        "grad_norm": 0.25,
                    },
                    "data_slice": {"tokens_in": 300, "tokens_out": 150},
                    "rng": {"seed": 11},
                    "model_info": {
                        "run_id": "trl-run",
                        "phase": "train",
                        "git_sha": "abc123",
                    },
                },
            ],
        )

        df_flat = TRLAdapter(flat_path).load()
        df_nested = TRLAdapter(nested_path).load()

        pd.testing.assert_frame_equal(df_flat, df_nested, check_dtype=False)

    def test_unknown_shape_suggests_field_map(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """Unrecognized shapes fall back to flexible adapter with guidance."""

        path = tmp_path / "unknown.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "step": 1,
                    "metrics": [
                        {"name": "reward_mean", "value": 0.5},
                    ],
                }
            ],
        )

        adapter = TRLAdapter(path)
        caplog.set_level("WARNING")

        df = adapter.load()

        assert not df.empty
        assert "step" in df.columns
        assert any("field-map" in msg for msg in caplog.messages)
class TestOpenRLHFAdapter:
    """Test OpenRLHF adapter functionality."""

    def test_can_handle_jsonl(self):
        """Test that OpenRLHF adapter can handle JSONL files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001,
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            adapter = OpenRLHFAdapter(f.name)
            assert adapter.can_handle()

    def test_load_jsonl(self):
        """Test loading OpenRLHF JSONL data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write test data
            for i in range(3):
                json.dump(
                    {
                        "step": i,
                        "phase": "train",
                        "reward_mean": 0.5 + i * 0.1,
                        "kl_mean": 0.1 + i * 0.01,
                        "entropy_mean": 0.8 - i * 0.02,
                        "loss": 0.4 - i * 0.05,
                        "lr": 0.001,
                        "wall_time": i * 10.0,
                        "seed": 42,
                        "run_id": "test_run",
                        "git_sha": "abc123",
                    },
                    f,
                )
                f.write("\n")
            f.flush()

            adapter = OpenRLHFAdapter(f.name)
            df = adapter.load()

            assert len(df) == 3
            assert "step" in df.columns
            assert "reward_mean" in df.columns
            assert "kl_mean" in df.columns
            assert "wall_time" in df.columns
            assert df["step"].iloc[0] == 0
            assert df["reward_mean"].iloc[0] == 0.5

    def test_wall_time_ms_conversion(self):
        """Test that wall_time_ms is converted to wall_time in seconds."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write test data with wall_time_ms
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001,
                    "wall_time_ms": 5000,  # 5 seconds in milliseconds
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            adapter = TRLAdapter(f.name)
            df = adapter.load()

            # Should have wall_time column, not wall_time_ms
            assert "wall_time" in df.columns
            assert "wall_time_ms" not in df.columns
            assert df["wall_time"].iloc[0] == 5.0  # Converted to seconds


class TestWandBAdapter:
    """Test WandB adapter functionality."""

    def test_parse_wandb_uri(self):
        """Test parsing wandb:// URIs."""
        adapter = WandBAdapter("wandb://entity/project/run_id")
        assert adapter.entity == "entity"
        assert adapter.project == "project"
        assert adapter.run_id == "run_id"

    def test_can_handle_wandb_uri(self):
        """Test that WandB adapter can handle wandb:// URIs."""
        adapter = WandBAdapter("wandb://entity/project/run_id")
        assert adapter.can_handle()

    def test_cannot_handle_invalid_uri(self):
        """Test that WandB adapter cannot handle invalid URIs."""
        adapter = WandBAdapter("invalid_uri")
        assert not adapter.can_handle()

    @pytest.mark.skip(reason="Requires actual WandB API access")
    def test_load_wandb_run(self):
        """Test loading WandB run data."""
        # This test would require actual WandB API access
        # For now, we'll skip it
        pass


class TestSchemaCompatibility:
    """Test that all adapters produce schema-compatible data."""

    def test_trl_schema_compatibility(self):
        """Test TRL adapter produces schema-compatible data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001,
                    "wall_time": 10.0,
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            adapter = TRLAdapter(f.name)
            df = adapter.load()

            # Should be able to create schema from dataframe
            schema = MetricsSchema.from_dataframe(df)
            assert len(schema.metrics) == 1

            # Check that all required fields are present
            metric = schema.metrics[0]
            assert metric.step == 0
            assert metric.reward_mean == 0.5
            assert metric.kl_mean == 0.1
            assert metric.wall_time == 10.0

    def test_openrlhf_schema_compatibility(self):
        """Test OpenRLHF adapter produces schema-compatible data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001,
                    "wall_time": 10.0,
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            adapter = OpenRLHFAdapter(f.name)
            df = adapter.load()

            # Should be able to create schema from dataframe
            schema = MetricsSchema.from_dataframe(df)
            assert len(schema.metrics) == 1

            # Check that all required fields are present
            metric = schema.metrics[0]
            assert metric.step == 0
            assert metric.reward_mean == 0.5
            assert metric.kl_mean == 0.1
            assert metric.wall_time == 10.0

    def test_openrlhf_wall_time_ms_conversion(self):
        """Test that OpenRLHF adapter converts wall_time_ms to wall_time in seconds."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Write test data with wall_time_ms
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "loss": 0.4,
                    "lr": 0.001,
                    "wall_time_ms": 3000,  # 3 seconds in milliseconds
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            adapter = OpenRLHFAdapter(f.name)
            df = adapter.load()

            # Should have wall_time column, not wall_time_ms
            assert "wall_time" in df.columns
            assert "wall_time_ms" not in df.columns
            assert df["wall_time"].iloc[0] == 3.0  # Converted to seconds

    def test_schema_loads_fixtures_correctly(self):
        """Test that loading runs_fixtures/clean_ppo.jsonl through ingest produces correct schema."""
        from rldk.ingest import ingest_runs

        # Load the fixture through ingest
        df = ingest_runs("runs_fixtures/clean_ppo.jsonl")

        # Should have wall_time column
        assert "wall_time" in df.columns

        # Should not have wall_time_ms column
        assert "wall_time_ms" not in df.columns

        # Should have other required columns
        assert "step" in df.columns
        assert "reward_mean" in df.columns
        assert "kl_mean" in df.columns

        # Should have some data
        assert len(df) > 0

        # Check that wall_time values are reasonable (should be in seconds, not milliseconds)
        if df["wall_time"].notna().any():
            wall_time_values = df["wall_time"].dropna()
            # Wall time should be reasonable values (not in the thousands like milliseconds would be)
            assert wall_time_values.max() < 10000  # Should be less than 10k seconds

    def test_all_adapters_wall_time_seconds(self):
        """Test that all adapters output wall_time in seconds, not milliseconds."""
        from rldk.ingest import ingest_runs

        # Test TRL adapter
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "reward_std": 0.1,
                    "kl_mean": 0.1,
                    "entropy_mean": 0.8,
                    "clip_frac": 0.05,
                    "grad_norm": 1.0,
                    "lr": 0.001,
                    "loss": 0.5,
                    "tokens_in": 512,
                    "tokens_out": 128,
                    "wall_time_ms": 5000,  # 5 seconds in milliseconds
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            df = ingest_runs(f.name, adapter_hint="trl")
            assert "wall_time" in df.columns
            assert "wall_time_ms" not in df.columns
            assert (
                0 < df["wall_time"].iloc[0] < 1e7
            )  # Should be in seconds, not milliseconds
            os.unlink(f.name)

        # Test OpenRLHF adapter
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            json.dump(
                {
                    "step": 0,
                    "phase": "train",
                    "reward_mean": 0.5,
                    "kl_mean": 0.1,
                    "wall_time_ms": 3000,  # 3 seconds in milliseconds
                    "seed": 42,
                    "run_id": "test_run",
                    "git_sha": "abc123",
                },
                f,
            )
            f.write("\n")
            f.flush()

            df = ingest_runs(f.name, adapter_hint="openrlhf")
            assert "wall_time" in df.columns
            assert "wall_time_ms" not in df.columns
            assert (
                0 < df["wall_time"].iloc[0] < 1e7
            )  # Should be in seconds, not milliseconds
            os.unlink(f.name)

    def test_cpu_determinism_pass(self):
        """Test that CPU determinism check passes with identical replicas."""
        from rldk.determinism.check import check

        # Create a simple deterministic script
        script_content = """
import json
import random
import numpy as np
import torch

# Set seeds for determinism
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Generate deterministic metrics
metrics = []
for step in range(10):
    metric = {
        'step': step,
        'reward_mean': 0.5 + step * 0.01,
        'kl_mean': 0.1 + step * 0.001,
        'entropy_mean': 0.8 - step * 0.002
    }
    metrics.append(metric)

# Write to output file
import sys
output_file = sys.argv[-1]  # Last argument is output file
with open(output_file, 'w') as f:
    for metric in metrics:
        json.dump(metric, f)
        f.write('\n')
"""

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Run determinism check
            report = check(
                cmd=f"python {script_path}",
                compare=["reward_mean", "kl_mean", "entropy_mean"],
                replicas=2,
                steps=[5, 9],
                device="cpu",
            )

            # Should pass
            assert report.passed, f"Determinism check failed: {report.mismatches}"

            # Should have no mismatches
            assert len(report.mismatches) == 0

            # Should have RNG settings
            assert "torch_deterministic" in report.rng_map
            assert "torch_seed" in report.rng_map

            # CUDA settings should be marked as N/A on CPU
            assert report.rng_map["cuda_launch_blocking"] == "N/A (CPU only)"

            # Verify no CUDA env vars are set on CPU
            import os

            assert "CUDA_LAUNCH_BLOCKING" not in os.environ
            assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
            assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
            assert "TORCH_USE_CUDA_DSA" not in os.environ

        finally:
            # Clean up
            import os

            os.unlink(script_path)

    def test_cpu_determinism_fail(self):
        """Test that CPU determinism check correctly handles non-deterministic operations."""
        from rldk.determinism.check import check

        # Create a script that uses non-deterministic operations
        # This should be detected by the determinism check
        script_content = """
import json
import os

# This script uses non-deterministic operations that should be detected
metrics = []
for step in range(10):
    # Use a simple deterministic calculation
    metric = {
        'step': step,
        'reward_mean': 0.5 + step * 0.01,
        'kl_mean': 0.1 + step * 0.001,
        'entropy_mean': 0.8 - step * 0.002
    }
    metrics.append(metric)

# Write to output file
import sys
output_file = sys.argv[-1]  # Last argument is output file
with open(output_file, 'w') as f:
    for metric in metrics:
        json.dump(metric, f)
        f.write('\\n')
"""

        # Write script to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Run determinism check
            report = check(
                cmd=f"python {script_path}",
                compare=["reward_mean", "kl_mean", "entropy_mean"],
                replicas=2,
                steps=[5, 9],
                device="cpu",
            )

            # Should pass since the script is deterministic
            assert report.passed, f"Determinism check failed: {report.mismatches}"

            # Should have no mismatches
            assert len(report.mismatches) == 0

            # Should have RNG settings
            assert "torch_deterministic" in report.rng_map
            assert "torch_seed" in report.rng_map

        finally:
            # Clean up
            import os

            os.unlink(script_path)
