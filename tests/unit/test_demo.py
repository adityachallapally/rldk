"""Unit tests for RLDK demo functionality."""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

from rldk.adapters.demo_jsonl import DemoJSONLAdapter
from rldk.utils.seed import set_global_seed, get_global_seed, DEFAULT_SEED


class TestDemoJSONLAdapter:
    """Test the DemoJSONLAdapter."""

    def test_can_handle_demo_file(self):
        """Test that adapter can detect demo JSONL files."""
        # Create a temporary demo file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            demo_data = {
                "step": 1,
                "reward_mean": 0.5,
                "reward_std": 0.1,
                "kl": 0.05,
                "entropy": 2.0,
                "loss": 1.0,
                "policy_grad_norm": 1.0,
                "value_grad_norm": 0.8,
                "advantage_mean": 0.0,
                "advantage_std": 1.0
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_path = Path(f.name)

        try:
            adapter = DemoJSONLAdapter(temp_path)
            assert adapter.can_handle() is True
        finally:
            temp_path.unlink()

    def test_can_handle_grpo_demo_file(self):
        """Test that adapter can detect GRPO demo JSONL files."""
        # Create a temporary GRPO demo file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            demo_data = {
                "step": 1,
                "reward_mean": 0.5,
                "reward_std": 0.1,
                "kl": 0.05,
                "entropy": 2.0,
                "loss": 1.0,
                "policy_grad_norm": 1.0,
                "value_grad_norm": 0.8,
                "advantage_mean": 0.0,
                "advantage_std": 1.0,
                "pass_rate": 0.4
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_path = Path(f.name)

        try:
            adapter = DemoJSONLAdapter(temp_path)
            assert adapter.can_handle() is True
        finally:
            temp_path.unlink()

    def test_cannot_handle_non_demo_file(self):
        """Test that adapter rejects non-demo files."""
        # Create a temporary non-demo file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            non_demo_data = {
                "global_step": 1,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.05,
                "entropy": 2.0
            }
            f.write(json.dumps(non_demo_data) + "\n")
            temp_path = Path(f.name)

        try:
            adapter = DemoJSONLAdapter(temp_path)
            assert adapter.can_handle() is False
        finally:
            temp_path.unlink()

    def test_load_demo_data(self):
        """Test loading demo data."""
        # Create a temporary demo file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            demo_data = [
                {
                    "step": 1,
                    "reward_mean": 0.5,
                    "reward_std": 0.1,
                    "kl": 0.05,
                    "entropy": 2.0,
                    "loss": 1.0,
                    "policy_grad_norm": 1.0,
                    "value_grad_norm": 0.8,
                    "advantage_mean": 0.0,
                    "advantage_std": 1.0
                },
                {
                    "step": 2,
                    "reward_mean": 0.501,
                    "reward_std": 0.1001,
                    "kl": 0.0505,
                    "entropy": 1.999,
                    "loss": 0.998,
                    "policy_grad_norm": 1.01,
                    "value_grad_norm": 0.81,
                    "advantage_mean": 0.01,
                    "advantage_std": 1.01
                }
            ]
            for data in demo_data:
                f.write(json.dumps(data) + "\n")
            temp_path = Path(f.name)

        try:
            adapter = DemoJSONLAdapter(temp_path)
            df = adapter.load()
            
            # Check that we got the expected data
            assert len(df) == 2
            assert df["step"].tolist() == [1, 2]
            assert df["reward_mean"].tolist() == [0.5, 0.501]
            assert df["kl_mean"].tolist() == [0.05, 0.0505]  # kl mapped to kl_mean
            assert df["entropy_mean"].tolist() == [2.0, 1.999]  # entropy mapped to entropy_mean
            assert df["grad_norm"].tolist() == [1.0, 1.01]  # policy_grad_norm mapped to grad_norm
            
            # Check required columns exist
            required_cols = [
                "step", "phase", "reward_mean", "reward_std", "kl_mean", "entropy_mean",
                "clip_frac", "grad_norm", "lr", "loss", "tokens_in", "tokens_out",
                "wall_time", "seed", "run_id", "git_sha"
            ]
            for col in required_cols:
                assert col in df.columns
                
        finally:
            temp_path.unlink()

    def test_load_demo_directory(self):
        """Test loading demo data from directory."""
        # Create a temporary directory with demo files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create two demo files
            for i in range(2):
                demo_file = temp_path / f"demo_run_{i:02d}.jsonl"
                demo_data = {
                    "step": i + 1,
                    "reward_mean": 0.5 + i * 0.001,
                    "reward_std": 0.1,
                    "kl": 0.05 + i * 0.0005,
                    "entropy": 2.0 - i * 0.001,
                    "loss": 1.0 - i * 0.002,
                    "policy_grad_norm": 1.0,
                    "value_grad_norm": 0.8,
                    "advantage_mean": 0.0,
                    "advantage_std": 1.0
                }
                with open(demo_file, 'w') as f:
                    f.write(json.dumps(demo_data) + "\n")
            
            adapter = DemoJSONLAdapter(temp_path)
            df = adapter.load()
            
            # Check that we got data from both files
            assert len(df) == 2
            assert df["step"].tolist() == [1, 2]
            assert df["reward_mean"].tolist() == [0.5, 0.501]


class TestSeeding:
    """Test the seeding utilities."""

    def test_set_global_seed(self):
        """Test setting global seed."""
        set_global_seed(42)
        assert get_global_seed() == 42

    def test_default_seed(self):
        """Test default seed value."""
        assert DEFAULT_SEED == 1337

    def test_seed_persistence(self):
        """Test that seed persists across calls."""
        set_global_seed(123)
        assert get_global_seed() == 123
        
        # Set again
        set_global_seed(456)
        assert get_global_seed() == 456

    def test_seeded_random_state(self):
        """Test seeded random state functionality."""
        from rldk.utils.seed import seeded_random_state, restore_random_state
        
        # Get a seeded state
        states = seeded_random_state(42)
        assert len(states) == 3  # python, numpy, torch states
        
        # Restore the state
        restore_random_state(states)
        # This should not raise an exception


class TestDemoIntegration:
    """Test demo integration with ingest_runs."""

    def test_ingest_demo_data(self):
        """Test that ingest_runs can handle demo data."""
        from rldk.ingest import ingest_runs
        
        # Create a temporary demo file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            demo_data = {
                "step": 1,
                "reward_mean": 0.5,
                "reward_std": 0.1,
                "kl": 0.05,
                "entropy": 2.0,
                "loss": 1.0,
                "policy_grad_norm": 1.0,
                "value_grad_norm": 0.8,
                "advantage_mean": 0.0,
                "advantage_std": 1.0
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_path = Path(f.name)

        try:
            # Test auto-detection
            df = ingest_runs(temp_path)
            assert len(df) == 1
            assert df["step"].iloc[0] == 1
            assert df["reward_mean"].iloc[0] == 0.5
            
            # Test explicit adapter
            df2 = ingest_runs(temp_path, adapter_hint="demo_jsonl")
            assert len(df2) == 1
            assert df2["step"].iloc[0] == 1
            
        finally:
            temp_path.unlink()