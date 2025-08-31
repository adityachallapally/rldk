"""Tests for ingest adapters."""

import pytest
import pandas as pd
import tempfile
import json
from pathlib import Path

from rldk.adapters.trl import TRLAdapter
from rldk.adapters.openrlhf import OpenRLHFAdapter
from rldk.adapters.wandb import WandBAdapter
from rldk.io.schema import TrainingMetrics, MetricsSchema


class TestTRLAdapter:
    """Test TRL adapter functionality."""
    
    def test_can_handle_jsonl(self):
        """Test that TRL adapter can handle JSONL files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 0,
                'phase': 'train',
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'entropy_mean': 0.8,
                'loss': 0.4,
                'lr': 0.001,
                'seed': 42,
                'run_id': 'test_run',
                'git_sha': 'abc123'
            }, f)
            f.write('\n')
            f.flush()
            
            adapter = TRLAdapter(f.name)
            assert adapter.can_handle()
    
    def test_load_jsonl(self):
        """Test loading TRL JSONL data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write test data
            for i in range(3):
                json.dump({
                    'step': i,
                    'phase': 'train',
                    'reward_mean': 0.5 + i * 0.1,
                    'kl_mean': 0.1 + i * 0.01,
                    'entropy_mean': 0.8 - i * 0.02,
                    'loss': 0.4 - i * 0.05,
                    'lr': 0.001,
                    'wall_time': i * 10.0,
                    'seed': 42,
                    'run_id': 'test_run',
                    'git_sha': 'abc123'
                }, f)
                f.write('\n')
            f.flush()
            
            adapter = TRLAdapter(f.name)
            df = adapter.load()
            
            assert len(df) == 3
            assert 'step' in df.columns
            assert 'reward_mean' in df.columns
            assert 'kl_mean' in df.columns
            assert 'wall_time' in df.columns
            assert df['step'].iloc[0] == 0
            assert df['reward_mean'].iloc[0] == 0.5
    
    def test_round_trip_schema(self):
        """Test round-trip conversion to and from schema."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write test data
            json.dump({
                'step': 0,
                'phase': 'train',
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'entropy_mean': 0.8,
                'loss': 0.4,
                'lr': 0.001,
                'wall_time': 10.0,
                'seed': 42,
                'run_id': 'test_run',
                'git_sha': 'abc123'
            }, f)
            f.write('\n')
            f.flush()
            
            adapter = TRLAdapter(f.name)
            df = adapter.load()
            
            # Convert to schema
            schema = MetricsSchema.from_dataframe(df)
            assert len(schema.metrics) == 1
            
            # Convert back to dataframe
            df_round_trip = schema.to_dataframe()
            assert len(df_round_trip) == 1
            assert df_round_trip['reward_mean'].iloc[0] == 0.5


class TestOpenRLHFAdapter:
    """Test OpenRLHF adapter functionality."""
    
    def test_can_handle_jsonl(self):
        """Test that OpenRLHF adapter can handle JSONL files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 0,
                'phase': 'train',
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'entropy_mean': 0.8,
                'loss': 0.4,
                'lr': 0.001,
                'seed': 42,
                'run_id': 'test_run',
                'git_sha': 'abc123'
            }, f)
            f.write('\n')
            f.flush()
            
            adapter = OpenRLHFAdapter(f.name)
            assert adapter.can_handle()
    
    def test_load_jsonl(self):
        """Test loading OpenRLHF JSONL data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write test data
            for i in range(3):
                json.dump({
                    'step': i,
                    'phase': 'train',
                    'reward_mean': 0.5 + i * 0.1,
                    'kl_mean': 0.1 + i * 0.01,
                    'entropy_mean': 0.8 - i * 0.02,
                    'loss': 0.4 - i * 0.05,
                    'lr': 0.001,
                    'wall_time': i * 10.0,
                    'seed': 42,
                    'run_id': 'test_run',
                    'git_sha': 'abc123'
                }, f)
                f.write('\n')
            f.flush()
            
            adapter = OpenRLHFAdapter(f.name)
            df = adapter.load()
            
            assert len(df) == 3
            assert 'step' in df.columns
            assert 'reward_mean' in df.columns
            assert 'kl_mean' in df.columns
            assert 'wall_time' in df.columns
            assert df['step'].iloc[0] == 0
            assert df['reward_mean'].iloc[0] == 0.5


class TestWandBAdapter:
    """Test WandB adapter functionality."""
    
    def test_parse_wandb_uri(self):
        """Test parsing wandb:// URIs."""
        adapter = WandBAdapter('wandb://entity/project/run_id')
        assert adapter.entity == 'entity'
        assert adapter.project == 'project'
        assert adapter.run_id == 'run_id'
    
    def test_can_handle_wandb_uri(self):
        """Test that WandB adapter can handle wandb:// URIs."""
        adapter = WandBAdapter('wandb://entity/project/run_id')
        assert adapter.can_handle()
    
    def test_cannot_handle_invalid_uri(self):
        """Test that WandB adapter cannot handle invalid URIs."""
        adapter = WandBAdapter('invalid_uri')
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 0,
                'phase': 'train',
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'entropy_mean': 0.8,
                'loss': 0.4,
                'lr': 0.001,
                'wall_time': 10.0,
                'seed': 42,
                'run_id': 'test_run',
                'git_sha': 'abc123'
            }, f)
            f.write('\n')
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 0,
                'phase': 'train',
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'entropy_mean': 0.8,
                'loss': 0.4,
                'lr': 0.001,
                'wall_time': 10.0,
                'seed': 42,
                'run_id': 'test_run',
                'git_sha': 'abc123'
            }, f)
            f.write('\n')
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