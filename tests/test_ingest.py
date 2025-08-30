"""Tests for the ingest module."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

from rldk.ingest import ingest_runs
from rldk.adapters import TRLAdapter, OpenRLHFAdapter


class TestTRLAdapter:
    """Test TRL adapter functionality."""
    
    def test_can_handle_trl_file(self):
        """Test TRL adapter can handle TRL files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 1,
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'trl': True
            }, f)
            f.write('\n')
        
        try:
            adapter = TRLAdapter(f.name)
            assert adapter.can_handle()
        finally:
            Path(f.name).unlink()
    
    def test_load_trl_data(self):
        """Test TRL adapter can load and convert data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(3):
                json.dump({
                    'step': i,
                    'reward_mean': 0.5 + i * 0.1,
                    'kl_mean': 0.1 + i * 0.01,
                    'trl': True
                }, f)
                f.write('\n')
        
        try:
            adapter = TRLAdapter(f.name)
            df = adapter.load()
            
            assert len(df) == 3
            assert 'step' in df.columns
            assert 'reward_mean' in df.columns
            assert 'kl_mean' in df.columns
            assert df['step'].iloc[0] == 0
            assert df['step'].iloc[2] == 2
        finally:
            Path(f.name).unlink()


class TestOpenRLHFAdapter:
    """Test OpenRLHF adapter functionality."""
    
    def test_can_handle_openrlhf_file(self):
        """Test OpenRLHF adapter can handle OpenRLHF files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 1,
                'reward_mean': 0.5,
                'kl_mean': 0.1,
                'openrlhf': True
            }, f)
            f.write('\n')
        
        try:
            adapter = OpenRLHFAdapter(f.name)
            assert adapter.can_handle()
        finally:
            Path(f.name).unlink()


class TestIngestRuns:
    """Test main ingest function."""
    
    def test_ingest_trl_data(self):
        """Test ingesting TRL data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for i in range(3):
                json.dump({
                    'step': i,
                    'reward_mean': 0.5 + i * 0.1,
                    'kl_mean': 0.1 + i * 0.01,
                    'trl': True
                }, f)
                f.write('\n')
        
        try:
            df = ingest_runs(f.name, adapter_hint='trl')
            
            assert len(df) == 3
            assert 'step' in df.columns
            assert 'reward_mean' in df.columns
            assert 'kl_mean' in df.columns
            
            # Check that all required columns exist
            required_cols = ['step', 'phase', 'reward_mean', 'reward_std', 'kl_mean', 
                           'entropy_mean', 'clip_frac', 'grad_norm', 'lr', 'loss',
                           'tokens_in', 'tokens_out', 'wall_time_ms', 'seed', 'run_id', 'git_sha']
            
            for col in required_cols:
                assert col in df.columns
        finally:
            Path(f.name).unlink()
    
    def test_auto_detection(self):
        """Test automatic adapter detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump({
                'step': 1,
                'reward_mean': 0.5,
                'trl': True
            }, f)
            f.write('\n')
        
        try:
            df = ingest_runs(f.name)  # No adapter hint
            assert len(df) > 0
        finally:
            Path(f.name).unlink()
