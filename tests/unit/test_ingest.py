"""Unit tests for ingest functionality."""

import pytest
import json
import tempfile
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from rldk.ingest.ingest import ingest_runs, _detect_adapter_type, _get_adapter_format_requirements
from rldk.adapters.demo_jsonl import DemoJSONLAdapter
from rldk.adapters.custom_jsonl import CustomJSONLAdapter
from rldk.adapters.trl import TRLAdapter
from rldk.adapters.openrlhf import OpenRLHFAdapter


class TestIngestRuns:
    """Test ingest_runs function."""
    
    def test_ingest_runs_with_demo_adapter(self):
        """Test ingest_runs with demo adapter."""
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
                "advantage_std": 1.0,
                "pass_rate": 0.4
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_file = f.name
        
        try:
            df = ingest_runs(temp_file, adapter_hint="demo_jsonl")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            assert df['kl_mean'].iloc[0] == 0.05
            assert df['pass_rate'].iloc[0] == 0.4
            
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_auto_detect_demo(self):
        """Test ingest_runs auto-detection with demo data."""
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
                "advantage_std": 1.0,
                "pass_rate": 0.4
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_file = f.name
        
        try:
            df = ingest_runs(temp_file)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_with_custom_adapter(self):
        """Test ingest_runs with custom adapter."""
        # Create a temporary custom JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            custom_data = {
                "global_step": 1,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.05,
                "entropy": 2.0,
                "loss": 1.0
            }
            f.write(json.dumps(custom_data) + "\n")
            temp_file = f.name
        
        try:
            df = ingest_runs(temp_file, adapter_hint="custom_jsonl")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            assert df['kl_mean'].iloc[0] == 0.05
            
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_with_trl_adapter(self):
        """Test ingest_runs with TRL adapter."""
        # Create a temporary TRL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            trl_data = {
                "step": 1,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.05,
                "entropy": 2.0,
                "loss": 1.0
            }
            f.write(json.dumps(trl_data) + "\n")
            temp_file = f.name
        
        try:
            df = ingest_runs(temp_file, adapter_hint="trl")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_with_openrlhf_adapter(self):
        """Test ingest_runs with OpenRLHF adapter."""
        # Create a temporary OpenRLHF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            openrlhf_data = {
                "step": 1,
                "ppo/rewards/mean": 0.5,
                "ppo/kl_div": 0.05,
                "ppo/entropy": 2.0,
                "ppo/loss": 1.0
            }
            f.write(json.dumps(openrlhf_data) + "\n")
            temp_file = f.name
        
        try:
            df = ingest_runs(temp_file, adapter_hint="openrlhf")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_invalid_adapter(self):
        """Test ingest_runs with invalid adapter hint."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"step": 1}\n')
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unknown adapter"):
                ingest_runs(temp_file, adapter_hint="invalid_adapter")
                
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_nonexistent_file(self):
        """Test ingest_runs with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            ingest_runs("nonexistent.jsonl")
    
    def test_ingest_runs_directory(self):
        """Test ingest_runs with directory."""
        # Create a temporary directory with demo files
        with tempfile.TemporaryDirectory() as temp_dir:
            demo_file1 = Path(temp_dir) / "demo1.jsonl"
            demo_file2 = Path(temp_dir) / "demo2.jsonl"
            
            demo_data1 = {
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
            
            demo_data2 = {
                "step": 2,
                "reward_mean": 0.6,
                "reward_std": 0.1,
                "kl": 0.06,
                "entropy": 1.9,
                "loss": 0.9,
                "policy_grad_norm": 0.9,
                "value_grad_norm": 0.7,
                "advantage_mean": 0.1,
                "advantage_std": 1.1,
                "pass_rate": 0.5
            }
            
            with open(demo_file1, 'w') as f:
                f.write(json.dumps(demo_data1) + "\n")
            
            with open(demo_file2, 'w') as f:
                f.write(json.dumps(demo_data2) + "\n")
            
            df = ingest_runs(temp_dir, adapter_hint="demo_jsonl")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert df['step'].iloc[0] == 1
            assert df['step'].iloc[1] == 2
    
    def test_ingest_runs_empty_file(self):
        """Test ingest_runs with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            pass  # Empty file
            temp_file = f.name
        
        try:
            df = ingest_runs(temp_file, adapter_hint="demo_jsonl")
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            
        finally:
            Path(temp_file).unlink()
    
    def test_ingest_runs_with_events(self):
        """Test ingest_runs with events parameter."""
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
            temp_file = f.name
        
        try:
            df, events = ingest_runs(temp_file, adapter_hint="demo_jsonl", events=True)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert isinstance(events, list)
            assert len(events) == 1
            assert events[0]['step'] == 1
            
        finally:
            Path(temp_file).unlink()


class TestDetectAdapterType:
    """Test _detect_adapter_type function."""
    
    def test_detect_adapter_type_demo(self):
        """Test detecting demo adapter type."""
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
                "advantage_std": 1.0,
                "pass_rate": 0.4
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "demo_jsonl"
            
        finally:
            Path(temp_file).unlink()
    
    def test_detect_adapter_type_custom(self):
        """Test detecting custom adapter type."""
        # Create a temporary custom file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            custom_data = {
                "global_step": 1,
                "reward_scalar": 0.5,
                "kl_to_ref": 0.05,
                "entropy": 2.0,
                "loss": 1.0
            }
            f.write(json.dumps(custom_data) + "\n")
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "custom_jsonl"
            
        finally:
            Path(temp_file).unlink()
    
    def test_detect_adapter_type_trl(self):
        """Test detecting TRL adapter type."""
        # Create a temporary TRL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            trl_data = {
                "step": 1,
                "phase": "train",
                "reward_mean": 0.5,
                "kl_mean": 0.05,
                "entropy": 2.0,
                "loss": 1.0
            }
            f.write(json.dumps(trl_data) + "\n")
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "trl"
            
        finally:
            Path(temp_file).unlink()
    
    def test_detect_adapter_type_openrlhf(self):
        """Test detecting OpenRLHF adapter type."""
        # Create a temporary OpenRLHF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            openrlhf_data = {
                "step": 1,
                "ppo/rewards/mean": 0.5,
                "ppo/kl_div": 0.05,
                "ppo/entropy": 2.0,
                "ppo/loss": 1.0
            }
            f.write(json.dumps(openrlhf_data) + "\n")
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "openrlhf"
            
        finally:
            Path(temp_file).unlink()
    
    def test_detect_adapter_type_fallback(self):
        """Test detecting adapter type with fallback to TRL."""
        # Create a temporary file that doesn't match any specific adapter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            generic_data = {
                "step": 1,
                "some_field": "value"
            }
            f.write(json.dumps(generic_data) + "\n")
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "trl"  # Fallback
            
        finally:
            Path(temp_file).unlink()
    
    def test_detect_adapter_type_nonexistent_file(self):
        """Test detecting adapter type with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            _detect_adapter_type("nonexistent.jsonl")
    
    def test_detect_adapter_type_invalid_json(self):
        """Test detecting adapter type with invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("invalid json content\n")
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "trl"  # Fallback
            
        finally:
            Path(temp_file).unlink()
    
    def test_detect_adapter_type_empty_file(self):
        """Test detecting adapter type with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            pass  # Empty file
            temp_file = f.name
        
        try:
            adapter_type = _detect_adapter_type(temp_file)
            assert adapter_type == "trl"  # Fallback
            
        finally:
            Path(temp_file).unlink()


class TestGetAdapterFormatRequirements:
    """Test _get_adapter_format_requirements function."""
    
    def test_get_adapter_format_requirements_demo(self):
        """Test getting demo adapter format requirements."""
        requirements = _get_adapter_format_requirements("demo_jsonl")
        
        assert "description" in requirements
        assert "file_extensions" in requirements
        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "examples" in requirements
        assert "suggestions" in requirements
        
        assert "demo" in requirements["description"].lower()
        assert ".jsonl" in requirements["file_extensions"]
        assert "step" in requirements["required_fields"]
        assert "reward_mean" in requirements["required_fields"]
        assert "pass_rate" in requirements["required_fields"]
    
    def test_get_adapter_format_requirements_custom(self):
        """Test getting custom adapter format requirements."""
        requirements = _get_adapter_format_requirements("custom_jsonl")
        
        assert "description" in requirements
        assert "file_extensions" in requirements
        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "examples" in requirements
        assert "suggestions" in requirements
        
        assert "custom" in requirements["description"].lower()
        assert ".jsonl" in requirements["file_extensions"]
        assert "global_step" in requirements["required_fields"]
        assert "reward_scalar" in requirements["required_fields"]
    
    def test_get_adapter_format_requirements_trl(self):
        """Test getting TRL adapter format requirements."""
        requirements = _get_adapter_format_requirements("trl")
        
        assert "description" in requirements
        assert "file_extensions" in requirements
        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "examples" in requirements
        assert "suggestions" in requirements
        
        assert "trl" in requirements["description"].lower()
        assert ".jsonl" in requirements["file_extensions"]
        assert "step" in requirements["required_fields"]
        assert "phase" in requirements["required_fields"]
    
    def test_get_adapter_format_requirements_openrlhf(self):
        """Test getting OpenRLHF adapter format requirements."""
        requirements = _get_adapter_format_requirements("openrlhf")
        
        assert "description" in requirements
        assert "file_extensions" in requirements
        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "examples" in requirements
        assert "suggestions" in requirements
        
        assert "openrlhf" in requirements["description"].lower()
        assert ".jsonl" in requirements["file_extensions"]
        assert "step" in requirements["required_fields"]
        assert "ppo/rewards/mean" in requirements["required_fields"]
    
    def test_get_adapter_format_requirements_invalid(self):
        """Test getting format requirements for invalid adapter."""
        requirements = _get_adapter_format_requirements("invalid_adapter")
        
        assert "description" in requirements
        assert "file_extensions" in requirements
        assert "required_fields" in requirements
        assert "optional_fields" in requirements
        assert "examples" in requirements
        assert "suggestions" in requirements
        
        assert "unknown" in requirements["description"].lower()
        assert requirements["file_extensions"] == []
        assert requirements["required_fields"] == []
        assert requirements["optional_fields"] == []