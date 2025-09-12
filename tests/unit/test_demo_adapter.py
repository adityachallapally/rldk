"""Unit tests for demo adapter functionality."""

import pytest
import json
import tempfile
import pandas as pd
from pathlib import Path

from rldk.adapters.demo_jsonl import DemoJSONLAdapter


class TestDemoJSONLAdapter:
    """Test DemoJSONLAdapter class."""
    
    def test_demo_adapter_creation(self):
        """Test creating a demo adapter."""
        adapter = DemoJSONLAdapter("test.jsonl")
        assert adapter.source == "test.jsonl"
    
    def test_can_handle_valid_demo_file(self):
        """Test can_handle with valid demo file."""
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
            adapter = DemoJSONLAdapter(temp_file)
            assert adapter.can_handle() is True
        finally:
            Path(temp_file).unlink()
    
    def test_can_handle_invalid_file(self):
        """Test can_handle with invalid file."""
        # Create a temporary non-demo file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            invalid_data = {
                "step": 1,
                "some_other_field": "value"
            }
            f.write(json.dumps(invalid_data) + "\n")
            temp_file = f.name
        
        try:
            adapter = DemoJSONLAdapter(temp_file)
            assert adapter.can_handle() is False
        finally:
            Path(temp_file).unlink()
    
    def test_can_handle_nonexistent_file(self):
        """Test can_handle with nonexistent file."""
        adapter = DemoJSONLAdapter("nonexistent.jsonl")
        assert adapter.can_handle() is False
    
    def test_can_handle_directory(self):
        """Test can_handle with directory containing demo files."""
        # Create a temporary directory with demo files
        with tempfile.TemporaryDirectory() as temp_dir:
            demo_file = Path(temp_dir) / "demo.jsonl"
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
            
            with open(demo_file, 'w') as f:
                f.write(json.dumps(demo_data) + "\n")
            
            adapter = DemoJSONLAdapter(temp_dir)
            assert adapter.can_handle() is True
    
    def test_load_demo_data(self):
        """Test loading demo data."""
        # Create a temporary demo file with multiple entries
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            demo_entries = [
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
                    "advantage_std": 1.0,
                    "pass_rate": 0.4
                },
                {
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
            ]
            
            for entry in demo_entries:
                f.write(json.dumps(entry) + "\n")
            
            temp_file = f.name
        
        try:
            adapter = DemoJSONLAdapter(temp_file)
            df = adapter.load()
            
            # Check DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            
            # Check required columns
            required_columns = [
                'step', 'reward_mean', 'reward_std', 'kl_mean', 'entropy',
                'loss', 'policy_grad_norm', 'value_grad_norm', 'advantage_mean',
                'advantage_std', 'pass_rate'
            ]
            for col in required_columns:
                assert col in df.columns
            
            # Check data values
            assert df['step'].iloc[0] == 1
            assert df['step'].iloc[1] == 2
            assert df['reward_mean'].iloc[0] == 0.5
            assert df['reward_mean'].iloc[1] == 0.6
            assert df['kl_mean'].iloc[0] == 0.05  # kl mapped to kl_mean
            assert df['kl_mean'].iloc[1] == 0.06
            assert df['pass_rate'].iloc[0] == 0.4
            assert df['pass_rate'].iloc[1] == 0.5
            
        finally:
            Path(temp_file).unlink()
    
    def test_load_demo_data_with_missing_fields(self):
        """Test loading demo data with missing fields."""
        # Create a temporary demo file with missing fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            demo_data = {
                "step": 1,
                "reward_mean": 0.5,
                "reward_std": 0.1,
                "kl": 0.05,
                "entropy": 2.0,
                "loss": 1.0,
                # Missing: policy_grad_norm, value_grad_norm, advantage_mean, advantage_std, pass_rate
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_file = f.name
        
        try:
            adapter = DemoJSONLAdapter(temp_file)
            df = adapter.load()
            
            # Check DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            
            # Check that missing fields are filled with defaults
            assert df['policy_grad_norm'].iloc[0] == 0.0
            assert df['value_grad_norm'].iloc[0] == 0.0
            assert df['advantage_mean'].iloc[0] == 0.0
            assert df['advantage_std'].iloc[0] == 0.0
            assert df['pass_rate'].iloc[0] == 0.0
            
            # Check that present fields are correct
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            assert df['kl_mean'].iloc[0] == 0.05
            
        finally:
            Path(temp_file).unlink()
    
    def test_load_demo_data_with_extra_fields(self):
        """Test loading demo data with extra fields."""
        # Create a temporary demo file with extra fields
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
                "pass_rate": 0.4,
                "extra_field": "extra_value",  # Extra field
                "another_field": 123  # Another extra field
            }
            f.write(json.dumps(demo_data) + "\n")
            temp_file = f.name
        
        try:
            adapter = DemoJSONLAdapter(temp_file)
            df = adapter.load()
            
            # Check DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            
            # Check that extra fields are ignored
            assert 'extra_field' not in df.columns
            assert 'another_field' not in df.columns
            
            # Check that required fields are present
            assert 'step' in df.columns
            assert 'reward_mean' in df.columns
            assert 'pass_rate' in df.columns
            
        finally:
            Path(temp_file).unlink()
    
    def test_load_demo_data_with_invalid_json(self):
        """Test loading demo data with invalid JSON."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("invalid json content\n")
            temp_file = f.name
        
        try:
            adapter = DemoJSONLAdapter(temp_file)
            
            # Should raise an exception
            with pytest.raises((json.JSONDecodeError, ValueError)):
                adapter.load()
                
        finally:
            Path(temp_file).unlink()
    
    def test_load_demo_data_empty_file(self):
        """Test loading demo data from empty file."""
        # Create a temporary empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            pass  # Empty file
            temp_file = f.name
        
        try:
            adapter = DemoJSONLAdapter(temp_file)
            df = adapter.load()
            
            # Should return empty DataFrame
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 0
            
        finally:
            Path(temp_file).unlink()
    
    def test_load_demo_data_directory(self):
        """Test loading demo data from directory."""
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
            
            adapter = DemoJSONLAdapter(temp_dir)
            df = adapter.load()
            
            # Should load data from both files
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            
            # Check that data from both files is present
            assert df['step'].iloc[0] == 1
            assert df['step'].iloc[1] == 2
    
    def test_extract_demo_metric(self):
        """Test extracting demo metrics from JSON object."""
        adapter = DemoJSONLAdapter("test.jsonl")
        
        # Test with all fields present
        json_obj = {
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
        
        metrics = adapter._extract_demo_metric(json_obj)
        
        assert metrics['step'] == 1
        assert metrics['reward_mean'] == 0.5
        assert metrics['kl_mean'] == 0.05  # kl mapped to kl_mean
        assert metrics['pass_rate'] == 0.4
        
        # Test with missing fields
        json_obj_missing = {
            "step": 1,
            "reward_mean": 0.5,
            "kl": 0.05,
            "entropy": 2.0,
            "loss": 1.0
        }
        
        metrics_missing = adapter._extract_demo_metric(json_obj_missing)
        
        assert metrics_missing['step'] == 1
        assert metrics_missing['reward_mean'] == 0.5
        assert metrics_missing['kl_mean'] == 0.05
        assert metrics_missing['pass_rate'] == 0.0  # Default value
        assert metrics_missing['policy_grad_norm'] == 0.0  # Default value
    
    def test_parse_file(self):
        """Test parsing a single demo file."""
        adapter = DemoJSONLAdapter("test.jsonl")
        
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
            df = adapter._parse_file(temp_file)
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert df['step'].iloc[0] == 1
            assert df['reward_mean'].iloc[0] == 0.5
            
        finally:
            Path(temp_file).unlink()
    
    def test_parse_file_nonexistent(self):
        """Test parsing a nonexistent file."""
        adapter = DemoJSONLAdapter("test.jsonl")
        
        with pytest.raises(FileNotFoundError):
            adapter._parse_file("nonexistent.jsonl")
    
    def test_adapter_source_property(self):
        """Test adapter source property."""
        adapter = DemoJSONLAdapter("test.jsonl")
        assert adapter.source == "test.jsonl"
        
        adapter.source = "new_test.jsonl"
        assert adapter.source == "new_test.jsonl"