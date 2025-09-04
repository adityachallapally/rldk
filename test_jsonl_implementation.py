#!/usr/bin/env python3
"""Simple test script to verify JSONL implementation."""

import sys
import json
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

from rldk.io.event_schema import Event, create_event_from_row
from rldk.io.validator import validate_jsonl_schema, validate_jsonl_file
from rldk.adapters.trl import TRLAdapter
from rldk.adapters.openrlhf import OpenRLHFAdapter


def test_event_schema():
    """Test Event schema creation and serialization."""
    print("Testing Event schema...")
    
    # Create test data
    test_data = {
        "step": 0,
        "phase": "train",
        "reward_mean": 0.5,
        "kl_mean": 0.1,
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
    }
    
    # Create Event object
    event = create_event_from_row(test_data, "test_run", "abc123")
    
    # Test serialization
    event_dict = event.to_dict()
    event_json = event.to_json()
    
    # Test deserialization
    recreated_event = Event.from_json(event_json)
    
    # Verify they match
    assert event_dict == recreated_event.to_dict()
    print("✅ Event schema test passed")


def test_jsonl_validation():
    """Test JSONL validation functionality."""
    print("Testing JSONL validation...")
    
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write valid JSON lines
        for i in range(3):
            data = {
                "step": i,
                "phase": "train",
                "reward_mean": 0.5 + i * 0.1,
                "kl_mean": 0.1 + i * 0.01,
                "entropy_mean": 0.8 - i * 0.02,
                "clip_frac": 0.2,
                "grad_norm": 1.0,
                "lr": 0.001,
                "loss": 0.4 - i * 0.05,
                "tokens_in": 1000,
                "tokens_out": 500,
                "wall_time": 10.0 + i * 5.0,
                "seed": 42,
                "run_id": "test_run",
                "git_sha": "abc123"
            }
            f.write(json.dumps(data) + "\n")
        
        # Write one malformed line
        f.write('{"step": 3, "phase": "train", "reward_mean": 0.8\n')  # Missing closing brace
        
        # Write another valid line
        data = {
            "step": 4,
            "phase": "train",
            "reward_mean": 0.9,
            "kl_mean": 0.15,
            "entropy_mean": 0.7,
            "clip_frac": 0.2,
            "grad_norm": 1.0,
            "lr": 0.001,
            "loss": 0.25,
            "tokens_in": 1000,
            "tokens_out": 500,
            "wall_time": 25.0,
            "seed": 42,
            "run_id": "test_run",
            "git_sha": "abc123"
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
        
        file_path = f.name
    
    try:
        # Test validation
        is_valid, errors = validate_jsonl_schema(Path(file_path), strict=False)
        
        # Should have errors due to malformed line
        assert not is_valid
        assert len(errors) > 0
        print(f"✅ JSONL validation test passed (found {len(errors)} errors)")
        
        # Test file validation
        is_valid = validate_jsonl_file(Path(file_path), output_errors=False)
        assert not is_valid
        print("✅ JSONL file validation test passed")
        
    finally:
        # Clean up
        os.unlink(file_path)


def test_adapters():
    """Test adapter functionality."""
    print("Testing adapters...")
    
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write valid JSON lines
        for i in range(3):
            data = {
                "step": i,
                "phase": "train",
                "reward_mean": 0.5 + i * 0.1,
                "kl_mean": 0.1 + i * 0.01,
                "entropy_mean": 0.8 - i * 0.02,
                "clip_frac": 0.2,
                "grad_norm": 1.0,
                "lr": 0.001,
                "loss": 0.4 - i * 0.05,
                "tokens_in": 1000,
                "tokens_out": 500,
                "wall_time": 10.0 + i * 5.0,
                "seed": 42,
                "run_id": "test_run",
                "git_sha": "abc123"
            }
            f.write(json.dumps(data) + "\n")
        f.flush()
        
        file_path = f.name
    
    try:
        # Test TRL adapter
        trl_adapter = TRLAdapter(file_path)
        assert trl_adapter.can_handle()
        
        df = trl_adapter.load()
        assert len(df) == 3
        assert "step" in df.columns
        assert "reward_mean" in df.columns
        print("✅ TRL adapter test passed")
        
        # Test OpenRLHF adapter
        openrlhf_adapter = OpenRLHFAdapter(file_path)
        assert openrlhf_adapter.can_handle()
        
        df = openrlhf_adapter.load()
        assert len(df) == 3
        assert "step" in df.columns
        assert "reward_mean" in df.columns
        print("✅ OpenRLHF adapter test passed")
        
    finally:
        # Clean up
        os.unlink(file_path)


def test_malformed_json():
    """Test handling of malformed JSON."""
    print("Testing malformed JSON handling...")
    
    # Create a temporary JSONL file with malformed JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        # Write valid JSON line
        data = {
            "step": 0,
            "phase": "train",
            "reward_mean": 0.5,
            "kl_mean": 0.1,
            "loss": 0.4,
            "run_id": "test_run"
        }
        f.write(json.dumps(data) + "\n")
        
        # Write completely invalid line
        f.write("This is not JSON at all\n")
        
        # Write another valid line
        data = {
            "step": 2,
            "phase": "train",
            "reward_mean": 0.7,
            "kl_mean": 0.3,
            "loss": 0.2,
            "run_id": "test_run"
        }
        f.write(json.dumps(data) + "\n")
        f.flush()
        
        file_path = f.name
    
    try:
        # Test TRL adapter with malformed JSON
        trl_adapter = TRLAdapter(file_path)
        df = trl_adapter.load()
        
        # Should only have 2 valid records (steps 0 and 2)
        assert len(df) == 2
        assert df["step"].iloc[0] == 0
        assert df["step"].iloc[1] == 2
        print("✅ Malformed JSON handling test passed")
        
    finally:
        # Clean up
        os.unlink(file_path)


def main():
    """Run all tests."""
    print("Running JSONL implementation tests...")
    print("=" * 50)
    
    try:
        test_event_schema()
        test_jsonl_validation()
        test_adapters()
        test_malformed_json()
        
        print("=" * 50)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()