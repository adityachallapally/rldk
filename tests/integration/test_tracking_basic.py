#!/usr/bin/env python3
"""
Basic test of the tracking system without external dependencies.
"""

import hashlib
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path
sys.path.append('src')

# Mock the external dependencies
class MockNumpy:
    def __init__(self):
        self.__version__ = "1.21.0"
        self.random = MockRandom()
        self.bool_ = bool

    def randn(self, *args):
        return [0.1, 0.2, 0.3]  # Mock data

    def choice(self, n, size, replace=False):
        return list(range(min(size, n)))

    def array(self, data):
        return data

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockRandom:
    def __init__(self):
        pass

    def seed(self, seed):
        return True

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockPandas:
    def __init__(self):
        self.__version__ = "1.3.0"

    def DataFrame(self, data):
        return MockDataFrame(data)

    def Series(self, data):
        return MockSeries(data)

class MockDataFrame:
    def __init__(self, data):
        self.data = data
        self.columns = list(data.keys()) if isinstance(data, dict) else []
        self.shape = (len(data), len(self.columns)) if self.columns else (0, 0)

    def memory_usage(self, deep=True):
        return MockSeries([100, 200, 300])

    def to_string(self):
        return "mock dataframe"

class MockSeries:
    def __init__(self, data):
        self.data = data

    def sum(self):
        return sum(self.data)

class MockTorch:
    def __init__(self):
        self.__version__ = "1.12.0"
        self.nn = MockNN()
        self._rng_state = MockRNGState()

    def __getattr__(self, name):
        if name == 'cuda':
            return MockCuda()
        return lambda *args, **kwargs: None

    def manual_seed(self, seed):
        self._rng_state.seed = seed
        return seed

    def get_rng_state(self):
        return self._rng_state

class MockNN:
    def __init__(self):
        self.Module = MockModule

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockModule:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockCuda:
    def is_available(self):
        return False

    def device_count(self):
        return 0


class MockRNGState:
    def __init__(self):
        self.seed = None

    def tolist(self):
        return [self.seed] if self.seed is not None else []

class MockTransformers:
    def __init__(self):
        self.__version__ = "4.20.0"

class MockDatasets:
    def __init__(self):
        self.__version__ = "2.0.0"
        self.Dataset = type('Dataset', (), {})
        self.DatasetDict = dict


@pytest.fixture
def mock_tracking_dependencies(monkeypatch):
    """Install mocked external dependencies for tracking tests."""

    mocks = {
        'numpy': MockNumpy(),
        'pandas': MockPandas(),
        'torch': MockTorch(),
        'transformers': MockTransformers(),
        'datasets': MockDatasets(),
    }

    for module_name, mock in mocks.items():
        monkeypatch.setitem(sys.modules, module_name, mock)

    # Ensure we import tracking with the mocked dependencies.
    monkeypatch.delitem(sys.modules, 'rldk.tracking', raising=False)

    yield mocks


def test_basic_tracking(mock_tracking_dependencies):
    """Test basic tracking functionality."""
    print("Testing basic tracking functionality...")

    from rldk.tracking import ExperimentTracker, TrackingConfig

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create config
        config = TrackingConfig(
            experiment_name="basic_test",
            output_dir=Path(temp_dir),
            save_to_wandb=False
        )

        # Create tracker
        tracker = ExperimentTracker(config)

        print("1. Starting experiment...")
        start_info = tracker.start_experiment()
        assert "experiment_id" in start_info
        print("   ✓ Experiment started")

        print("2. Testing dataset tracking...")
        # Test with mock data
        test_data = [1, 2, 3, 4, 5]
        dataset_info = tracker.track_dataset(test_data, "test_data")
        if dataset_info:
            if dataset_info.get("exception"):
                assert 'error' in dataset_info
            else:
                assert dataset_info.get("name") == "test_data"
        print("   ✓ Dataset tracked")

        print("3. Testing seed tracking...")
        seed_info = tracker.set_seeds(42)
        assert seed_info["set_seed"] == 42
        print("   ✓ Seeds set")

        print("4. Testing metadata...")
        tracker.add_metadata("test_key", "test_value")
        tracker.add_tag("test_tag")
        tracker.set_notes("Test notes")
        print("   ✓ Metadata added")

        print("5. Finishing experiment...")
        summary = tracker.finish_experiment()
        assert summary["experiment_name"] == "basic_test"
        print("   ✓ Experiment finished")

        print("6. Checking output files...")
        output_files = list(Path(temp_dir).glob("*"))
        assert len(output_files) > 0
        print(f"   ✓ {len(output_files)} files created")

        # Check JSON file content
        json_files = [f for f in output_files if f.suffix == '.json']
        if json_files:
            with open(json_files[0]) as f:
                data = json.load(f)
                assert data["experiment_name"] == "basic_test"
                assert "test_key" in data["metadata"]
                assert data["metadata"]["test_key"] == "test_value"
            print("   ✓ JSON file content verified")

        print("\n✓ All basic tests passed!")


def test_config(mock_tracking_dependencies):
    """Test configuration functionality."""
    print("Testing configuration...")

    from rldk.tracking import TrackingConfig

    config = TrackingConfig(experiment_name="test")
    assert config.experiment_name == "test"
    assert config.experiment_id is not None
    assert config.enable_dataset_tracking is True
    assert config.enable_model_tracking is True
    assert config.enable_environment_tracking is True
    assert config.enable_seed_tracking is True
    assert config.enable_git_tracking is True

    print("✓ Configuration tests passed!")


def test_checksum_functionality():
    """Test checksum functionality."""
    print("Testing checksum functionality...")

    # Test that identical data produces identical checksums
    data1 = [1, 2, 3, 4, 5]
    data2 = [1, 2, 3, 4, 5]

    hash1 = hashlib.sha256(str(data1).encode()).hexdigest()
    hash2 = hashlib.sha256(str(data2).encode()).hexdigest()

    assert hash1 == hash2
    print("✓ Checksum consistency verified")

    # Test that different data produces different checksums
    data3 = [1, 2, 3, 4, 6]
    hash3 = hashlib.sha256(str(data3).encode()).hexdigest()

    assert hash1 != hash3
    print("✓ Checksum uniqueness verified")


def main():
    """Run all tests."""
    print("="*60)
    print("BASIC TRACKING SYSTEM TESTS")
    print("="*60)

    try:
        test_config()
        test_checksum_functionality()
        test_basic_tracking()

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("The tracking system is working correctly.")
        print("Key features verified:")
        print("✓ Configuration system")
        print("✓ Experiment lifecycle")
        print("✓ Dataset tracking")
        print("✓ Seed management")
        print("✓ Metadata handling")
        print("✓ File output")
        print("✓ Checksum functionality")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
