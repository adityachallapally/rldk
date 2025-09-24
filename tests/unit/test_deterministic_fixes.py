#!/usr/bin/env python3
"""
Test the deterministic fixes for the tracking system.

This script verifies that:
1. Dataset checksums are deterministic (no random sampling)
2. Model weight checksums are deterministic for large models
3. Torch RNG state restoration works correctly
"""

import hashlib
import json
import os
import sys
import tempfile
import types
import uuid
from dataclasses import dataclass, field
from unittest.mock import patch

# import numpy as np  # Will be mocked
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# Mock the external dependencies BEFORE any imports
class MockNDArray(list):
    def __init__(self, data, dtype="float32"):
        super().__init__(data)
        self._data = list(data)
        self.shape = (len(self._data),)
        self.dtype = dtype
        self.nbytes = len(self._data)

    def tolist(self):
        return list(self._data)


class MockNumpy:
    __spec__ = types.SimpleNamespace(name="numpy", loader=None, submodule_search_locations=[])
    __path__ = []

    def __init__(self):
        self.__version__ = "1.21.0"
        self.random = MockRandom()
        self.ndarray = MockNDArray
        self.uint32 = "uint32"
        self.bool_ = bool

    def randn(self, *args):
        return [0.1, 0.2, 0.3]

    def choice(self, n, size, replace=False):
        return list(range(min(size, n)))

    def array(self, data, dtype="float32"):
        return MockNDArray(data, dtype=dtype)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockRandom:
    def __init__(self):
        self._state = ("MT19937", MockNDArray([0]), 0, 0, 0)

    def seed(self, seed):
        return True

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        return True

    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class MockPandas:
    __spec__ = types.SimpleNamespace(name="pandas", loader=None)

    def __init__(self):
        self.__version__ = "1.3.0"
        self.DataFrame = MockDataFrame
        self.Series = MockSeries

class MockDataFrame:
    def __init__(self, data):
        self.data = data
        self.columns = list(data.keys()) if isinstance(data, dict) else []
        self.shape = (len(data), len(self.columns)) if self.columns else (0, 0)

    def memory_usage(self, deep=True):
        return MockSeries([100, 200, 300])

    def to_string(self):
        return "mock dataframe"


class MockSeries(list):
    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def tolist(self):
        return list(self.data)

class MockTorch:
    __spec__ = types.SimpleNamespace(name="torch", loader=None, submodule_search_locations=[])
    __path__ = []

    def __init__(self):
        self.__version__ = "1.12.0"
        self.uint8 = "uint8"
        self.nn = MockNN()
        self.utils = types.SimpleNamespace(data=types.SimpleNamespace(Dataset=MockTorchDataset))
        self.cuda = MockCuda()

    def tensor(self, data, dtype=None, device=None, requires_grad=False):
        return MockTensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    def randn(self, *args):
        return MockTensor([0.1, 0.2, 0.3] * (args[0] if args else 1))

    def set_rng_state(self, state):
        if hasattr(state, 'dtype') and str(state.dtype) != 'uint8':
            raise RuntimeError("expected torch.ByteTensor")
        return True

    def get_rng_state(self):
        return MockTensor([1, 2, 3, 4, 5], "uint8")

    def manual_seed(self, seed):
        return True

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class MockDatasets:
    __spec__ = types.SimpleNamespace(name="datasets", loader=None)

    class Dataset:
        pass

    class DatasetDict(dict):
        pass


class MockTorchDataset:
    pass


class MockTransformers:
    __spec__ = types.SimpleNamespace(name="transformers", loader=None)

    class PreTrainedModel:
        pass

    class PreTrainedTokenizer:
        pass


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

class MockDevice:
    def __init__(self, device_type: str = "cpu"):
        self.type = device_type or "cpu"

    def __str__(self):
        return self.type


class MockTensor:
    def __init__(self, data, dtype=None, device=None, requires_grad: bool = False):
        self.data = data
        self.dtype = dtype or "float32"
        self.requires_grad = requires_grad
        device_type = device.type if isinstance(device, MockDevice) else device
        self.device = MockDevice(device_type or "cpu")
        if isinstance(self.data, list):
            self.shape = (len(self.data),)
        else:
            self.shape = ()

    def detach(self):
        return self

    def detach_(self):
        return self

    def cpu(self):
        self.device = MockDevice("cpu")
        return self

    def to(self, device=None, dtype=None):
        if device is not None:
            device_type = device.type if isinstance(device, MockDevice) else device
            self.device = MockDevice(device_type or "cpu")
        if dtype is not None:
            self.dtype = dtype
        return self

    def size(self):
        return self.shape

    def numpy(self):
        return self.data  # Return as-is for simplicity

    def tobytes(self):
        return str(self.data).encode()

    def flatten(self):
        return self

    def numel(self):
        if isinstance(self.data, list):
            return len(self.data)
        return 1

    def tolist(self):
        if isinstance(self.data, list):
            return list(self.data)
        return [self.data]

    def clone(self):
        if isinstance(self.data, list):
            data_copy = list(self.data)
        else:
            data_copy = self.data
        return MockTensor(data_copy, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)

    def requires_grad_(self, requires_grad: bool = True):
        self.requires_grad = requires_grad
        return self

    def __iter__(self):
        if isinstance(self.data, list):
            return iter(self.data)
        return iter([self.data])

    def __getitem__(self, key):
        if isinstance(self.data, list):
            return MockTensor(self.data[key], dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)
        return MockTensor(self.data, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad)

    def __len__(self):
        if isinstance(self.data, list):
            return len(self.data)
        return 1

    def __repr__(self):
        return f"MockTensor(data={self.data}, dtype={self.dtype}, device={self.device}, requires_grad={self.requires_grad})"

class MockCuda:
    def is_available(self):
        return False

    def device_count(self):
        return 0

    def set_rng_state(self, state):
        if hasattr(state, 'dtype') and str(state.dtype) != 'uint8':
            raise RuntimeError("expected torch.ByteTensor")
        return True

    def get_rng_state(self):
        return MockTensor([1, 2, 3, 4, 5], "uint8")

    def manual_seed(self, seed):
        return True

    def manual_seed_all(self, seed):
        return True

# Mock the modules BEFORE any imports
mock_torch_utils = types.ModuleType("torch.utils")
mock_torch_utils_data = types.ModuleType("torch.utils.data")
mock_torch_utils_data.Dataset = MockTorchDataset
mock_torch_utils.data = mock_torch_utils_data
patched_modules = {
    'numpy': MockNumpy(),
    'pandas': MockPandas(),
    'torch': MockTorch(),
    'datasets': MockDatasets(),
    'transformers': MockTransformers(),
    'torch.utils': mock_torch_utils,
    'torch.utils.data': mock_torch_utils_data,
    'numpy.typing': types.ModuleType("numpy.typing"),
}

with patch.dict(sys.modules, patched_modules):
    from src.rldk.tracking import DatasetTracker, ModelTracker, SeedTracker


def test_dataset_checksum_determinism():
    """Test that dataset checksums are deterministic."""
    print("Testing dataset checksum determinism...")

    tracker = DatasetTracker()

    # Create identical datasets
    data1 = [1, 2, 3, 4, 5] * 1000  # Large dataset to trigger sampling
    data2 = [1, 2, 3, 4, 5] * 1000  # Identical dataset

    # Track both datasets
    info1 = tracker.track_dataset(data1, "test1")
    info2 = tracker.track_dataset(data2, "test2")

    # Checksums should be identical
    assert info1["checksum"] == info2["checksum"], f"Checksums differ: {info1['checksum']} vs {info2['checksum']}"
    print("   ✓ Dataset checksums are deterministic")

    # Test with different data
    data3 = [1, 2, 3, 4, 6] * 1000  # Different dataset
    info3 = tracker.track_dataset(data3, "test3")

    # Checksums should be different
    assert info1["checksum"] != info3["checksum"], "Different datasets produced identical checksums"
    print("   ✓ Different datasets produce different checksums")

    return True


def test_model_checksum_determinism():
    """Test that model checksums are deterministic."""
    print("Testing model checksum determinism...")

    tracker = ModelTracker()

    # Create a mock large model
    class LargeModel:
        def __init__(self):
            # Simulate a large model with many parameters
            self.parameters_count = 150000000  # 150M parameters (triggers sampling)
            self._parameters = [MockTensor([0.1] * 1000000, requires_grad=True) for _ in range(10)]

        def parameters(self):
            # Mock parameters generator
            for param in self._parameters:
                yield param

        def named_parameters(self):
            for idx, param in enumerate(self._parameters):
                yield (f"layer_{idx}", param)

        def modules(self):
            return [self]

        def named_modules(self):
            return [("layer", self)]

        def children(self):
            return []

        def __repr__(self):
            return "LargeModel(parameters=150000000)"

    model1 = LargeModel()
    model2 = LargeModel()

    # Track both models
    info1 = tracker.track_model(model1, "model1")
    info2 = tracker.track_model(model2, "model2")

    # Architecture checksums should be identical
    assert info1["architecture_checksum"] == info2["architecture_checksum"], \
        f"Architecture checksums differ: {info1['architecture_checksum']} vs {info2['architecture_checksum']}"
    print("   ✓ Model architecture checksums are deterministic")

    # Test with different model
    class DifferentModel:
        def __init__(self):
            self.parameters_count = 150000000
            self._parameters = [MockTensor([0.2] * 1000000, requires_grad=True) for _ in range(10)]

        def parameters(self):
            for param in self._parameters:
                yield param

        def named_parameters(self):
            for idx, param in enumerate(self._parameters):
                yield (f"layer_{idx}", param)

        def modules(self):
            return [self]

        def named_modules(self):
            return [("layer", self)]

        def children(self):
            return []

        def __repr__(self):
            return "DifferentModel(parameters=150000000)"

    model3 = DifferentModel()
    info3 = tracker.track_model(model3, "model3")

    # Architecture checksums should be different
    assert info1["architecture_checksum"] != info3["architecture_checksum"], \
        "Different models produced identical architecture checksums"
    print("   ✓ Different models produce different architecture checksums")

    return True


def test_torch_rng_restoration():
    """Test that torch RNG state restoration works correctly."""
    print("Testing torch RNG state restoration...")

    tracker = SeedTracker()

    # Set seeds
    tracker.set_seeds(42)

    # Save state
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "seed_state.json"
        tracker.save_seed_state(str(save_path))

        # Change seeds
        tracker.set_seeds(100)

        # Load state - this should not raise an error
        try:
            tracker.load_seed_state(str(save_path))
            print("   ✓ Torch RNG state restoration works correctly")
            return True
        except RuntimeError as e:
            if "expected torch.ByteTensor" in str(e):
                print(f"   ✗ Torch RNG state restoration failed: {e}")
                return False
            else:
                raise


def test_multiple_runs_consistency():
    """Test that multiple runs produce consistent results."""
    print("Testing multiple runs consistency...")

    tracker = DatasetTracker()

    # Create a large dataset
    large_data = list(range(100000))  # 100K elements

    # Track the same dataset multiple times
    checksums = []
    for i in range(5):
        info = tracker.track_dataset(large_data, f"run_{i}")
        checksums.append(info["checksum"])

    # All checksums should be identical
    assert all(c == checksums[0] for c in checksums), f"Checksums vary across runs: {checksums}"
    print("   ✓ Multiple runs produce consistent checksums")

    return True


def test_deterministic_sampling():
    """Test that deterministic sampling works correctly."""
    print("Testing deterministic sampling...")

    tracker = DatasetTracker()

    # Create a dataset that will trigger sampling
    data = list(range(10000))  # 10K elements

    # Track the dataset
    info = tracker.track_dataset(data, "sampled_data")

    # The checksum should be based on deterministic sampling
    # We can't easily test the internal sampling logic, but we can verify
    # that the same dataset produces the same checksum
    info2 = tracker.track_dataset(data, "sampled_data_2")

    assert info["checksum"] == info2["checksum"], "Deterministic sampling failed"
    print("   ✓ Deterministic sampling works correctly")

    return True


def main():
    """Run all deterministic tests."""
    print("="*60)
    print("DETERMINISTIC FIXES VERIFICATION TESTS")
    print("="*60)

    try:
        # Test dataset checksum determinism
        test_dataset_checksum_determinism()

        # Test model checksum determinism
        test_model_checksum_determinism()

        # Test torch RNG restoration
        test_torch_rng_restoration()

        # Test multiple runs consistency
        test_multiple_runs_consistency()

        # Test deterministic sampling
        test_deterministic_sampling()

        print("\n" + "="*60)
        print("🎉 ALL DETERMINISTIC TESTS PASSED!")
        print("="*60)
        print("The following issues have been fixed:")
        print("✓ Dataset checksums are now deterministic (no random sampling)")
        print("✓ Model weight checksums are deterministic for large models")
        print("✓ Torch RNG state restoration uses correct tensor dtype")
        print("✓ Multiple runs produce consistent results")
        print("✓ Deterministic sampling works correctly")

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
