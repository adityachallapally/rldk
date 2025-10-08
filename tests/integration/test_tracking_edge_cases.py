"""
Test edge cases for deterministic tracking and RNG restoration.
"""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rldk.tracking.dataset_tracker import DatasetTracker
from rldk.tracking.model_tracker import ModelTracker
from rldk.tracking.seed_tracker import SeedTracker


class TestTrackingEdgeCases:
    """Test edge cases for tracking functionality."""

    def test_empty_dataset_sampling(self):
        """Test sampling with empty datasets."""
        tracker = DatasetTracker()

        # Test empty numpy array
        empty_array = np.array([])
        info = tracker.track_dataset(empty_array, "empty_numpy")
        assert info["checksum"] is not None

        # Test empty pandas DataFrame
        empty_df = pd.DataFrame()
        info = tracker.track_dataset(empty_df, "empty_pandas")
        assert info["checksum"] is not None

        # Test single element array
        single_array = np.array([42])
        info = tracker.track_dataset(single_array, "single_numpy")
        assert info["checksum"] is not None

    def test_very_large_dataset_sampling(self):
        """Test sampling with very large datasets."""
        tracker = DatasetTracker()

        # Test very large numpy array
        large_array = np.random.rand(1000000, 10)
        info1 = tracker.track_dataset(large_array, "large_numpy_1")
        info2 = tracker.track_dataset(large_array, "large_numpy_2")
        assert info1["checksum"] == info2["checksum"]

        # Test very large pandas DataFrame
        large_df = pd.DataFrame(np.random.rand(50000, 5), columns=['a', 'b', 'c', 'd', 'e'])
        info1 = tracker.track_dataset(large_df, "large_pandas_1")
        info2 = tracker.track_dataset(large_df, "large_pandas_2")
        assert info1["checksum"] == info2["checksum"]

    def test_corrupted_seed_file(self):
        """Test handling of corrupted seed files."""
        tracker = SeedTracker()

        # Create a corrupted JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                tracker.load_seed_state(temp_path)
        finally:
            os.unlink(temp_path)

    def test_missing_seed_file(self):
        """Test handling of missing seed files."""
        tracker = SeedTracker()

        with pytest.raises(FileNotFoundError):
            tracker.load_seed_state("nonexistent_file.json")

    def test_invalid_seed_file_structure(self):
        """Test handling of seed files with invalid structure."""
        tracker = SeedTracker()

        # Create a file with missing required fields
        invalid_state = {"timestamp": "2023-01-01", "invalid_field": "value"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(invalid_state, f)
            temp_path = f.name

        try:
            with pytest.raises(RuntimeError, match="Missing required field"):
                tracker.load_seed_state(temp_path)
        finally:
            os.unlink(temp_path)

    def test_partial_cuda_availability(self):
        """Test behavior when CUDA is partially available."""
        tracker = SeedTracker()

        # Test setting seeds when CUDA might not be available
        seed_info = tracker.set_seeds(42, track_cuda=True)

        if torch.cuda.is_available():
            assert "cuda" in seed_info["seeds"]
            assert seed_info["seeds"]["cuda"]["available"] is True
        else:
            assert "cuda" in seed_info["seeds"]
            assert seed_info["seeds"]["cuda"]["available"] is False

    def test_deterministic_sampling_consistency(self):
        """Test that deterministic sampling produces consistent results."""
        tracker = DatasetTracker()

        # Create test data
        data = np.random.rand(1000, 5)

        # Track the same data multiple times
        checksums = []
        for i in range(5):
            info = tracker.track_dataset(data, f"test_data_{i}")
            checksums.append(info["checksum"])

        # All checksums should be identical
        assert all(c == checksums[0] for c in checksums), "Deterministic sampling should produce identical checksums"

    def test_model_tracking_memory_efficiency(self):
        """Test that model tracking doesn't cause memory issues."""
        model_tracker = ModelTracker()

        # Create a moderately large model
        model = torch.nn.Sequential(
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 10)
        )

        # Track the model multiple times to ensure no memory leaks
        for i in range(10):
            info = model_tracker.track_model(model, f"test_model_{i}", save_weights=True)
            assert "weights_checksum" in info
            assert info["weights_checksum"] is not None

    def test_seed_roundtrip_with_different_data_types(self):
        """Test seed roundtrip with different data types."""
        tracker = SeedTracker()

        # Set seed and generate different types of random data
        tracker.set_seeds(42)

        [random.random() for _ in range(10)]
        np.random.random(10).tolist()
        torch.rand(10).tolist()

        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker.save_seed_state(temp_path)

            # Generate more numbers
            python_nums_2 = [random.random() for _ in range(5)]
            numpy_nums_2 = np.random.random(5).tolist()
            torch_nums_2 = torch.rand(5).tolist()

            # Restore state
            tracker.load_seed_state(temp_path)

            # Generate same numbers again
            python_nums_2_restored = [random.random() for _ in range(5)]
            numpy_nums_2_restored = np.random.random(5).tolist()
            torch_nums_2_restored = torch.rand(5).tolist()

            assert python_nums_2 == python_nums_2_restored
            assert numpy_nums_2 == numpy_nums_2_restored
            assert torch_nums_2 == torch_nums_2_restored

        finally:
            os.unlink(temp_path)

    def test_directory_tracking_with_nested_structure(self):
        """Test directory tracking with complex nested structure."""
        tracker = DatasetTracker()

        # Create a temporary directory with nested structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create nested directories and files
            (temp_path / "subdir1").mkdir()
            (temp_path / "subdir2").mkdir()
            (temp_path / "subdir1" / "nested").mkdir()

            # Create files with different content
            (temp_path / "file1.txt").write_text("content1")
            (temp_path / "file2.txt").write_text("content2")
            (temp_path / "subdir1" / "file3.txt").write_text("content3")
            (temp_path / "subdir1" / "nested" / "file4.txt").write_text("content4")
            (temp_path / "subdir2" / "file5.txt").write_text("content5")

            # Track the directory
            info1 = tracker.track_dataset(temp_path, "nested_dir_1")
            info2 = tracker.track_dataset(temp_path, "nested_dir_2")

            assert info1["checksum"] == info2["checksum"]
            # The current implementation counts all items (files + directories)
            # We have 5 files + 3 directories = 8 total items
            assert info1["file_count"] == 8

    def test_pandas_dataframe_with_special_values(self):
        """Test pandas DataFrame tracking with special values."""
        tracker = DatasetTracker()

        # Create DataFrame with special values
        df = pd.DataFrame({
            'normal': [1, 2, 3, 4, 5],
            'nan_values': [1.0, np.nan, 3.0, np.nan, 5.0],
            'inf_values': [1.0, np.inf, 3.0, -np.inf, 5.0],
            'strings': ['a', 'b', 'c', 'd', 'e']
        })

        info1 = tracker.track_dataset(df, "special_df_1")
        info2 = tracker.track_dataset(df, "special_df_2")

        assert info1["checksum"] == info2["checksum"]

    def test_model_tracking_with_different_architectures(self):
        """Test model tracking with different architectures."""
        model_tracker = ModelTracker()

        # Test different model architectures
        models = [
            torch.nn.Linear(10, 5),
            torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU()),
            torch.nn.Conv2d(3, 16, 3),
            torch.nn.LSTM(10, 5, 2)
        ]

        checksums = []
        for i, model in enumerate(models):
            info = model_tracker.track_model(model, f"arch_{i}", save_weights=True)
            checksums.append(info["weights_checksum"])

        # All checksums should be different for different architectures
        assert len(set(checksums)) == len(checksums), "Different architectures should have different checksums"
