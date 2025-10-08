"""
Test seed roundtrip functionality to ensure proper RNG state restoration.
"""

import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from rldk.tracking.seed_tracker import SeedTracker


class TestSeedRoundtrip:
    """Test seed roundtrip functionality."""

    def test_python_random_roundtrip(self):
        """Test Python random state roundtrip."""
        tracker = SeedTracker()

        # Set a seed and generate some random numbers
        tracker.set_seeds(42)
        [random.random() for _ in range(10)]

        # Save state to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker.save_seed_state(temp_path)

            # Generate more numbers
            random_numbers_2 = [random.random() for _ in range(5)]

            # Load state and verify next numbers match
            tracker.load_seed_state(temp_path)
            random_numbers_2_restored = [random.random() for _ in range(5)]

            assert random_numbers_2 == random_numbers_2_restored, "Python random state not properly restored"
        finally:
            os.unlink(temp_path)

    def test_numpy_random_roundtrip(self):
        """Test NumPy random state roundtrip."""
        tracker = SeedTracker()

        # Set a seed and generate some random numbers
        tracker.set_seeds(42)
        np.random.random(10).tolist()

        # Save state to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker.save_seed_state(temp_path)

            # Generate more numbers
            random_numbers_2 = np.random.random(5).tolist()

            # Load state and verify next numbers match
            tracker.load_seed_state(temp_path)
            random_numbers_2_restored = np.random.random(5).tolist()

            assert random_numbers_2 == random_numbers_2_restored, "NumPy random state not properly restored"
        finally:
            os.unlink(temp_path)

    def test_torch_random_roundtrip(self):
        """Test PyTorch random state roundtrip."""
        tracker = SeedTracker()

        # Set a seed and generate some random numbers
        tracker.set_seeds(42)
        torch.rand(10).tolist()

        # Save state to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker.save_seed_state(temp_path)

            # Generate more numbers
            random_tensor_2 = torch.rand(5).tolist()

            # Load state and verify next numbers match
            tracker.load_seed_state(temp_path)
            random_tensor_2_restored = torch.rand(5).tolist()

            assert random_tensor_2 == random_tensor_2_restored, "PyTorch random state not properly restored"
        finally:
            os.unlink(temp_path)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_random_roundtrip(self):
        """Test CUDA random state roundtrip."""
        tracker = SeedTracker()

        # Set a seed and generate some random numbers
        tracker.set_seeds(42)
        torch.cuda.rand(10).cpu().tolist()

        # Save state to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker.save_seed_state(temp_path)

            # Generate more numbers
            random_tensor_2 = torch.cuda.rand(5).cpu().tolist()

            # Load state and verify next numbers match
            tracker.load_seed_state(temp_path)
            random_tensor_2_restored = torch.cuda.rand(5).cpu().tolist()

            assert random_tensor_2 == random_tensor_2_restored, "CUDA random state not properly restored"
        finally:
            os.unlink(temp_path)

    def test_all_rngs_roundtrip(self):
        """Test all RNG states together."""
        tracker = SeedTracker()

        # Set a seed and generate some random numbers
        tracker.set_seeds(42)
        [random.random() for _ in range(5)]
        np.random.random(5).tolist()
        torch.rand(5).tolist()

        # Save state to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker.save_seed_state(temp_path)

            # Generate more numbers
            python_nums_2 = [random.random() for _ in range(3)]
            numpy_nums_2 = np.random.random(3).tolist()
            torch_nums_2 = torch.rand(3).tolist()

            # Load state and verify next numbers match
            tracker.load_seed_state(temp_path)
            python_nums_2_restored = [random.random() for _ in range(3)]
            numpy_nums_2_restored = np.random.random(3).tolist()
            torch_nums_2_restored = torch.rand(3).tolist()

            assert python_nums_2 == python_nums_2_restored, "Python random state not properly restored in combined test"
            assert numpy_nums_2 == numpy_nums_2_restored, "NumPy random state not properly restored in combined test"
            assert torch_nums_2 == torch_nums_2_restored, "PyTorch random state not properly restored in combined test"
        finally:
            os.unlink(temp_path)

    def test_fresh_instance_restoration(self):
        """Test restoration into a fresh SeedTracker instance."""
        tracker1 = SeedTracker()

        # Set a seed and generate some random numbers
        tracker1.set_seeds(42)
        [random.random() for _ in range(5)]
        np.random.random(5).tolist()
        torch.rand(5).tolist()

        # Save state to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            tracker1.save_seed_state(temp_path)

            # Generate more numbers
            python_nums_2 = [random.random() for _ in range(3)]
            numpy_nums_2 = np.random.random(3).tolist()
            torch_nums_2 = torch.rand(3).tolist()

            # Create fresh instance and load state
            tracker2 = SeedTracker()
            tracker2.load_seed_state(temp_path)

            # Verify next numbers match
            python_nums_2_restored = [random.random() for _ in range(3)]
            numpy_nums_2_restored = np.random.random(3).tolist()
            torch_nums_2_restored = torch.rand(3).tolist()

            assert python_nums_2 == python_nums_2_restored, "Python random state not properly restored in fresh instance"
            assert numpy_nums_2 == numpy_nums_2_restored, "NumPy random state not properly restored in fresh instance"
            assert torch_nums_2 == torch_nums_2_restored, "PyTorch random state not properly restored in fresh instance"
        finally:
            os.unlink(temp_path)

    def test_create_reproducible_environment(self):
        """Test create_reproducible_environment helper."""
        tracker = SeedTracker()

        # Create reproducible environment
        seed_info = tracker.create_reproducible_environment(42)

        # Verify seed info structure
        assert "reproducibility_settings" in seed_info
        assert "seeds" in seed_info

        # Generate some random numbers
        python_nums = [random.random() for _ in range(5)]
        numpy_nums = np.random.random(5).tolist()
        torch_nums = torch.rand(5).tolist()

        # Create another reproducible environment with same seed
        tracker2 = SeedTracker()
        tracker2.create_reproducible_environment(42)

        # Generate same numbers
        python_nums_2 = [random.random() for _ in range(5)]
        numpy_nums_2 = np.random.random(5).tolist()
        torch_nums_2 = torch.rand(5).tolist()

        assert python_nums == python_nums_2, "create_reproducible_environment not deterministic for Python random"
        assert numpy_nums == numpy_nums_2, "create_reproducible_environment not deterministic for NumPy random"
        assert torch_nums == torch_nums_2, "create_reproducible_environment not deterministic for PyTorch random"

    def test_deterministic_settings(self):
        """Test that deterministic settings are properly configured."""
        tracker = SeedTracker()

        # Create reproducible environment
        seed_info = tracker.create_reproducible_environment(42)

        # Check PyTorch deterministic settings
        assert torch.are_deterministic_algorithms_enabled(), "PyTorch deterministic algorithms not enabled"

        if torch.cuda.is_available():
            assert torch.backends.cudnn.deterministic, "cuDNN deterministic mode not enabled"
            assert not torch.backends.cudnn.benchmark, "cuDNN benchmark mode not disabled"

        # Check environment variable
        assert os.environ.get('PYTHONHASHSEED') == '42', "PYTHONHASHSEED not set correctly"

        # Verify settings in seed_info
        settings = seed_info["reproducibility_settings"]
        assert settings["PYTHONHASHSEED"] == '42'
        assert settings["torch_deterministic"] is True
