#!/usr/bin/env python3
"""
Simple test for seed roundtrip functionality.
"""

import random
import tempfile

import numpy as np
import torch

# Direct import to avoid full module dependencies
import _path_setup  # noqa: F401
from rldk.tracking.seed_tracker import SeedTracker


def test_python_random_roundtrip():
    """Test Python random state roundtrip."""
    print("Testing Python random roundtrip...")
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
        print("✓ Python random roundtrip passed")
    finally:
        os.unlink(temp_path)


def test_numpy_random_roundtrip():
    """Test NumPy random state roundtrip."""
    print("Testing NumPy random roundtrip...")
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
        print("✓ NumPy random roundtrip passed")
    finally:
        os.unlink(temp_path)


def test_torch_random_roundtrip():
    """Test PyTorch random state roundtrip."""
    print("Testing PyTorch random roundtrip...")
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
        print("✓ PyTorch random roundtrip passed")
    finally:
        os.unlink(temp_path)


def test_create_reproducible_environment():
    """Test create_reproducible_environment helper."""
    print("Testing create_reproducible_environment...")
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
    print("✓ create_reproducible_environment passed")


if __name__ == "__main__":
    print("Running seed roundtrip tests...")

    try:
        test_python_random_roundtrip()
        test_numpy_random_roundtrip()
        test_torch_random_roundtrip()
        test_create_reproducible_environment()
        print("\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
