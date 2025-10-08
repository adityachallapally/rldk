#!/usr/bin/env python3
"""
Direct test for seed roundtrip functionality without full module imports.
"""

import os
import random
import sys
import tempfile

import numpy as np
import torch

import _path_setup  # noqa: F401
from rldk.tracking.seed_tracker import SeedTracker


def test_seed_roundtrip():
    """Test complete seed roundtrip functionality."""
    print("Testing seed roundtrip functionality...")

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

        assert python_nums_2 == python_nums_2_restored, "Python random state not properly restored"
        assert numpy_nums_2 == numpy_nums_2_restored, "NumPy random state not properly restored"
        assert torch_nums_2 == torch_nums_2_restored, "PyTorch random state not properly restored"

        print("✓ Seed roundtrip test passed")

        # Test reproducible environment
        tracker2 = SeedTracker()
        seed_info = tracker2.create_reproducible_environment(42)

        # Verify seed info structure
        assert "reproducibility_settings" in seed_info
        assert "seeds" in seed_info

        # Generate numbers with reproducible environment
        python_repro = [random.random() for _ in range(3)]
        numpy_repro = np.random.random(3).tolist()
        torch_repro = torch.rand(3).tolist()

        # Create another reproducible environment
        tracker3 = SeedTracker()
        tracker3.create_reproducible_environment(42)

        # Generate same numbers
        python_repro_2 = [random.random() for _ in range(3)]
        numpy_repro_2 = np.random.random(3).tolist()
        torch_repro_2 = torch.rand(3).tolist()

        assert python_repro == python_repro_2, "Reproducible environment not deterministic for Python"
        assert numpy_repro == numpy_repro_2, "Reproducible environment not deterministic for NumPy"
        assert torch_repro == torch_repro_2, "Reproducible environment not deterministic for PyTorch"

        print("✓ Reproducible environment test passed")

        return True

    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    print("Running direct seed tracker tests...")

    try:
        success = test_seed_roundtrip()
        if success:
            print("\n✅ All seed roundtrip tests passed!")
            sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
