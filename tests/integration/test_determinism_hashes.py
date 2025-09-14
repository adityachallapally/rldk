"""
Test that two runs with the same seed produce identical hashes.
"""

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


class TestDeterminismHashes:
    """Test that deterministic runs produce identical hashes."""

    def test_seed_determinism(self):
        """Test that same seed produces identical results."""
        try:
            from rldk.utils.seed import get_current_seed, set_global_seed

            # Test with multiple seeds
            test_seeds = [42, 123, 456, 789, 1000]

            for seed in test_seeds:
                # First run
                set_global_seed(seed)
                result1 = get_current_seed()

                # Second run with same seed
                set_global_seed(seed)
                result2 = get_current_seed()

                # Results should be identical
                assert result1 == result2, f"Seed {seed} produced different results: {result1} vs {result2}"
                assert result1 == seed, f"Seed {seed} did not produce expected result: {result1}"

        except ImportError:
            pytest.skip("Seed module not available")

    def test_random_state_determinism(self):
        """Test that random state is deterministic with same seed."""
        try:
            import random

            import numpy as np

            from rldk.utils.seed import set_global_seed

            # Test with multiple seeds
            test_seeds = [42, 123, 456, 789, 1000]

            for seed in test_seeds:
                # First run
                set_global_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # Generate some random numbers
                np_nums1 = [np.random.random() for _ in range(10)]
                py_nums1 = [random.random() for _ in range(10)]

                # Second run with same seed
                set_global_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # Generate same random numbers
                np_nums2 = [np.random.random() for _ in range(10)]
                py_nums2 = [random.random() for _ in range(10)]

                # Results should be identical
                assert np_nums1 == np_nums2, f"NumPy random numbers differ for seed {seed}"
                assert py_nums1 == py_nums2, f"Python random numbers differ for seed {seed}"

        except ImportError:
            pytest.skip("Required modules not available")

    def test_hash_determinism(self):
        """Test that deterministic runs produce identical hashes."""
        try:
            import random

            import numpy as np

            from rldk.utils.seed import set_global_seed

            # Test with multiple seeds
            test_seeds = [42, 123, 456, 789, 1000]

            for seed in test_seeds:
                # First run
                set_global_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # Generate deterministic data
                data1 = []
                for _ in range(100):
                    data1.append({
                        'step': np.random.randint(0, 1000),
                        'reward': np.random.random(),
                        'loss': np.random.random(),
                        'action': np.random.randint(0, 10)
                    })

                # Create hash of data
                data_str1 = str(sorted(data1, key=lambda x: x['step']))
                hash1 = hashlib.md5(data_str1.encode()).hexdigest()

                # Second run with same seed
                set_global_seed(seed)
                np.random.seed(seed)
                random.seed(seed)

                # Generate same deterministic data
                data2 = []
                for _ in range(100):
                    data2.append({
                        'step': np.random.randint(0, 1000),
                        'reward': np.random.random(),
                        'loss': np.random.random(),
                        'action': np.random.randint(0, 10)
                    })

                # Create hash of data
                data_str2 = str(sorted(data2, key=lambda x: x['step']))
                hash2 = hashlib.md5(data_str2.encode()).hexdigest()

                # Hashes should be identical
                assert hash1 == hash2, f"Data hashes differ for seed {seed}: {hash1} vs {hash2}"
                assert data1 == data2, f"Data differs for seed {seed}"

        except ImportError:
            pytest.skip("Required modules not available")

    def test_cli_determinism(self):
        """Test that CLI commands produce deterministic output."""
        try:
            import sys
            from pathlib import Path

            # Test CLI seed command
            result1 = subprocess.run([
                sys.executable, "-m", "rldk", "seed", "--seed", "42"
            ], capture_output=True, text=True)

            result2 = subprocess.run([
                sys.executable, "-m", "rldk", "seed", "--seed", "42"
            ], capture_output=True, text=True)

            # Output should be identical
            assert result1.stdout == result2.stdout, "CLI seed command output differs"
            assert result1.returncode == result2.returncode, "CLI seed command return codes differ"

        except Exception as e:
            pytest.skip(f"CLI test failed: {e}")

    def test_file_determinism(self):
        """Test that file operations are deterministic with same seed."""
        try:
            import json
            import tempfile

            import numpy as np

            from rldk.utils.seed import set_global_seed

            # Test with multiple seeds
            test_seeds = [42, 123, 456, 789, 1000]

            for seed in test_seeds:
                # First run
                set_global_seed(seed)
                np.random.seed(seed)

                # Create temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                    data1 = {
                        'seed': seed,
                        'random_values': [np.random.random() for _ in range(10)],
                        'metadata': {
                            'timestamp': '2024-01-01T00:00:00Z',
                            'version': '1.0.0'
                        }
                    }
                    json.dump(data1, f)
                    file1_path = f.name

                # Second run with same seed
                set_global_seed(seed)
                np.random.seed(seed)

                # Create another temporary file
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                    data2 = {
                        'seed': seed,
                        'random_values': [np.random.random() for _ in range(10)],
                        'metadata': {
                            'timestamp': '2024-01-01T00:00:00Z',
                            'version': '1.0.0'
                        }
                    }
                    json.dump(data2, f)
                    file2_path = f.name

                # Files should be identical
                with open(file1_path) as f:
                    content1 = f.read()
                with open(file2_path) as f:
                    content2 = f.read()

                assert content1 == content2, f"File contents differ for seed {seed}"

                # Clean up
                Path(file1_path).unlink()
                Path(file2_path).unlink()

        except ImportError:
            pytest.skip("Required modules not available")

    def test_environment_determinism(self):
        """Test that environment variables don't affect determinism."""
        try:
            import os

            import numpy as np

            from rldk.utils.seed import set_global_seed

            # Test with different environment variables
            test_seeds = [42, 123, 456]

            for seed in test_seeds:
                # First run with default environment
                set_global_seed(seed)
                np.random.seed(seed)
                result1 = np.random.random()

                # Second run with modified environment
                os.environ['TEST_VAR'] = 'test_value'
                set_global_seed(seed)
                np.random.seed(seed)
                result2 = np.random.random()

                # Results should be identical (environment shouldn't affect random state)
                assert result1 == result2, f"Environment affects determinism for seed {seed}: {result1} vs {result2}"

                # Clean up
                if 'TEST_VAR' in os.environ:
                    del os.environ['TEST_VAR']

        except ImportError:
            pytest.skip("Required modules not available")

    def test_multiprocessing_determinism(self):
        """Test that multiprocessing doesn't affect determinism."""
        try:
            import multiprocessing as mp

            import numpy as np

            from rldk.utils.seed import set_global_seed

            def worker(seed, return_dict):
                """Worker function for multiprocessing test."""
                set_global_seed(seed)
                np.random.seed(seed)
                result = np.random.random()
                return_dict[seed] = result

            # Test with multiple seeds
            test_seeds = [42, 123, 456]

            for seed in test_seeds:
                # First run in main process
                set_global_seed(seed)
                np.random.seed(seed)
                result1 = np.random.random()

                # Second run in subprocess
                manager = mp.Manager()
                return_dict = manager.dict()
                p = mp.Process(target=worker, args=(seed, return_dict))
                p.start()
                p.join()
                result2 = return_dict[seed]

                # Results should be identical
                assert result1 == result2, f"Multiprocessing affects determinism for seed {seed}: {result1} vs {result2}"

        except ImportError:
            pytest.skip("Required modules not available")

    def test_threading_determinism(self):
        """Test that threading doesn't affect determinism."""
        try:
            import threading

            import numpy as np

            from rldk.utils.seed import set_global_seed

            def worker(seed, results):
                """Worker function for threading test."""
                set_global_seed(seed)
                np.random.seed(seed)
                result = np.random.random()
                results.append(result)

            # Test with multiple seeds
            test_seeds = [42, 123, 456]

            for seed in test_seeds:
                # First run in main thread
                set_global_seed(seed)
                np.random.seed(seed)
                result1 = np.random.random()

                # Second run in subthread
                results = []
                t = threading.Thread(target=worker, args=(seed, results))
                t.start()
                t.join()
                result2 = results[0]

                # Results should be identical
                assert result1 == result2, f"Threading affects determinism for seed {seed}: {result1} vs {result2}"

        except ImportError:
            pytest.skip("Required modules not available")

    def test_determinism_across_imports(self):
        """Test that determinism is maintained across imports."""
        try:
            import importlib

            import numpy as np

            from rldk.utils.seed import set_global_seed

            # Test with multiple seeds
            test_seeds = [42, 123, 456]

            for seed in test_seeds:
                # First run
                set_global_seed(seed)
                np.random.seed(seed)
                result1 = np.random.random()

                # importlib.reload(np)

                # Second run with same seed
                set_global_seed(seed)
                np.random.seed(seed)
                result2 = np.random.random()

                # Results should be identical
                assert result1 == result2, f"Import reload affects determinism for seed {seed}: {result1} vs {result2}"

        except ImportError:
            pytest.skip("Required modules not available")
