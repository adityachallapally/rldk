"""Tests for determinism harness."""

from unittest.mock import MagicMock, patch

import pandas as pd

from rldk.determinism.check import check


class TestDeterminismHarness:
    """Test determinism harness functionality."""

    def test_identical_runs_pass(self):
        """Test that identical runs pass determinism check."""
        # Create identical data for multiple replicas
        steps = list(range(50))
        base_data = {
            "step": steps,
            "reward_mean": [0.5 + i * 0.01 for i in steps],
            "kl_mean": [0.1 + i * 0.001 for i in steps],
            "entropy_mean": [0.8 - i * 0.002 for i in steps],
            "lr": [0.001] * len(steps),
            "seed": [42] * len(steps),
            "run_id": ["test_run"] * len(steps),
            "git_sha": ["abc123"] * len(steps),
        }

        # Mock subprocess.run to return identical results
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.metrics_df = pd.DataFrame(base_data)

        with patch("subprocess.run", return_value=mock_result), patch(
            "rldk.determinism.check._run_deterministic_cmd", return_value=mock_result
        ):
            report = check(
                cmd="python train.py",
                compare=["reward_mean", "kl_mean", "entropy_mean"],
                steps=[10, 20, 30],
                replicas=3,
                device="cpu",
            )

            assert report.passed
            assert len(report.mismatches) == 0
            assert len(report.fixes) > 0

    def test_nondeterministic_runs_fail(self):
        """Test that nondeterministic runs fail determinism check."""
        # Create slightly different data for each replica
        steps = list(range(50))
        base_data = {
            "step": steps,
            "reward_mean": [0.5 + i * 0.01 for i in steps],
            "kl_mean": [0.1 + i * 0.001 for i in steps],
            "entropy_mean": [0.8 - i * 0.002 for i in steps],
        }

        # Create different results for each replica
        replica_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""

            # Add small random differences to simulate nondeterminism
            modified_data = base_data.copy()
            modified_data["reward_mean"] = [
                v + 0.001 * i for v in modified_data["reward_mean"]
            ]
            modified_data["kl_mean"] = [
                v + 0.0001 * i for v in modified_data["kl_mean"]
            ]

            mock_result.metrics_df = pd.DataFrame(modified_data)
            replica_results.append(mock_result)

        with patch("subprocess.run", side_effect=replica_results), patch(
            "rldk.determinism.check._run_deterministic_cmd", side_effect=replica_results
        ):
            report = check(
                cmd="python train.py",
                compare=["reward_mean", "kl_mean", "entropy_mean"],
                steps=[10, 20, 30],
                replicas=3,
                device="cpu",
            )

            assert not report.passed
            assert len(report.mismatches) > 0
            assert len(report.fixes) > 0

    def test_device_detection(self):
        """Test automatic device detection."""
        # Test with explicit device specification
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
            }
        )

        with patch("subprocess.run", return_value=mock_result), patch(
            "rldk.determinism.check._run_deterministic_cmd", return_value=mock_result
        ):
            # Test CPU device
            report = check(
                cmd="python train.py", compare=["reward_mean"], replicas=1, device="cpu"
            )
            assert report.passed

            # Test CUDA device
            report = check(
                cmd="python train.py",
                compare=["reward_mean"],
                replicas=1,
                device="cuda",
            )
            assert report.passed

    def test_nondeterministic_ops_detection(self):
        """Test detection of non-deterministic operations."""
        # Create mock results with non-deterministic operation warnings
        mock_result1 = MagicMock()
        mock_result1.returncode = 0
        mock_result1.stderr = "cuDNN convolution is non-deterministic"
        mock_result1.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
                "kl_mean": [0.1 + i * 0.001 for i in range(10)],
                "entropy_mean": [0.8 - i * 0.002 for i in range(10)],
            }
        )

        mock_result2 = MagicMock()
        mock_result2.returncode = 0
        mock_result2.stderr = "dropout operation is non-deterministic"
        mock_result2.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
                "kl_mean": [0.1 + i * 0.001 for i in range(10)],
                "entropy_mean": [0.8 - i * 0.002 for i in range(10)],
            }
        )

        with patch("subprocess.run", side_effect=[mock_result1, mock_result2]), patch(
            "rldk.determinism.check._run_deterministic_cmd",
            side_effect=[mock_result1, mock_result2],
        ):
            report = check(
                cmd="python train.py", compare=["reward_mean"], replicas=2, device="cpu"
            )

            # Should detect non-deterministic operations
            assert report.culprit is not None
            assert len(report.fixes) > 0
            assert any("cudnn.deterministic" in fix for fix in report.fixes)

    def test_replica_variance_calculation(self):
        """Test calculation of variance across replicas."""
        # Create mock results with varying data
        mock_results = []
        for i in range(3):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            mock_result.metrics_df = pd.DataFrame(
                {
                    "step": list(range(10)),
                    "reward_mean": [0.5 + i * 0.1 + j * 0.01 for j in range(10)],
                    "kl_mean": [0.1 + i * 0.01 + j * 0.001 for j in range(10)],
                }
            )
            mock_results.append(mock_result)

        with patch("subprocess.run", side_effect=mock_results), patch(
            "rldk.determinism.check._run_deterministic_cmd", side_effect=mock_results
        ):
            report = check(
                cmd="python train.py",
                compare=["reward_mean", "kl_mean"],
                replicas=3,
                device="cpu",
            )

            # Should calculate variance
            assert "reward_mean" in report.replica_variance
            assert "kl_mean" in report.replica_variance
            assert report.replica_variance["reward_mean"] > 0  # Should have variance
            assert report.replica_variance["kl_mean"] > 0

    def test_rng_map_creation(self):
        """Test creation of RNG settings map."""
        # Create mock result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
                "kl_mean": [0.1 + i * 0.001 for i in range(10)],
                "entropy_mean": [0.8 - i * 0.002 for i in range(10)],
            }
        )

        with patch("subprocess.run", return_value=mock_result), patch(
            "rldk.determinism.check._run_deterministic_cmd", return_value=mock_result
        ):
            report = check(
                cmd="python train.py", compare=["reward_mean"], replicas=1, device="cpu"
            )

            # Should create RNG map
            assert "torch_seed" in report.rng_map
            assert "numpy_seed" in report.rng_map
            assert "python_hash_seed" in report.rng_map

    def test_insufficient_replicas(self):
        """Test handling of insufficient replicas."""
        # Create only one replica result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
                "kl_mean": [0.1 + i * 0.001 for i in range(10)],
                "entropy_mean": [0.8 - i * 0.002 for i in range(10)],
            }
        )

        with patch("subprocess.run", return_value=mock_result), patch(
            "rldk.determinism.check._run_deterministic_cmd", return_value=mock_result
        ):
            report = check(
                cmd="python train.py", compare=["reward_mean"], replicas=1, device="cpu"
            )

            # Should still pass with only one replica
            assert report.passed
            assert len(report.mismatches) == 0

    def test_command_timeout(self):
        """Test handling of command timeout."""
        # Create first replica with data
        mock_result1 = MagicMock()
        mock_result1.returncode = 0
        mock_result1.stderr = ""
        mock_result1.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
                "kl_mean": [0.1 + i * 0.001 for i in range(10)],
                "entropy_mean": [0.8 - i * 0.002 for i in range(10)],
            }
        )

        # Create second replica that times out
        mock_result2 = MagicMock()
        mock_result2.returncode = -1
        mock_result2.stderr = "Command timed out"
        mock_result2.metrics_df = pd.DataFrame()  # Empty due to timeout

        with patch("subprocess.run", side_effect=[mock_result1, mock_result2]), patch(
            "rldk.determinism.check._run_deterministic_cmd",
            side_effect=[mock_result1, mock_result2],
        ):
            report = check(
                cmd="python train.py", compare=["reward_mean"], replicas=2, device="cpu"
            )

            # Should fail due to timeout
            assert not report.passed
            assert len(report.mismatches) > 0

    def test_missing_metrics_file(self):
        """Test handling of missing metrics file."""
        # Create first replica with data
        mock_result1 = MagicMock()
        mock_result1.returncode = 0
        mock_result1.stderr = ""
        mock_result1.metrics_df = pd.DataFrame(
            {
                "step": list(range(10)),
                "reward_mean": [0.5 + i * 0.01 for i in range(10)],
                "kl_mean": [0.1 + i * 0.001 for i in range(10)],
                "entropy_mean": [0.8 - i * 0.002 for i in range(10)],
            }
        )

        # Create second replica with empty data
        mock_result2 = MagicMock()
        mock_result2.returncode = 0
        mock_result2.stderr = ""
        mock_result2.metrics_df = pd.DataFrame()  # Empty dataframe

        with patch("subprocess.run", side_effect=[mock_result1, mock_result2]), patch(
            "rldk.determinism.check._run_deterministic_cmd",
            side_effect=[mock_result1, mock_result2],
        ):
            report = check(
                cmd="python train.py", compare=["reward_mean"], replicas=2, device="cpu"
            )

            # Should fail due to missing metrics
            assert not report.passed
            assert len(report.mismatches) > 0

    def test_step_specific_comparison(self):
        """Test comparison at specific steps."""
        steps = list(range(50))
        base_data = {
            "step": steps,
            "reward_mean": [0.5 + i * 0.01 for i in steps],
            "kl_mean": [0.1 + i * 0.001 for i in steps],
            "entropy_mean": [0.8 - i * 0.002 for i in steps],
        }

        # Create identical results
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_result.metrics_df = pd.DataFrame(base_data)

        with patch("subprocess.run", return_value=mock_result), patch(
            "rldk.determinism.check._run_deterministic_cmd", return_value=mock_result
        ):
            report = check(
                cmd="python train.py",
                compare=["reward_mean", "kl_mean"],
                steps=[10, 25, 40],  # Specific steps
                replicas=3,
                device="cpu",
            )

            assert report.passed
            assert len(report.mismatches) == 0
