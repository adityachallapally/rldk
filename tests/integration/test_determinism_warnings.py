"""Tests for determinism dependency warnings."""

import builtins
import os
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rldk.determinism.check import DeterminismReport, check


class TestDeterminismWarnings:
    """Test determinism warning functionality."""

    def test_pytorch_cuda_kernels_warning_when_missing(self):
        """Test that PyTorch CUDA kernels warning is shown when torch is missing."""
        with patch.dict(os.environ, {"RLDK_SILENCE_DETERMINISM_WARN": "0"}):
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Mock the rest of the check function to avoid actual execution
                    with patch("rldk.determinism.check._detect_device", return_value="cpu"), \
                         patch("rldk.determinism.check._get_deterministic_env", return_value={}), \
                         patch("rldk.determinism.check._run_deterministic_cmd") as mock_run, \
                         patch("rldk.determinism.check._compare_replicas", return_value=[]), \
                         patch("rldk.determinism.check._parse_nondeterministic_ops", return_value=(None, [])), \
                         patch("rldk.determinism.check._calculate_replica_variance", return_value={}), \
                         patch("rldk.determinism.check._create_rng_map", return_value={}), \
                         patch("rldk.determinism.check._detect_dataloader_issues", return_value=[]):

                        mock_result = MagicMock()
                        mock_result.metrics_df = pd.DataFrame()
                        mock_run.return_value = mock_result

                        report = check(
                            cmd="python train.py",
                            compare=["reward_mean"],
                            replicas=2,
                            device="cpu"
                        )

                        # Check that warning was issued
                        warning_messages = [str(warning.message) for warning in w]
                        assert any(
                            "Skipped PyTorch CUDA kernels check" in msg for msg in warning_messages
                        )
                        assert any("Install torch>=2.0.0 to enable" in msg for msg in warning_messages)

                        # Check that skipped_checks contains the right value
                        assert "pytorch_cuda_kernels" in report.skipped_checks

    def test_tensorflow_warning_when_missing(self):
        """Test that TensorFlow warning is shown when tensorflow is missing."""
        with patch.dict(os.environ, {"RLDK_SILENCE_DETERMINISM_WARN": "0"}):
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "tensorflow":
                    raise ImportError("No module named 'tensorflow'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Mock the rest of the check function
                    with patch("rldk.determinism.check._detect_device", return_value="cpu"), \
                         patch("rldk.determinism.check._get_deterministic_env", return_value={}), \
                         patch("rldk.determinism.check._run_deterministic_cmd") as mock_run, \
                         patch("rldk.determinism.check._compare_replicas", return_value=[]), \
                         patch("rldk.determinism.check._parse_nondeterministic_ops", return_value=(None, [])), \
                         patch("rldk.determinism.check._calculate_replica_variance", return_value={}), \
                         patch("rldk.determinism.check._create_rng_map", return_value={}), \
                         patch("rldk.determinism.check._detect_dataloader_issues", return_value=[]):

                        mock_result = MagicMock()
                        mock_result.metrics_df = pd.DataFrame()
                        mock_run.return_value = mock_result

                        report = check(
                            cmd="python train.py",
                            compare=["reward_mean"],
                            replicas=2,
                            device="cpu"
                        )

                        # Check that warning was issued
                        warning_messages = [str(warning.message) for warning in w]
                        assert any(
                            "Skipped TensorFlow determinism check" in msg for msg in warning_messages
                        )
                        assert any("Install tensorflow>=2.8.0 to enable" in msg for msg in warning_messages)

                        # Check that skipped_checks contains the right value
                        assert "tensorflow_determinism" in report.skipped_checks

    def test_jax_warning_when_missing(self):
        """Test that JAX warning is shown when jax is missing."""
        with patch.dict(os.environ, {"RLDK_SILENCE_DETERMINISM_WARN": "0"}):
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "jax":
                    raise ImportError("No module named 'jax'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Mock the rest of the check function
                    with patch("rldk.determinism.check._detect_device", return_value="cpu"), \
                         patch("rldk.determinism.check._get_deterministic_env", return_value={}), \
                         patch("rldk.determinism.check._run_deterministic_cmd") as mock_run, \
                         patch("rldk.determinism.check._compare_replicas", return_value=[]), \
                         patch("rldk.determinism.check._parse_nondeterministic_ops", return_value=(None, [])), \
                         patch("rldk.determinism.check._calculate_replica_variance", return_value={}), \
                         patch("rldk.determinism.check._create_rng_map", return_value={}), \
                         patch("rldk.determinism.check._detect_dataloader_issues", return_value=[]):

                        mock_result = MagicMock()
                        mock_result.metrics_df = pd.DataFrame()
                        mock_run.return_value = mock_result

                        report = check(
                            cmd="python train.py",
                            compare=["reward_mean"],
                            replicas=2,
                            device="cpu"
                        )

                        # Check that warning was issued
                        warning_messages = [str(warning.message) for warning in w]
                        assert any(
                            "Skipped JAX determinism check" in msg for msg in warning_messages
                        )
                        assert any("Install jax>=0.4.0 to enable" in msg for msg in warning_messages)

                        # Check that skipped_checks contains the right value
                        assert "jax_determinism" in report.skipped_checks

    def test_silence_flag_prevents_warnings(self):
        """Test that RLDK_SILENCE_DETERMINISM_WARN=1 prevents warnings."""
        with patch.dict(os.environ, {"RLDK_SILENCE_DETERMINISM_WARN": "1"}):
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Mock the rest of the check function
                    with patch("rldk.determinism.check._detect_device", return_value="cpu"), \
                         patch("rldk.determinism.check._get_deterministic_env", return_value={}), \
                         patch("rldk.determinism.check._run_deterministic_cmd") as mock_run, \
                         patch("rldk.determinism.check._compare_replicas", return_value=[]), \
                         patch("rldk.determinism.check._parse_nondeterministic_ops", return_value=(None, [])), \
                         patch("rldk.determinism.check._calculate_replica_variance", return_value={}), \
                         patch("rldk.determinism.check._create_rng_map", return_value={}), \
                         patch("rldk.determinism.check._detect_dataloader_issues", return_value=[]):

                        mock_result = MagicMock()
                        mock_result.metrics_df = pd.DataFrame()
                        mock_run.return_value = mock_result

                        report = check(
                            cmd="python train.py",
                            compare=["reward_mean"],
                            replicas=2,
                            device="cpu"
                        )

                        # Check that no warnings were issued
                        assert len(w) == 0

                        # Check that skipped_checks still contains the right value
                        assert "pytorch_cuda_kernels" in report.skipped_checks

    def test_multiple_missing_dependencies(self):
        """Test that multiple missing dependencies generate multiple warnings."""
        with patch.dict(os.environ, {"RLDK_SILENCE_DETERMINISM_WARN": "0"}):
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                elif name == "tensorflow":
                    raise ImportError("No module named 'tensorflow'")
                elif name == "jax":
                    raise ImportError("No module named 'jax'")
                else:
                    return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Mock the rest of the check function
                    with patch("rldk.determinism.check._detect_device", return_value="cpu"), \
                         patch("rldk.determinism.check._get_deterministic_env", return_value={}), \
                         patch("rldk.determinism.check._run_deterministic_cmd") as mock_run, \
                         patch("rldk.determinism.check._compare_replicas", return_value=[]), \
                         patch("rldk.determinism.check._parse_nondeterministic_ops", return_value=(None, [])), \
                         patch("rldk.determinism.check._calculate_replica_variance", return_value={}), \
                         patch("rldk.determinism.check._create_rng_map", return_value={}), \
                         patch("rldk.determinism.check._detect_dataloader_issues", return_value=[]):

                        mock_result = MagicMock()
                        mock_result.metrics_df = pd.DataFrame()
                        mock_run.return_value = mock_result

                        report = check(
                            cmd="python train.py",
                            compare=["reward_mean"],
                            replicas=2,
                            device="cpu"
                        )

                        # Check that three warnings were issued
                        assert len(w) == 3

                        warning_messages = [str(warning.message) for warning in w]
                        assert any("PyTorch CUDA kernels check" in msg for msg in warning_messages)
                        assert any("TensorFlow determinism check" in msg for msg in warning_messages)
                        assert any("JAX determinism check" in msg for msg in warning_messages)

                        # Check that all skipped checks are recorded
                        assert "pytorch_cuda_kernels" in report.skipped_checks
                        assert "tensorflow_determinism" in report.skipped_checks
                        assert "jax_determinism" in report.skipped_checks

    def test_no_warnings_when_dependencies_available(self):
        """Test that no warnings are issued when all dependencies are available."""
        with patch.dict(os.environ, {"RLDK_SILENCE_DETERMINISM_WARN": "0"}):
            # Mock successful imports
            with patch("rldk.determinism.check._check_pytorch_cuda_kernels", return_value=True), \
                 patch("rldk.determinism.check._check_tensorflow_determinism", return_value=True), \
                 patch("rldk.determinism.check._check_jax_determinism", return_value=True):

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # Mock the rest of the check function
                    with patch("rldk.determinism.check._detect_device", return_value="cpu"), \
                         patch("rldk.determinism.check._get_deterministic_env", return_value={}), \
                         patch("rldk.determinism.check._run_deterministic_cmd") as mock_run, \
                         patch("rldk.determinism.check._compare_replicas", return_value=[]), \
                         patch("rldk.determinism.check._parse_nondeterministic_ops", return_value=(None, [])), \
                         patch("rldk.determinism.check._calculate_replica_variance", return_value={}), \
                         patch("rldk.determinism.check._create_rng_map", return_value={}), \
                         patch("rldk.determinism.check._detect_dataloader_issues", return_value=[]):

                        mock_result = MagicMock()
                        mock_result.metrics_df = pd.DataFrame()
                        mock_run.return_value = mock_result

                        report = check(
                            cmd="python train.py",
                            compare=["reward_mean"],
                            replicas=2,
                            device="cpu"
                        )

                        # Check that no warnings were issued
                        assert len(w) == 0

                        # Check that no checks were skipped
                        assert len(report.skipped_checks) == 0

    def test_determinism_report_includes_skipped_checks(self):
        """Test that DeterminismReport includes skipped_checks field."""
        report = DeterminismReport(
            passed=True,
            culprit=None,
            fixes=[],
            replica_variance={},
            rng_map={},
            mismatches=[],
            dataloader_notes=[],
            skipped_checks=["pytorch_cuda_kernels", "tensorflow_determinism"]
        )

        assert hasattr(report, "skipped_checks")
        assert report.skipped_checks == ["pytorch_cuda_kernels", "tensorflow_determinism"]
