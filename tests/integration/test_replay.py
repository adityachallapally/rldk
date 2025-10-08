"""Tests for the seeded replay functionality."""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from rldk.replay import (
    ReplayReport,
    ReplayResult,
    _cleanup_temp_file,
    _compare_metrics,
    _prepare_replay_command,
    replay,
)


class TestReplayCommandPreparation:
    """Test replay command preparation."""

    def test_add_seed_to_command(self):
        """Test adding seed argument to command."""
        command = "python train.py --epochs 10"
        seed = 42

        result = _prepare_replay_command(command, seed)
        expected = "python train.py --epochs 10 --seed 42"
        assert result == expected

    def test_replace_existing_seed(self):
        """Test replacing existing seed argument."""
        command = "python train.py --seed 100 --epochs 10"
        seed = 42

        result = _prepare_replay_command(command, seed)
        expected = "python train.py --seed 42 --epochs 10"
        assert result == expected

    def test_command_with_multiple_seed_occurrences(self):
        """Test command with multiple seed occurrences (should replace first)."""
        command = "python train.py --seed 100 --other --seed 200"
        seed = 42

        result = _prepare_replay_command(command, seed)
        expected = "python train.py --seed 42 --other --seed 200"
        assert result == expected


class TestMetricComparison:
    """Test metric comparison functionality."""

    def test_compare_metrics_within_tolerance(self):
        """Test metric comparison when values are within tolerance."""
        # Create test data
        original_data = {
            "step": [0, 1, 2],
            "reward_mean": [0.5, 0.6, 0.7],
            "kl_mean": [0.1, 0.11, 0.12],
        }
        original_df = pd.DataFrame(original_data)

        replay_data = {
            "step": [0, 1, 2],
            "reward_mean": [0.5001, 0.6001, 0.7001],
            "kl_mean": [0.1001, 0.1101, 0.1201],
        }
        replay_df = pd.DataFrame(replay_data)

        metrics = ["reward_mean", "kl_mean"]
        tolerance = 0.01

        mismatches, stats = _compare_metrics(original_df, replay_df, metrics, tolerance)

        # Should have no mismatches
        assert len(mismatches) == 0

        # Check stats
        assert "reward_mean" in stats
        assert "kl_mean" in stats
        assert stats["reward_mean"]["tolerance_violations"] == 0
        assert stats["kl_mean"]["tolerance_violations"] == 0

    def test_compare_metrics_with_tolerance_violations(self):
        """Test metric comparison when values exceed tolerance."""
        # Create test data with violations
        original_data = {
            "step": [0, 1, 2],
            "reward_mean": [0.5, 0.6, 0.7],
            "kl_mean": [0.1, 0.11, 0.12],
        }
        original_df = pd.DataFrame(original_data)

        replay_data = {
            "step": [0, 1, 2],
            "reward_mean": [0.5, 0.6, 0.7],  # Within tolerance
            "kl_mean": [0.1, 0.15, 0.12],  # Step 1 exceeds tolerance
        }
        replay_df = pd.DataFrame(replay_data)

        metrics = ["reward_mean", "kl_mean"]
        tolerance = 0.01

        mismatches, stats = _compare_metrics(original_df, replay_df, metrics, tolerance)

        # Should have one mismatch
        assert len(mismatches) == 1
        assert mismatches[0]["step"] == 1
        assert mismatches[0]["metric"] == "kl_mean"

        # Check stats
        assert stats["reward_mean"]["tolerance_violations"] == 0
        assert stats["kl_mean"]["tolerance_violations"] == 1

    def test_compare_metrics_with_missing_data(self):
        """Test metric comparison with missing data."""
        original_data = {
            "step": [0, 1, 2],
            "reward_mean": [0.5, 0.6, 0.7],
            "kl_mean": [0.1, 0.11, 0.12],
        }
        original_df = pd.DataFrame(original_data)

        replay_data = {
            "step": [0, 1, 2],
            "reward_mean": [0.5, 0.6, 0.7],
            "kl_mean": [0.1, None, 0.12],  # Missing data at step 1
        }
        replay_df = pd.DataFrame(replay_data)

        metrics = ["reward_mean", "kl_mean"]
        tolerance = 0.01

        mismatches, stats = _compare_metrics(original_df, replay_df, metrics, tolerance)

        # Should have no mismatches (missing data is skipped)
        assert len(mismatches) == 0

    def test_compare_metrics_no_common_steps(self):
        """Test metric comparison with no common steps."""
        original_data = {"step": [0, 1], "reward_mean": [0.5, 0.6]}
        original_df = pd.DataFrame(original_data)

        replay_data = {"step": [5, 6], "reward_mean": [0.5, 0.6]}
        replay_df = pd.DataFrame(replay_data)

        metrics = ["reward_mean"]
        tolerance = 0.01

        with pytest.raises(ValueError, match="No common steps"):
            _compare_metrics(original_df, replay_df, metrics, tolerance)


class TestReplayFunction:
    """Test the main replay function."""

    @patch("rldk.replay.replay._run_replay")
    @patch("rldk.replay.replay.ingest_runs")
    def test_replay_success(self, mock_ingest, mock_run_replay):
        """Test successful replay execution."""
        # Mock original run data
        original_data = {
            "step": [0, 1, 2],
            "seed": [42, 42, 42],
            "reward_mean": [0.5, 0.6, 0.7],
            "kl_mean": [0.1, 0.11, 0.12],
            "phase": ["train", "train", "train"],
            "reward_std": [0.1, 0.1, 0.1],
            "entropy_mean": [0.8, 0.8, 0.8],
            "clip_frac": [0.1, 0.1, 0.1],
            "grad_norm": [1.0, 1.0, 1.0],
            "lr": [0.001, 0.001, 0.001],
            "loss": [0.5, 0.5, 0.5],
            "tokens_in": [512, 512, 512],
            "tokens_out": [128, 128, 128],
            "wall_time": [100.0, 101.0, 102.0],
            "run_id": ["test", "test", "test"],
            "git_sha": ["abc123", "abc123", "abc123"],
        }
        original_df = pd.DataFrame(original_data)
        mock_ingest.return_value = original_df

        # Mock replay run data (identical to original)
        mock_run_replay.return_value = original_df.copy()

        # Run replay
        with tempfile.TemporaryDirectory() as temp_dir:
            report = replay(
                run_path="test_run.jsonl",
                training_command="python train.py",
                metrics_to_compare=["reward_mean", "kl_mean"],
                tolerance=0.01,
                output_dir=temp_dir,
            )

        # Verify report
        assert isinstance(report, ReplayReport)
        assert report.passed is True
        assert report.original_seed == 42
        assert report.replay_seed == 42
        assert len(report.mismatches) == 0
        assert "reward_mean" in report.metrics_compared
        assert "kl_mean" in report.metrics_compared

    @patch("rldk.replay.replay._run_replay")
    @patch("rldk.replay.replay.ingest_runs")
    def test_replay_with_tolerance_violations(self, mock_ingest, mock_run_replay):
        """Test replay with tolerance violations."""
        # Mock original run data
        original_data = {
            "step": [0, 1, 2],
            "seed": [42, 42, 42],
            "reward_mean": [0.5, 0.6, 0.7],
            "kl_mean": [0.1, 0.11, 0.12],
            "phase": ["train", "train", "train"],
            "reward_std": [0.1, 0.1, 0.1],
            "entropy_mean": [0.8, 0.8, 0.8],
            "clip_frac": [0.1, 0.1, 0.1],
            "grad_norm": [1.0, 1.0, 1.0],
            "lr": [0.001, 0.001, 0.001],
            "loss": [0.5, 0.5, 0.5],
            "tokens_in": [512, 512, 512],
            "tokens_out": [128, 128, 128],
            "wall_time": [100.0, 101.0, 102.0],
            "run_id": ["test", "test", "test"],
            "git_sha": ["abc123", "abc123", "abc123"],
        }
        original_df = pd.DataFrame(original_data)
        mock_ingest.return_value = original_df

        # Mock replay run data with violations
        replay_data = original_data.copy()
        replay_data["reward_mean"] = [0.5, 0.65, 0.7]  # Step 1 exceeds tolerance
        replay_df = pd.DataFrame(replay_data)
        mock_run_replay.return_value = replay_df

        # Run replay
        with tempfile.TemporaryDirectory() as temp_dir:
            report = replay(
                run_path="test_run.jsonl",
                training_command="python train.py",
                metrics_to_compare=["reward_mean", "kl_mean"],
                tolerance=0.01,
                output_dir=temp_dir,
            )

        # Verify report
        assert isinstance(report, ReplayReport)
        assert report.passed is False
        assert len(report.mismatches) > 0

        # Check that reward_mean has violations
        reward_stats = report.comparison_stats.get("reward_mean", {})
        assert reward_stats.get("tolerance_violations", 0) > 0

    def test_replay_missing_seed(self):
        """Test replay with missing seed in original run."""
        # Create DataFrame without seed column
        original_data = {"step": [0, 1, 2], "reward_mean": [0.5, 0.6, 0.7]}
        original_df = pd.DataFrame(original_data)

        with patch("rldk.replay.replay.ingest_runs", return_value=original_df):
            with pytest.raises(ValueError, match="must contain 'seed' column"):
                replay(
                    run_path="test_run.jsonl",
                    training_command="python train.py",
                    metrics_to_compare=["reward_mean"],
                    tolerance=0.01,
                )

    def test_replay_nan_seed(self):
        """Test replay with NaN seed in original run."""
        # Create DataFrame with NaN seed
        original_data = {
            "step": [0, 1, 2],
            "seed": [None, None, None],
            "reward_mean": [0.5, 0.6, 0.7],
        }
        original_df = pd.DataFrame(original_data)

        with patch("rldk.replay.replay.ingest_runs", return_value=original_df):
            with pytest.raises(ValueError, match="seed is missing or NaN"):
                replay(
                    run_path="test_run.jsonl",
                    training_command="python train.py",
                    metrics_to_compare=["reward_mean"],
                    tolerance=0.01,
                )


class TestReplayReport:
    """Test the ReplayReport dataclass."""

    def test_replay_report_creation(self):
        """Test creating a ReplayReport instance."""
        report = ReplayReport(
            passed=True,
            original_seed=42,
            replay_seed=42,
            metrics_compared=["reward_mean"],
            tolerance=0.01,
            mismatches=[],
            original_metrics=pd.DataFrame(),
            replay_metrics=pd.DataFrame(),
            comparison_stats={},
            replay_command="python train.py",
            replay_duration=10.5,
        )

        assert report.passed is True
        assert report.original_seed == 42
        assert report.replay_seed == 42
        assert report.metrics_compared == ["reward_mean"]
        assert report.tolerance == 0.01
        assert len(report.mismatches) == 0
        assert report.replay_command == "python train.py"
        assert report.replay_duration == 10.5


class TestCleanupFunctionality:
    """Test cleanup functionality with explicit error handling."""

    def test_cleanup_success(self, caplog):
        """Test successful cleanup of temp file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        # Ensure file exists
        assert os.path.exists(temp_file)

        # Clean up
        _cleanup_temp_file(temp_file)

        # Verify file is deleted
        assert not os.path.exists(temp_file)

        # Check for debug log
        assert any("Successfully cleaned up temp file" in record.message for record in caplog.records)

    def test_cleanup_file_not_found(self, caplog):
        """Test cleanup when file doesn't exist (FileNotFoundError)."""
        non_existent_file = "/tmp/non_existent_file_12345"

        # Clean up non-existent file
        _cleanup_temp_file(non_existent_file)

        # Check for debug log about file not found
        assert any("Temp file not found during cleanup" in record.message for record in caplog.records)

    def test_cleanup_permission_error(self, caplog):
        """Test cleanup with permission error."""
        # Create a file and make it read-only
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name

        try:
            # Make file read-only
            os.chmod(temp_file, 0o444)

            # Clean up (should fail with PermissionError)
            _cleanup_temp_file(temp_file)

            # Check for warning log with actionable message
            warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
            assert len(warning_logs) > 0
            assert any("Permission denied cleaning up temp file" in record.message for record in warning_logs)
            assert any("Please check file permissions" in record.message for record in warning_logs)

        finally:
            # Clean up manually
            try:
                os.chmod(temp_file, 0o644)
                os.unlink(temp_file)
            except OSError:
                pass

    def test_cleanup_other_os_error(self, caplog):
        """Test cleanup with other OS error."""
        # Mock os.unlink to raise OSError
        with patch('os.unlink', side_effect=OSError("Mock OS error")):
            _cleanup_temp_file("/tmp/test_file")

            # Check for warning log with actionable message
            warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
            assert len(warning_logs) > 0
            assert any("Failed to clean up temp file" in record.message for record in warning_logs)
            assert any("File may need manual cleanup" in record.message for record in warning_logs)


class TestReplayResult:
    """Test the ReplayResult dataclass."""

    def test_replay_result_success(self):
        """Test creating a successful ReplayResult."""
        df = pd.DataFrame({"step": [1, 2], "reward": [0.5, 0.6]})
        result = ReplayResult(
            success=True,
            return_code=0,
            stdout="Success output",
            stderr="",
            metrics_data=df
        )

        assert result.success is True
        assert result.return_code == 0
        assert result.stdout == "Success output"
        assert result.stderr == ""
        assert result.error_message is None
        assert len(result.metrics_data) == 2

    def test_replay_result_failure(self):
        """Test creating a failed ReplayResult."""
        result = ReplayResult(
            success=False,
            return_code=1,
            stdout="",
            stderr="Error occurred",
            metrics_data=pd.DataFrame(),
            error_message="Test error"
        )

        assert result.success is False
        assert result.return_code == 1
        assert result.stdout == ""
        assert result.stderr == "Error occurred"
        assert result.error_message == "Test error"
        assert len(result.metrics_data) == 0


class TestTempFileCleanup:
    """Test that temporary files are cleaned up in all code paths."""

    @patch("rldk.replay.replay._cleanup_temp_file")
    @patch("rldk.replay.replay._detect_device")
    @patch("rldk.replay.replay._get_deterministic_env")
    def test_temp_file_cleanup_on_command_parse_error(self, mock_env, mock_device, mock_cleanup):
        """Test that temp file is cleaned up when command parsing fails."""
        from rldk.replay.replay import _run_replay

        mock_device.return_value = "cpu"
        mock_env.return_value = {}

        # Test with invalid command that will fail parsing
        invalid_command = "python train.py --invalid-quote \""

        result = _run_replay(invalid_command, Path("/tmp"), "cpu")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify result indicates failure
        assert result.success is False
        assert "Failed to parse command" in result.error_message

    @patch("rldk.replay.replay._cleanup_temp_file")
    @patch("rldk.replay.replay._detect_device")
    @patch("rldk.replay.replay._get_deterministic_env")
    @patch("subprocess.run")
    def test_temp_file_cleanup_on_subprocess_timeout(self, mock_run, mock_env, mock_device, mock_cleanup):
        """Test that temp file is cleaned up when subprocess times out."""
        import subprocess

        from rldk.replay.replay import _run_replay

        mock_device.return_value = "cpu"
        mock_env.return_value = {}
        mock_run.side_effect = subprocess.TimeoutExpired("python", 30)

        result = _run_replay("python train.py", Path("/tmp"), "cpu")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify result indicates timeout
        assert result.success is False
        assert "timed out" in result.error_message

    @patch("rldk.replay.replay._cleanup_temp_file")
    @patch("rldk.replay.replay._detect_device")
    @patch("rldk.replay.replay._get_deterministic_env")
    @patch("subprocess.run")
    def test_temp_file_cleanup_on_subprocess_error(self, mock_run, mock_env, mock_device, mock_cleanup):
        """Test that temp file is cleaned up when subprocess raises exception."""
        from rldk.replay.replay import _run_replay

        mock_device.return_value = "cpu"
        mock_env.return_value = {}
        mock_run.side_effect = OSError("Command not found")

        result = _run_replay("python train.py", Path("/tmp"), "cpu")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify result indicates execution error
        assert result.success is False
        assert "Failed to execute replay command" in result.error_message

    @patch("rldk.replay.replay._cleanup_temp_file")
    @patch("rldk.replay.replay._detect_device")
    @patch("rldk.replay.replay._get_deterministic_env")
    @patch("subprocess.run")
    def test_temp_file_cleanup_on_command_failure(self, mock_run, mock_env, mock_device, mock_cleanup):
        """Test that temp file is cleaned up when command returns non-zero exit code."""
        from unittest.mock import MagicMock

        from rldk.replay.replay import _run_replay

        mock_device.return_value = "cpu"
        mock_env.return_value = {}

        # Mock subprocess result with failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = "Error output"
        mock_result.stderr = "Error occurred"
        mock_run.return_value = mock_result

        result = _run_replay("python train.py", Path("/tmp"), "cpu")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify result indicates command failure
        assert result.success is False
        assert "failed with return code 1" in result.error_message

    @patch("rldk.replay.replay._cleanup_temp_file")
    @patch("rldk.replay.replay._detect_device")
    @patch("rldk.replay.replay._get_deterministic_env")
    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_temp_file_cleanup_on_no_metrics_file(self, mock_exists, mock_run, mock_env, mock_device, mock_cleanup):
        """Test that temp file is cleaned up when no metrics file is found."""
        from unittest.mock import MagicMock

        from rldk.replay.replay import _run_replay

        mock_device.return_value = "cpu"
        mock_env.return_value = {}

        # Mock subprocess result with success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock that metrics file doesn't exist
        mock_exists.return_value = False

        result = _run_replay("python train.py", Path("/tmp"), "cpu")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify result indicates no metrics file found
        assert result.success is False
        assert "No metrics file found after replay" in result.error_message

    @patch("rldk.replay.replay._cleanup_temp_file")
    @patch("rldk.replay.replay._detect_device")
    @patch("rldk.replay.replay._get_deterministic_env")
    @patch("subprocess.run")
    @patch("os.path.exists")
    @patch("pandas.read_json")
    def test_temp_file_cleanup_on_metrics_parse_error(self, mock_read_json, mock_exists, mock_run, mock_env, mock_device, mock_cleanup):
        """Test that temp file is cleaned up when metrics parsing fails."""
        from unittest.mock import MagicMock

        from rldk.replay.replay import _run_replay

        mock_device.return_value = "cpu"
        mock_env.return_value = {}

        # Mock subprocess result with success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        # Mock that metrics file exists
        mock_exists.return_value = True

        # Mock pandas.read_json to raise exception
        mock_read_json.side_effect = Exception("Invalid JSON")

        result = _run_replay("python train.py", Path("/tmp"), "cpu")

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

        # Verify result indicates parsing error
        assert result.success is False
        assert "Could not load replay metrics" in result.error_message


if __name__ == "__main__":
    pytest.main([__file__])
