"""Tests for the seeded replay functionality."""

import pytest
import pandas as pd
import tempfile
from unittest.mock import patch

from rldk.replay import replay, ReplayReport, _compare_metrics, _prepare_replay_command


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


if __name__ == "__main__":
    pytest.main([__file__])
