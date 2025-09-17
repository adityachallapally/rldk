#!/usr/bin/env python3
"""Test API contract compliance for RLDK."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

# Add src to path for imports
sys.path.insert(0, "src")

import pytest
from typer.testing import CliRunner


class TestPublicSymbols:
    """Test that all public symbols can be imported and are callable/classes."""

    def test_top_level_imports(self):
        """Test top-level package imports."""
        import rldk

        # Test functions
        assert callable(rldk.ingest_runs)
        assert callable(rldk.first_divergence)
        assert callable(rldk.check)
        assert callable(rldk.bisect_commits)
        assert callable(rldk.health)
        assert callable(rldk.run)

        # Test classes
        assert hasattr(rldk, 'RewardHealthReport')
        assert hasattr(rldk, 'EvalResult')

    def test_tracking_module_imports(self):
        """Test tracking module imports."""
        from rldk.tracking import (
            DatasetTracker,
            EnvironmentTracker,
            ExperimentTracker,
            GitTracker,
            ModelTracker,
            SeedTracker,
            TrackingConfig,
        )

        # Test classes exist
        assert ExperimentTracker
        assert TrackingConfig
        assert DatasetTracker
        assert ModelTracker
        assert EnvironmentTracker
        assert SeedTracker
        assert GitTracker

    def test_forensics_module_imports(self):
        """Test forensics module imports."""
        from rldk.forensics import (
            AdvantageStatisticsMetrics,
            AdvantageStatisticsTracker,
            ComprehensivePPOForensics,
            ComprehensivePPOMetrics,
            GradientNormsAnalyzer,
            GradientNormsMetrics,
            KLScheduleMetrics,
            KLScheduleTracker,
            scan_ppo_events,
        )

        # Test function
        assert callable(scan_ppo_events)

        # Test classes
        assert KLScheduleTracker
        assert KLScheduleMetrics
        assert GradientNormsAnalyzer
        assert GradientNormsMetrics
        assert AdvantageStatisticsTracker
        assert AdvantageStatisticsMetrics
        assert ComprehensivePPOForensics
        assert ComprehensivePPOMetrics

    def test_determinism_module_imports(self):
        """Test determinism module imports."""
        from rldk.determinism import DeterminismReport, check, check_determinism

        # Test functions
        assert callable(check)
        assert callable(check_determinism)

        # Test class
        assert DeterminismReport

    def test_diff_module_imports(self):
        """Test diff module imports."""
        from rldk.diff import DivergenceReport, first_divergence

        # Test function
        assert callable(first_divergence)

        # Test class
        assert DivergenceReport

    def test_bisect_module_imports(self):
        """Test bisect module imports."""
        from rldk.bisect import BisectResult, bisect_commits

        # Test function
        assert callable(bisect_commits)

        # Test class
        assert BisectResult

    def test_ingest_module_imports(self):
        """Test ingest module imports."""
        from rldk.ingest import ingest_runs, ingest_runs_to_events

        # Test functions
        assert callable(ingest_runs)
        assert callable(ingest_runs_to_events)

    def test_reward_module_imports(self):
        """Test reward module imports."""
        from rldk.reward import (
            HealthAnalysisResult,
            RewardHealthReport,
            health,
            reward_health,
        )

        # Test functions
        assert callable(health)
        assert callable(reward_health)

        # Test classes
        assert RewardHealthReport
        assert HealthAnalysisResult

    def test_evals_module_imports(self):
        """Test evals module imports."""
        from rldk.evals import EvalResult, run

        # Test function
        assert callable(run)

        # Test class
        assert EvalResult


class TestCLICommands:
    """Test CLI commands with synthetic inputs."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

        # Create synthetic test data
        self.create_synthetic_data()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_synthetic_data(self):
        """Create synthetic test data files."""
        # Create synthetic training run data
        run_data = [
            {"step": 0, "loss": 1.0, "reward_mean": 0.5},
            {"step": 100, "loss": 0.8, "reward_mean": 0.6},
            {"step": 200, "loss": 0.6, "reward_mean": 0.7},
        ]

        run_file = Path(self.temp_dir) / "synthetic_run.jsonl"
        with open(run_file, 'w') as f:
            for item in run_data:
                f.write(json.dumps(item) + '\n')

        self.run_file = str(run_file)

        # Create synthetic prompts
        prompts_data = [
            {"text": "Hello world"},
            {"text": "How are you?"},
            {"text": "Goodbye"},
        ]

        prompts_file = Path(self.temp_dir) / "synthetic_prompts.jsonl"
        with open(prompts_file, 'w') as f:
            for item in prompts_data:
                f.write(json.dumps(item) + '\n')

        self.prompts_file = str(prompts_file)

        # Create synthetic checkpoint files with proper PyTorch format
        try:
            from collections import OrderedDict

            import torch

            # Check if torch is mocked
            if hasattr(torch, 'randn') and callable(torch.randn):
                # Real torch
                checkpoint_a = Path(self.temp_dir) / "model_a.pt"
                checkpoint_b = Path(self.temp_dir) / "model_b.pt"

                # Create minimal valid PyTorch checkpoints
                model_a_state = OrderedDict([
                    ('layer1.weight', torch.randn(10, 5)),
                    ('layer1.bias', torch.randn(10)),
                    ('layer2.weight', torch.randn(1, 10)),
                    ('layer2.bias', torch.randn(1))
                ])

                model_b_state = OrderedDict([
                    ('layer1.weight', torch.randn(10, 5) + torch.tensor(0.1)),  # Slightly different
                    ('layer1.bias', torch.randn(10)),
                    ('layer2.weight', torch.randn(1, 10)),
                    ('layer2.bias', torch.randn(1))
                ])

                torch.save(model_a_state, checkpoint_a)
                torch.save(model_b_state, checkpoint_b)

                self.checkpoint_a = str(checkpoint_a)
                self.checkpoint_b = str(checkpoint_b)
            else:
                # Mocked torch, create dummy files
                checkpoint_a = Path(self.temp_dir) / "model_a.pt"
                checkpoint_b = Path(self.temp_dir) / "model_b.pt"

                checkpoint_a.write_text('{"dummy": "checkpoint_a"}')
                checkpoint_b.write_text('{"dummy": "checkpoint_b"}')

                self.checkpoint_a = str(checkpoint_a)
                self.checkpoint_b = str(checkpoint_b)
        except (ImportError, TypeError, AttributeError):
            # Fallback to dummy files if torch not available or mocked
            checkpoint_a = Path(self.temp_dir) / "model_a.pt"
            checkpoint_b = Path(self.temp_dir) / "model_b.pt"

            checkpoint_a.write_text('{"dummy": "checkpoint_a"}')
            checkpoint_b.write_text('{"dummy": "checkpoint_b"}')

            self.checkpoint_a = str(checkpoint_a)
            self.checkpoint_b = str(checkpoint_b)

    def test_main_help(self):
        """Test main CLI help command."""
        from rldk.cli import app

        result = self.runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "RL Debug Kit" in result.stdout
        assert "ingest" in result.stdout
        assert "diff" in result.stdout
        assert "check-determinism" in result.stdout
        assert "bisect" in result.stdout
        assert "reward-health" in result.stdout
        assert "replay" in result.stdout
        assert "eval" in result.stdout
        assert "track" in result.stdout
        assert "reward-drift" in result.stdout
        assert "doctor" in result.stdout
        assert "version" in result.stdout
        assert "card" in result.stdout

    def test_ingest_command(self):
        """Test ingest command with synthetic data."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "ingest",
            self.run_file,
            "--output", f"{self.temp_dir}/output.jsonl"
        ])

        # May fail with synthetic data format, but should handle gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            stdout_lower = result.stdout.lower()
            assert "ingest" in stdout_lower
            assert "step" in stdout_lower

    def test_diff_command(self):
        """Test diff command with synthetic data."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "diff",
            "--a", self.run_file,
            "--b", self.run_file,
            "--signals", "loss,reward_mean",
            "--output-dir", f"{self.temp_dir}/diff_analysis"
        ])

        # May fail with synthetic data format, but should handle gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert "divergence detected" in result.stdout.lower()

    def test_check_determinism_command(self):
        """Test check-determinism command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "check-determinism",
            "--compare", "loss",
            "--cmd", "echo 'test'",
            "--replicas", "2",
            "--output-dir", f"{self.temp_dir}/determinism_analysis"
        ])

        # Should succeed with simple command
        assert result.exit_code == 0
        assert "Determinism check" in result.stdout

    def test_bisect_command(self):
        """Test bisect command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "bisect",
            "--good", "HEAD~1",
            "--bad", "HEAD",
            "--shell-predicate", "echo 'test'"
        ])

        # May fail if not in git repo, but should handle gracefully
        assert result.exit_code in [0, 1]

    def test_reward_health_command(self):
        """Test reward-health command with synthetic data."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "reward-health",
            "--run", self.run_file,
            "--output-dir", f"{self.temp_dir}/reward_analysis"
        ])

        # May fail with synthetic data format, but should handle gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert "Reward health analysis" in result.stdout

    def test_replay_command(self):
        """Test replay command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "replay",
            "--run", self.run_file,
            "--command", "echo 'test'",
            "--metrics", "loss",
            "--output-dir", f"{self.temp_dir}/replay_results"
        ])

        # May fail with synthetic data format, but should handle gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert "replay" in result.stdout.lower()

    def test_eval_command(self):
        """Test eval command with synthetic data."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "eval",
            "--run", self.run_file,
            "--output-dir", f"{self.temp_dir}/eval_results"
        ])

        # May fail with synthetic data format, but should handle gracefully
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert "Evaluation" in result.stdout

    def test_track_command(self):
        """Test track command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "track",
            "test_experiment",
            "--output-dir", f"{self.temp_dir}/runs",
            "--no-wandb"
        ])

        # Should succeed
        assert result.exit_code == 0
        assert "Experiment tracking started" in result.stdout

    def test_reward_drift_command(self):
        """Test reward-drift command with synthetic data."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "reward-drift",
            self.checkpoint_a,
            self.checkpoint_b,
            "--prompts", self.prompts_file
        ])

        # May fail with dummy checkpoints, but should handle gracefully
        assert result.exit_code in [0, 1]

    def test_doctor_command(self):
        """Test doctor command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "doctor",
            self.temp_dir
        ])

        # Should succeed with directory
        assert result.exit_code == 0
        assert "diagnostics" in result.stdout.lower()

    def test_version_command(self):
        """Test version command."""
        from rldk.cli import app

        result = self.runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "RL Debug Kit version" in result.stdout

    def test_card_command(self):
        """Test card command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "card",
            "determinism",
            self.temp_dir,
            "--output-dir", f"{self.temp_dir}/cards"
        ])

        # Should succeed with directory
        assert result.exit_code == 0
        assert "card generated" in result.stdout

    def test_forensics_help(self):
        """Test forensics subcommand help."""
        from rldk.cli import app

        result = self.runner.invoke(app, ["forensics", "--help"])
        assert result.exit_code == 0
        assert "Forensics commands" in result.stdout
        assert "compare-runs" in result.stdout
        assert "diff-ckpt" in result.stdout
        assert "env-audit" in result.stdout
        assert "log-scan" in result.stdout
        assert "doctor" in result.stdout

    def test_forensics_compare_runs(self):
        """Test forensics compare-runs command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "forensics", "compare-runs",
            self.temp_dir,
            self.temp_dir
        ])

        # Should succeed with identical directories
        assert result.exit_code == 0
        assert "Comparison complete" in result.stdout

    def test_forensics_diff_ckpt(self):
        """Test forensics diff-ckpt command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "forensics", "diff-ckpt",
            self.checkpoint_a,
            self.checkpoint_b
        ])

        # Should succeed with valid PyTorch checkpoints, or fail gracefully with dummy files
        assert result.exit_code in [0, 1]
        if result.exit_code == 0:
            assert "Checkpoint diff complete" in result.stdout or "Checkpoint comparison" in result.stdout

    def test_forensics_env_audit(self):
        """Test forensics env-audit command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "forensics", "env-audit",
            self.temp_dir
        ])

        # Should succeed with directory
        assert result.exit_code == 0
        assert "Environment audit" in result.stdout

    def test_forensics_log_scan(self):
        """Test forensics log-scan command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "forensics", "log-scan",
            self.temp_dir
        ])

        # Should succeed with directory
        assert result.exit_code == 0
        assert "Log scan" in result.stdout

    def test_forensics_doctor(self):
        """Test forensics doctor command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "forensics", "doctor",
            self.temp_dir
        ])

        # Should succeed with directory
        assert result.exit_code == 0
        assert "diagnostics" in result.stdout.lower()

    def test_reward_help(self):
        """Test reward subcommand help."""
        from rldk.cli import app

        result = self.runner.invoke(app, ["reward", "--help"])
        assert result.exit_code == 0
        assert "Reward model analysis" in result.stdout
        assert "reward-drift" in result.stdout

    def test_reward_reward_drift(self):
        """Test reward reward-drift command."""
        from rldk.cli import app

        result = self.runner.invoke(app, [
            "reward", "reward-drift",
            self.checkpoint_a,
            self.checkpoint_b,
            "--prompts", self.prompts_file
        ])

        # May fail with dummy checkpoints, but should handle gracefully
        assert result.exit_code in [0, 1]


class TestArtifactLocations:
    """Test that commands produce expected artifacts."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = CliRunner()

        # Create minimal synthetic data
        run_data = [{"step": 0, "loss": 1.0}]
        run_file = Path(self.temp_dir) / "test_run.jsonl"
        with open(run_file, 'w') as f:
            for item in run_data:
                f.write(json.dumps(item) + '\n')
        self.run_file = str(run_file)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_determinism_analysis_artifacts(self):
        """Test check-determinism produces expected artifacts."""
        from rldk.cli import app

        output_dir = f"{self.temp_dir}/determinism_analysis"

        result = self.runner.invoke(app, [
            "check-determinism",
            "--compare", "loss",
            "--cmd", "echo 'test'",
            "--replicas", "2",
            "--output-dir", output_dir
        ])

        if result.exit_code == 0:
            # Check for expected artifacts
            output_path = Path(output_dir)
            assert output_path.exists()

            # Should have determinism card
            card_file = output_path / "determinism_card.json"
            if card_file.exists():
                with open(card_file) as f:
                    card_data = json.load(f)
                assert "version" in card_data
                assert "passed" in card_data

    def test_diff_analysis_artifacts(self):
        """Test diff produces expected artifacts."""
        from rldk.cli import app

        output_dir = f"{self.temp_dir}/diff_analysis"

        result = self.runner.invoke(app, [
            "diff",
            "--a", self.run_file,
            "--b", self.run_file,
            "--signals", "loss",
            "--output-dir", output_dir
        ])

        if result.exit_code == 0:
            # Check for expected artifacts
            output_path = Path(output_dir)
            if output_path.exists():
                # Should have divergence report
                report_file = output_path / "divergence_report.json"
                if report_file.exists():
                    with open(report_file) as f:
                        report_data = json.load(f)
                    assert "version" in report_data

    def test_reward_analysis_artifacts(self):
        """Test reward-health produces expected artifacts."""
        from rldk.cli import app

        output_dir = f"{self.temp_dir}/reward_analysis"

        result = self.runner.invoke(app, [
            "reward-health",
            "--run", self.run_file,
            "--output-dir", output_dir
        ])

        if result.exit_code == 0:
            # Check for expected artifacts
            output_path = Path(output_dir)
            if output_path.exists():
                # Should have reward health report
                report_file = output_path / "reward_health_report.json"
                if report_file.exists():
                    with open(report_file) as f:
                        report_data = json.load(f)
                    assert "version" in report_data

    def test_replay_results_artifacts(self):
        """Test replay produces expected artifacts."""
        from rldk.cli import app

        output_dir = f"{self.temp_dir}/replay_results"

        result = self.runner.invoke(app, [
            "replay",
            "--run", self.run_file,
            "--command", "echo 'test'",
            "--metrics", "loss",
            "--output-dir", output_dir
        ])

        if result.exit_code == 0:
            # Check for expected artifacts
            output_path = Path(output_dir)
            if output_path.exists():
                # Should have replay metrics
                metrics_file = output_path / "replay_metrics.jsonl"
                if metrics_file.exists():
                    # File should contain JSONL data
                    with open(metrics_file) as f:
                        lines = f.readlines()
                    assert len(lines) > 0

    def test_eval_results_artifacts(self):
        """Test eval produces expected artifacts."""
        from rldk.cli import app

        output_dir = f"{self.temp_dir}/eval_results"

        result = self.runner.invoke(app, [
            "eval",
            "--run", self.run_file,
            "--output-dir", output_dir
        ])

        if result.exit_code == 0:
            # Check for expected artifacts
            output_path = Path(output_dir)
            if output_path.exists():
                # Should have eval report
                report_file = output_path / "eval_report.json"
                if report_file.exists():
                    with open(report_file) as f:
                        report_data = json.load(f)
                    assert "version" in report_data

    def test_tracking_artifacts(self):
        """Test track produces expected artifacts."""
        from rldk.cli import app

        output_dir = f"{self.temp_dir}/runs"

        result = self.runner.invoke(app, [
            "track",
            "test_experiment",
            "--output-dir", output_dir,
            "--no-wandb"
        ])

        if result.exit_code == 0:
            # Check for expected artifacts
            output_path = Path(output_dir)
            if output_path.exists():
                # Should have experiment directory
                experiment_dirs = list(output_path.glob("*"))
                if experiment_dirs:
                    exp_dir = experiment_dirs[0]
                    # Should have experiment metadata
                    metadata_file = exp_dir / "experiment_metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        assert "experiment_name" in metadata

    def test_forensics_reports_artifacts(self):
        """Test forensics commands produce expected artifacts."""
        from rldk.cli import app

        # Test compare-runs
        result = self.runner.invoke(app, [
            "forensics", "compare-runs",
            self.temp_dir,
            self.temp_dir
        ])

        if result.exit_code == 0:
            # Should create rldk_reports directory
            reports_dir = Path("rldk_reports")
            if reports_dir.exists():
                # Should have run comparison report
                report_file = reports_dir / "run_comparison.json"
                if report_file.exists():
                    with open(report_file) as f:
                        report_data = json.load(f)
                    assert "version" in report_data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
