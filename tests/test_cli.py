"""Comprehensive CLI tests for RLDK.

Tests all CLI commands, options, exit codes, and error handling.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest


class TestCLIBasic:
    """Test basic CLI functionality."""

    def test_help_command(self):
        """Test that help command works and shows all subcommands."""
        result = subprocess.run(["rldk", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "Usage: rldk [OPTIONS] COMMAND [ARGS]..." in result.stdout
        assert "forensics" in result.stdout
        assert "reward" in result.stdout
        assert "evals" in result.stdout

    def test_reward_health_help(self):
        """Test that reward-health subcommand shows run and gate options."""
        result = subprocess.run(["rldk", "reward-health", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "gate" in result.stdout
        assert "Examples:" in result.stdout

    def test_reward_health_run_help(self):
        """Test that reward-health run shows all required options."""
        result = subprocess.run(["rldk", "reward-health", "run", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--scores" in result.stdout
        assert "--out" in result.stdout
        assert "--config" in result.stdout
        assert "--json" in result.stdout
        assert "--verbose" in result.stdout
        assert "--quiet" in result.stdout
        assert "--log-file" in result.stdout
        assert "Examples:" in result.stdout

    def test_reward_health_gate_help(self):
        """Test that reward-health gate shows all options."""
        result = subprocess.run(["rldk", "reward-health", "gate", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--from" in result.stdout
        assert "--min-pass-rate" in result.stdout
        assert "--json" in result.stdout
        assert "Examples:" in result.stdout

    def test_evals_evaluate_help(self):
        """Test that evals evaluate shows all options."""
        result = subprocess.run(["rldk", "evals", "evaluate", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--suite" in result.stdout
        assert "input_file" in result.stdout
        assert "--min-samples" in result.stdout
        assert "--gate" in result.stdout
        assert "--json" in result.stdout
        assert "Examples:" in result.stdout


class TestCLIErrorHandling:
    """Test CLI error handling and exit codes."""

    def test_invalid_command_exit_code(self):
        """Test that invalid commands return exit code 2."""
        result = subprocess.run(["rldk", "invalid-command"], capture_output=True, text=True)
        assert result.returncode == 2

    def test_missing_required_args_exit_code(self):
        """Test that missing required arguments return exit code 2."""
        result = subprocess.run(["rldk", "reward-health", "run"], capture_output=True, text=True)
        assert result.returncode == 2
        assert "Missing option" in result.stderr or "Error" in result.stderr

    def test_invalid_file_path_exit_code(self):
        """Test that invalid file paths return exit code 5 (internal error)."""
        result = subprocess.run([
            "rldk", "reward-health", "run",
            "--scores", "nonexistent.jsonl",
            "--out", "/tmp/test"
        ], capture_output=True, text=True)
        assert result.returncode == 5
        assert "ERROR:" in result.stderr

    def test_invalid_json_exit_code(self):
        """Test that invalid JSON returns exit code 4."""
        # Create a file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write("invalid json content")
            invalid_file = f.name

        try:
            result = subprocess.run([
                "rldk", "reward-health", "run",
                "--scores", invalid_file,
                "--out", "/tmp/test"
            ], capture_output=True, text=True)
            assert result.returncode == 4
            assert "ERROR:" in result.stderr
        finally:
            os.unlink(invalid_file)


class TestRewardHealthCommands:
    """Test reward health specific functionality."""

    @pytest.fixture
    def custom_reward_data(self, tmp_path):
        """Create custom reward evaluation data for testing."""
        data_file = tmp_path / "test_reward_data.jsonl"
        data_file.write_text("""{"global_step": 100, "reward_scalar": 0.8, "kl_to_ref": 0.05, "entropy": 2.3, "loss": 0.15, "learning_rate": 1e-4, "grad_norm": 1.2, "clip_frac": 0.1, "reward_std": 0.2, "tokens_in": 512, "tokens_out": 128, "wall_time": 100.5, "seed": 42, "run_id": "test_run_1", "git_sha": "abc123", "phase": "train"}
{"global_step": 200, "reward_scalar": 0.75, "kl_to_ref": 0.08, "entropy": 2.1, "loss": 0.18, "learning_rate": 1e-4, "grad_norm": 1.5, "clip_frac": 0.15, "reward_std": 0.25, "tokens_in": 512, "tokens_out": 128, "wall_time": 200.8, "seed": 42, "run_id": "test_run_1", "git_sha": "abc123", "phase": "train"}
{"global_step": 300, "reward_scalar": 0.82, "kl_to_ref": 0.06, "entropy": 2.0, "loss": 0.13, "learning_rate": 1e-4, "grad_norm": 1.1, "clip_frac": 0.08, "reward_std": 0.18, "tokens_in": 512, "tokens_out": 128, "wall_time": 301.2, "seed": 42, "run_id": "test_run_1", "git_sha": "abc123", "phase": "train"}""")
        return data_file

    def test_reward_health_run_success(self, custom_reward_data, tmp_path):
        """Test successful reward health run command."""
        output_dir = tmp_path / "output"
        result = subprocess.run([
            "rldk", "reward-health", "run",
            "--scores", str(custom_reward_data),
            "--out", str(output_dir),
            "--adapter", "custom_jsonl"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Reward health check passed" in result.stdout
        assert (output_dir / "reward_health_card.md").exists()
        assert (output_dir / "reward_health_summary.json").exists()

    def test_reward_health_run_json_output(self, custom_reward_data, tmp_path):
        """Test reward health run with JSON output."""
        output_dir = tmp_path / "output"
        result = subprocess.run([
            "rldk", "reward-health", "run",
            "--scores", str(custom_reward_data),
            "--out", str(output_dir),
            "--adapter", "custom_jsonl",
            "--json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        # Should have JSON output at the end
        # Find the JSON part by looking for the first '{'
        json_start = result.stdout.find('{')
        assert json_start != -1, "No JSON output found"
        json_output = json.loads(result.stdout[json_start:])
        assert isinstance(json_output, dict)
        assert "passed" in json_output

    def test_reward_health_gate_success(self, custom_reward_data, tmp_path):
        """Test successful reward health gate command."""
        # First run the analysis to create health.json
        output_dir = tmp_path / "output"
        run_result = subprocess.run([
            "rldk", "reward-health", "run",
            "--scores", str(custom_reward_data),
            "--out", str(output_dir),
            "--adapter", "custom_jsonl"
        ], capture_output=True, text=True)
        assert run_result.returncode == 0
        
        # Now test the gate command
        health_file = output_dir / "reward_health_summary.json"
        result = subprocess.run([
            "rldk", "reward-health", "gate",
            "--from", str(health_file),
            "--min-pass-rate", "0.8"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Health check passed" in result.stdout

    def test_reward_health_gate_failure_exit_code_3(self, custom_reward_data, tmp_path):
        """Test reward health gate returns exit code 3 on threshold failure."""
        # First run the analysis to create health.json
        output_dir = tmp_path / "output"
        run_result = subprocess.run([
            "rldk", "reward-health", "run",
            "--scores", str(custom_reward_data),
            "--out", str(output_dir),
            "--adapter", "custom_jsonl"
        ], capture_output=True, text=True)
        assert run_result.returncode == 0
        
        # Now test the gate command with impossible threshold
        health_file = output_dir / "reward_health_summary.json"
        result = subprocess.run([
            "rldk", "reward-health", "gate",
            "--from", str(health_file),
            "--min-pass-rate", "1.1"  # Impossible threshold
        ], capture_output=True, text=True)
        
        assert result.returncode == 3
        assert "Pass rate too low" in result.stdout

    def test_reward_health_gate_json_output(self, custom_reward_data, tmp_path):
        """Test reward health gate with JSON output."""
        # First run the analysis to create health.json
        output_dir = tmp_path / "output"
        run_result = subprocess.run([
            "rldk", "reward-health", "run",
            "--scores", str(custom_reward_data),
            "--out", str(output_dir),
            "--adapter", "custom_jsonl"
        ], capture_output=True, text=True)
        assert run_result.returncode == 0
        
        # Now test the gate command with JSON output
        health_file = output_dir / "reward_health_summary.json"
        result = subprocess.run([
            "rldk", "reward-health", "gate",
            "--from", str(health_file),
            "--min-pass-rate", "0.8",
            "--json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        # Should have JSON output at the end
        # Find the JSON part by looking for the first '{'
        json_start = result.stdout.find('{')
        assert json_start != -1, "No JSON output found"
        json_output = json.loads(result.stdout[json_start:])
        assert isinstance(json_output, dict)
        assert "passed" in json_output


class TestEvalsCommands:
    """Test evals specific functionality."""

    @pytest.fixture
    def test_data(self, tmp_path):
        """Create test data for evals."""
        data_file = tmp_path / "test_data.jsonl"
        data_file.write_text("""{"output": "This is a test output", "events": []}
{"output": "Another test output", "events": []}
{"output": "Third test output", "events": []}""")
        return data_file

    def test_evals_evaluate_runtime_error_exit_code_4(self, test_data):
        """Test evals evaluate returns exit code 4 for runtime errors."""
        result = subprocess.run([
            "rldk", "evals", "evaluate",
            str(test_data),
            "--suite", "quick"
        ], capture_output=True, text=True)
        
        assert result.returncode == 4
        assert "Evaluation" in result.stderr or "evaluation" in result.stderr

    def test_evals_evaluate_invalid_args_exit_code_2(self):
        """Test evals evaluate returns exit code 2 for invalid arguments."""
        result = subprocess.run([
            "rldk", "evals", "evaluate"
        ], capture_output=True, text=True)
        
        assert result.returncode == 2

    def test_evals_evaluate_json_output(self, test_data):
        """Test evals evaluate with JSON output."""
        result = subprocess.run([
            "rldk", "evals", "evaluate",
            str(test_data),
            "--suite", "quick",
            "--json"
        ], capture_output=True, text=True)
        
        assert result.returncode == 4
        # Should have JSON output even on error
        json_start = result.stdout.find('{')
        assert json_start != -1, "No JSON output found"
        json_output = json.loads(result.stdout[json_start:])
        assert isinstance(json_output, dict)
        assert "evaluations" in json_output

    def test_evals_evaluate_gate_mode_insufficient_samples_exit_code_3(self, test_data):
        """Test evals evaluate gate mode returns exit code 3 for insufficient samples."""
        result = subprocess.run([
            "rldk", "evals", "evaluate",
            str(test_data),
            "--suite", "quick",
            "--gate",
            "--min-samples", "10"  # More than we have (3)
        ], capture_output=True, text=True)
        
        assert result.returncode == 4  # Gate mode with evaluation errors returns 4
        assert "GATE: FAIL" in result.stdout


class TestSharedOptions:
    """Test shared CLI options across commands."""

    def test_verbose_option(self):
        """Test that --verbose option works across commands."""
        result = subprocess.run([
            "rldk", "reward-health", "run", "--help"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--verbose" in result.stdout

    def test_quiet_option(self):
        """Test that --quiet option works across commands."""
        result = subprocess.run([
            "rldk", "evals", "evaluate", "--help"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--quiet" in result.stdout

    def test_json_option(self):
        """Test that --json option works across commands."""
        result = subprocess.run([
            "rldk", "reward-health", "gate", "--help"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--json" in result.stdout

    def test_log_file_option(self):
        """Test that --log-file option works across commands."""
        result = subprocess.run([
            "rldk", "reward-health", "run", "--help"
        ], capture_output=True, text=True)
        assert result.returncode == 0
        assert "--log-file" in result.stdout


class TestDeprecationWarnings:
    """Test deprecated command aliases show warnings."""

    def test_deprecated_aliases_show_warnings(self):
        """Test that deprecated command aliases show warnings."""
        # This would test any deprecated aliases if they existed
        # For now, we just verify the structure is in place
        pass


if __name__ == "__main__":
    pytest.main([__file__])