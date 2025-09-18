"""
Integration tests for CLI against minimal run fixture.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def minirun_path():
    """Path to the minimal run fixture."""
    return Path(__file__).parent.parent / "fixtures" / "minirun"


@pytest.fixture
def rldk_cmd():
    """RLDK command prefix."""
    return [sys.executable, "-m", "rldk"]


class TestCLIMinirun:
    """Test CLI functionality against minimal run fixture."""

    def _check_cli_result(self, result, test_name):
        """Helper function to check CLI results with proper error reporting."""
        # Check that CLI doesn't crash with unexpected errors
        if result.returncode not in [0, 1]:
            pytest.fail(f"CLI crashed with unexpected return code {result.returncode} in {test_name}. "
                       f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}")

        # If it fails, it should be a known/expected failure, not a crash
        if result.returncode == 1:
            # Should not contain Python tracebacks or unexpected errors
            assert "Traceback" not in result.stderr, f"Unexpected Python traceback in {test_name}: {result.stderr}"
            assert "Exception" not in result.stderr or "RLDKError" in result.stderr, f"Unexpected exception in {test_name}: {result.stderr}"

    @pytest.mark.parametrize("command,args", [
        (["ingest"], ["--adapter", "generic"]),
        (["replay"], ["--command", "echo test"]),
        (["bisect"], ["--good", "abc123", "--bad", "def456", "--cmd", "echo test"]),
        (["check-determinism"], ["--cmd", "echo test", "--replicas", "2"]),
        (["forensics"], []),
        (["reward", "health"], []),
        (["evals", "bias"], []),
        (["evals", "toxicity"], []),
        (["evals", "throughput"], []),
        (["track", "start"], []),
        (["track", "stop"], []),
        (["track", "status"], []),
        (["track", "logs"], []),
        (["track", "export"], []),
        (["track", "cleanup"], []),
    ])
    def test_cli_commands_with_minirun(self, rldk_cmd, minirun_path, command, args):
        """Test CLI commands that may fail but should not crash."""
        full_command = rldk_cmd + command + [str(minirun_path)] + args
        result = subprocess.run(full_command, capture_output=True, text=True)
        self._check_cli_result(result, f"test_cli_{'_'.join(command)}")

    def test_cli_help(self, rldk_cmd):
        """Test that CLI help works."""
        result = subprocess.run(rldk_cmd + ["--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "RLDK" in result.stdout
        assert "Usage:" in result.stdout

    def test_cli_seed_help(self, rldk_cmd):
        """Test that seed command help works."""
        result = subprocess.run(rldk_cmd + ["seed", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "seed" in result.stdout.lower()

    def test_cli_seed_set(self, rldk_cmd):
        """Test setting a seed via CLI."""
        result = subprocess.run(rldk_cmd + ["seed", "--seed", "42"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "42" in result.stdout

    def test_cli_seed_show(self, rldk_cmd):
        """Test showing current seed via CLI."""
        result = subprocess.run(rldk_cmd + ["seed", "--show"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "seed" in result.stdout.lower()

    def test_cli_forensics_help(self, rldk_cmd):
        """Test that forensics command help works."""
        result = subprocess.run(rldk_cmd + ["forensics", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "forensics" in result.stdout.lower()

    def test_cli_reward_help(self, rldk_cmd):
        """Test that reward command help works."""
        result = subprocess.run(rldk_cmd + ["reward", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "reward" in result.stdout.lower()

    def test_cli_evals_help(self, rldk_cmd):
        """Test that evals command help works."""
        result = subprocess.run(rldk_cmd + ["evals", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "evals" in result.stdout.lower()

    def test_cli_track_help(self, rldk_cmd):
        """Test that track command help works."""
        result = subprocess.run(rldk_cmd + ["track", "--help"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "track" in result.stdout.lower()

    def test_cli_ingest_minirun(self, rldk_cmd, minirun_path):
        """Test ingesting the minimal run fixture."""
        result = subprocess.run(
            rldk_cmd + ["ingest", str(minirun_path), "--adapter", "generic"],
            capture_output=True,
            text=True
        )

        self._check_cli_result(result, "test_cli_ingest_minirun")

    def test_cli_diff_minirun(self, rldk_cmd, minirun_path):
        """Test diffing the minimal run fixture with itself."""
        result = subprocess.run(
            rldk_cmd
            + [
                "diff",
                "--a",
                str(minirun_path),
                "--b",
                str(minirun_path),
                "--signals",
                "reward_mean",
            ],
            capture_output=True,
            text=True
        )

        self._check_cli_result(result, "test_cli_diff_minirun")

    def test_cli_replay_minirun(self, rldk_cmd, minirun_path):
        """Test replaying the minimal run fixture."""
        result = subprocess.run(
            rldk_cmd + ["replay", str(minirun_path), "--command", "echo test"],
            capture_output=True,
            text=True
        )

        self._check_cli_result(result, "test_cli_replay_minirun")

    def test_cli_forensics_minirun(self, rldk_cmd, minirun_path):
        """Test running forensics on the minimal run fixture."""
        result = subprocess.run(
            rldk_cmd + ["forensics", str(minirun_path)],
            capture_output=True,
            text=True
        )

        self._check_cli_result(result, "test_cli_forensics_minirun")

