"""Tests for the CLI module."""

from typer.testing import CliRunner
from rldk.cli import app

runner = CliRunner()


class TestCLI:
    """Test CLI functionality."""

    def test_version(self):
        """Test version command."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "RL Debug Kit version" in result.stdout

    def test_help(self):
        """Test help command."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "RL Debug Kit" in result.stdout
        assert "ingest" in result.stdout
        assert "diff" in result.stdout
        assert "check-determinism" in result.stdout
        assert "bisect" in result.stdout

    def test_ingest_help(self):
        """Test ingest help."""
        result = runner.invoke(app, ["ingest", "--help"])
        assert result.exit_code == 0
        assert "Ingest training runs" in result.stdout

    def test_diff_help(self):
        """Test diff help."""
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        assert "Find first divergence" in result.stdout

    def test_check_determinism_help(self):
        """Test check-determinism help."""
        result = runner.invoke(app, ["check-determinism", "--help"])
        assert result.exit_code == 0
        assert "Check if a training command" in result.stdout

    def test_bisect_help(self):
        """Test bisect help."""
        result = runner.invoke(app, ["bisect", "--help"])
        assert result.exit_code == 0
        assert "Find regression using git bisect" in result.stdout

    def test_reward_health_help(self):
        """Test reward-health help."""
        result = runner.invoke(app, ["reward-health", "--help"])
        assert result.exit_code == 0
        assert "Analyze reward model health" in result.stdout

    def test_eval_help(self):
        """Test eval help."""
        result = runner.invoke(app, ["eval", "--help"])
        assert result.exit_code == 0
        assert "Run evaluation suite" in result.stdout
