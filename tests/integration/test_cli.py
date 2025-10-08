#!/usr/bin/env python3
"""Test suite for CLI functionality."""

import sys

import pytest
from typer.testing import CliRunner

sys.path.insert(0, "src")

from rldk.cli import app


class TestCLI:
    """Test cases for CLI functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_app_import(self):
        """Test that CLI app can be imported successfully."""
        assert app is not None
        assert hasattr(app, 'callback')

    def test_version_command(self):
        """Test version command functionality."""
        result = self.runner.invoke(app, ["version"])

        # Should exit successfully
        assert result.exit_code == 0, f"Version command failed with exit code {result.exit_code}"

        # Should produce some output
        assert result.stdout is not None
        assert len(result.stdout.strip()) > 0, "Version command produced no output"

        # Should not have errors
        assert result.stderr is None or result.stderr.strip() == "", f"Version command had errors: {result.stderr}"

    def test_help_command(self):
        """Test help command functionality."""
        result = self.runner.invoke(app, ["--help"])

        # Should exit successfully
        assert result.exit_code == 0, f"Help command failed with exit code {result.exit_code}"

        # Should contain help text
        assert "help" in result.stdout.lower() or "usage" in result.stdout.lower(), "Help command missing help text"

        # Should not have errors
        assert result.stderr is None or result.stderr.strip() == "", f"Help command had errors: {result.stderr}"

    def test_invalid_command(self):
        """Test behavior with invalid command."""
        result = self.runner.invoke(app, ["invalid-command"])

        # Should fail with non-zero exit code
        assert result.exit_code != 0, "Invalid command should fail"

        # Should have error output (Typer typically writes to stdout, not stderr)
        error_output = result.stderr or result.stdout
        assert error_output is not None and len(error_output.strip()) > 0, "Invalid command should produce error message"

    def test_no_arguments(self):
        """Test CLI behavior with no arguments."""
        result = self.runner.invoke(app, [])

        # Should either show help or fail gracefully
        # The exact behavior depends on CLI implementation
        assert result.exit_code in [0, 1, 2], f"Unexpected exit code {result.exit_code} for no arguments"

    def test_cli_runner_initialization(self):
        """Test that CLI runner can be initialized properly."""
        runner = CliRunner()
        assert runner is not None
        assert hasattr(runner, 'invoke')
