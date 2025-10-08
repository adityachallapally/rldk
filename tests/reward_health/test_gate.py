"""Tests for reward health gate functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from rldk.reward.health_config.exit_codes import get_exit_code, raise_on_failure


class TestExitCodeMapping:
    """Test exit code mapping functionality."""

    def test_get_exit_code_passed_true(self):
        """Test that passed=True returns exit code 0."""
        assert get_exit_code(True) == 0

    def test_get_exit_code_passed_false(self):
        """Test that passed=False returns exit code 3."""
        assert get_exit_code(False) == 3


class TestRaiseOnFailure:
    """Test raise_on_failure function."""

    def test_raise_on_failure_passed_true(self):
        """Test that passed=True exits with code 0."""
        health_data = {
            "passed": True,
            "warnings": ["Minor issue"],
            "failures": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(health_data, f)
            health_path = f.name

        try:
            with patch('sys.exit') as mock_exit:
                with patch('sys.stderr'):
                    raise_on_failure(health_path)
                mock_exit.assert_called_once_with(0)
        finally:
            Path(health_path).unlink()

    def test_raise_on_failure_passed_false(self):
        """Test that passed=False exits with code 3."""
        health_data = {
            "passed": False,
            "warnings": ["Warning message"],
            "failures": ["Failure message"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(health_data, f)
            health_path = f.name

        try:
            with patch('sys.exit') as mock_exit:
                with patch('sys.stderr'):
                    raise_on_failure(health_path)
                mock_exit.assert_called_once_with(3)
        finally:
            Path(health_path).unlink()

    def test_raise_on_failure_file_not_found(self):
        """Test that missing file exits with code 1."""
        with patch('sys.exit') as mock_exit:
            with patch('sys.stderr'):
                raise_on_failure("nonexistent.json")
            mock_exit.assert_called_once_with(1)

    def test_raise_on_failure_invalid_json(self):
        """Test that invalid JSON exits with code 1."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            health_path = f.name

        try:
            with patch('sys.exit') as mock_exit:
                with patch('sys.stderr'):
                    raise_on_failure(health_path)
                mock_exit.assert_called_once_with(1)
        finally:
            Path(health_path).unlink()

    def test_raise_on_failure_missing_passed_field(self):
        """Test that missing 'passed' field exits with code 1."""
        health_data = {
            "warnings": ["Warning message"],
            "failures": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(health_data, f)
            health_path = f.name

        try:
            with patch('sys.exit') as mock_exit:
                with patch('sys.stderr'):
                    raise_on_failure(health_path)
                mock_exit.assert_called_once_with(1)
        finally:
            Path(health_path).unlink()

    def test_raise_on_failure_with_warnings_and_failures(self):
        """Test output formatting with warnings and failures."""
        health_data = {
            "passed": False,
            "warnings": ["Warning 1", "Warning 2"],
            "failures": ["Failure 1", "Failure 2"]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(health_data, f)
            health_path = f.name

        try:
            with patch('sys.exit') as mock_exit:
                with patch('sys.stderr'):
                    with patch('builtins.print') as mock_print:
                        raise_on_failure(health_path)

                        # Check that appropriate messages were printed
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert "🚨 Health check failed" in print_calls
                        assert "Failures: 2" in print_calls
                        assert "Warnings: 2" in print_calls
                        assert "Failure 1" in print_calls
                        assert "Warning 1" in print_calls

                mock_exit.assert_called_once_with(3)
        finally:
            Path(health_path).unlink()

    def test_raise_on_failure_passed_with_warnings(self):
        """Test output formatting when passed=True with warnings."""
        health_data = {
            "passed": True,
            "warnings": ["Warning 1", "Warning 2"],
            "failures": []
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(health_data, f)
            health_path = f.name

        try:
            with patch('sys.exit') as mock_exit:
                with patch('sys.stderr'):
                    with patch('builtins.print') as mock_print:
                        raise_on_failure(health_path)

                        # Check that appropriate messages were printed
                        print_calls = [call[0][0] for call in mock_print.call_args_list]
                        assert "✅ Health check passed" in print_calls
                        assert "Warnings: 2" in print_calls
                        assert "Warning 1" in print_calls

                mock_exit.assert_called_once_with(0)
        finally:
            Path(health_path).unlink()
