"""Test forensics CLI commands."""

import subprocess


def test_cli_help():
    """Test that CLI help works."""
    result = subprocess.run(["rldk", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "rldk" in result.stdout


def test_env_audit_help():
    """Test env-audit help."""
    result = subprocess.run(
        ["rldk", "env-audit", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "env-audit" in result.stdout


def test_log_scan_help():
    """Test log-scan help."""
    result = subprocess.run(
        ["rldk", "log-scan", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "log-scan" in result.stdout


def test_diff_ckpt_help():
    """Test diff-ckpt help."""
    result = subprocess.run(
        ["rldk", "diff-ckpt", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "diff-ckpt" in result.stdout


def test_reward_drift_help():
    """Test reward-drift help."""
    result = subprocess.run(
        ["rldk", "reward-drift", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "reward-drift" in result.stdout


def test_doctor_help():
    """Test doctor help."""
    result = subprocess.run(
        ["rldk", "doctor", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "doctor" in result.stdout


def test_compare_runs_help():
    """Test compare-runs help."""
    result = subprocess.run(
        ["rldk", "compare-runs", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "compare-runs" in result.stdout
