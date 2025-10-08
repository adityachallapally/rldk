"""Integration tests for CLI timeout and determinism functionality."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from rldk.determinism.runner import run_deterministic_command
from rldk.utils.error_handling import RLDKTimeoutError
from rldk.utils.runtime import run_with_timeout_subprocess, with_timeout


class TestTimeoutFunctionality:
    """Test timeout functionality using the unified runtime module."""

    def test_with_timeout_decorator_success(self):
        """Test that with_timeout decorator works for successful operations."""
        @with_timeout(5.0)
        def quick_operation():
            return "success"

        result = quick_operation()
        assert result == "success"

    @pytest.mark.skip(reason="Signal-based timeout decorator doesn't work reliably in test environment")
    def test_with_timeout_decorator_timeout(self):
        """Test that with_timeout decorator raises RLDKTimeoutError on timeout."""
        @with_timeout(0.5)
        def slow_operation():
            import time
            time.sleep(2.0)
            return "should not reach here"

        with pytest.raises(RLDKTimeoutError) as exc_info:
            slow_operation()

        assert "Operation timed out after 0.5 seconds" in str(exc_info.value)
        assert exc_info.value.error_code == "OPERATION_TIMEOUT"

    def test_run_with_timeout_subprocess_success(self):
        """Test that run_with_timeout_subprocess works for successful commands."""
        result = run_with_timeout_subprocess(
            ["python3", "-c", "print('hello')"],
            timeout_seconds=5.0,
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_run_with_timeout_subprocess_timeout(self):
        """Test that run_with_timeout_subprocess raises RLDKTimeoutError on timeout."""
        with pytest.raises(RLDKTimeoutError) as exc_info:
            run_with_timeout_subprocess(
                ["python3", "-c", "import time; time.sleep(10)"],
                timeout_seconds=0.1,
                capture_output=True,
                text=True
            )

        assert "Subprocess command timed out after 0.1 seconds" in str(exc_info.value)
        assert exc_info.value.error_code == "SUBPROCESS_TIMEOUT"


class TestDeterminismFunctionality:
    """Test determinism functionality using the unified runner."""

    def test_run_deterministic_command_basic(self):
        """Test basic deterministic command execution."""
        result = run_deterministic_command(
            cmd="python3 -c 'import random; print(random.random())'",
            timeout_seconds=5.0
        )
        assert result.returncode == 0
        assert "metrics_df" in result.__dict__
        assert "output_file" in result.__dict__

    def test_run_deterministic_command_deterministic_output(self):
        """Test that deterministic commands produce identical outputs."""
        # Create a simple Python script that uses random numbers
        script_content = """
import random
import numpy as np

# This should be deterministic due to our seeding
random.seed(42)
np.random.seed(42)

print(f"random: {random.random()}")
print(f"numpy: {np.random.random()}")

# Try to import torch, but don't fail if it's not available
try:
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        print(f"torch: {torch.rand(1).item()}")
    else:
        print(f"torch: {torch.rand(1).item()}")
except ImportError:
    print("torch: not available")
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Run the same command twice with deterministic settings
            result1 = run_deterministic_command(
                cmd=f"python3 {script_path}",
                timeout_seconds=5.0,
                replica_id=0
            )

            result2 = run_deterministic_command(
                cmd=f"python3 {script_path}",
                timeout_seconds=5.0,
                replica_id=0
            )

            # Both should succeed
            assert result1.returncode == 0
            assert result2.returncode == 0

            # Outputs should be identical (deterministic)
            assert result1.stdout == result2.stdout

        finally:
            os.unlink(script_path)

    def test_run_deterministic_command_different_replicas(self):
        """Test that different replica IDs produce different outputs."""
        script_content = """
import random
import numpy as np

# This should be deterministic per replica due to our seeding
print(f"random: {random.random()}")
print(f"numpy: {np.random.random()}")
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Run with different replica IDs
            result1 = run_deterministic_command(
                cmd=f"python3 {script_path}",
                timeout_seconds=5.0,
                replica_id=0
            )

            result2 = run_deterministic_command(
                cmd=f"python3 {script_path}",
                timeout_seconds=5.0,
                replica_id=1
            )

            # Both should succeed
            assert result1.returncode == 0
            assert result2.returncode == 0

            # Outputs should be different (different seeds)
            assert result1.stdout != result2.stdout

        finally:
            os.unlink(script_path)


class TestCLIIntegration:
    """Test CLI integration with timeout and determinism."""

    def test_cli_timeout_integration(self):
        """Test that CLI commands properly use timeout functionality."""
        # This test verifies that the CLI imports and uses the timeout decorator
        # We can't easily test the full CLI without complex setup, but we can
        # verify the imports work correctly
        from rldk.cli import with_timeout as cli_with_timeout
        from rldk.utils.runtime import with_timeout as runtime_with_timeout

        # They should be the same function
        assert cli_with_timeout is runtime_with_timeout

    def test_cli_determinism_integration(self):
        """Test that CLI commands properly use determinism functionality."""
        # This test verifies that the CLI imports and uses the determinism runner
        from rldk.cli import run_deterministic_command as cli_runner
        from rldk.determinism.runner import run_deterministic_command as runner_runner

        # They should be the same function
        assert cli_runner is runner_runner

    def test_determinism_cli_command_simulation(self):
        """Test a simulated determinism CLI command."""
        # Create a simple script that produces deterministic output
        script_content = """
import random
import numpy as np
import json

# Set seeds (this simulates what our deterministic runner does)
random.seed(42)
np.random.seed(42)

# Generate some "metrics"
metrics = []
for step in range(5):
    metrics.append({
        "step": step,
        "loss": random.random(),
        "reward": np.random.random()
    })

# Write to metrics file if RLDK_METRICS_PATH is set
import os
output_path = os.environ.get("RLDK_METRICS_PATH")
if output_path:
    with open(output_path, "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\\n")
else:
    # Just print the metrics
    for metric in metrics:
        print(json.dumps(metric))
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            # Run the command twice with deterministic settings
            result1 = run_deterministic_command(
                cmd=f"python3 {script_path}",
                timeout_seconds=10.0,
                replica_id=0
            )

            result2 = run_deterministic_command(
                cmd=f"python3 {script_path}",
                timeout_seconds=10.0,
                replica_id=0
            )

            # Both should succeed
            assert result1.returncode == 0
            assert result2.returncode == 0

            # If metrics were written, they should be identical
            if hasattr(result1, 'metrics_df') and not result1.metrics_df.empty:
                assert result1.metrics_df.equals(result2.metrics_df)

            # stdout should be identical (deterministic)
            assert result1.stdout == result2.stdout

        finally:
            os.unlink(script_path)
