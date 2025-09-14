"""Git bisect wrapper for finding regressions."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BisectResult:
    """Result of git bisect operation."""

    culprit_sha: str
    iterations: int
    logs_path: str


def bisect_commits(
    good_sha: str,
    bad_sha: str = "HEAD",
    cmd: Optional[str] = None,
    metric: Optional[str] = None,
    cmp: Optional[str] = None,
    window: int = 100,
    shell_predicate: Optional[str] = None,
) -> BisectResult:
    """
    Find regression using git bisect.

    Args:
        good_sha: Known good commit SHA
        bad_sha: Known bad commit SHA (default: HEAD)
        cmd: Command to run for testing
        metric: Metric name to monitor
        cmp: Comparison operator (e.g., "> 0.2")
        window: Window size for metric statistics
        shell_predicate: Shell command that returns non-zero on failure

    Returns:
        BisectResult with culprit commit and iteration count
    """
    # Validate inputs
    if not cmd and not shell_predicate:
        raise ValueError("Either cmd or shell_predicate must be provided")

    if metric and not cmp:
        raise ValueError("cmp must be provided when using metric")

    # Start git bisect
    _start_bisect(good_sha, bad_sha)

    # Create bisect script
    if shell_predicate:
        script_content = _create_shell_bisect_script(shell_predicate)
    else:
        script_content = _create_metric_bisect_script(cmd, metric, cmp, window)

    script_path = Path("bisect_script.sh")
    with open(script_path, "w") as f:
        f.write(script_content)

    script_path.chmod(0o755)

    try:
        # Run git bisect
        result = _run_git_bisect(script_path)

        # Parse results
        culprit_sha, iterations = _parse_bisect_result(result)

        return BisectResult(
            culprit_sha=culprit_sha, iterations=iterations, logs_path="bisect_logs.txt"
        )

    finally:
        # Clean up
        if script_path.exists():
            script_path.unlink()

        # Reset bisect
        subprocess.run(["git", "bisect", "reset"], check=True)


def _start_bisect(good_sha: str, bad_sha: str) -> None:
    """Start git bisect process."""
    try:
        subprocess.run(["git", "bisect", "start"], check=True)
        subprocess.run(["git", "bisect", "bad", bad_sha], check=True)
        subprocess.run(["git", "bisect", "good", good_sha], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to start git bisect: {e}")


def _create_shell_bisect_script(shell_predicate: str) -> str:
    """Create shell script for shell predicate bisect."""
    return f"""#!/bin/bash
set -e

# Run the shell predicate
{shell_predicate}

# If we get here, the predicate succeeded (returned 0)
exit 0
"""


def _create_metric_bisect_script(cmd: str, metric: str, cmp: str, window: int) -> str:
    """Create shell script for metric-based bisect."""
    return f"""#!/bin/bash
set -e

# Run the command
{cmd}

# Check if metrics.jsonl was created
if [ ! -f "metrics.jsonl" ]; then
    echo "No metrics.jsonl found, marking as bad"
    exit 1
fi

# Parse the comparison
python3 -c "
import pandas as pd
import sys

try:
    # Read metrics
    df = pd.read_json('metrics.jsonl', lines=True)

    if '{metric}' not in df.columns:
        print('Metric {metric} not found in data')
        sys.exit(1)

    # Calculate window statistic
    if len(df) >= {window}:
        recent_data = df.tail({window})['{metric}']
    else:
        recent_data = df['{metric}']

    if recent_data.isna().all():
        print('No valid data for metric {metric}')
        sys.exit(1)

    # Calculate mean over window
    metric_value = recent_data.mean()

    # Apply comparison
    cmp_expr = f'{{metric_value}} {cmp}'
    result = eval(cmp_expr)

    if result:
        print(f'Metric {metric} = {{metric_value}} {cmp} - GOOD')
        sys.exit(0)
    else:
        print(f'Metric {metric} = {{metric_value}} {cmp} - BAD')
        sys.exit(1)

except Exception as e:
    print(f'Error evaluating metric: {{e}}')
    sys.exit(1)
"
"""


def _run_git_bisect(script_path: Path) -> subprocess.CompletedProcess:
    """Run git bisect with the given script."""
    try:
        result = subprocess.run(
            ["git", "bisect", "run", str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )
        return result
    except subprocess.TimeoutExpired:
        # Force reset bisect on timeout
        subprocess.run(["git", "bisect", "reset"], check=True)
        raise RuntimeError("Git bisect timed out")


def _parse_bisect_result(result: subprocess.CompletedProcess) -> tuple[str, int]:
    """Parse git bisect output to find culprit and iteration count."""
    output = result.stdout + result.stderr

    # Look for the culprit commit
    culprit_match = None
    for line in output.split("\n"):
        if "is the first bad commit" in line:
            culprit_match = line.split()[0]
            break

    if not culprit_match:
        raise RuntimeError("Could not find culprit commit in bisect output")

    # Count iterations (rough estimate based on output)
    iterations = output.count("Bisecting:") + 1

    return culprit_match, iterations
