"""Seeded replay utility for training runs."""

import json
import logging
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from rldk.determinism.check import _detect_device, _get_deterministic_env
from rldk.ingest import ingest_runs

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ReplayResult:
    """Result of a replay command execution."""

    success: bool
    return_code: int
    stdout: str
    stderr: str
    metrics_data: pd.DataFrame
    error_message: Optional[str] = None


@dataclass
class ReplayReport:
    """Report of seeded replay results."""

    passed: bool
    original_seed: int
    replay_seed: int
    metrics_compared: List[str]
    tolerance: float
    mismatches: List[Dict[str, Any]]
    original_metrics: pd.DataFrame
    replay_metrics: pd.DataFrame
    comparison_stats: Dict[str, Dict[str, float]]
    replay_command: str
    replay_duration: float


def replay(
    run_path: Union[str, Path],
    training_command: str,
    metrics_to_compare: List[str],
    tolerance: float = 0.01,
    max_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> ReplayReport:
    """
    Replay a training run with the original seed and verify metrics match.

    Args:
        run_path: Path to the original training run data
        training_command: Command to run for replay (should accept --seed argument)
        metrics_to_compare: List of metric names to compare
        tolerance: Tolerance for metric differences (relative)
        max_steps: Maximum number of steps to replay (None for all)
        output_dir: Directory to save replay results (None for temp)
        device: Device to use (auto-detected if None)

    Returns:
        ReplayReport with comparison results
    """
    # Load original run data
    print(f"Loading original run from: {run_path}")
    original_df = ingest_runs(run_path)

    # Extract original seed
    if "seed" not in original_df.columns:
        raise ValueError("Original run data must contain 'seed' column")

    original_seed = original_df["seed"].iloc[0]
    if pd.isna(original_seed):
        raise ValueError("Original run seed is missing or NaN")

    print(f"Original run used seed: {original_seed}")

    # Limit steps if specified
    if max_steps is not None:
        original_df = original_df[original_df["step"] <= max_steps].copy()
        print(f"Limiting replay to {max_steps} steps")

    # Prepare replay command
    replay_command = _prepare_replay_command(training_command, original_seed)
    print(f"Replay command: {replay_command}")

    # Set up output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="replay_")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run replay
    print("Running replay...")
    replay_start_time = pd.Timestamp.now()

    replay_result = _run_replay(replay_command, output_path, device)
    replay_duration = (pd.Timestamp.now() - replay_start_time).total_seconds()

    if not replay_result.success:
        error_msg = f"Replay failed: {replay_result.error_message}"
        if replay_result.stderr:
            error_msg += f"\nstderr: {replay_result.stderr}"
        print(error_msg)
        raise RuntimeError(error_msg)

    replay_df = replay_result.metrics_data
    print(f"Replay completed in {replay_duration:.2f} seconds")

    # Compare metrics
    print("Comparing metrics...")
    mismatches, comparison_stats = _compare_metrics(
        original_df, replay_df, metrics_to_compare, tolerance
    )

    # Determine if replay passed
    passed = len(mismatches) == 0

    # Create report
    report = ReplayReport(
        passed=passed,
        original_seed=original_seed,
        replay_seed=original_seed,  # Same seed used for replay
        metrics_compared=metrics_to_compare,
        tolerance=tolerance,
        mismatches=mismatches,
        original_metrics=original_df,
        replay_metrics=replay_df,
        comparison_stats=comparison_stats,
        replay_command=replay_command,
        replay_duration=replay_duration,
    )

    # Save replay results
    _save_replay_results(report, output_path)

    return report


def _prepare_replay_command(command: str, seed: int) -> str:
    """Prepare the replay command with seed argument."""
    # Check if command already has --seed
    if "--seed" in command:
        # Replace first existing seed only
        import re

        command = re.sub(r"--seed\s+\d+", f"--seed {seed}", command, count=1)
    else:
        # Add seed argument
        command = f"{command} --seed {seed}"

    return command


def _run_replay(command: str, output_path: Path, device: Optional[str]) -> ReplayResult:
    """Run the replay command and capture metrics."""
    # Auto-detect device
    if device is None:
        device = _detect_device()

    # Set deterministic environment
    env = _get_deterministic_env(device)

    # Create temporary output file for metrics
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        metrics_file = f.name

    # Set environment variable for metrics output
    env["RLDK_METRICS_PATH"] = metrics_file

    # Ensure temp file cleanup in all code paths
    try:
        # Parse command into argv for secure execution
        try:
            argv = shlex.split(command)
        except ValueError as e:
            error_msg = f"Failed to parse command '{command}': {e}"
            logger.error(error_msg)
            return ReplayResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=error_msg,
                metrics_data=pd.DataFrame(),
                error_message=error_msg
            )

        # Run the command securely
        logger.info(f"Executing: {command}")
        try:
            result = subprocess.run(
                argv,
                shell=False,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                check=False  # We'll handle errors explicitly
            )
        except subprocess.TimeoutExpired as e:
            error_msg = f"Replay command timed out after 3600 seconds: {e}"
            logger.error(error_msg)
            return ReplayResult(
                success=False,
                return_code=-1,
                stdout=e.stdout or "",
                stderr=e.stderr or "",
                metrics_data=pd.DataFrame(),
                error_message=error_msg
            )
        except Exception as e:
            error_msg = f"Failed to execute replay command: {e}"
            logger.error(error_msg)
            return ReplayResult(
                success=False,
                return_code=-1,
                stdout="",
                stderr=str(e),
                metrics_data=pd.DataFrame(),
                error_message=error_msg
            )

        # Check if command succeeded
        if result.returncode != 0:
            error_msg = f"Replay command failed with return code {result.returncode}"
            logger.error(f"{error_msg}\nstdout: {result.stdout}\nstderr: {result.stderr}")
            return ReplayResult(
                success=False,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                metrics_data=pd.DataFrame(),
                error_message=error_msg
            )

        # Load replay metrics
        metrics_file_path = metrics_file
        if not os.path.exists(metrics_file_path):
            # Try to find metrics in the output directory
            metrics_files = list(output_path.glob("*.jsonl"))
            if metrics_files:
                metrics_file_path = str(metrics_files[0])
            else:
                error_msg = "No metrics file found after replay"
                logger.error(error_msg)
                return ReplayResult(
                    success=False,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    metrics_data=pd.DataFrame(),
                    error_message=error_msg
                )

        try:
            replay_df = pd.read_json(metrics_file_path, lines=True)
        except Exception as e:
            logger.warning(f"Failed to load replay metrics from file: {e}")
            # Try to parse from stdout if available
            if result.stdout:
                logger.info("Attempting to parse metrics from stdout...")
                try:
                    replay_df = _parse_metrics_from_stdout(result.stdout)
                except Exception as parse_error:
                    error_msg = f"Could not load replay metrics from file or stdout: {parse_error}"
                    logger.error(error_msg)
                    return ReplayResult(
                        success=False,
                        return_code=result.returncode,
                        stdout=result.stdout,
                        stderr=result.stderr,
                        metrics_data=pd.DataFrame(),
                        error_message=error_msg
                    )
            else:
                error_msg = "Could not load replay metrics and no stdout available"
                logger.error(error_msg)
                return ReplayResult(
                    success=False,
                    return_code=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    metrics_data=pd.DataFrame(),
                    error_message=error_msg
                )

        return ReplayResult(
            success=True,
            return_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            metrics_data=replay_df,
            error_message=None
        )

    finally:
        # Clean up temp file with explicit error handling - this runs in ALL code paths
        _cleanup_temp_file(metrics_file)


def _cleanup_temp_file(file_path: str) -> None:
    """Clean up temporary file with explicit error handling."""
    try:
        os.unlink(file_path)
        logger.debug(f"Successfully cleaned up temp file: {file_path}")
    except FileNotFoundError:
        # File was already deleted or never created - this is not an error
        logger.debug(f"Temp file not found during cleanup (already deleted): {file_path}")
    except PermissionError as e:
        # Permission denied - log warning with actionable message
        logger.warning(
            f"Permission denied cleaning up temp file {file_path}: {e}. "
            f"Please check file permissions or run with appropriate privileges."
        )
    except OSError as e:
        # Other OS errors - log warning with actionable message
        logger.warning(
            f"Failed to clean up temp file {file_path}: {e}. "
            f"File may need manual cleanup."
        )


def _parse_metrics_from_stdout(stdout: str) -> pd.DataFrame:
    """Parse metrics from command stdout as fallback."""
    # This is a fallback method - in practice, the training script should
    # output metrics to the file specified by RLDK_METRICS_PATH

    # Look for JSON lines in stdout
    lines = stdout.strip().split("\n")
    metrics_lines = []

    for line in lines:
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                json.loads(line)
                metrics_lines.append(line)
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue

    if not metrics_lines:
        raise RuntimeError("No valid JSON metrics found in stdout")

    # Create DataFrame from parsed lines
    metrics_data = [json.loads(line) for line in metrics_lines]
    df = pd.DataFrame(metrics_data)

    # Ensure required columns exist
    required_cols = [
        "step",
        "phase",
        "reward_mean",
        "reward_std",
        "kl_mean",
        "entropy_mean",
        "clip_frac",
        "grad_norm",
        "lr",
        "loss",
        "tokens_in",
        "tokens_out",
        "wall_time",
        "seed",
        "run_id",
        "git_sha",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    return df[required_cols]


def _compare_metrics(
    original_df: pd.DataFrame,
    replay_df: pd.DataFrame,
    metrics: List[str],
    tolerance: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    """Compare metrics between original and replay runs."""
    mismatches = []
    comparison_stats = {}

    # Ensure both DataFrames have the same step range
    common_steps = set(original_df["step"]) & set(replay_df["step"])
    if not common_steps:
        raise ValueError("No common steps between original and replay runs")

    print(f"Comparing {len(common_steps)} common steps")

    for metric in metrics:
        if metric not in original_df.columns or metric not in replay_df.columns:
            print(f"Warning: Metric '{metric}' not found in both runs")
            continue

        metric_mismatches = []
        metric_stats = {
            "max_diff": 0.0,
            "max_diff_step": None,
            "mean_diff": 0.0,
            "std_diff": 0.0,
            "tolerance_violations": 0,
        }

        diffs = []

        for step in sorted(common_steps):
            orig_val = original_df[original_df["step"] == step][metric].iloc[0]
            replay_val = replay_df[replay_df["step"] == step][metric].iloc[0]

            if pd.isna(orig_val) or pd.isna(replay_val):
                continue

            # Calculate relative difference
            if abs(orig_val) > 1e-8:  # Avoid division by zero
                rel_diff = abs(replay_val - orig_val) / abs(orig_val)
            else:
                rel_diff = abs(replay_val - orig_val)

            diffs.append(rel_diff)

            # Check tolerance
            if rel_diff > tolerance:
                metric_stats["tolerance_violations"] += 1
                metric_mismatches.append(
                    {
                        "step": step,
                        "metric": metric,
                        "original": orig_val,
                        "replay": replay_val,
                        "absolute_diff": abs(replay_val - orig_val),
                        "relative_diff": rel_diff,
                        "tolerance": tolerance,
                    }
                )

        if diffs:
            diffs = np.array(diffs)
            metric_stats["max_diff"] = float(np.max(diffs))
            metric_stats["max_diff_step"] = int(sorted(common_steps)[np.argmax(diffs)])
            metric_stats["mean_diff"] = float(np.mean(diffs))
            metric_stats["std_diff"] = float(np.std(diffs))

        comparison_stats[metric] = metric_stats

        if metric_mismatches:
            mismatches.extend(metric_mismatches)
            print(f"  {metric}: {len(metric_mismatches)} tolerance violations")
        else:
            print(f"  {metric}: ✅ within tolerance")

    return mismatches, comparison_stats


def _save_replay_results(report: ReplayReport, output_path: Path):
    """Save replay results to output directory."""
    # Save replay metrics
    replay_file = output_path / "replay_metrics.jsonl"
    report.replay_metrics.to_json(replay_file, orient="records", lines=True)

    # Save comparison report
    comparison_file = output_path / "replay_comparison.json"

    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    comparison_data = {
        "passed": report.passed,
        "original_seed": convert_numpy_types(report.original_seed),
        "replay_seed": convert_numpy_types(report.replay_seed),
        "metrics_compared": report.metrics_compared,
        "tolerance": report.tolerance,
        "tolerance_violations": len(report.mismatches),
        "comparison_stats": convert_numpy_types(report.comparison_stats),
        "replay_command": report.replay_command,
        "replay_duration": report.replay_duration,
    }

    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2)

    # Save mismatches if any
    if report.mismatches:
        mismatches_file = output_path / "replay_mismatches.json"
        with open(mismatches_file, "w") as f:
            json.dump(report.mismatches, f, indent=2)

    print(f"Replay results saved to: {output_path}")


def _detect_device() -> str:
    """Auto-detect available device."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"
