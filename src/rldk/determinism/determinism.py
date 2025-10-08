"""Determinism checking for training runs."""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..io import read_metrics_jsonl
from .runner import run_deterministic_command


def _deduplicate_deterministic(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order deterministically."""
    unique_items = []
    seen = set()
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


@dataclass
class DeterminismReport:
    """Report of determinism check results."""

    passed: bool
    mismatches: List[Dict[str, Any]]
    enforced_settings: Dict[str, str]
    culprit: Optional[str]
    fixes: List[str]
    report_path: str


def check_determinism(
    cmd: str,
    compare: List[str],
    steps: Optional[List[int]] = None,
    stride: int = 50,
    device: Optional[str] = None,
) -> DeterminismReport:
    """
    Check if a training command is deterministic.

    Args:
        cmd: Command to run
        compare: List of metric names or tensor tags to compare
        steps: Specific steps to compare, or None for stride-based
        stride: Step interval for comparison if steps not specified
        device: Device to use (auto-detected if None)

    Returns:
        DeterminismReport with analysis results
    """
    # Auto-detect device
    if device is None:
        device = _detect_device()

    # Set deterministic environment
    env = _get_deterministic_env(device)

    # Run command twice
    print("Running first execution...")
    result1 = _run_deterministic_cmd(cmd, env)

    print("Running second execution...")
    result2 = _run_deterministic_cmd(cmd, env)

    # Compare results
    mismatches = _compare_executions(result1, result2, compare, steps, stride)

    # Parse stderr for non-deterministic operations
    culprit, fixes = _parse_nondeterministic_ops(result1.stderr + result2.stderr)

    # Determine if passed
    passed = len(mismatches) == 0

    # Create report
    report = DeterminismReport(
        passed=passed,
        mismatches=mismatches,
        enforced_settings=env,
        culprit=culprit,
        fixes=fixes,
        report_path="determinism_report.md",
    )

    return report


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


def _get_deterministic_env(device: str) -> Dict[str, str]:
    """Get environment variables for deterministic execution."""
    env = os.environ.copy()

    # PyTorch deterministic settings
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"

    # Python hash seed for deterministic behavior
    # This prevents hash randomization from affecting set/dict operations
    env["PYTHONHASHSEED"] = "42"

    # PyTorch deterministic settings
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    if device == "cuda":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["CUDA_LAUNCH_BLOCKING"] = "1"

    return env


def _run_deterministic_cmd(
    cmd: str, env: Dict[str, str]
) -> subprocess.CompletedProcess:
    """Run command with deterministic environment using the unified runner."""
    return run_deterministic_command(
        cmd=cmd,
        env=env,
        timeout_seconds=300,  # 5 minute timeout
        replica_id=0
    )


def _compare_executions(
    result1: subprocess.CompletedProcess,
    result2: subprocess.CompletedProcess,
    compare: List[str],
    steps: Optional[List[int]],
    stride: int,
) -> List[Dict[str, Any]]:
    """Compare two execution results."""
    mismatches = []

    # Try to read metrics from both runs
    try:
        # Look for metrics.jsonl files
        metrics_files = list(Path(".").glob("metrics.jsonl"))
        if len(metrics_files) >= 2:
            df1 = read_metrics_jsonl(metrics_files[0])
            df2 = read_metrics_jsonl(metrics_files[1])

            # Compare specified metrics
            for metric in compare:
                if metric in df1.columns and metric in df2.columns:
                    mismatch = _compare_metric(df1, df2, metric, steps, stride)
                    if mismatch:
                        mismatches.append(mismatch)

        # Compare return codes
        if result1.returncode != result2.returncode:
            mismatches.append(
                {
                    "metric": "return_code",
                    "details": f"Return codes differ: {result1.returncode} vs {result2.returncode}",
                }
            )

        # Compare stdout length (rough proxy for consistency)
        if abs(len(result1.stdout) - len(result2.stdout)) > 100:
            mismatches.append(
                {
                    "metric": "stdout_consistency",
                    "details": f"Output lengths differ significantly: {len(result1.stdout)} vs {len(result2.stdout)}",
                }
            )

    except Exception as e:
        mismatches.append(
            {"metric": "comparison_error", "details": f"Error during comparison: {e}"}
        )

    return mismatches


def _compare_metric(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    metric: str,
    steps: Optional[List[int]],
    stride: int,
) -> Optional[Dict[str, Any]]:
    """Compare a specific metric between two dataframes."""
    if steps is None:
        # Use stride-based comparison
        steps = list(range(0, min(len(df1), len(df2)), stride))

    for step in steps:
        if step < len(df1) and step < len(df2):
            val1 = df1.iloc[step][metric]
            val2 = df2.iloc[step][metric]

            if pd.notna(val1) and pd.notna(val2):
                # Check for significant difference (1% tolerance)
                if abs(val1 - val2) / max(abs(val1), 1e-8) > 0.01:
                    return {
                        "metric": metric,
                        "details": f"Step {step}: {val1} vs {val2} (diff: {abs(val1 - val2)})",
                    }

    return None


def _parse_nondeterministic_ops(stderr: str) -> tuple[Optional[str], List[str]]:
    """Parse stderr for known non-deterministic operations."""
    # Known non-deterministic operations and their fixes
    nondet_patterns = {
        r"aten::rand": [
            "Use torch.manual_seed() and torch.cuda.manual_seed() before random operations",
            "Set torch.use_deterministic_algorithms(True)",
        ],
        r"aten::multinomial": [
            "Use torch.manual_seed() before multinomial sampling",
            "Consider using torch.distributions.Categorical for deterministic sampling",
        ],
        r"aten::dropout": [
            "Set torch.backends.cudnn.deterministic = True",
            "Use torch.nn.Dropout(p=0.0) for evaluation",
        ],
        r"aten::max_pool": [
            "Set torch.backends.cudnn.deterministic = True",
            "Use torch.backends.cudnn.benchmark = False",
        ],
        r"aten::convolution": [
            "Set torch.backends.cudnn.deterministic = True",
            "Use torch.backends.cudnn.benchmark = False",
        ],
    }

    culprit = None
    fixes = []

    for pattern, pattern_fixes in nondet_patterns.items():
        if re.search(pattern, stderr):
            culprit = pattern
            fixes.extend(pattern_fixes)

    # Add general fixes
    fixes.extend(
        [
            "Set environment variable CUBLAS_WORKSPACE_CONFIG=:4096:8",
            "Set environment variable CUDA_LAUNCH_BLOCKING=1",
            "Use torch.use_deterministic_algorithms(True)",
            "Set torch.backends.cudnn.deterministic = True",
            "Set torch.backends.cudnn.benchmark = False",
        ]
    )

    return culprit, _deduplicate_deterministic(fixes)
