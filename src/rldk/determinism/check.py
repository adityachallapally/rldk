"""Determinism checking for training runs."""

import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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


def _log_determinism_warning(message: str) -> None:
    """Log a determinism warning if not silenced."""
    if os.getenv("RLDK_SILENCE_DETERMINISM_WARN", "0") != "1":
        warnings.warn(message, UserWarning, stacklevel=3)


def _check_pytorch_cuda_kernels() -> bool:
    """Check if PyTorch CUDA kernels are available for determinism checks."""
    try:
        import torch
        if torch.cuda.is_available():
            # Try to create a simple tensor to verify CUDA kernels work
            x = torch.tensor([1.0], device='cuda')
            _ = torch.nn.functional.relu(x)
            return True
        return False
    except ImportError:
        _log_determinism_warning(
            "Determinism: Skipped PyTorch CUDA kernels check. Install torch>=2.0.0 to enable. "
            "Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress."
        )
        return False
    except Exception:
        # CUDA available but kernels not working properly
        return False


def _check_tensorflow_determinism() -> bool:
    """Check if TensorFlow determinism features are available."""
    try:
        import tensorflow as tf
        # Check if TensorFlow has the required determinism features
        # without modifying global configuration
        return hasattr(tf.config.experimental, 'enable_op_determinism')
    except ImportError:
        _log_determinism_warning(
            "Determinism: Skipped TensorFlow determinism check. Install tensorflow>=2.8.0 to enable. "
            "Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress."
        )
        return False
    except Exception:
        # TensorFlow available but determinism not supported
        return False


def _check_jax_determinism() -> bool:
    """Check if JAX determinism features are available."""
    try:
        import jax
        # Check if JAX is available and has the required config options
        # without modifying global configuration
        return hasattr(jax.config, 'update')
    except ImportError:
        _log_determinism_warning(
            "Determinism: Skipped JAX determinism check. Install jax>=0.4.0 to enable. "
            "Set RLDK_SILENCE_DETERMINISM_WARN=1 to suppress."
        )
        return False
    except Exception:
        # JAX available but determinism not supported
        return False


@dataclass
class DeterminismReport:
    """Report of determinism check results."""

    passed: bool
    culprit: Optional[str]
    fixes: List[str]
    replica_variance: Dict[str, float]
    rng_map: Dict[str, str]
    mismatches: List[Dict[str, Any]]
    dataloader_notes: List[str]
    skipped_checks: List[str]


def check(
    cmd: str,
    compare: List[str],
    steps: Optional[List[int]] = None,
    replicas: int = 5,
    device: Optional[str] = None,
) -> DeterminismReport:
    """
    Check if a training command is deterministic.

    Args:
        cmd: Command to run
        compare: List of metric names to compare
        steps: Specific steps to compare, or None for all
        replicas: Number of replicas to run
        device: Device to use (auto-detected if None)

    Returns:
        DeterminismReport with analysis results
    """
    # Auto-detect device
    if device is None:
        device = _detect_device()

    # Check for available determinism features and track skipped checks
    skipped_checks = []

    if not _check_pytorch_cuda_kernels():
        skipped_checks.append("pytorch_cuda_kernels")

    if not _check_tensorflow_determinism():
        skipped_checks.append("tensorflow_determinism")

    if not _check_jax_determinism():
        skipped_checks.append("jax_determinism")

    # Set deterministic environment
    env = _get_deterministic_env(device)

    # Run multiple replicas
    print(f"Running {replicas} replicas for determinism check...")
    replica_results = []

    for i in range(replicas):
        print(f"Running replica {i+1}/{replicas}...")
        result = _run_deterministic_cmd(cmd, env, replica_id=i)
        replica_results.append(result)

    # Compare results
    mismatches = _compare_replicas(replica_results, compare, steps)

    # Parse stderr for non-deterministic operations
    culprit, fixes = _parse_nondeterministic_ops([r.stderr for r in replica_results])

    # Calculate variance across replicas
    replica_variance = _calculate_replica_variance(replica_results, compare)

    # Detect CUDA availability for truthful RNG map
    cuda_available = _detect_device() == "cuda"

    # Create RNG map
    rng_map = _create_rng_map(env, cuda_available)

    # Detect DataLoader issues
    dataloader_notes = _detect_dataloader_issues(replica_results)

    # Determine if passed
    passed = len(mismatches) == 0

    return DeterminismReport(
        passed=passed,
        culprit=culprit,
        fixes=fixes,
        replica_variance=replica_variance,
        rng_map=rng_map,
        mismatches=mismatches,
        dataloader_notes=dataloader_notes,
        skipped_checks=skipped_checks,
    )


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

    # Set Python deterministic settings (only once)
    # PYTHONHASHSEED ensures consistent hash values across runs
    # This prevents hash randomization from affecting set/dict operations
    env.update(
        {
            "PYTHONHASHSEED": "42",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )

    # Set CUDA-specific settings only when CUDA is actually available
    if device == "cuda" or (device is None and _detect_device() == "cuda"):
        env.update(
            {
                "CUDA_LAUNCH_BLOCKING": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
                "TORCH_USE_CUDA_DSA": "1",
            }
        )

    return env


def _run_deterministic_cmd(
    cmd: str, env: Dict[str, str], replica_id: int
) -> subprocess.CompletedProcess:
    """Run a command with deterministic settings using the unified runner."""
    return run_deterministic_command(
        cmd=cmd,
        env=env,
        timeout_seconds=300,  # 5 minute timeout
        replica_id=replica_id
    )


def _compare_replicas(
    replica_results: List[subprocess.CompletedProcess],
    compare: List[str],
    steps: Optional[List[int]],
) -> List[Dict[str, Any]]:
    """Compare metrics across replicas."""
    mismatches = []

    # Get the first replica as reference
    if not replica_results or replica_results[0].metrics_df.empty:
        return mismatches

    reference_df = replica_results[0].metrics_df

    # Determine steps to compare
    if steps is None:
        steps_to_compare = reference_df["step"].tolist()
    else:
        steps_to_compare = [s for s in steps if s in reference_df["step"].values]

    # Compare each replica against the reference
    for i, result in enumerate(replica_results[1:], 1):
        if result.metrics_df.empty:
            mismatches.append(
                {
                    "replica": i,
                    "issue": "No metrics data available",
                    "details": "Replica failed to produce metrics",
                }
            )
            continue

        df = result.metrics_df

        for step in steps_to_compare:
            if step not in df["step"].values:
                continue

            ref_row = reference_df[reference_df["step"] == step].iloc[0]
            rep_row = df[df["step"] == step].iloc[0]

            for metric in compare:
                if metric not in ref_row or metric not in rep_row:
                    continue

                ref_val = ref_row[metric]
                rep_val = rep_row[metric]

                if pd.isna(ref_val) or pd.isna(rep_val):
                    continue

                # Check for significant difference
                if abs(ref_val - rep_val) > 1e-6:
                    mismatches.append(
                        {
                            "replica": i,
                            "step": step,
                            "metric": metric,
                            "reference_value": ref_val,
                            "replica_value": rep_val,
                            "difference": abs(ref_val - rep_val),
                            "issue": f"Metric {metric} differs at step {step}",
                        }
                    )

    return mismatches


def _parse_nondeterministic_ops(
    stderr_list: List[str],
) -> tuple[Optional[str], List[str]]:
    """Parse stderr for non-deterministic operations."""
    culprit = None
    fixes = []

    # Common non-deterministic operation patterns
    patterns = {
        "cudnn": {
            "pattern": r"cuDNN.*non-deterministic",
            "fix": "Set torch.backends.cudnn.deterministic = True",
        },
        "dropout": {
            "pattern": r"dropout.*non-deterministic",
            "fix": "Use torch.nn.Dropout with deterministic=True",
        },
        "convolution": {
            "pattern": r"convolution.*non-deterministic",
            "fix": "Set torch.backends.cudnn.benchmark = False",
        },
        "reduction": {
            "pattern": r"reduction.*non-deterministic",
            "fix": "Use deterministic reduction operations",
        },
    }

    for stderr in stderr_list:
        for op_name, info in patterns.items():
            if re.search(info["pattern"], stderr, re.IGNORECASE):
                culprit = op_name
                if info["fix"] not in fixes:
                    fixes.append(info["fix"])

    # Add general fixes if no specific culprit found
    if not fixes:
        fixes.extend(
            [
                "Set torch.backends.cudnn.deterministic = True",
                "Set torch.backends.cudnn.benchmark = False",
                "Use torch.manual_seed() consistently",
                "Disable dropout or use deterministic=True",
                "Use deterministic reduction operations",
            ]
        )

    return culprit, fixes


def _calculate_replica_variance(
    replica_results: List[subprocess.CompletedProcess], compare: List[str]
) -> Dict[str, float]:
    """Calculate variance of metrics across replicas."""
    variance = {}

    # Collect all metrics data
    all_dfs = []
    for result in replica_results:
        if not result.metrics_df.empty:
            all_dfs.append(result.metrics_df)

    if len(all_dfs) < 2:
        return variance

    # Concatenate all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Calculate variance for each metric
    for metric in compare:
        if metric in combined_df.columns:
            values = combined_df[metric].dropna()
            if len(values) > 1:
                variance[metric] = float(values.var())
            else:
                variance[metric] = 0.0

    return variance


def _create_rng_map(
    env: Dict[str, str], cuda_available: bool = False
) -> Dict[str, str]:
    """Create a map of RNG settings that were actually enforced."""
    rng_map = {}

    # Python settings
    rng_map["python_hash_seed"] = env.get("PYTHONHASHSEED", "Not set")
    rng_map["omp_threads"] = env.get("OMP_NUM_THREADS", "Not set")
    rng_map["mkl_threads"] = env.get("MKL_NUM_THREADS", "Not set")
    rng_map["numexpr_threads"] = env.get("NUMEXPR_NUM_THREADS", "Not set")

    # PyTorch settings
    rng_map["torch_deterministic"] = "True (enforced)"
    rng_map["torch_seed"] = "Set per replica (42 + replica_id)"
    rng_map["numpy_seed"] = "Set per replica (42 + replica_id)"
    rng_map["random_seed"] = "Set per replica (42 + replica_id)"

    # CUDA settings (based on actual availability, not env key presence)
    if cuda_available:
        rng_map["cuda_launch_blocking"] = env.get("CUDA_LAUNCH_BLOCKING", "Not set")
        rng_map["cublas_workspace"] = env.get("CUBLAS_WORKSPACE_CONFIG", "Not set")
        rng_map["cudnn_deterministic"] = "True (enforced)"
        rng_map["cudnn_benchmark"] = "False (enforced)"
        rng_map["tf32_disabled"] = "True (enforced)"
        rng_map["cuda_seed"] = "Set per replica (42 + replica_id)"
    else:
        rng_map["cuda_launch_blocking"] = "N/A (CPU only)"
        rng_map["cublas_workspace"] = "N/A (CPU only)"
        rng_map["cudnn_deterministic"] = "N/A (CPU only)"
        rng_map["cudnn_benchmark"] = "N/A (CPU only)"
        rng_map["tf32_disabled"] = "N/A (CPU only)"
        rng_map["cuda_seed"] = "N/A (CPU only)"

    return rng_map


def _detect_dataloader_issues(
    replica_results: List[subprocess.CompletedProcess],
) -> List[str]:
    """Detect potential DataLoader-related determinism issues."""
    issues = []

    # Check for obvious batch order mismatches
    if len(replica_results) >= 2:
        for i, result in enumerate(replica_results[1:], 1):
            if not result.metrics_df.empty and not replica_results[0].metrics_df.empty:
                ref_df = replica_results[0].metrics_df
                rep_df = result.metrics_df

                # Check if the first few steps have identical values (suggesting same batch order)
                # This is a simple heuristic - identical early steps might indicate same data order
                if len(ref_df) >= 3 and len(rep_df) >= 3:
                    # Use common columns between the two dataframes
                    # Get common columns deterministically (sorted for consistency)
                    ref_cols = sorted(ref_df.columns)
                    rep_cols = sorted(rep_df.columns)
                    common_cols = [col for col in ref_cols if col in rep_cols]
                    # Filter out non-numeric columns
                    numeric_cols = [
                        col
                        for col in common_cols
                        if col != "step" and pd.api.types.is_numeric_dtype(ref_df[col])
                    ]

                    if numeric_cols:
                        ref_early = ref_df.head(3)[numeric_cols].values
                        rep_early = rep_df.head(3)[numeric_cols].values

                        if np.array_equal(ref_early, rep_early):
                            issues.append(
                                f"Replica {i} shows identical early metrics - may need worker_init_fn and shuffle=False"
                            )

    # Check stderr for DataLoader warnings
    for i, result in enumerate(replica_results):
        if result.stderr and "DataLoader" in result.stderr:
            if "num_workers" in result.stderr and "shuffle" in result.stderr:
                issues.append(
                    f"Replica {i} uses DataLoader - ensure worker_init_fn and shuffle=False for determinism"
                )

    return issues
