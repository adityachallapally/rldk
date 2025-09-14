"""Unified deterministic command runner for RLDK."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..io import read_metrics_jsonl
from ..utils.runtime import run_with_timeout_subprocess


def run_deterministic_command(
    cmd: str,
    env: Optional[Dict[str, str]] = None,
    timeout_seconds: float = 300,
    replica_id: int = 0,
    output_file: Optional[str] = None,
    device: Optional[str] = None
) -> subprocess.CompletedProcess:
    """Run a command with deterministic settings.

    This is the unified function for running deterministic commands in RLDK.
    All other deterministic command runners should use this function.

    Args:
        cmd: Command to run
        env: Environment variables (will be merged with deterministic settings)
        timeout_seconds: Maximum time to wait for command completion
        replica_id: Replica ID for seeding (default: 0)
        output_file: Optional output file for metrics
        device: Device to use (auto-detected if None)

    Returns:
        CompletedProcess with additional attributes:
        - metrics_df: DataFrame with metrics if output_file exists
        - output_file: Path to the output file used
    """
    # Auto-detect device if not provided
    if device is None:
        device = _detect_device()

    # Get deterministic environment
    deterministic_env = _get_deterministic_env(device)

    # Merge with provided environment
    if env:
        deterministic_env.update(env)

    # Create temporary output file if not provided
    if output_file is None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_file = f.name

    # Set the output file in environment for user code to use
    deterministic_env["RLDK_METRICS_PATH"] = output_file
    modified_cmd = cmd

    # Check if the command is a simple Python command that we can wrap
    if modified_cmd.startswith("python3 -c "):
        # Extract the Python code from the command
        python_code = modified_cmd[11:]  # Remove "python3 -c "
        if python_code.startswith("'") and python_code.endswith("'"):
            python_code = python_code[1:-1]  # Remove quotes
        elif python_code.startswith('"') and python_code.endswith('"'):
            python_code = python_code[1:-1]  # Remove quotes

        # Sanitize the Python code to prevent injection
        # Replace any triple quotes that could break out of our wrapper
        python_code = python_code.replace('"""', '\\"\\"\\"').replace("'''", "\\'\\'\\'")

        # Create a deterministic wrapper using safer string formatting
        deterministic_wrapper = f"""
import torch
import random
import numpy as np
import os

# Always set seeds regardless of device
random.seed(42 + {replica_id})
np.random.seed(42 + {replica_id})
torch.manual_seed(42 + {replica_id})

# Set deterministic algorithms
torch.use_deterministic_algorithms(True)

# Set CUDA-specific settings only if CUDA is available
if torch.cuda.is_available():
    torch.cuda.manual_seed(42 + {replica_id})
    torch.cuda.manual_seed_all(42 + {replica_id})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

# Execute the original Python code
{python_code}
"""

        final_cmd = f'python3 -c "{deterministic_wrapper}"'
    else:
        # For other commands, use subprocess with deterministic environment
        final_cmd = modified_cmd

    try:
        # Run the command with timeout
        result = run_with_timeout_subprocess(
            final_cmd,
            timeout_seconds=timeout_seconds,
            shell=True,
            env=deterministic_env,
            capture_output=True,
            text=True
        )

        # Read output file if it exists
        if Path(output_file).exists():
            try:
                df = read_metrics_jsonl(output_file)
                result.metrics_df = df
            except Exception as e:
                print(f"Warning: Could not read metrics from {output_file}: {e}")
                result.metrics_df = pd.DataFrame()
        else:
            result.metrics_df = pd.DataFrame()

        result.output_file = output_file

    except Exception as e:
        # Create a CompletedProcess with error info
        result = subprocess.CompletedProcess(
            args=final_cmd,
            returncode=-1,
            stdout="",
            stderr=str(e)
        )
        result.metrics_df = pd.DataFrame()
        result.output_file = output_file

    return result


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
    if device == "cuda":
        env.update(
            {
                "CUDA_LAUNCH_BLOCKING": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128",
                "TORCH_USE_CUDA_DSA": "1",
            }
        )

    return env
