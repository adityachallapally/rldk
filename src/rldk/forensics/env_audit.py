"""Environment audit for determinism and reproducibility."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


def audit_environment(repo_or_run: str) -> Tuple[Dict[str, Any], str]:
    """Audit environment for determinism and reproducibility."""
    repo_or_run = Path(repo_or_run)

    # Collect environment information
    env_info = collect_env_info()

    # Check for nondeterminism hints
    nondeterminism_hints = check_nondeterminism_hints(env_info)

    # Create determinism card
    determinism_card = {
        "version": "1",
        "rng": {
            "python": env_info.get("python_seed"),
            "torch": env_info.get("torch_seed"),
        },
        "flags": {
            "cudnn_deterministic": env_info.get("cudnn_deterministic", False),
            "cudnn_benchmark": env_info.get("cudnn_benchmark", True),
            "tokenizers_parallelism": env_info.get("tokenizers_parallelism"),
        },
        "nondeterminism_hints": nondeterminism_hints,
        "pass": len(nondeterminism_hints) == 0,
    }

    # Generate lock file content
    lock_content = generate_lock_content(env_info)

    return determinism_card, lock_content


def collect_env_info() -> Dict[str, Any]:
    """Collect environment information."""
    info = {}

    # Python and PyTorch versions
    info["python_version"] = sys.version
    info["torch_version"] = torch.__version__

    # CUDA information
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()

    # Environment variables
    info["tokenizers_parallelism"] = os.environ.get("TOKENIZERS_PARALLELISM")

    # Convert environment variables to boolean flags
    cudnn_deterministic_env = os.environ.get("CUDNN_DETERMINISTIC")
    info[
        "cudnn_deterministic"
    ] = cudnn_deterministic_env is not None and cudnn_deterministic_env.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    cudnn_benchmark_env = os.environ.get("CUDNN_BENCHMARK")
    info[
        "cudnn_benchmark"
    ] = cudnn_benchmark_env is None or cudnn_benchmark_env.lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    # PyTorch settings
    info["torch_cudnn_deterministic"] = torch.backends.cudnn.deterministic
    info["torch_cudnn_benchmark"] = torch.backends.cudnn.benchmark

    # Seeds (if available)
    info["python_seed"] = None  # Would need to be set by user
    info["torch_seed"] = None  # Would need to be set by user

    # Locale and timezone
    info["locale"] = os.environ.get("LANG", "unknown")
    info["timezone"] = os.environ.get("TZ", "unknown")

    return info


def check_nondeterminism_hints(env_info: Dict[str, Any]) -> List[str]:
    """Check for potential nondeterminism issues."""
    hints = []

    # Check tokenizers parallelism
    if env_info.get("tokenizers_parallelism") != "false":
        hints.append("TOKENIZERS_PARALLELISM not set to 'false'")

    # Check CUDNN settings
    if not env_info.get("torch_cudnn_deterministic"):
        hints.append("CUDNN deterministic mode not enabled")

    if env_info.get("torch_cudnn_benchmark"):
        hints.append("CUDNN benchmark mode enabled (can cause nondeterminism)")

    # Check CUDA availability
    if env_info.get("cuda_available"):
        hints.append("CUDA available (can introduce nondeterminism)")

    # Check for missing seeds
    if env_info.get("python_seed") is None:
        hints.append("Python random seed not set")

    if env_info.get("torch_seed") is None:
        hints.append("PyTorch random seed not set")

    return hints


def generate_lock_content(env_info: Dict[str, Any]) -> str:
    """Generate lock file content."""
    lines = []

    lines.append("# RL Debug Kit Environment Lock")
    lines.append(f"# Generated: {__import__('datetime').datetime.now().isoformat()}")
    lines.append("")

    lines.append("## Environment")
    lines.append(f"Python: {env_info['python_version']}")
    lines.append(f"PyTorch: {env_info['torch_version']}")
    lines.append(f"CUDA Available: {env_info['cuda_available']}")

    if env_info.get("cuda_version"):
        lines.append(f"CUDA Version: {env_info['cuda_version']}")

    if env_info.get("cudnn_version"):
        lines.append(f"CUDNN Version: {env_info['cudnn_version']}")

    lines.append("")

    lines.append("## Environment Variables")
    for key, value in env_info.items():
        if key in [
            "tokenizers_parallelism",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "locale",
            "timezone",
        ]:
            lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("## PyTorch Settings")
    lines.append(f"CUDNN Deterministic: {env_info['torch_cudnn_deterministic']}")
    lines.append(f"CUDNN Benchmark: {env_info['torch_cudnn_benchmark']}")

    lines.append("")

    # Try to get pip freeze output
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            lines.append("## Dependencies")
            lines.append(result.stdout)
    except Exception:
        lines.append("## Dependencies")
        lines.append("# Failed to capture pip freeze output")

    return "\n".join(lines)
