"""Determinism card generation for RL training runs."""

import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..io.event_schema import Event

# Import removed - using simplified check instead


def generate_determinism_card(
    events: List[Event], run_path: str, output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a determinism card for a training run.

    Args:
        events: List of Event objects from the training run
        run_path: Path to the training run
        output_dir: Directory to save the card (defaults to runs/run_id/rldk_cards)

    Returns:
        Dictionary containing the determinism card data
    """
    # Extract run_id from events
    run_id = events[0].model_info["run_id"] if events else "unknown"

    # Set output directory
    if output_dir is None:
        output_dir = f"runs/{run_id}/rldk_cards"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Perform basic determinism check (simplified for card generation)
    determinism_result = _basic_determinism_check(run_path)

    # Analyze RNG consistency across events
    rng_analysis = _analyze_rng_consistency(events)

    # Check for non-deterministic patterns
    nondeterminism_hints = _detect_nondeterminism_patterns(events)

    # Create the card data
    card_data = {
        "version": "1.0",
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "passed": bool(determinism_result["pass"]),
        "replicas": int(
            len(
                set(
                    event.rng.get("seed")
                    for event in events
                    if event.rng.get("seed") is not None
                )
            )
        ),
        "metrics_compared": list(events[0].metrics.keys()) if events else [],
        "replica_variance": _calculate_replica_variance(events),
        "rng_map": {
            "python_hash_seed": (
                events[0].rng.get("python_hash_seed") if events else None
            ),
            "torch_deterministic": bool(
                determinism_result["flags"]["cudnn_deterministic"]
            ),
            "torch_seed": (
                "Set per replica"
                if rng_analysis["multiple_seeds"]
                else (
                    str(events[0].rng.get("torch_seed"))
                    if events and events[0].rng.get("torch_seed") is not None
                    else None
                )
            ),
            "numpy_seed": (
                "Set per replica"
                if rng_analysis["multiple_seeds"]
                else (
                    str(events[0].rng.get("numpy_seed"))
                    if events and events[0].rng.get("numpy_seed") is not None
                    else None
                )
            ),
            "random_seed": (
                "Set per replica"
                if rng_analysis["multiple_seeds"]
                else (
                    str(events[0].rng.get("random_seed"))
                    if events and events[0].rng.get("random_seed") is not None
                    else None
                )
            ),
        },
        "mismatches": _find_metric_mismatches(events),
        "fixes": _generate_determinism_fixes(determinism_result, nondeterminism_hints),
        "nondeterminism_hints": nondeterminism_hints,
        "flags": determinism_result["flags"],
    }

    # Save JSON card
    json_path = output_path / "determinism_card.json"
    with open(json_path, "w") as f:
        # Convert numpy types to native Python types for JSON serialization
        json.dump(
            card_data,
            f,
            indent=2,
            default=lambda x: float(x) if hasattr(x, "item") else x,
        )

    # Generate and save PNG visualization
    png_path = output_path / "determinism_card.png"
    _generate_determinism_visualization(card_data, png_path)

    return card_data


def _basic_determinism_check(run_path: str) -> Dict[str, Any]:
    """Perform a comprehensive determinism check."""
    # Check environment variables
    env_vars = {
        "cudnn_deterministic": os.environ.get("CUDNN_DETERMINISTIC", "false").lower() == "true",
        "cudnn_benchmark": os.environ.get("CUDNN_BENCHMARK", "true").lower() == "true",
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "torch_deterministic": os.environ.get("TORCH_DETERMINISTIC", "false").lower() == "true",
        "torch_use_deterministic_algorithms": os.environ.get("TORCH_USE_DETERMINISTIC_ALGORITHMS", "false").lower() == "true",
    }

    # Check PyTorch backend settings
    try:
        import torch
        env_vars["torch_cudnn_deterministic"] = torch.backends.cudnn.deterministic
        env_vars["torch_cudnn_benchmark"] = torch.backends.cudnn.benchmark
        env_vars["torch_use_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except ImportError:
        env_vars["torch_cudnn_deterministic"] = False
        env_vars["torch_cudnn_benchmark"] = True
        env_vars["torch_use_deterministic_algorithms"] = False

    # Check for common non-deterministic patterns in the run path
    nondeterminism_hints = []
    
    # Check for multiple runs that could be compared
    run_dir = Path(run_path)
    if run_dir.exists():
        # Look for multiple run directories that could indicate replicas
        parent_dir = run_dir.parent
        if parent_dir.exists():
            run_dirs = [d for d in parent_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
            if len(run_dirs) > 1:
                # Check if we can compare metrics across runs
                nondeterminism_hints.append(f"Found {len(run_dirs)} potential replica runs for comparison")
        
        # Check for log files that might indicate non-deterministic behavior
        log_files = list(run_dir.glob("*.log")) + list(run_dir.glob("**/*.log"))
        
        # Import settings
        from ..config.settings import get_settings
        settings = get_settings()
        
        # Limit number of files processed to prevent performance issues
        max_files = settings.file.max_files_to_process
        if len(log_files) > max_files:
            log_files = log_files[:max_files]
            nondeterminism_hints.append(f"Processing only first {max_files} log files (found {len(log_files)})")
        
        for log_file in log_files:
            try:
                # Check file size first to avoid reading huge files
                file_size = log_file.stat().st_size
                max_file_size = settings.file.max_file_size_mb * 1024 * 1024
                if file_size > max_file_size:
                    nondeterminism_hints.append(f"Skipping large log file {log_file.name} ({file_size // (1024*1024)}MB)")
                    continue
                
                # Additional safety checks
                if not log_file.is_file() or not log_file.exists():
                    continue
                
                # Read file with proper encoding handling and size limits
                with open(log_file, 'r', encoding=settings.file.encoding, errors='ignore') as f:
                    # Read only first portion to avoid memory issues
                    max_read_size = settings.file.max_read_size_kb * 1024
                    content = f.read(max_read_size).lower()
                    
                    # Check for non-deterministic behavior indicators
                    if "non-deterministic" in content or "nondeterministic" in content:
                        nondeterminism_hints.append(f"Non-deterministic behavior detected in {log_file.name}")
                    
                    # Check for RNG-related warnings
                    if "warning" in content and ("seed" in content or "random" in content):
                        nondeterminism_hints.append(f"RNG warnings found in {log_file.name}")
                        
            except (OSError, UnicodeDecodeError, MemoryError, PermissionError) as e:
                # Log the error but continue processing other files
                error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
                nondeterminism_hints.append(f"Could not read log file {log_file.name}: {error_msg}")
                continue
            except Exception as e:
                # Catch any other unexpected errors
                error_msg = str(e)[:50] if len(str(e)) > 50 else str(e)
                nondeterminism_hints.append(f"Unexpected error reading {log_file.name}: {error_msg}")
                continue

    # Comprehensive pass/fail logic
    deterministic_conditions = [
        env_vars["cudnn_deterministic"],
        not env_vars["cudnn_benchmark"],
        env_vars.get("torch_cudnn_deterministic", False),
        not env_vars.get("torch_cudnn_benchmark", True),
        env_vars.get("torch_use_deterministic_algorithms", False),
    ]
    
    pass_check = all(deterministic_conditions)
    
    # Add specific hints for failed conditions
    if not env_vars["cudnn_deterministic"]:
        nondeterminism_hints.append("CUDNN_DETERMINISTIC not set to true")
    if env_vars["cudnn_benchmark"]:
        nondeterminism_hints.append("CUDNN_BENCHMARK is enabled (should be false)")
    if not env_vars.get("torch_cudnn_deterministic", False):
        nondeterminism_hints.append("torch.backends.cudnn.deterministic is False")
    if env_vars.get("torch_cudnn_benchmark", True):
        nondeterminism_hints.append("torch.backends.cudnn.benchmark is True")
    if not env_vars.get("torch_use_deterministic_algorithms", False):
        nondeterminism_hints.append("torch.use_deterministic_algorithms is False")

    return {
        "pass": pass_check, 
        "flags": env_vars, 
        "nondeterminism_hints": nondeterminism_hints,
        "deterministic_conditions_met": sum(deterministic_conditions),
        "total_conditions": len(deterministic_conditions)
    }


def _analyze_rng_consistency(events: List[Event]) -> Dict[str, Any]:
    """Analyze RNG consistency across events."""
    if not events:
        return {"multiple_seeds": False, "seed_changes": 0}

    seeds = [
        event.rng.get("seed") for event in events if event.rng.get("seed") is not None
    ]
    unique_seeds = set(seeds)

    # Count seed changes
    seed_changes = 0
    for i in range(1, len(seeds)):
        if seeds[i] != seeds[i - 1]:
            seed_changes += 1

    return {
        "multiple_seeds": len(unique_seeds) > 1,
        "seed_changes": seed_changes,
        "unique_seeds": list(unique_seeds),
    }


def _detect_nondeterminism_patterns(events: List[Event]) -> List[str]:
    """Detect patterns that might indicate non-determinism."""
    hints = []

    if not events:
        return hints

    # Check for high variance in metrics
    metrics_df = pd.DataFrame([event.metrics for event in events])

    for metric in metrics_df.columns:
        if metric in ["reward_mean", "kl_mean", "entropy_mean"]:
            variance = metrics_df[metric].var()
            if variance > 0.1:  # High variance threshold
                hints.append(f"High variance in {metric}: {variance:.4f}")

    # Check for unusual patterns in wall time
    wall_times = [event.wall_time for event in events]
    if len(wall_times) > 1:
        time_variance = np.var(wall_times)
        if time_variance > 100:  # High time variance
            hints.append(f"High variance in wall time: {time_variance:.2f}")

    # Check for notes indicating issues
    for event in events:
        for note in event.notes:
            if "detected" in note.lower():
                hints.append(note)

    return list(set(hints))  # Remove duplicates


def _calculate_replica_variance(events: List[Event]) -> Dict[str, float]:
    """Calculate variance for key metrics across events."""
    if not events:
        return {}

    metrics_df = pd.DataFrame([event.metrics for event in events])
    variance = {}

    for metric in ["reward_mean", "kl_mean", "entropy_mean"]:
        if metric in metrics_df.columns:
            variance[metric] = float(metrics_df[metric].var())

    return variance


def _find_metric_mismatches(events: List[Event]) -> List[Dict[str, Any]]:
    """Find metric mismatches that might indicate non-determinism."""
    mismatches = []

    if len(events) < 2:
        return mismatches

    # Look for sudden changes in metrics
    for i in range(1, len(events)):
        event_prev = events[i - 1]
        event_curr = events[i]

        for metric_name, curr_value in event_curr.metrics.items():
            if metric_name in event_prev.metrics:
                prev_value = event_prev.metrics[metric_name]
                change = abs(curr_value - prev_value)

                # Flag large changes
                if change > 0.1:  # Threshold for significant change
                    mismatches.append(
                        {
                            "step": event_curr.step,
                            "metric": metric_name,
                            "replica_1": prev_value,
                            "replica_2": curr_value,
                            "variance": change,
                        }
                    )

    return mismatches[:10]  # Limit to first 10 mismatches


def _generate_determinism_fixes(
    determinism_result: Dict[str, Any], nondeterminism_hints: List[str]
) -> List[str]:
    """Generate fixes for determinism issues."""
    fixes = []

    # Add standard fixes based on determinism check
    if not determinism_result["flags"]["cudnn_deterministic"]:
        fixes.append("Set torch.backends.cudnn.deterministic = True")

    if determinism_result["flags"]["cudnn_benchmark"]:
        fixes.append("Set torch.backends.cudnn.benchmark = False")

    if determinism_result["flags"]["tokenizers_parallelism"] != "false":
        fixes.append("Set TOKENIZERS_PARALLELISM=false")

    # Add fixes based on hints
    for hint in nondeterminism_hints:
        if "variance" in hint.lower():
            fixes.append("Use consistent seeds across all components")
        if "wall time" in hint.lower():
            fixes.append("Ensure consistent hardware and environment")

    # Add general fixes
    fixes.extend(
        [
            "Use torch.manual_seed() consistently",
            "Set numpy.random.seed() consistently",
            "Use random.seed() consistently",
            "Ensure deterministic data loading",
        ]
    )

    return list(set(fixes))  # Remove duplicates


def _generate_determinism_visualization(
    card_data: Dict[str, Any], output_path: Path
) -> None:
    """Generate a visual representation of the determinism card."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"Determinism Card - {card_data['run_id']}", fontsize=16, fontweight="bold"
    )

    # Overall status
    status_color = "green" if card_data["passed"] else "red"
    status_text = "PASS" if card_data["passed"] else "FAIL"

    ax1.text(
        0.5,
        0.5,
        status_text,
        fontsize=48,
        ha="center",
        va="center",
        color=status_color,
        fontweight="bold",
    )
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title("Overall Status", fontsize=14, fontweight="bold")

    # RNG configuration
    rng_data = card_data["rng_map"]
    rng_text = "\n".join([f"{k}: {v}" for k, v in rng_data.items()])
    ax2.text(0.1, 0.9, rng_text, fontsize=10, va="top", transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("RNG Configuration", fontsize=14, fontweight="bold")

    # Metric variance
    if card_data["replica_variance"]:
        metrics = list(card_data["replica_variance"].keys())
        variances = list(card_data["replica_variance"].values())

        bars = ax3.bar(
            metrics,
            variances,
            color=["red" if v > 0.05 else "green" for v in variances],
        )
        ax3.set_title("Metric Variance", fontsize=14, fontweight="bold")
        ax3.set_ylabel("Variance")
        ax3.tick_params(axis="x", rotation=45)

        # Add threshold line
        ax3.axhline(
            y=0.05, color="orange", linestyle="--", alpha=0.7, label="Threshold"
        )
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No variance data", ha="center", va="center")
        ax3.set_title("Metric Variance", fontsize=14, fontweight="bold")
        ax3.axis("off")

    # Issues and fixes
    issues = card_data["nondeterminism_hints"]
    fixes = card_data["fixes"][:5]  # Show first 5 fixes

    issues_text = (
        "\n".join([f"• {issue}" for issue in issues[:3]])
        if issues
        else "No issues detected"
    )
    fixes_text = "\n".join([f"• {fix}" for fix in fixes])

    ax4.text(
        0.05,
        0.95,
        "Issues:",
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax4.transAxes,
    )
    ax4.text(0.05, 0.85, issues_text, fontsize=9, va="top", transform=ax4.transAxes)
    ax4.text(
        0.05,
        0.45,
        "Fixes:",
        fontsize=12,
        fontweight="bold",
        va="top",
        transform=ax4.transAxes,
    )
    ax4.text(0.05, 0.35, fixes_text, fontsize=9, va="top", transform=ax4.transAxes)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis("off")
    ax4.set_title("Issues & Fixes", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
