"""Determinism card generation for RL training runs."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io.event_schema import Event


def _deduplicate_deterministic(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order deterministically."""
    unique_items = []
    seen = set()
    for item in items:
        if item not in seen:
            unique_items.append(item)
            seen.add(item)
    return unique_items


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
                # Count unique seeds deterministically
                _get_unique_seeds_deterministic(events)
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
    """Perform a basic determinism check for card generation."""
    # This is a simplified check that doesn't require running multiple replicas
    # In a real implementation, this would use the full determinism check

    # Check environment variables
    env_vars = {
        "cudnn_deterministic": os.environ.get("CUDNN_DETERMINISTIC", "false").lower()
        == "true",
        "cudnn_benchmark": os.environ.get("CUDNN_BENCHMARK", "true").lower() == "true",
        "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
    }

    # Basic pass/fail logic
    pass_check = env_vars["cudnn_deterministic"] and not env_vars["cudnn_benchmark"]

    return {"pass": pass_check, "flags": env_vars, "nondeterminism_hints": []}


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

    return _deduplicate_deterministic(hints)


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

    return _deduplicate_deterministic(fixes)


def _get_unique_seeds_deterministic(events: List[Event]) -> List[Any]:
    """Get unique seeds deterministically (sorted for consistency)."""
    seeds = [
        event.rng.get("seed")
        for event in events
        if event.rng.get("seed") is not None
    ]
    # Sort seeds for deterministic ordering, then remove duplicates
    unique_seeds = []
    seen = set()
    for seed in sorted(seeds):
        if seed not in seen:
            unique_seeds.append(seed)
            seen.add(seed)
    return unique_seeds


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

        ax3.bar(
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
