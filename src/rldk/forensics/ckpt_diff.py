"""Checkpoint diff analysis."""

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from rldk.io.readers import read_checkpoint


def diff_checkpoints(ckpt_a: str, ckpt_b: str) -> Dict[str, Any]:
    """Compare two model checkpoints and identify parameter differences."""
    ckpt_a = Path(ckpt_a)
    ckpt_b = Path(ckpt_b)

    # Load checkpoints
    state_a = read_checkpoint(ckpt_a)
    state_b = read_checkpoint(ckpt_b)

    # Ensure both are OrderedDict
    if not isinstance(state_a, OrderedDict):
        state_a = OrderedDict(state_a)
    if not isinstance(state_b, OrderedDict):
        state_b = OrderedDict(state_b)

    # Get all parameter names from both checkpoints
    all_param_names = set(state_a.keys()) | set(state_b.keys())
    common_param_names = set(state_a.keys()) & set(state_b.keys())
    only_in_a = set(state_a.keys()) - set(state_b.keys())
    only_in_b = set(state_b.keys()) - set(state_a.keys())

    if not all_param_names:
        raise ValueError("No parameters found in either checkpoint")

    # Compute differences
    differences = []
    l2_norms = []
    cosine_similarities = []

    for name in all_param_names:
        if name in common_param_names:
            # Parameter exists in both checkpoints
            param_a = state_a[name]
            param_b = state_b[name]

            # Ensure same shape
            if param_a.shape != param_b.shape:
                differences.append({
                    "name": name,
                    "l2": float('inf'),
                    "cosine": 0.0,
                    "note": f"Shape mismatch: {param_a.shape} vs {param_b.shape}"
                })
                continue

            # Compute L2 norm of difference
            diff = param_a - param_b
            l2_norm = torch.norm(diff).item()

            # Compute cosine similarity
            norm_a = torch.norm(param_a).item()
            norm_b = torch.norm(param_b).item()

            if norm_a > 0 and norm_b > 0:
                dot_product = torch.dot(param_a.flatten(), param_b.flatten()).item()
                cosine_sim = dot_product / (norm_a * norm_b)
            else:
                cosine_sim = 1.0 if l2_norm == 0 else 0.0

            differences.append({"name": name, "l2": l2_norm, "cosine": cosine_sim})

            l2_norms.append(l2_norm)
            cosine_similarities.append(cosine_sim)

        elif name in only_in_a:
            # Parameter only exists in checkpoint A
            param_a = state_a[name]
            l2_norm = torch.norm(param_a).item()
            differences.append({
                "name": name,
                "l2": l2_norm,
                "cosine": 0.0,
                "note": "Only in checkpoint A"
            })
            l2_norms.append(l2_norm)
            cosine_similarities.append(0.0)

        elif name in only_in_b:
            # Parameter only exists in checkpoint B
            param_b = state_b[name]
            l2_norm = torch.norm(param_b).item()
            differences.append({
                "name": name,
                "l2": l2_norm,
                "cosine": 0.0,
                "note": "Only in checkpoint B"
            })
            l2_norms.append(l2_norm)
            cosine_similarities.append(0.0)

    # Sort by L2 norm (descending)
    differences.sort(key=lambda x: x["l2"], reverse=True)

    # Compute summary statistics
    l2_norms = np.array(l2_norms)
    cosine_similarities = np.array(cosine_similarities)

    # Handle infinite values in L2 norms for percentile calculations
    finite_l2_norms = l2_norms[np.isfinite(l2_norms)]

    # Handle empty cosine_similarities array (e.g., when all common params have shape mismatches)
    if len(cosine_similarities) > 0:
        avg_cosine = float(np.mean(cosine_similarities))
    else:
        avg_cosine = 0.0  # Default to 0 when no comparable tensors exist

    summary = {
        "num_params": len(differences),
        "num_common_params": len(common_param_names),
        "num_only_in_a": len(only_in_a),
        "num_only_in_b": len(only_in_b),
        "avg_cosine": avg_cosine,
    }

    if len(finite_l2_norms) > 0:
        summary.update({
            "l2_p05": float(np.percentile(finite_l2_norms, 5)),
            "l2_p50": float(np.percentile(finite_l2_norms, 50)),
            "l2_p95": float(np.percentile(finite_l2_norms, 95)),
        })
    else:
        summary.update({
            "l2_p05": 0.0,
            "l2_p50": 0.0,
            "l2_p95": 0.0,
        })

    # Generate notes
    notes = []

    # Report parameter differences
    if only_in_a:
        notes.append(f"Checkpoint A has {len(only_in_a)} unique parameters: {list(only_in_a)}")
    if only_in_b:
        notes.append(f"Checkpoint B has {len(only_in_b)} unique parameters: {list(only_in_b)}")
    if len(common_param_names) > 0:
        notes.append(f"Both checkpoints share {len(common_param_names)} parameters")

    # Report if no comparable parameters exist (all have shape mismatches)
    if len(cosine_similarities) == 0 and len(common_param_names) > 0:
        notes.append("No comparable parameters found - all common parameters have shape mismatches")

    # Check for optimizer states
    optimizer_keys = [k for k in state_a.keys() if "optimizer" in k.lower()]
    if optimizer_keys:
        notes.append(
            f"Checkpoint A contains {len(optimizer_keys)} optimizer state keys"
        )

    optimizer_keys = [k for k in state_b.keys() if "optimizer" in k.lower()]
    if optimizer_keys:
        notes.append(
            f"Checkpoint B contains {len(optimizer_keys)} optimizer state keys"
        )

    # Check for RNG states
    rng_keys = [
        k for k in state_a.keys() if "rng" in k.lower() or "random" in k.lower()
    ]
    if rng_keys:
        notes.append(f"Checkpoint A contains {len(rng_keys)} RNG state keys")

    rng_keys = [
        k for k in state_b.keys() if "rng" in k.lower() or "random" in k.lower()
    ]
    if rng_keys:
        notes.append(f"Checkpoint B contains {len(rng_keys)} RNG state keys")

    # Create report
    report = {
        "version": "1",
        "summary": summary,
        "top_movers": differences[:20],  # Top 20 parameter changes
        "notes": notes,
    }

    return report
