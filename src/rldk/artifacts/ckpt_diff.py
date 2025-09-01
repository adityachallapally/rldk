"""Checkpoint diff analysis."""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from collections import OrderedDict

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
    
    # Get common parameter names
    param_names = set(state_a.keys()) & set(state_b.keys())
    
    if not param_names:
        raise ValueError("No common parameters found between checkpoints")
    
    # Compute differences
    differences = []
    l2_norms = []
    cosine_similarities = []
    
    for name in param_names:
        param_a = state_a[name]
        param_b = state_b[name]
        
        # Ensure same shape
        if param_a.shape != param_b.shape:
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
        
        differences.append({
            "name": name,
            "l2": l2_norm,
            "cosine": cosine_sim
        })
        
        l2_norms.append(l2_norm)
        cosine_similarities.append(cosine_sim)
    
    # Sort by L2 norm (descending)
    differences.sort(key=lambda x: x["l2"], reverse=True)
    
    # Compute summary statistics
    l2_norms = np.array(l2_norms)
    cosine_similarities = np.array(cosine_similarities)
    
    summary = {
        "num_params": len(differences),
        "avg_cosine": float(np.mean(cosine_similarities)),
        "l2_p05": float(np.percentile(l2_norms, 5)),
        "l2_p50": float(np.percentile(l2_norms, 50)),
        "l2_p95": float(np.percentile(l2_norms, 95))
    }
    
    # Generate notes
    notes = []
    
    # Check for optimizer states
    optimizer_keys = [k for k in state_a.keys() if "optimizer" in k.lower()]
    if optimizer_keys:
        notes.append(f"Checkpoint A contains {len(optimizer_keys)} optimizer state keys")
    
    optimizer_keys = [k for k in state_b.keys() if "optimizer" in k.lower()]
    if optimizer_keys:
        notes.append(f"Checkpoint B contains {len(optimizer_keys)} optimizer state keys")
    
    # Check for RNG states
    rng_keys = [k for k in state_a.keys() if "rng" in k.lower() or "random" in k.lower()]
    if rng_keys:
        notes.append(f"Checkpoint A contains {len(rng_keys)} RNG state keys")
    
    rng_keys = [k for k in state_b.keys() if "rng" in k.lower() or "random" in k.lower()]
    if rng_keys:
        notes.append(f"Checkpoint B contains {len(rng_keys)} RNG state keys")
    
    # Create report
    report = {
        "version": "1",
        "summary": summary,
        "top_movers": differences[:20],  # Top 20 parameter changes
        "notes": notes
    }
    
    return report