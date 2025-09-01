"""Drift card generation for comparing RL training runs."""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..io.event_schema import Event
from ..diff.diff import first_divergence_events


def generate_drift_card(
    events_a: List[Event],
    events_b: List[Event],
    run_a_path: str,
    run_b_path: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a drift card comparing two training runs.
    
    Args:
        events_a: List of Event objects from run A
        events_b: List of Event objects from run B
        run_a_path: Path to run A
        run_b_path: Path to run B
        output_dir: Directory to save the card (defaults to runs/run_id/rldk_cards)
    
    Returns:
        Dictionary containing the drift card data
    """
    # Extract run IDs
    run_id_a = events_a[0].model_info["run_id"] if events_a else "unknown_a"
    run_id_b = events_b[0].model_info["run_id"] if events_b else "unknown_b"
    
    # Set output directory
    if output_dir is None:
        output_dir = f"runs/{run_id_a}_vs_{run_id_b}/rldk_cards"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Perform divergence analysis
    signals = ['reward_mean', 'kl_mean', 'entropy_mean', 'grad_norm']
    divergence_report = first_divergence_events(
        events_a, events_b, signals, k_consecutive=3, window=50, tolerance=2.0
    )
    
    # Analyze metric correlations
    correlations = _calculate_metric_correlations(events_a, events_b)
    
    # Detect specific drift patterns
    drift_patterns = _detect_drift_patterns(events_a, events_b)
    
    # Generate suspected causes
    suspected_causes = _generate_suspected_causes(divergence_report, events_a, events_b)
    
    # Create the card data
    card_data = {
        "version": "1.0",
        "run_a": run_id_a,
        "run_b": run_id_b,
        "generated_at": datetime.now().isoformat(),
        "diverged": divergence_report.diverged,
        "first_step": divergence_report.first_step,
        "tripped_signals": divergence_report.tripped_signals,
        "suspected_causes": suspected_causes,
        "repro": {
            "command": f"rldk card drift {run_a_path} {run_b_path}",
            "changes": _extract_changes(events_a, events_b)
        },
        "details": {
            "kl_divergence": _calculate_kl_divergence(events_a, events_b),
            "reward_drift": {
                "correlation": correlations.get("reward_mean", 0),
                "mae": _calculate_mae(events_a, events_b, "reward_mean")
            },
            "metric_correlations": correlations,
            "drift_patterns": drift_patterns
        },
        "notes": divergence_report.notes
    }
    
    # Save JSON card
    json_path = output_path / "drift_card.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json.dump(card_data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    
    # Generate and save PNG visualization
    png_path = output_path / "drift_card.png"
    _generate_drift_visualization(card_data, png_path)
    
    return card_data


def _calculate_metric_correlations(events_a: List[Event], events_b: List[Event]) -> Dict[str, float]:
    """Calculate correlations between metrics from two runs."""
    if not events_a or not events_b:
        return {}
    
    # Convert events to DataFrames
    from ..io.event_schema import events_to_dataframe
    
    df_a = events_to_dataframe(events_a)
    df_b = events_to_dataframe(events_b)
    
    # Align by step
    common_steps = set(df_a['step']) & set(df_b['step'])
    if len(common_steps) < 10:  # Need minimum data points
        return {}
    
    df_a_aligned = df_a[df_a['step'].isin(common_steps)].set_index('step').sort_index()
    df_b_aligned = df_b[df_b['step'].isin(common_steps)].set_index('step').sort_index()
    
    correlations = {}
    metric_fields = ['reward_mean', 'kl_mean', 'entropy_mean', 'grad_norm', 'loss']
    
    for metric in metric_fields:
        if metric in df_a_aligned.columns and metric in df_b_aligned.columns:
            corr = df_a_aligned[metric].corr(df_b_aligned[metric])
            correlations[metric] = float(corr) if not pd.isna(corr) else 0.0
    
    return correlations


def _detect_drift_patterns(events_a: List[Event], events_b: List[Event]) -> Dict[str, Any]:
    """Detect specific patterns in the drift."""
    patterns = {}
    
    if not events_a or not events_b:
        return patterns
    
    # Convert to DataFrames
    from ..io.event_schema import events_to_dataframe
    
    df_a = events_to_dataframe(events_a)
    df_b = events_to_dataframe(events_b)
    
    # Check for systematic bias
    for metric in ['reward_mean', 'kl_mean', 'entropy_mean']:
        if metric in df_a.columns and metric in df_b.columns:
            mean_a = df_a[metric].mean()
            mean_b = df_b[metric].mean()
            bias = mean_b - mean_a
            patterns[f"{metric}_bias"] = float(bias)
    
    # Check for trend differences
    for metric in ['reward_mean', 'kl_mean']:
        if metric in df_a.columns and metric in df_b.columns:
            # Calculate trend (slope of linear fit)
            x_a = np.arange(len(df_a))
            x_b = np.arange(len(df_b))
            
            if len(x_a) > 1 and len(x_b) > 1:
                slope_a = np.polyfit(x_a, df_a[metric], 1)[0]
                slope_b = np.polyfit(x_b, df_b[metric], 1)[0]
                patterns[f"{metric}_trend_diff"] = float(slope_b - slope_a)
    
    return patterns


def _calculate_kl_divergence(events_a: List[Event], events_b: List[Event]) -> Dict[str, float]:
    """Calculate KL divergence between runs at specific steps."""
    if not events_a or not events_b:
        return {}
    
    # Convert to DataFrames
    from ..io.event_schema import events_to_dataframe
    
    df_a = events_to_dataframe(events_a)
    df_b = events_to_dataframe(events_b)
    
    # Find common steps
    common_steps = set(df_a['step']) & set(df_b['step'])
    if not common_steps:
        return {}
    
    kl_divergences = {}
    for step in sorted(common_steps)[:10]:  # First 10 common steps
        a_row = df_a[df_a['step'] == step]
        b_row = df_b[df_b['step'] == step]
        
        if not a_row.empty and not b_row.empty and 'kl_mean' in a_row.columns and 'kl_mean' in b_row.columns:
            kl_a = a_row['kl_mean'].iloc[0]
            kl_b = b_row['kl_mean'].iloc[0]
            kl_divergences[f"step_{step}"] = float(abs(kl_b - kl_a))
    
    return kl_divergences


def _calculate_mae(events_a: List[Event], events_b: List[Event], metric: str) -> float:
    """Calculate Mean Absolute Error for a specific metric."""
    if not events_a or not events_b:
        return 0.0
    
    # Convert to DataFrames
    from ..io.event_schema import events_to_dataframe
    
    df_a = events_to_dataframe(events_a)
    df_b = events_to_dataframe(events_b)
    
    # Align by step
    common_steps = set(df_a['step']) & set(df_b['step'])
    if len(common_steps) < 5:
        return 0.0
    
    df_a_aligned = df_a[df_a['step'].isin(common_steps)].set_index('step').sort_index()
    df_b_aligned = df_b[df_b['step'].isin(common_steps)].set_index('step').sort_index()
    
    if metric not in df_a_aligned.columns or metric not in df_b_aligned.columns:
        return 0.0
    
    mae = np.mean(np.abs(df_a_aligned[metric] - df_b_aligned[metric]))
    return float(mae)


def _extract_changes(events_a: List[Event], events_b: List[Event]) -> List[str]:
    """Extract potential changes between runs."""
    changes = []
    
    if not events_a or not events_b:
        return changes
    
    # Compare model info
    info_a = events_a[0].model_info
    info_b = events_b[0].model_info
    
    for key in ['model_name', 'optimizer', 'scheduler']:
        if key in info_a and key in info_b:
            if info_a[key] != info_b[key]:
                changes.append(f"{key}: {info_a[key]} -> {info_b[key]}")
    
    # Compare RNG settings
    rng_a = events_a[0].rng
    rng_b = events_b[0].rng
    
    for key in ['seed', 'torch_seed', 'numpy_seed']:
        if key in rng_a and key in rng_b:
            if rng_a[key] != rng_b[key]:
                changes.append(f"{key}: {rng_a[key]} -> {rng_b[key]}")
    
    # Compare data slice settings
    slice_a = events_a[0].data_slice
    slice_b = events_b[0].data_slice
    
    for key in ['tokens_in', 'tokens_out', 'batch_size']:
        if key in slice_a and key in slice_b:
            if slice_a[key] != slice_b[key]:
                changes.append(f"{key}: {slice_a[key]} -> {slice_b[key]}")
    
    return changes


def _generate_suspected_causes(
    divergence_report: Any, 
    events_a: List[Event], 
    events_b: List[Event]
) -> List[str]:
    """Generate suspected causes for the divergence."""
    causes = []
    
    # Add causes from divergence report
    if hasattr(divergence_report, 'suspected_causes'):
        causes.extend(divergence_report.suspected_causes)
    
    # Analyze specific patterns
    if not events_a or not events_b:
        return causes
    
    # Check for different seeds
    seed_a = events_a[0].rng.get("seed") if events_a else None
    seed_b = events_b[0].rng.get("seed") if events_b else None
    if seed_a != seed_b:
        causes.append("Different random seeds")
    
    # Check for different model configurations
    model_a = events_a[0].model_info.get("model_name") if events_a else None
    model_b = events_b[0].model_info.get("model_name") if events_b else None
    if model_a != model_b:
        causes.append("Different model configurations")
    
    # Check for different data processing
    tokens_a = events_a[0].data_slice.get("tokens_in") if events_a else None
    tokens_b = events_b[0].data_slice.get("tokens_in") if events_b else None
    if tokens_a != tokens_b:
        causes.append("Different tokenization settings")
    
    # Check for gradient norm issues
    grad_norms_a = [e.metrics.get("grad_norm", 0) for e in events_a if "grad_norm" in e.metrics]
    grad_norms_b = [e.metrics.get("grad_norm", 0) for e in events_b if "grad_norm" in e.metrics]
    
    if grad_norms_a and grad_norms_b:
        max_grad_a = max(grad_norms_a)
        max_grad_b = max(grad_norms_b)
        if abs(max_grad_a - max_grad_b) > 5.0:
            causes.append("Significant gradient norm differences")
    
    # Add general causes if no specific ones found
    if not causes:
        causes.extend([
            "Tokenizer configuration change",
            "Different pad_direction setting",
            "Modified truncate_at parameter",
            "Environment variable changes",
            "Hardware differences"
        ])
    
    return list(set(causes))  # Remove duplicates


def _generate_drift_visualization(card_data: Dict[str, Any], output_path: Path) -> None:
    """Generate a visual representation of the drift card."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Drift Card - {card_data['run_a']} vs {card_data['run_b']}", 
                 fontsize=16, fontweight='bold')
    
    # Overall status
    status_color = 'red' if card_data['diverged'] else 'green'
    status_text = 'DIVERGED' if card_data['diverged'] else 'CONSISTENT'
    
    ax1.text(0.5, 0.5, status_text, fontsize=36, ha='center', va='center', 
             color=status_color, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Divergence Status', fontsize=14, fontweight='bold')
    
    # First divergence step
    if card_data['first_step'] is not None:
        ax2.text(0.5, 0.5, f"First Divergence:\nStep {card_data['first_step']}", 
                 fontsize=16, ha='center', va='center', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No divergence\ndetected", fontsize=16, ha='center', va='center')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Divergence Point', fontsize=14, fontweight='bold')
    
    # Metric correlations
    correlations = card_data['details']['metric_correlations']
    if correlations:
        metrics = list(correlations.keys())
        corr_values = list(correlations.values())
        
        bars = ax3.bar(metrics, corr_values, 
                      color=['red' if v < 0.8 else 'green' for v in corr_values])
        ax3.set_title('Metric Correlations', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Correlation')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(-1, 1)
        
        # Add threshold line
        ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Threshold')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'No correlation data', ha='center', va='center')
        ax3.set_title('Metric Correlations', fontsize=14, fontweight='bold')
        ax3.axis('off')
    
    # Tripped signals
    signals = card_data['tripped_signals']
    causes = card_data['suspected_causes'][:5]  # Show first 5 causes
    
    signals_text = '\n'.join([f"• {signal}" for signal in signals]) if signals else "No signals tripped"
    causes_text = '\n'.join([f"• {cause}" for cause in causes])
    
    ax4.text(0.05, 0.95, 'Tripped Signals:', fontsize=12, fontweight='bold', va='top', transform=ax4.transAxes)
    ax4.text(0.05, 0.85, signals_text, fontsize=9, va='top', transform=ax4.transAxes)
    ax4.text(0.05, 0.45, 'Suspected Causes:', fontsize=12, fontweight='bold', va='top', transform=ax4.transAxes)
    ax4.text(0.05, 0.35, causes_text, fontsize=9, va='top', transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Signals & Causes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()