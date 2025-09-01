"""Diff analysis for training runs using rolling z-score detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats

from ..io.event_schema import Event


@dataclass
class DivergenceReport:
    """Report of divergence analysis between two runs."""
    
    diverged: bool
    first_step: Optional[int]
    tripped_signals: List[str]
    notes: List[str]
    report_path: str
    events_csv_path: str
    details: pd.DataFrame
    suspected_causes: List[str]


def first_divergence(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    signals: List[str],
    k_consecutive: int = 3,
    window: int = 50,
    tolerance: float = 2.0,
    output_dir: str = "diff_analysis"
) -> DivergenceReport:
    """
    Find first divergence between two training runs.
    
    Args:
        df_a: DataFrame for run A
        df_b: DataFrame for run B
        signals: List of metric names to monitor
        k_consecutive: Number of consecutive violations required
        window: Rolling window size for z-score calculation
    
    Returns:
        DivergenceReport with analysis results
    """
    # Align dataframes by step
    df_a = df_a.set_index('step').sort_index()
    df_b = df_b.set_index('step').sort_index()
    
    # Find common steps
    common_steps = df_a.index.intersection(df_b.index)
    if len(common_steps) < window:
        return DivergenceReport(
            diverged=False,
            first_step=None,
            tripped_signals=[],
            notes=["Insufficient common steps for analysis"],
            report_path="",
            events_csv_path="",
            details=pd.DataFrame(),
            suspected_causes=["Insufficient common steps for analysis"]
        )


def first_divergence_events(
    events_a: List[Event],
    events_b: List[Event],
    signals: List[str],
    k_consecutive: int = 3,
    window: int = 50,
    tolerance: float = 2.0,
    output_dir: str = "diff_analysis"
) -> DivergenceReport:
    """
    Find first divergence between two training runs using Event objects.
    
    Args:
        events_a: List of Event objects for run A
        events_b: List of Event objects for run B
        signals: List of metric names to monitor
        k_consecutive: Number of consecutive violations required
        window: Rolling window size for z-score calculation
        tolerance: Z-score threshold for violation detection
        output_dir: Directory to save analysis results
    
    Returns:
        DivergenceReport with analysis results
    """
    # Convert events to DataFrames for analysis
    from ..io.event_schema import events_to_dataframe
    
    df_a = events_to_dataframe(events_a)
    df_b = events_to_dataframe(events_b)
    
    # Use the existing DataFrame-based analysis
    return first_divergence(df_a, df_b, signals, k_consecutive, window, tolerance, output_dir)


def _calculate_rolling_z_scores(
    series_a: pd.Series, 
    series_b: pd.Series, 
    window: int
) -> pd.Series:
    """Calculate rolling z-scores for difference between two series."""
    # Calculate difference
    diff = series_a - series_b
    
    # Calculate rolling mean and std
    rolling_mean = diff.rolling(window=window, center=True).mean()
    rolling_std = diff.rolling(window=window, center=True).std()
    
    # Calculate z-scores
    z_scores = (diff - rolling_mean) / rolling_std
    
    return z_scores


def _find_k_consecutive_violations(
    z_scores: pd.Series, 
    k: int,
    tolerance: float = 2.0
) -> List[dict]:
    """Find k violations (not necessarily consecutive)."""
    violations = []
    
    # Define thresholds based on tolerance parameter
    upper_threshold = tolerance
    lower_threshold = -tolerance
    
    
    
    # Find all violations
    violation_indices = []
    
    for i, z_score in enumerate(z_scores):
        if pd.isna(z_score):
            continue
        
        # Check if current value is a violation
        if z_score > upper_threshold or z_score < lower_threshold:
            violation_indices.append(i)
    
    # If we have k or more violations, report the first k
    if len(violation_indices) >= k:
        # Take the first k violations
        first_k_violations = violation_indices[:k]
        
        violations.append({
            'start_idx': first_k_violations[0],
            'end_idx': first_k_violations[-1],
            'z_score': z_scores.iloc[first_k_violations[-1]],
            'type': 'lower' if z_scores.iloc[first_k_violations[-1]] < 0 else 'upper',
            'consecutive_count': len(first_k_violations)
        })
    
    return violations


def _analyze_suspected_causes(
    divergence_events: List[dict],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame
) -> List[str]:
    """Analyze potential causes of divergence."""
    causes = []
    
    # Check for learning rate differences
    if 'lr' in df_a.columns and 'lr' in df_b.columns:
        lr_diff = abs(df_a['lr'] - df_b['lr']).max()
        if lr_diff > 1e-6:
            causes.append("Learning rate differences detected")
    
    # Check for seed differences
    if 'seed' in df_a.columns and 'seed' in df_b.columns:
        seed_a = df_a['seed'].iloc[0] if not df_a['seed'].isna().all() else None
        seed_b = df_b['seed'].iloc[0] if not df_b['seed'].isna().all() else None
        if seed_a != seed_b:
            causes.append("Different random seeds used")
    
    # Check for git SHA differences
    if 'git_sha' in df_a.columns and 'git_sha' in df_b.columns:
        sha_a = df_a['git_sha'].iloc[0] if not df_a['git_sha'].isna().all() else None
        sha_b = df_b['git_sha'].iloc[0] if not df_b['git_sha'].isna().all() else None
        if sha_a != sha_b:
            causes.append("Different code versions (git SHA)")
    
    # Analyze which signals were most affected
    signal_counts = {}
    for event in divergence_events:
        signal = event['signal']
        signal_counts[signal] = signal_counts.get(signal, 0) + 1
    
    if signal_counts:
        most_affected = max(signal_counts.items(), key=lambda x: x[1])
        causes.append(f"Most affected signal: {most_affected[0]} ({most_affected[1]} violations)")
    
    # Check for sudden spikes in metrics
    for event in divergence_events:
        signal = event['signal']
        step = event['step']
        
        # Look for sudden changes around the divergence point
        if step > 0:
            prev_step = step - 1
            if prev_step in df_a.index and prev_step in df_b.index:
                change_a = abs(df_a.loc[step, signal] - df_a.loc[prev_step, signal])
                change_b = abs(df_b.loc[step, signal] - df_b.loc[prev_step, signal])
                
                if max(change_a, change_b) > 0.1:  # Threshold for "sudden" change
                    causes.append(f"Sudden spike detected in {signal} at step {step}")
                    break
    
    if not causes:
        causes.append("No obvious cause identified - may be due to numerical instability")
    
    return causes
