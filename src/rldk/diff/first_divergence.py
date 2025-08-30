"""First divergence detection using rolling z-score analysis."""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class DivergenceReport:
    """Report of divergence analysis between two runs."""
    
    diverged: bool
    first_step: Optional[int]
    tripped_signals: List[str]
    details: pd.DataFrame
    suspected_causes: List[str]


def first_divergence(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    signals: List[str],
    k_consecutive: int = 3,
    window: int = 50,
    tolerance: float = 2.0
) -> DivergenceReport:
    """
    Find first divergence between two training runs.
    
    Args:
        df_a: DataFrame for run A
        df_b: DataFrame for run B
        signals: List of metric names to monitor
        k_consecutive: Number of consecutive violations required
        window: Rolling window size for z-score calculation
        tolerance: Z-score threshold for violation detection
    
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
            details=pd.DataFrame(),
            suspected_causes=["Insufficient common steps for analysis"]
        )
    
    # Align data
    df_a_aligned = df_a.loc[common_steps]
    df_b_aligned = df_b.loc[common_steps]
    
    # Calculate rolling z-scores for each signal
    divergence_events = []
    tripped_signals = set()
    first_divergence_step = None
    
    for signal in signals:
        if signal not in df_a_aligned.columns or signal not in df_b_aligned.columns:
            continue
        
        # Calculate rolling z-scores
        z_scores = _calculate_rolling_z_scores(
            df_a_aligned[signal], 
            df_b_aligned[signal], 
            window
        )
        
        # Find violations of k-consecutive rule
        violations = _find_k_consecutive_violations(z_scores, k_consecutive, tolerance)
        
        for violation in violations:
            step = common_steps[violation['end_idx']]
            divergence_events.append({
                'step': step,
                'signal': signal,
                'z_score': violation['z_score'],
                'run_a_value': df_a_aligned.loc[step, signal],
                'run_b_value': df_b_aligned.loc[step, signal],
                'violation_type': violation['type'],
                'consecutive_count': violation['count']
            })
            
            tripped_signals.add(signal)
            
            if first_divergence_step is None or step < first_divergence_step:
                first_divergence_step = step
    
    # Create details DataFrame
    details_df = pd.DataFrame(divergence_events)
    
    # Determine if divergence occurred
    diverged = len(divergence_events) > 0
    
    # Analyze suspected causes (even if no divergence events)
    suspected_causes = _analyze_suspected_causes(divergence_events, df_a_aligned, df_b_aligned)
    
    return DivergenceReport(
        diverged=diverged,
        first_step=first_divergence_step,
        tripped_signals=list(tripped_signals),
        details=details_df,
        suspected_causes=suspected_causes
    )


def _calculate_rolling_z_scores(
    series_a: pd.Series, 
    series_b: pd.Series, 
    window: int
) -> np.ndarray:
    """Calculate rolling z-scores of the difference between two series."""
    # Calculate difference
    diff = series_a - series_b
    
    # Calculate rolling mean and std
    rolling_mean = diff.rolling(window=window, center=True).mean()
    rolling_std = diff.rolling(window=window, center=True).std()
    
    # Calculate z-scores
    z_scores = np.zeros_like(diff)
    valid_mask = ~(rolling_mean.isna() | rolling_std.isna())
    z_scores[valid_mask] = (diff[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]
    
    return z_scores


def _find_k_consecutive_violations(
    z_scores: np.ndarray, 
    k: int, 
    tolerance: float
) -> List[Dict[str, Any]]:
    """Find k consecutive violations of the z-score threshold."""
    violations = []
    
    # Find points above tolerance
    above_threshold = np.abs(z_scores) > tolerance
    
    # Find consecutive runs
    consecutive_count = 0
    start_idx = None
    
    for i, is_violation in enumerate(above_threshold):
        if is_violation:
            if consecutive_count == 0:
                start_idx = i
            consecutive_count += 1
        else:
            if consecutive_count >= k:
                # Record the violation
                violations.append({
                    'start_idx': start_idx,
                    'end_idx': i - 1,
                    'count': consecutive_count,
                    'z_score': z_scores[i - 1],
                    'type': 'consecutive_violation'
                })
            consecutive_count = 0
            start_idx = None
    
    # Handle case where violation extends to end
    if consecutive_count >= k:
        violations.append({
            'start_idx': start_idx,
            'end_idx': len(z_scores) - 1,
            'count': consecutive_count,
            'z_score': z_scores[-1],
            'type': 'consecutive_violation'
        })
    
    return violations


def _analyze_suspected_causes(
    divergence_events: List[Dict[str, Any]],
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