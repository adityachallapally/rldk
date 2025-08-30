"""Diff analysis for training runs using rolling z-score detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from scipy import stats


@dataclass
class DivergenceReport:
    """Report of divergence analysis between two runs."""
    
    diverged: bool
    first_step: Optional[int]
    tripped_signals: List[str]
    notes: List[str]
    report_path: str
    events_csv_path: str


def first_divergence(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    signals: List[str],
    k_consecutive: int = 3,
    window: int = 50
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
            events_csv_path=""
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
        violations = _find_k_consecutive_violations(z_scores, k_consecutive)
        
        for violation in violations:
            step = common_steps[violation['end_idx']]
            divergence_events.append({
                'step': step,
                'signal': signal,
                'z_score': violation['z_score'],
                'run_a_value': df_a_aligned.loc[step, signal],
                'run_b_value': df_b_aligned.loc[step, signal],
                'violation_type': violation['type']
            })
            
            tripped_signals.add(signal)
            
            if first_divergence_step is None or step < first_divergence_step:
                first_divergence_step = step
    
    # Determine if divergence occurred
    diverged = len(divergence_events) > 0
    
    # Generate notes
    notes = []
    if diverged:
        notes.append(f"Divergence detected using {k_consecutive}-consecutive rule")
        notes.append(f"Rolling window size: {window}")
        notes.append(f"Signals monitored: {', '.join(signals)}")
    else:
        notes.append("No significant divergence detected")
        notes.append(f"Threshold: {k_consecutive} consecutive violations")
        notes.append(f"Rolling window size: {window}")
    
    # Create report paths
    output_dir = Path("diff_analysis")
    report_path = str(output_dir / "diff_report.md")
    events_csv_path = str(output_dir / "diff_events.csv")
    
    # Save events to CSV
    if divergence_events:
        events_df = pd.DataFrame(divergence_events)
        events_df.to_csv(events_csv_path, index=False)
    
    return DivergenceReport(
        diverged=diverged,
        first_step=first_divergence_step,
        tripped_signals=list(tripped_signals),
        notes=notes,
        report_path=report_path,
        events_csv_path=events_csv_path
    )


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
    k: int
) -> List[dict]:
    """Find k-consecutive violations of z-score threshold."""
    violations = []
    
    # Define thresholds (can be made configurable)
    upper_threshold = 2.0  # 2 standard deviations
    lower_threshold = -2.0
    
    # Find consecutive violations
    consecutive_count = 0
    violation_start = None
    violation_type = None
    
    for i, z_score in enumerate(z_scores):
        if pd.isna(z_score):
            consecutive_count = 0
            violation_start = None
            continue
        
        # Check if current value is a violation
        is_violation = False
        current_type = None
        
        if z_score > upper_threshold:
            is_violation = True
            current_type = 'upper'
        elif z_score < lower_threshold:
            is_violation = True
            current_type = 'lower'
        
        if is_violation:
            if consecutive_count == 0:
                violation_start = i
                violation_type = current_type
            
            consecutive_count += 1
            
            # Check if we have k consecutive violations
            if consecutive_count >= k:
                violations.append({
                    'start_idx': violation_start,
                    'end_idx': i,
                    'z_score': z_score,
                    'type': violation_type,
                    'consecutive_count': consecutive_count
                })
                consecutive_count = 0
                violation_start = None
        else:
            consecutive_count = 0
            violation_start = None
    
    return violations
