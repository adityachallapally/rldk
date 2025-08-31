"""Reward drift detection for reward models."""

from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, wasserstein_distance


def detect_reward_drift(run_data: pd.DataFrame,
                       reference_data: pd.DataFrame,
                       reward_col: str,
                       step_col: str,
                       threshold: float = 0.1) -> Tuple[bool, pd.DataFrame]:
    """
    Detect reward drift between current run and reference data.
    
    Args:
        run_data: Current training run data
        reference_data: Reference run data for comparison
        reward_col: Column name for reward values
        step_col: Column name for training steps
        threshold: P-value threshold for drift detection (lower = more sensitive)
    
    Returns:
        Tuple of (drift_detected, drift_metrics)
    """
    # Ensure both datasets have the required columns
    if reward_col not in run_data.columns or reward_col not in reference_data.columns:
        raise ValueError(f"Reward column '{reward_col}' not found in one or both datasets")
    
    if step_col not in run_data.columns or step_col not in reference_data.columns:
        raise ValueError(f"Step column '{step_col}' not found in one or both datasets")
    
    # Sort by step
    run_data = run_data.sort_values(step_col).reset_index(drop=True)
    reference_data = reference_data.sort_values(step_col).reset_index(drop=True)
    
    # Initialize drift metrics
    drift_events = []
    
    # Method 1: Overall distribution comparison
    overall_drift = _compare_overall_distributions(
        run_data[reward_col], reference_data[reward_col], threshold
    )
    
    if overall_drift['detected']:
        drift_events.append({
            'step': 'overall',
            'drift_type': 'overall_distribution',
            'p_value': overall_drift['p_value'],
            'statistic': overall_drift['statistic'],
            'effect_size': overall_drift['effect_size'],
            'detected': True
        })
    
    # Method 2: Temporal drift analysis (compare early vs late in training)
    temporal_drift = _detect_temporal_drift(run_data, reward_col, step_col, threshold)
    if temporal_drift['detected']:
        drift_events.append({
            'step': 'temporal',
            'drift_type': 'temporal_drift',
            'p_value': temporal_drift['p_value'],
            'statistic': temporal_drift['statistic'],
            'effect_size': temporal_drift['effect_size'],
            'detected': True
        })
    
    # Method 3: Rolling window drift detection
    rolling_drift = _detect_rolling_drift(run_data, reference_data, reward_col, step_col, threshold)
    drift_events.extend(rolling_drift)
    
    # Create drift metrics DataFrame
    if drift_events:
        drift_metrics = pd.DataFrame(drift_events)
        drift_detected = True
    else:
        drift_metrics = pd.DataFrame()
        drift_detected = False
    
    return drift_detected, drift_metrics


def _compare_overall_distributions(sample1: pd.Series, 
                                 sample2: pd.Series, 
                                 threshold: float) -> Dict[str, Any]:
    """Compare overall distributions using KS test and Wasserstein distance."""
    
    # Remove NaN values
    sample1_clean = sample1.dropna()
    sample2_clean = sample2.dropna()
    
    if len(sample1_clean) < 10 or len(sample2_clean) < 10:
        return {
            'detected': False,
            'p_value': 1.0,
            'statistic': 0.0,
            'effect_size': 0.0,
            'reason': 'Insufficient data for comparison'
        }
    
    # KS test for distribution similarity
    try:
        ks_statistic, ks_p_value = ks_2samp(sample1_clean, sample2_clean)
    except Exception:
        # Fallback to basic statistics
        return {
            'detected': False,
            'p_value': 1.0,
            'statistic': 0.0,
            'effect_size': 0.0,
            'reason': 'KS test failed'
        }
    
    # Wasserstein distance for distribution difference
    try:
        wasserstein_dist = wasserstein_distance(sample1_clean, sample2_clean)
    except Exception:
        wasserstein_dist = np.nan
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(sample1_clean) - 1) * sample1_clean.var() + 
                          (len(sample2_clean) - 1) * sample2_clean.var()) / 
                         (len(sample1_clean) + len(sample2_clean) - 2))
    
    if pooled_std > 0:
        effect_size = abs(sample1_clean.mean() - sample2_clean.mean()) / pooled_std
    else:
        effect_size = 0.0
    
    # Determine if drift is detected
    detected = ks_p_value < threshold
    
    return {
        'detected': detected,
        'p_value': ks_p_value,
        'statistic': ks_statistic,
        'effect_size': effect_size,
        'wasserstein_distance': wasserstein_dist,
        'sample1_mean': sample1_clean.mean(),
        'sample2_mean': sample2_clean.mean(),
        'sample1_std': sample1_clean.std(),
        'sample2_std': sample2_clean.std()
    }


def _detect_temporal_drift(run_data: pd.DataFrame, 
                          reward_col: str, 
                          step_col: str, 
                          threshold: float) -> Dict[str, Any]:
    """Detect drift within a single run over time."""
    
    if len(run_data) < 20:
        return {
            'detected': False,
            'p_value': 1.0,
            'statistic': 0.0,
            'effect_size': 0.0,
            'reason': 'Insufficient data for temporal analysis'
        }
    
    # Split data into early and late periods
    mid_point = len(run_data) // 2
    early_data = run_data.iloc[:mid_point][reward_col].dropna()
    late_data = run_data.iloc[mid_point:][reward_col].dropna()
    
    if len(early_data) < 10 or len(late_data) < 10:
        return {
            'detected': False,
            'p_value': 1.0,
            'statistic': 0.0,
            'effect_size': 0.0,
            'reason': 'Insufficient data in early/late periods'
        }
    
    # Compare early vs late distributions
    try:
        ks_statistic, ks_p_value = ks_2samp(early_data, late_data)
    except Exception:
        return {
            'detected': False,
            'p_value': 1.0,
            'statistic': 0.0,
            'effect_size': 0.0,
            'reason': 'KS test failed'
        }
    
    # Calculate effect size
    pooled_std = np.sqrt(((len(early_data) - 1) * early_data.var() + 
                          (len(late_data) - 1) * late_data.var()) / 
                         (len(early_data) + len(late_data) - 2))
    
    if pooled_std > 0:
        effect_size = abs(early_data.mean() - late_data.mean()) / pooled_std
    else:
        effect_size = 0.0
    
    detected = ks_p_value < threshold
    
    return {
        'detected': detected,
        'p_value': ks_p_value,
        'statistic': ks_statistic,
        'effect_size': effect_size,
        'early_mean': early_data.mean(),
        'late_mean': late_data.mean(),
        'early_std': early_data.std(),
        'late_std': late_data.std()
    }


def _detect_rolling_drift(run_data: pd.DataFrame,
                         reference_data: pd.DataFrame,
                         reward_col: str,
                         step_col: str,
                         threshold: float,
                         window_size: int = 50) -> list:
    """Detect drift using rolling window comparisons."""
    
    drift_events = []
    
    if len(run_data) < window_size:
        return drift_events
    
    # Use rolling windows to detect drift
    for i in range(window_size, len(run_data), window_size // 2):  # 50% overlap
        window_data = run_data.iloc[i-window_size:i][reward_col].dropna()
        
        if len(window_data) < window_size // 2:  # Need at least half window size
            continue
        
        # Compare window to reference data
        reference_sample = reference_data[reward_col].dropna()
        if len(reference_sample) < 10:
            continue
        
        try:
            ks_statistic, ks_p_value = ks_2samp(window_data, reference_sample)
            
            if ks_p_value < threshold:
                # Calculate effect size
                pooled_std = np.sqrt(((len(window_data) - 1) * window_data.var() + 
                                    (len(reference_sample) - 1) * reference_sample.var()) / 
                                   (len(window_data) + len(reference_sample) - 2))
                
                if pooled_std > 0:
                    effect_size = abs(window_data.mean() - reference_sample.mean()) / pooled_std
                else:
                    effect_size = 0.0
                
                drift_events.append({
                    'step': run_data.iloc[i][step_col],
                    'drift_type': 'rolling_window',
                    'p_value': ks_p_value,
                    'statistic': ks_statistic,
                    'effect_size': effect_size,
                    'detected': True,
                    'window_start': i - window_size,
                    'window_end': i,
                    'window_mean': window_data.mean(),
                    'reference_mean': reference_sample.mean()
                })
                
        except Exception:
            # Skip this window if analysis fails
            continue
    
    return drift_events


def calculate_drift_summary(drift_metrics: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for drift detection results."""
    
    if drift_metrics.empty:
        return {
            'total_events': 0,
            'drift_types': {},
            'mean_effect_size': 0.0,
            'max_effect_size': 0.0,
            'most_common_drift': None
        }
    
    summary = {
        'total_events': len(drift_metrics),
        'drift_types': drift_metrics['drift_type'].value_counts().to_dict(),
        'mean_effect_size': drift_metrics['effect_size'].mean(),
        'max_effect_size': drift_metrics['effect_size'].max(),
        'most_common_drift': drift_metrics['drift_type'].mode().iloc[0] if not drift_metrics.empty else None
    }
    
    # Add step-specific information if available
    if 'step' in drift_metrics.columns and drift_metrics['step'].dtype in ['int64', 'float64']:
        step_metrics = drift_metrics[drift_metrics['step'] != 'overall']
        if not step_metrics.empty:
            summary['first_drift_step'] = step_metrics['step'].min()
            summary['last_drift_step'] = step_metrics['step'].max()
    
    return summary