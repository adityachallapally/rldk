"""Diff analysis for training runs using rolling z-score detection."""

from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np
import os
import json
import logging

from ..io.event_schema import Event

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    debug_info: dict  # New field for debugging information


def first_divergence(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    signals: List[str],
    k_consecutive: int = 2,  # Reduced from 3 to 2 for more sensitivity
    window: int = 20,        # Reduced from 50 to 20 for more sensitivity
    tolerance: float = 1.5,  # Reduced from 2.0 to 1.5 for more sensitivity
    output_dir: str = "diff_analysis",
    debug: bool = False,     # New parameter for debugging output
) -> DivergenceReport:
    """
    Find first divergence between two training runs.

    Args:
        df_a: DataFrame for run A
        df_b: DataFrame for run B
        signals: List of metric names to monitor
        k_consecutive: Number of consecutive violations required (default: 2)
        window: Rolling window size for z-score calculation (default: 20)
        tolerance: Z-score threshold for violation detection (default: 1.5)
        output_dir: Directory to save analysis results
        debug: Enable detailed debugging output (default: False)

    Returns:
        DivergenceReport with analysis results
    """
    # Initialize debug information
    debug_info = {
        "parameters": {
            "k_consecutive": k_consecutive,
            "window": window,
            "tolerance": tolerance,
            "signals": signals
        },
        "signal_analysis": {},
        "z_scores": {},
        "violations": {}
    }
    
    if debug:
        logger.info(f"Starting divergence analysis with parameters: k_consecutive={k_consecutive}, "
                   f"window={window}, tolerance={tolerance}")
    
    # Validate input dataframes
    if df_a.empty or df_b.empty:
        return DivergenceReport(
            diverged=False,
            first_step=None,
            tripped_signals=[],
            notes=["Empty dataframes provided"],
            report_path="",
            events_csv_path="",
            details=pd.DataFrame(),
            suspected_causes=["Empty dataframes provided"],
            debug_info=debug_info,
        )
    
    if "step" not in df_a.columns or "step" not in df_b.columns:
        return DivergenceReport(
            diverged=False,
            first_step=None,
            tripped_signals=[],
            notes=["Missing 'step' column in input dataframes"],
            report_path="",
            events_csv_path="",
            details=pd.DataFrame(),
            suspected_causes=["Missing 'step' column in input dataframes"],
            debug_info=debug_info,
        )

    # Align dataframes by step
    df_a = df_a.set_index("step").sort_index()
    df_b = df_b.set_index("step").sort_index()

    # Find common steps
    common_steps = df_a.index.intersection(df_b.index)
    debug_info["common_steps_count"] = len(common_steps)
    debug_info["total_steps_a"] = len(df_a)
    debug_info["total_steps_b"] = len(df_b)
    
    if debug:
        logger.info(f"Found {len(common_steps)} common steps between runs")
    
    if len(common_steps) < window:
        return DivergenceReport(
            diverged=False,
            first_step=None,
            tripped_signals=[],
            notes=[f"Insufficient common steps for analysis: {len(common_steps)} < {window}"],
            report_path="",
            events_csv_path="",
            details=pd.DataFrame(),
            suspected_causes=[f"Insufficient common steps for analysis: {len(common_steps)} < {window}"],
            debug_info=debug_info,
        )
    
    # Filter dataframes to common steps
    df_a_common = df_a.loc[common_steps]
    df_b_common = df_b.loc[common_steps]
    
    # Initialize tracking variables
    diverged = False
    first_step = None
    tripped_signals = []
    all_violations = []
    notes = []
    
    # Analyze each signal
    for signal in signals:
        if signal not in df_a_common.columns or signal not in df_b_common.columns:
            if debug:
                logger.warning(f"Signal '{signal}' not found in both dataframes")
            continue
        
        if debug:
            logger.info(f"Analyzing signal: {signal}")
        
        try:
            # Calculate rolling z-scores for this signal
            z_scores = _calculate_rolling_z_scores(
                df_a_common[signal], df_b_common[signal], window, debug=debug
            )
            
            # Also calculate relative change detection for additional sensitivity
            relative_violations = _detect_relative_changes(
                df_a_common[signal], df_b_common[signal], window, tolerance, debug=debug
            )
            
            # Store debug information for this signal
            debug_info["z_scores"][signal] = {
                "mean": float(z_scores.mean()) if not z_scores.empty else None,
                "std": float(z_scores.std()) if not z_scores.empty else None,
                "max": float(z_scores.max()) if not z_scores.empty else None,
                "min": float(z_scores.min()) if not z_scores.empty else None,
                "valid_count": int(z_scores.notna().sum())
            }
            
            if debug:
                logger.info(f"Z-scores for {signal}: mean={debug_info['z_scores'][signal]['mean']:.3f}, "
                           f"std={debug_info['z_scores'][signal]['std']:.3f}, "
                           f"max={debug_info['z_scores'][signal]['max']:.3f}, "
                           f"min={debug_info['z_scores'][signal]['min']:.3f}")
            
            # Find violations from both z-score and relative change detection
            violations = _find_k_consecutive_violations(z_scores, k_consecutive, tolerance, debug=debug)
            
            # Combine with relative violations if any
            if relative_violations:
                violations.extend(relative_violations)
                if debug:
                    logger.info(f"Added {len(relative_violations)} relative change violations")
            
            debug_info["violations"][signal] = len(violations)
            
            if debug and violations:
                logger.info(f"Found {len(violations)} violations for signal {signal}")
            
            if violations:
                diverged = True
                signal_violations = []
                
                for violation in violations:
                    # Convert index position to actual step value
                    if violation["start_idx"] < len(common_steps):
                        step_idx = common_steps.tolist()[violation["start_idx"]]
                        if first_step is None or step_idx < first_step:
                            first_step = step_idx
                        
                        signal_violations.append({
                            "signal": signal,
                            "step": step_idx,
                            "z_score": violation["z_score"],
                            "type": violation["type"],
                            "consecutive_count": violation["consecutive_count"]
                        })
                        
                        if debug:
                            logger.info(f"Violation detected: signal={signal}, step={step_idx}, "
                                       f"z_score={violation['z_score']:.3f}, type={violation['type']}")
                
                if signal_violations:  # Only add if we have valid violations
                    tripped_signals.append({
                        "signal": signal,
                        "violations": signal_violations
                    })
                    
                    all_violations.extend(signal_violations)
                    
        except Exception as e:
            # If analysis fails for this signal, add a note but continue
            error_msg = f"Warning: Analysis failed for signal '{signal}': {str(e)}"
            notes.append(error_msg)
            if debug:
                logger.error(error_msg)
            continue
    
    # Create details DataFrame
    if all_violations:
        details = pd.DataFrame(all_violations)
    else:
        details = pd.DataFrame()
    
    # Analyze suspected causes
    suspected_causes = _analyze_suspected_causes(all_violations, df_a_common, df_b_common)
    
    # Create and return report first (before file operations)
    report = DivergenceReport(
        diverged=diverged,
        first_step=first_step,
        tripped_signals=tripped_signals,
        notes=notes,
        report_path="",  # Will be set after successful file creation
        events_csv_path="",  # Will be set after successful file creation
        details=details,
        suspected_causes=suspected_causes,
        debug_info=debug_info,
    )
    
    # Try to create output files, but don't fail if unable to
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate file paths
        report_path = os.path.join(output_dir, "divergence_report.json")
        events_csv_path = os.path.join(output_dir, "divergence_events.csv")
        
        # Save details to CSV
        if not details.empty:
            details.to_csv(events_csv_path, index=False)
            report.events_csv_path = events_csv_path
        
        # Save report
        def json_serializer(obj):
            """Custom JSON serializer for numpy types."""
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            elif hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            else:
                return str(obj)
        
        with open(report_path, 'w') as f:
            json.dump(report.__dict__, f, indent=2, default=json_serializer)
        
        report.report_path = report_path
        
    except Exception as e:
        # If file operations fail, add a note but still return the report
        report.notes.append(f"Warning: Could not save report files: {str(e)}")
    
    return report


def first_divergence_events(
    events_a: List[Event],
    events_b: List[Event],
    signals: List[str],
    k_consecutive: int = 2,
    window: int = 20,
    tolerance: float = 1.5,
    output_dir: str = "diff_analysis",
    debug: bool = False,
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
    return first_divergence(
        df_a, df_b, signals, k_consecutive, window, tolerance, output_dir, debug
    )


def _calculate_rolling_z_scores(
    series_a: pd.Series, series_b: pd.Series, window: int, debug: bool = False
) -> pd.Series:
    """Calculate rolling z-scores for difference between two series."""
    # Calculate difference
    diff = series_a - series_b

    if debug:
        logger.info(f"Difference statistics: mean={diff.mean():.4f}, std={diff.std():.4f}, "
                   f"min={diff.min():.4f}, max={diff.max():.4f}")

    # Calculate rolling mean and std with min_periods to handle edge cases better
    rolling_mean = diff.rolling(window=window, min_periods=max(1, window//2)).mean()
    rolling_std = diff.rolling(window=window, min_periods=max(1, window//2)).std()

    if debug:
        logger.info(f"Rolling statistics (window={window}): mean_mean={rolling_mean.mean():.4f}, "
                   f"mean_std={rolling_std.mean():.4f}")

    # Calculate z-scores, handling division by zero
    z_scores = pd.Series(index=diff.index, dtype=float)
    
    # Only calculate z-scores where rolling_std is not zero or NaN
    # Use a smaller threshold to be more sensitive
    valid_mask = (rolling_std > 1e-8) & (~rolling_std.isna()) & (~rolling_mean.isna())
    z_scores[valid_mask] = (diff[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]
    
    # Set invalid z-scores to NaN
    z_scores[~valid_mask] = float('nan')

    if debug:
        valid_z_scores = z_scores[valid_mask]
        if len(valid_z_scores) > 0:
            logger.info(f"Valid z-scores: count={len(valid_z_scores)}, "
                       f"mean={valid_z_scores.mean():.4f}, std={valid_z_scores.std():.4f}, "
                       f"max={valid_z_scores.max():.4f}, min={valid_z_scores.min():.4f}")

    return z_scores


def _find_k_consecutive_violations(
    z_scores: pd.Series, k: int, tolerance: float = 1.5, debug: bool = False
) -> List[dict]:
    """Find k consecutive violations above tolerance threshold."""
    violations = []

    # Define thresholds based on tolerance parameter
    upper_threshold = tolerance
    lower_threshold = -tolerance

    if debug:
        logger.info(f"Looking for {k} consecutive violations with tolerance {tolerance} "
                   f"(thresholds: {lower_threshold} to {upper_threshold})")

    # Find consecutive violations
    consecutive_count = 0
    start_idx = None
    total_violations = 0
    
    for i, z_score in enumerate(z_scores):
        if pd.isna(z_score):
            consecutive_count = 0
            start_idx = None
            continue

        # Check if current value is a violation
        is_violation = z_score > upper_threshold or z_score < lower_threshold
        
        if is_violation:
            total_violations += 1
            if consecutive_count == 0:
                start_idx = i
            consecutive_count += 1
            
            if debug and consecutive_count == 1:
                logger.info(f"First violation at index {i}: z_score={z_score:.3f}")
            
            # Check if we have k consecutive violations
            if consecutive_count >= k:
                violations.append({
                    "start_idx": start_idx,
                    "end_idx": i,
                    "z_score": z_score,
                    "type": "lower" if z_score < 0 else "upper",
                    "consecutive_count": consecutive_count,
                })
                if debug:
                    logger.info(f"Found {k} consecutive violations starting at index {start_idx}")
                # Return the first occurrence of k consecutive violations
                return violations
        else:
            # Reset consecutive count if not a violation
            if debug and consecutive_count > 0:
                logger.info(f"Breaking consecutive sequence at index {i}: z_score={z_score:.3f}")
            consecutive_count = 0
            start_idx = None

    if debug:
        logger.info(f"Total violations found: {total_violations}, consecutive violations: 0")

    return violations


def _analyze_suspected_causes(
    divergence_events: List[dict], df_a: pd.DataFrame, df_b: pd.DataFrame
) -> List[str]:
    """Analyze potential causes of divergence."""
    causes = []

    # Check for learning rate differences
    if "lr" in df_a.columns and "lr" in df_b.columns:
        lr_diff = abs(df_a["lr"] - df_b["lr"]).max()
        if lr_diff > 1e-6:
            causes.append("Learning rate differences detected")

    # Check for seed differences
    if "seed" in df_a.columns and "seed" in df_b.columns:
        seed_a = df_a["seed"].iloc[0] if not df_a["seed"].isna().all() else None
        seed_b = df_b["seed"].iloc[0] if not df_b["seed"].isna().all() else None
        if seed_a != seed_b:
            causes.append("Different random seeds used")

    # Check for git SHA differences
    if "git_sha" in df_a.columns and "git_sha" in df_b.columns:
        sha_a = df_a["git_sha"].iloc[0] if not df_a["git_sha"].isna().all() else None
        sha_b = df_b["git_sha"].iloc[0] if not df_b["git_sha"].isna().all() else None
        if sha_a != sha_b:
            causes.append("Different code versions (git SHA)")

    # Analyze which signals were most affected
    signal_counts = {}
    for event in divergence_events:
        # Handle both old and new data structures
        if isinstance(event, dict):
            if "signal" in event:
                signal = event["signal"]
            else:
                # This is a violation event from the new structure
                continue
        else:
            continue
            
        signal_counts[signal] = signal_counts.get(signal, 0) + 1

    if signal_counts:
        most_affected = max(signal_counts.items(), key=lambda x: x[1])
        causes.append(
            f"Most affected signal: {most_affected[0]} ({most_affected[1]} violations)"
        )

    # Check for sudden spikes in metrics
    for event in divergence_events:
        # Handle both old and new data structures
        if isinstance(event, dict):
            if "signal" in event:
                signal = event["signal"]
                step = event["step"]
            else:
                # This is a violation event from the new structure
                continue
        else:
            continue

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
        causes.append(
            "No obvious cause identified - may be due to numerical instability"
        )

    return causes


def _detect_relative_changes(
    series_a: pd.Series, series_b: pd.Series, window: int, tolerance: float, debug: bool = False
) -> List[dict]:
    """Detect relative changes between two series that might indicate divergence."""
    violations = []
    
    if debug:
        logger.info("Running relative change detection")
    
    # Calculate relative differences (percentage change from baseline)
    # Use the first window values as baseline
    if len(series_a) < window or len(series_b) < window:
        return violations
    
    baseline_a = series_a.iloc[:window].mean()
    baseline_b = series_b.iloc[:window].mean()
    
    if debug:
        logger.info(f"Baseline values: A={baseline_a:.4f}, B={baseline_b:.4f}")
    
    # Calculate relative changes from baseline
    rel_change_a = (series_a - baseline_a) / baseline_a
    rel_change_b = (series_b - baseline_b) / baseline_b
    
    # Find where relative changes diverge significantly
    rel_diff = rel_change_a - rel_change_b
    
    # Calculate rolling statistics of the relative difference
    rolling_mean = rel_diff.rolling(window=window, min_periods=max(1, window//2)).mean()
    rolling_std = rel_diff.rolling(window=window, min_periods=max(1, window//2)).std()
    
    # Find points where relative difference exceeds threshold
    threshold = tolerance * 0.1  # Use a smaller threshold for relative changes
    violations_found = 0
    
    for i, (mean_val, std_val) in enumerate(zip(rolling_mean, rolling_std)):
        if pd.isna(mean_val) or pd.isna(std_val) or std_val < 1e-8:
            continue
            
        # Check if relative difference is significant
        if abs(mean_val) > threshold:
            violations_found += 1
            violations.append({
                "start_idx": i,
                "end_idx": i,
                "z_score": mean_val / std_val if std_val > 1e-8 else 0,
                "type": "relative_change",
                "consecutive_count": 1,
                "relative_change": float(mean_val),
                "step": series_a.index[i] if i < len(series_a) else None
            })
            
            # Only return the first significant relative change
            if violations_found >= 1:
                break
    
    if debug and violations:
        logger.info(f"Found {len(violations)} relative change violations")
    
    return violations
