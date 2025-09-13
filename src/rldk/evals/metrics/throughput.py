"""Throughput evaluation metrics for RL Debug Kit."""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

from ...utils.math_utils import try_divide, safe_rate, nan_aware_mean, nan_aware_std

logger = logging.getLogger(__name__)


def parse_event_logs(log_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse event logs to extract token counts and timestamps.
    
    Args:
        log_data: List of event log entries
        
    Returns:
        List of parsed throughput events with token counts and timestamps
    """
    throughput_events = []
    
    for event in log_data:
        if not isinstance(event, dict):
            continue
            
        # Look for token generation events
        if event.get("event_type") in ["token_generated", "generation_complete", "inference_step"]:
            timestamp = event.get("timestamp")
            token_count = event.get("token_count") or event.get("tokens_generated") or event.get("num_tokens")
            
            if timestamp and token_count is not None:
                throughput_events.append({
                    "timestamp": timestamp,
                    "token_count": int(token_count),
                    "event_type": event.get("event_type")
                })
        
        # Look for batch processing events
        elif event.get("event_type") in ["batch_complete", "training_step"]:
            timestamp = event.get("timestamp")
            batch_size = event.get("batch_size") or event.get("samples_processed")
            processing_time = event.get("processing_time") or event.get("step_time")
            
            if timestamp and batch_size is not None and processing_time is not None:
                throughput_events.append({
                    "timestamp": timestamp,
                    "batch_size": int(batch_size),
                    "processing_time": float(processing_time),
                    "event_type": event.get("event_type")
                })
    
    return throughput_events


def calculate_tokens_per_second(events: List[Dict[str, Any]]) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Calculate tokens per second from event logs with robust division handling.
    
    Args:
        events: List of throughput events
        
    Returns:
        Tuple of (mean_tokens_per_sec, std_tokens_per_sec, total_tokens, counters)
        where counters is a dict with sample tracking information
    """
    if not events:
        return 0.0, 0.0, 0.0, {
            "samples_seen": 0,
            "samples_used": 0,
            "zero_denominator_skipped": 0,
            "non_positive_time_skipped": 0,
            "other_skip_reasons": []
        }
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x.get("timestamp", 0))
    
    # Calculate time intervals and token rates with robust division
    intervals = []
    token_rates = []
    total_tokens = 0
    
    # Counters for provenance tracking
    samples_seen = 0
    samples_used = 0
    zero_denominator_skipped = 0
    non_positive_time_skipped = 0
    other_skip_reasons = []
    
    for i in range(1, len(sorted_events)):
        prev_event = sorted_events[i-1]
        curr_event = sorted_events[i]
        
        prev_time = prev_event.get("timestamp")
        curr_time = curr_event.get("timestamp")
        
        if prev_time is None or curr_time is None:
            other_skip_reasons.append("missing_timestamp")
            continue
            
        # Convert timestamps to seconds if needed
        if isinstance(prev_time, str):
            try:
                prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse previous timestamp '{prev_time}': {e}")
                other_skip_reasons.append("invalid_previous_timestamp")
                continue
                
        if isinstance(curr_time, str):
            try:
                curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00')).timestamp()
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to parse current timestamp '{curr_time}': {e}")
                other_skip_reasons.append("invalid_current_timestamp")
                continue
        
        time_interval = curr_time - prev_time
        samples_seen += 1
        
        if time_interval <= 0:
            non_positive_time_skipped += 1
            logger.debug(f"Skipping non-positive time interval: {time_interval}")
            continue
            
        # Calculate tokens for this interval
        tokens_this_interval = 0
        
        if "token_count" in curr_event:
            tokens_this_interval = curr_event["token_count"]
            total_tokens += tokens_this_interval
        elif "batch_size" in curr_event and "processing_time" in curr_event:
            # For batch events, calculate effective tokens per second
            batch_size = curr_event["batch_size"]
            processing_time = curr_event["processing_time"]
            
            # Use robust division for processing time
            tokens_per_sec, used, reason = safe_rate(batch_size, processing_time, on_zero="skip")
            if used:
                tokens_this_interval = tokens_per_sec
                total_tokens += batch_size
            else:
                if reason == "zero_denominator_skipped":
                    zero_denominator_skipped += 1
                other_skip_reasons.append(f"batch_processing_{reason}")
                continue
        
        if tokens_this_interval > 0:
            # Use robust division for time interval
            token_rate, used, reason = safe_rate(tokens_this_interval, time_interval, on_zero="skip")
            if used:
                intervals.append(time_interval)
                token_rates.append(token_rate)
                samples_used += 1
            else:
                if reason == "zero_denominator_skipped":
                    zero_denominator_skipped += 1
                other_skip_reasons.append(f"token_rate_{reason}")
        else:
            other_skip_reasons.append("no_tokens_in_interval")
    
    # Calculate statistics using nan-aware functions
    if not token_rates:
        mean_tokens_per_sec = 0.0
        std_tokens_per_sec = 0.0
    else:
        mean_tokens_per_sec = nan_aware_mean(token_rates)
        std_tokens_per_sec = nan_aware_std(token_rates)
    
    counters = {
        "samples_seen": samples_seen,
        "samples_used": samples_used,
        "zero_denominator_skipped": zero_denominator_skipped,
        "non_positive_time_skipped": non_positive_time_skipped,
        "other_skip_reasons": other_skip_reasons
    }
    
    return mean_tokens_per_sec, std_tokens_per_sec, total_tokens, counters


def calculate_confidence_interval(scores: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for throughput scores using nan-aware functions.
    
    Args:
        scores: List of throughput scores (may contain NaN values)
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 0.0)
    
    # Filter out NaN values
    valid_scores = [s for s in scores if not (isinstance(s, float) and (s != s))]  # s != s checks for NaN
    
    if not valid_scores:
        return (float("nan"), float("nan"))
    
    if len(valid_scores) < 2:
        score = valid_scores[0]
        return (score, score)
    
    try:
        # Use bootstrap method for confidence interval
        # Check if we have enough samples for bootstrap
        if len(valid_scores) < 3:
            # Not enough samples for bootstrap, use normal approximation
            mean_score = nan_aware_mean(valid_scores)
            std_score = nan_aware_std(valid_scores, ddof=1)
            if std_score != std_score:  # Check for NaN
                return (mean_score, mean_score)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * std_score / np.sqrt(len(valid_scores))
            return mean_score - margin_of_error, mean_score + margin_of_error
        
        bootstrap_result = stats.bootstrap((valid_scores,), np.mean, confidence_level=confidence_level, n_resamples=min(1000, len(valid_scores) * 10))
        return bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high
    except Exception as e:
        logger.warning(f"Bootstrap failed, using normal approximation: {e}")
        # Fallback to normal approximation
        mean_score = nan_aware_mean(valid_scores)
        std_score = nan_aware_std(valid_scores, ddof=1)
        if std_score != std_score:  # Check for NaN
            return (mean_score, mean_score)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_score / np.sqrt(len(valid_scores))
        return mean_score - margin_of_error, mean_score + margin_of_error


def evaluate_throughput(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Evaluate model throughput and processing capacity.
    
    Measures tokens per second using elapsed time and token count from logs.
    Computes average throughput and confidence intervals.
    
    Args:
        data: Training run data containing event logs
        **kwargs: Additional arguments including:
            - log_column: Column name containing JSON event logs (default: "events")
            - confidence_level: Confidence level for intervals (default: 0.95)
            - min_samples: Minimum samples required for evaluation (default: 10)
            
    Returns:
        Dictionary with throughput score, details, and confidence intervals
    """
    log_column = kwargs.get("log_column", "events")
    confidence_level = kwargs.get("confidence_level", 0.95)
    min_samples = kwargs.get("min_samples", 3)
    
    logger.info("Starting throughput evaluation")
    
    # Check if we have event logs
    if log_column not in data.columns:
        logger.warning(f"Log column '{log_column}' not found in data")
        return {
            "score": 0.0,
            "details": f"No event logs found in column '{log_column}'",
            "method": "event_log_analysis",
            "num_samples": 0,
            "error": "missing_log_column"
        }
    
    # Parse event logs
    all_events = []
    valid_samples = 0
    
    for idx, row in data.iterrows():
        try:
            events_raw = row[log_column]
            
            if isinstance(events_raw, str):
                events = json.loads(events_raw)
            elif isinstance(events_raw, list):
                events = events_raw
            else:
                continue
                
            if not isinstance(events, list):
                continue
                
            parsed_events = parse_event_logs(events)
            if parsed_events:
                all_events.extend(parsed_events)
                valid_samples += 1
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.debug(f"Failed to parse events at row {idx}: {e}")
            continue
    
    if valid_samples < min_samples:
        logger.warning(f"Insufficient valid samples: {valid_samples} < {min_samples}")
        return {
            "score": 0.0,
            "details": f"Insufficient valid samples ({valid_samples} < {min_samples})",
            "method": "event_log_analysis",
            "num_samples": valid_samples,
            "error": "insufficient_samples"
        }
    
    if not all_events:
        logger.warning("No valid throughput events found")
        return {
            "score": 0.0,
            "details": "No valid throughput events found in logs",
            "method": "event_log_analysis",
            "num_samples": valid_samples,
            "error": "no_throughput_events"
        }
    
    # Calculate throughput metrics
    mean_tokens_per_sec, std_tokens_per_sec, total_tokens, counters = calculate_tokens_per_second(all_events)
    
    # Normalize score to [0, 1] range (assuming reasonable max of 1000 tokens/sec)
    max_expected_throughput = 1000.0
    normalized_score = min(1.0, mean_tokens_per_sec / max_expected_throughput)
    
    # Calculate confidence interval using the token rates from calculate_tokens_per_second
    if len(all_events) >= 2:
        # Re-calculate token rates for confidence interval using robust division
        token_rates = []
        sorted_events = sorted(all_events, key=lambda x: x.get("timestamp", 0))
        
        for i in range(1, len(sorted_events)):
            prev_event = sorted_events[i-1]
            curr_event = sorted_events[i]
            
            prev_time = prev_event.get("timestamp")
            curr_time = curr_event.get("timestamp")
            
            if prev_time is None or curr_time is None:
                continue
                
            # Convert timestamps
            if isinstance(prev_time, str):
                try:
                    prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00')).timestamp()
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse previous timestamp '{prev_time}': {e}")
                    continue
                    
            if isinstance(curr_time, str):
                try:
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00')).timestamp()
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse current timestamp '{curr_time}': {e}")
                    continue
            
            time_interval = curr_time - prev_time
            if time_interval <= 0:
                continue
                
            tokens_this_interval = 0
            if "token_count" in curr_event:
                tokens_this_interval = curr_event["token_count"]
            elif "batch_size" in curr_event and "processing_time" in curr_event:
                batch_size = curr_event["batch_size"]
                processing_time = curr_event["processing_time"]
                
                # Use robust division for processing time
                tokens_per_sec, used, _ = safe_rate(batch_size, processing_time, on_zero="skip")
                if used:
                    tokens_this_interval = tokens_per_sec
            
            if tokens_this_interval > 0:
                # Use robust division for time interval
                token_rate, used, _ = safe_rate(tokens_this_interval, time_interval, on_zero="skip")
                if used:
                    token_rates.append(token_rate)
        
        if token_rates:
            ci_lower, ci_upper = calculate_confidence_interval(token_rates, confidence_level)
            ci_lower_norm = min(1.0, ci_lower / max_expected_throughput)
            ci_upper_norm = min(1.0, ci_upper / max_expected_throughput)
        else:
            ci_lower_norm = ci_upper_norm = normalized_score
    else:
        ci_lower_norm = ci_upper_norm = normalized_score
    
    # Calculate additional metrics
    throughput_stability = max(0, 1 - std_tokens_per_sec / mean_tokens_per_sec) if mean_tokens_per_sec > 0 else 0
    
    logger.info(f"Throughput evaluation complete: {mean_tokens_per_sec:.2f} tokens/sec (score: {normalized_score:.3f})")
    
    return {
        "score": float(normalized_score),
        "details": f"Throughput: {mean_tokens_per_sec:.2f} ± {std_tokens_per_sec:.2f} tokens/sec",
        "method": "event_log_analysis",
        "num_samples": valid_samples,
        "metrics": {
            "mean_tokens_per_sec": float(mean_tokens_per_sec),
            "std_tokens_per_sec": float(std_tokens_per_sec),
            "total_tokens": int(total_tokens),
            "throughput_stability": float(throughput_stability),
            "confidence_interval": {
                "lower": float(ci_lower_norm),
                "upper": float(ci_upper_norm),
                "level": confidence_level
            }
        },
        "raw_data": {
            "num_events": len(all_events),
            "event_types": list(set(event.get("event_type") for event in all_events))
        },
        "counters": counters
    }