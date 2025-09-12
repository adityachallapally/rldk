"""Throughput evaluation metrics for RL Debug Kit."""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

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


def calculate_tokens_per_second(events: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """
    Calculate tokens per second from event logs.
    
    Args:
        events: List of throughput events
        
    Returns:
        Tuple of (mean_tokens_per_sec, std_tokens_per_sec, total_tokens)
    """
    if not events:
        return 0.0, 0.0, 0.0
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x.get("timestamp", 0))
    
    # Calculate time intervals and token rates
    intervals = []
    token_rates = []
    total_tokens = 0
    
    for i in range(1, len(sorted_events)):
        prev_event = sorted_events[i-1]
        curr_event = sorted_events[i]
        
        prev_time = prev_event.get("timestamp")
        curr_time = curr_event.get("timestamp")
        
        if prev_time is None or curr_time is None:
            continue
            
        # Convert timestamps to seconds if needed
        if isinstance(prev_time, str):
            try:
                prev_time = datetime.fromisoformat(prev_time.replace('Z', '+00:00')).timestamp()
            except:
                continue
                
        if isinstance(curr_time, str):
            try:
                curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00')).timestamp()
            except:
                continue
        
        time_interval = curr_time - prev_time
        
        if time_interval <= 0:
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
            if processing_time > 0:
                tokens_this_interval = batch_size / processing_time
                total_tokens += batch_size
        
        if tokens_this_interval > 0:
            token_rate = tokens_this_interval / time_interval
            intervals.append(time_interval)
            token_rates.append(token_rate)
    
    if not token_rates:
        return 0.0, 0.0, total_tokens
    
    mean_tokens_per_sec = np.mean(token_rates)
    std_tokens_per_sec = np.std(token_rates)
    
    return mean_tokens_per_sec, std_tokens_per_sec, total_tokens


def calculate_confidence_interval(scores: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for throughput scores.
    
    Args:
        scores: List of throughput scores
        confidence_level: Confidence level (default: 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not scores:
        return (0.0, 0.0)
    
    if len(scores) < 2:
        score = scores[0] if scores else 0.0
        return (score, score)
    
    try:
        # Use bootstrap method for confidence interval
        # Check if we have enough samples for bootstrap
        if len(scores) < 3:
            # Not enough samples for bootstrap, use normal approximation
            mean_score = np.mean(scores)
            std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = z_score * std_score / np.sqrt(len(scores))
            return mean_score - margin_of_error, mean_score + margin_of_error
        
        bootstrap_result = stats.bootstrap((scores,), np.mean, confidence_level=confidence_level, n_resamples=min(1000, len(scores) * 10))
        return bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high
    except Exception as e:
        logger.warning(f"Bootstrap failed, using normal approximation: {e}")
        # Fallback to normal approximation
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_score / np.sqrt(len(scores))
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
            - alternative_columns: List of alternative column names to try (default: ["logs", "event_logs", "training_logs"])
            - confidence_level: Confidence level for intervals (default: 0.95)
            - min_samples: Minimum samples required for evaluation (default: 10)
            - fallback_to_other_metrics: Whether to try alternative metrics if no event logs (default: True)
            
    Returns:
        Dictionary with throughput score, details, and confidence intervals
    """
    log_column = kwargs.get("log_column", "events")
    alternative_columns = kwargs.get("alternative_columns", ["logs", "event_logs", "training_logs", "metrics", "performance_logs"])
    confidence_level = kwargs.get("confidence_level", 0.95)
    min_samples = kwargs.get("min_samples", 3)
    fallback_to_other_metrics = kwargs.get("fallback_to_other_metrics", True)
    
    logger.info("Starting throughput evaluation")
    
    # Try to find the log column with fallbacks
    actual_log_column = None
    available_columns = list(data.columns)
    
    # First try the specified column
    if log_column in data.columns:
        actual_log_column = log_column
    else:
        # Try alternative columns
        for alt_col in alternative_columns:
            if alt_col in data.columns:
                actual_log_column = alt_col
                logger.info(f"Using alternative column '{alt_col}' instead of '{log_column}'")
                break
    
    # If no log column found, provide detailed error message
    if actual_log_column is None:
        error_msg = f"Required column '{log_column}' not found in data. "
        error_msg += f"Available columns: {available_columns}. "
        error_msg += f"Tried alternatives: {alternative_columns}. "
        
        if fallback_to_other_metrics:
            error_msg += "Consider using alternative metrics like 'tokens_per_second', 'throughput_rate', or 'processing_speed' if available."
        
        logger.warning(error_msg)
        return {
            "score": 0.0,
            "details": error_msg,
            "method": "event_log_analysis",
            "num_samples": 0,
            "error": "missing_log_column",
            "available_columns": available_columns,
            "suggested_alternatives": alternative_columns
        }
    
    # Parse event logs
    all_events = []
    valid_samples = 0
    
    for idx, row in data.iterrows():
        try:
            events_raw = row[actual_log_column]
            
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
    
    # If no valid event logs found, try fallback metrics
    if valid_samples < min_samples and fallback_to_other_metrics:
        logger.info("No valid event logs found, trying fallback throughput metrics")
        return _evaluate_throughput_fallback(data, **kwargs)
    
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
    mean_tokens_per_sec, std_tokens_per_sec, total_tokens = calculate_tokens_per_second(all_events)
    
    # Normalize score to [0, 1] range (assuming reasonable max of 1000 tokens/sec)
    max_expected_throughput = 1000.0
    normalized_score = min(1.0, mean_tokens_per_sec / max_expected_throughput)
    
    # Calculate confidence interval
    if len(all_events) >= 2:
        # Extract individual token rates for confidence interval calculation
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
                except:
                    continue
                    
            if isinstance(curr_time, str):
                try:
                    curr_time = datetime.fromisoformat(curr_time.replace('Z', '+00:00')).timestamp()
                except:
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
                if processing_time > 0:
                    tokens_this_interval = batch_size / processing_time
            
            if tokens_this_interval > 0:
                token_rate = tokens_this_interval / time_interval
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
        }
    }


def _evaluate_throughput_fallback(data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Fallback throughput evaluation using alternative metrics when event logs are not available.
    
    Args:
        data: Training run data
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with throughput score and details
    """
    logger.info("Using fallback throughput evaluation")
    
    # Look for alternative throughput metrics
    throughput_columns = [
        "tokens_per_second", "throughput_rate", "processing_speed", 
        "inference_speed", "batch_throughput", "tps", "throughput"
    ]
    
    found_metrics = []
    for col in throughput_columns:
        if col in data.columns:
            values = pd.to_numeric(data[col], errors='coerce').dropna()
            if len(values) > 0:
                found_metrics.append((col, values))
    
    if not found_metrics:
        logger.warning("No alternative throughput metrics found")
        return {
            "score": 0.0,
            "details": "No event logs or alternative throughput metrics found",
            "method": "fallback_analysis",
            "num_samples": 0,
            "error": "no_throughput_data",
            "available_columns": list(data.columns),
            "suggested_metrics": throughput_columns
        }
    
    # Use the first available metric
    metric_name, values = found_metrics[0]
    mean_throughput = values.mean()
    std_throughput = values.std()
    
    # Normalize score to [0, 1] range (assuming reasonable max of 1000 tokens/sec)
    max_expected_throughput = 1000.0
    normalized_score = min(1.0, mean_throughput / max_expected_throughput)
    
    # Calculate stability
    throughput_stability = max(0, 1 - std_throughput / mean_throughput) if mean_throughput > 0 else 0
    
    logger.info(f"Fallback throughput evaluation complete: {mean_throughput:.2f} {metric_name} (score: {normalized_score:.3f})")
    
    return {
        "score": float(normalized_score),
        "details": f"Throughput: {mean_throughput:.2f} ± {std_throughput:.2f} {metric_name}",
        "method": "fallback_analysis",
        "num_samples": len(values),
        "metrics": {
            f"mean_{metric_name}": float(mean_throughput),
            f"std_{metric_name}": float(std_throughput),
            "throughput_stability": float(throughput_stability),
            "metric_used": metric_name
        },
        "raw_data": {
            "fallback_used": True,
            "available_metrics": [name for name, _ in found_metrics]
        }
    }