"""Throughput evaluation metrics for RL Debug Kit."""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...utils.math_utils import safe_divide, safe_rate_calculation

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
                tokens_this_interval = safe_rate_calculation(batch_size, processing_time, 0.0)
                total_tokens += batch_size

        if tokens_this_interval > 0 and time_interval > 0:
            token_rate = safe_rate_calculation(tokens_this_interval, time_interval, 0.0)
            # Always add both interval and rate together to maintain array consistency
            intervals.append(time_interval)
            token_rates.append(token_rate)

    if not token_rates:
        return 0.0, 0.0, total_tokens

    mean_tokens_per_sec = np.mean(token_rates)
    std_tokens_per_sec = np.std(token_rates)

    return mean_tokens_per_sec, std_tokens_per_sec, total_tokens


def calculate_token_throughput(events: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Backward compatible helper that wraps :func:`calculate_tokens_per_second`."""

    mean, std, total = calculate_tokens_per_second(events)
    return float(mean), float(std), float(total)


def calculate_batch_throughput(events: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute batch throughput metrics while guarding against zero denominators."""

    if not events:
        return {
            "average_batches_per_sec": 0.0,
            "max_batches_per_sec": 0.0,
            "min_batches_per_sec": 0.0,
            "num_batches": 0.0,
        }

    rates: List[float] = []
    total_batches = 0.0

    for event in events:
        batch_size = event.get("batch_size")
        processing_time = event.get("processing_time")

        if batch_size is None or processing_time is None:
            continue

        rate = safe_rate_calculation(batch_size, processing_time, 0.0)
        rates.append(float(rate))
        total_batches += float(batch_size)

    if not rates:
        return {
            "average_batches_per_sec": 0.0,
            "max_batches_per_sec": 0.0,
            "min_batches_per_sec": 0.0,
            "num_batches": float(total_batches),
        }

    return {
        "average_batches_per_sec": float(np.mean(rates)),
        "max_batches_per_sec": float(np.max(rates)),
        "min_batches_per_sec": float(np.min(rates)),
        "num_batches": float(total_batches),
    }


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
            margin_of_error = safe_divide(z_score * std_score, np.sqrt(len(scores)), 0.0)
            return mean_score - margin_of_error, mean_score + margin_of_error

        bootstrap_result = stats.bootstrap((scores,), np.mean, confidence_level=confidence_level, n_resamples=min(1000, len(scores) * 10))
        return bootstrap_result.confidence_interval.low, bootstrap_result.confidence_interval.high
    except Exception as e:
        logger.warning(f"Bootstrap failed, using normal approximation: {e}")
        # Fallback to normal approximation
        mean_score = np.mean(scores)
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = safe_divide(z_score * std_score, np.sqrt(len(scores)), 0.0)
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
            "score": np.nan,
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
            "score": np.nan,
            "details": f"Insufficient valid samples ({valid_samples} < {min_samples})",
            "method": "event_log_analysis",
            "num_samples": valid_samples,
            "error": "insufficient_samples"
        }

    if not all_events:
        logger.warning("No valid throughput events found")
        return {
            "score": np.nan,
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
                tokens_this_interval = safe_rate_calculation(batch_size, processing_time, 0.0)

            if tokens_this_interval > 0 and time_interval > 0:
                token_rate = safe_rate_calculation(tokens_this_interval, time_interval, 0.0)
                # Add rate regardless of value to maintain consistency
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
    throughput_stability = max(0, 1 - safe_divide(std_tokens_per_sec, mean_tokens_per_sec, 0.0))

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
            "event_types": list({event.get("event_type") for event in all_events})
        }
    }
