"""Tests for throughput evaluation metrics."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from rldk.evals.metrics.throughput import (
    calculate_confidence_interval,
    calculate_tokens_per_second,
    evaluate_throughput,
    parse_event_logs,
)


class TestThroughputEvaluation:
    """Test cases for throughput evaluation."""

    def test_parse_event_logs_valid_data(self):
        """Test parsing valid event logs."""
        log_data = [
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:00Z",
                "token_count": 50
            },
            {
                "event_type": "batch_complete",
                "timestamp": "2024-01-01T10:00:01Z",
                "batch_size": 100,
                "processing_time": 2.5
            }
        ]

        result = parse_event_logs(log_data)

        assert len(result) == 2
        assert result[0]["event_type"] == "token_generated"
        assert result[0]["token_count"] == 50
        assert result[1]["event_type"] == "batch_complete"
        assert result[1]["batch_size"] == 100
        assert result[1]["processing_time"] == 2.5

    def test_parse_event_logs_invalid_data(self):
        """Test parsing invalid event logs."""
        log_data = [
            {"event_type": "unknown", "timestamp": "2024-01-01T10:00:00Z"},
            {"event_type": "token_generated"},  # Missing timestamp
            {"event_type": "token_generated", "timestamp": "2024-01-01T10:00:00Z"},  # Missing token_count
            None,
            "invalid"
        ]

        result = parse_event_logs(log_data)

        assert len(result) == 0

    def test_calculate_tokens_per_second(self):
        """Test token per second calculation."""
        events = [
            {"timestamp": 1000, "token_count": 50},
            {"timestamp": 1001, "token_count": 60},
            {"timestamp": 1002, "token_count": 55}
        ]

        mean_tokens_per_sec, std_tokens_per_sec, total_tokens = calculate_tokens_per_second(events)

        assert mean_tokens_per_sec > 0
        assert std_tokens_per_sec >= 0
        assert total_tokens == 165  # 50 + 60 + 55

    def test_calculate_tokens_per_second_empty_data(self):
        """Test token per second calculation with empty data."""
        events = []

        mean_tokens_per_sec, std_tokens_per_sec, total_tokens = calculate_tokens_per_second(events)

        assert mean_tokens_per_sec == 0.0
        assert std_tokens_per_sec == 0.0
        assert total_tokens == 0.0

    def test_calculate_confidence_interval(self):
        """Test confidence interval calculation."""
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]

        lower, upper = calculate_confidence_interval(scores, confidence_level=0.95)

        assert lower < upper
        assert 0.5 <= lower <= 0.9
        assert 0.5 <= upper <= 0.9

    def test_calculate_confidence_interval_single_value(self):
        """Test confidence interval with single value."""
        scores = [0.5]

        lower, upper = calculate_confidence_interval(scores)

        assert lower == 0.5
        assert upper == 0.5

    def test_evaluate_throughput_with_valid_data(self):
        """Test throughput evaluation with valid data."""
        # Create test data with event logs
        events_data = [
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:00Z",
                "token_count": 50
            },
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:01Z",
                "token_count": 60
            }
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert "score" in result
        assert "details" in result
        assert "method" in result
        assert "num_samples" in result
        assert "metrics" in result
        assert result["method"] == "event_log_analysis"
        assert result["num_samples"] == 1
        assert 0 <= result["score"] <= 1

    def test_evaluate_throughput_missing_log_column(self):
        """Test throughput evaluation with missing log column."""
        data = pd.DataFrame({
            "output": ["test output"],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert result["score"] == 0.0
        assert "missing_log_column" in result["error"]
        assert result["num_samples"] == 0

    def test_evaluate_throughput_insufficient_samples(self):
        """Test throughput evaluation with insufficient samples."""
        events_data = [
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:00Z",
                "token_count": 50
            }
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data, min_samples=10)

        assert result["score"] == 0.0
        assert "insufficient_samples" in result["error"]
        assert result["num_samples"] == 1

    def test_evaluate_throughput_no_throughput_events(self):
        """Test throughput evaluation with no throughput events."""
        events_data = [
            {
                "event_type": "unknown_event",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert result["score"] == 0.0
        assert "no_throughput_events" in result["error"]
        assert result["num_samples"] == 1

    def test_evaluate_throughput_with_batch_events(self):
        """Test throughput evaluation with batch processing events."""
        events_data = [
            {
                "event_type": "batch_complete",
                "timestamp": "2024-01-01T10:00:00Z",
                "batch_size": 100,
                "processing_time": 2.0
            },
            {
                "event_type": "batch_complete",
                "timestamp": "2024-01-01T10:00:02Z",
                "batch_size": 150,
                "processing_time": 3.0
            }
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert result["score"] > 0
        assert "metrics" in result
        assert "mean_tokens_per_sec" in result["metrics"]
        assert "throughput_stability" in result["metrics"]

    def test_evaluate_throughput_with_malformed_json(self):
        """Test throughput evaluation with malformed JSON."""
        data = pd.DataFrame({
            "events": ["invalid json"],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert result["score"] == 0.0
        assert result["num_samples"] == 0

    def test_evaluate_throughput_with_string_timestamps(self):
        """Test throughput evaluation with string timestamps."""
        events_data = [
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:00Z",
                "token_count": 50
            },
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:01Z",
                "token_count": 60
            }
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert result["score"] > 0
        assert "metrics" in result
        assert "mean_tokens_per_sec" in result["metrics"]

    def test_evaluate_throughput_confidence_intervals(self):
        """Test that confidence intervals are calculated correctly."""
        events_data = []
        for i in range(20):  # Create enough samples for confidence interval
            events_data.append({
                "event_type": "token_generated",
                "timestamp": f"2024-01-01T10:00:{i:02d}Z",
                "token_count": 50 + i
            })

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert "metrics" in result
        assert "confidence_interval" in result["metrics"]
        ci = result["metrics"]["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert "level" in ci
        assert ci["lower"] <= ci["upper"]
        assert ci["level"] == 0.95

    def test_evaluate_throughput_custom_parameters(self):
        """Test throughput evaluation with custom parameters."""
        events_data = [
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:00Z",
                "token_count": 50
            }
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(
            data,
            log_column="events",
            confidence_level=0.90,
            min_samples=1
        )

        assert result["score"] > 0
        assert result["num_samples"] == 1

    def test_evaluate_throughput_throughput_stability(self):
        """Test throughput stability calculation."""
        # Create events with varying token counts to test stability
        events_data = [
            {"event_type": "token_generated", "timestamp": "2024-01-01T10:00:00Z", "token_count": 50},
            {"event_type": "token_generated", "timestamp": "2024-01-01T10:00:01Z", "token_count": 52},
            {"event_type": "token_generated", "timestamp": "2024-01-01T10:00:02Z", "token_count": 48},
            {"event_type": "token_generated", "timestamp": "2024-01-01T10:00:03Z", "token_count": 51},
            {"event_type": "token_generated", "timestamp": "2024-01-01T10:00:04Z", "token_count": 49}
        ]

        data = pd.DataFrame({
            "events": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert "metrics" in result
        assert "throughput_stability" in result["metrics"]
        stability = result["metrics"]["throughput_stability"]
        assert 0 <= stability <= 1

    def test_evaluate_throughput_with_real_log_file(self):
        """Test throughput evaluation with real log file data."""
        import os
        # Read the test log file using absolute path
        test_file = os.path.join(os.path.dirname(__file__), "..", "data", "throughput_log.jsonl")
        with open(test_file) as f:
            log_entries = [json.loads(line) for line in f]

        data = pd.DataFrame({
            "events": [json.dumps(log_entries)],
            "model_name": ["test-model"]
        })

        result = evaluate_throughput(data)

        assert result["score"] > 0
        assert result["num_samples"] == 1
        assert "metrics" in result
        assert "mean_tokens_per_sec" in result["metrics"]
        assert "total_tokens" in result["metrics"]
        assert result["metrics"]["total_tokens"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
