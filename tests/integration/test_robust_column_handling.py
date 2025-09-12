"""Tests for robust column handling in evaluation metrics."""

import pandas as pd
import pytest
import json
from unittest.mock import patch

from rldk.evals.metrics.throughput import evaluate_throughput
from rldk.evals.metrics.toxicity import evaluate_toxicity
from rldk.evals.metrics.bias import evaluate_bias
from rldk.evals.column_config import ColumnConfig, get_evaluation_kwargs, detect_columns, suggest_columns


class TestRobustColumnHandling:
    """Test cases for robust column handling in evaluation metrics."""
    
    def test_throughput_missing_events_column_with_alternatives(self):
        """Test throughput evaluation with missing events column but alternative columns available."""
        # Create test data with alternative column names
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
            "logs": [json.dumps(events_data)],  # Alternative column name
            "model_name": ["test-model"]
        })
        
        result = evaluate_throughput(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 1
        assert result["method"] == "event_log_analysis"
        assert "metrics" in result
    
    def test_throughput_missing_events_column_with_fallback_metrics(self):
        """Test throughput evaluation with missing events column but fallback metrics available."""
        data = pd.DataFrame({
            "tokens_per_second": [100, 120, 110, 105, 115],
            "model_name": ["test-model"]
        })
        
        result = evaluate_throughput(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 5
        assert result["method"] == "fallback_analysis"
        assert "metrics" in result
        assert "metric_used" in result["metrics"]
        assert result["metrics"]["metric_used"] == "tokens_per_second"
    
    def test_throughput_no_suitable_columns(self):
        """Test throughput evaluation with no suitable columns."""
        data = pd.DataFrame({
            "random_column": ["value1", "value2"],
            "another_column": [1, 2]
        })
        
        result = evaluate_throughput(data)
        
        assert result["score"] == 0.0
        assert result["num_samples"] == 0
        assert "error" in result
        assert "available_columns" in result
        assert "suggested_alternatives" in result
        assert "random_column" in result["available_columns"]
        assert "another_column" in result["available_columns"]
    
    def test_toxicity_missing_output_column_with_alternatives(self):
        """Test toxicity evaluation with missing output column but alternative columns available."""
        data = pd.DataFrame({
            "response": [
                "This is a helpful response.",
                "You should kill yourself.",
                "This is informative content."
            ]
        })
        
        result = evaluate_toxicity(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["method"] == "content_analysis"
        assert "metrics" in result
    
    def test_toxicity_missing_output_column_with_fallback_metrics(self):
        """Test toxicity evaluation with missing output column but fallback metrics available."""
        data = pd.DataFrame({
            "toxicity_score": [0.1, 0.8, 0.3, 0.9, 0.2],
            "model_name": ["test-model"]
        })
        
        result = evaluate_toxicity(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 5
        assert result["method"] == "fallback_analysis"
        assert "metrics" in result
        assert "metric_used" in result["metrics"]
        assert result["metrics"]["metric_used"] == "toxicity_score"
    
    def test_bias_missing_output_column_with_alternatives(self):
        """Test bias evaluation with missing output column but alternative columns available."""
        data = pd.DataFrame({
            "generated_text": [
                "Women are intelligent and capable leaders.",
                "Men are strong and confident in their abilities.",
                "Women are emotional and should not be leaders."
            ]
        })
        
        result = evaluate_bias(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["method"] == "demographic_analysis"
        assert "metrics" in result
    
    def test_bias_missing_output_column_with_fallback_metrics(self):
        """Test bias evaluation with missing output column but fallback metrics available."""
        data = pd.DataFrame({
            "bias_score": [0.2, 0.7, 0.3, 0.8, 0.1],
            "model_name": ["test-model"]
        })
        
        result = evaluate_bias(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 5
        assert result["method"] == "fallback_analysis"
        assert "metrics" in result
        assert "metric_used" in result["metrics"]
        assert result["metrics"]["metric_used"] == "bias_score"
    
    def test_custom_alternative_columns(self):
        """Test evaluation with custom alternative column names."""
        events_data = [
            {
                "event_type": "token_generated",
                "timestamp": "2024-01-01T10:00:00Z",
                "token_count": 50
            }
        ]
        
        data = pd.DataFrame({
            "custom_logs": [json.dumps(events_data)],
            "model_name": ["test-model"]
        })
        
        result = evaluate_throughput(
            data, 
            alternative_columns=["custom_logs", "my_events"],
            min_samples=1
        )
        
        assert result["score"] > 0
        assert result["num_samples"] == 1
    
    def test_disable_fallback_metrics(self):
        """Test evaluation with fallback metrics disabled."""
        data = pd.DataFrame({
            "tokens_per_second": [100, 120, 110],
            "model_name": ["test-model"]
        })
        
        result = evaluate_throughput(
            data, 
            fallback_to_other_metrics=False
        )
        
        assert result["score"] == 0.0
        assert result["num_samples"] == 0
        assert "error" in result
    
    def test_column_config_utility(self):
        """Test column configuration utility functions."""
        config = ColumnConfig()
        
        # Test getting default config
        throughput_config = config.get_config("throughput")
        assert "primary_column" in throughput_config
        assert "alternative_columns" in throughput_config
        assert "fallback_metrics" in throughput_config
        
        # Test setting primary column
        config.set_primary_column("throughput", "my_events")
        updated_config = config.get_config("throughput")
        assert updated_config["primary_column"] == "my_events"
        
        # Test adding alternative column
        config.add_alternative_column("throughput", "custom_logs")
        updated_config = config.get_config("throughput")
        assert "custom_logs" in updated_config["alternative_columns"]
    
    def test_detect_columns(self):
        """Test column detection functionality."""
        data_columns = ["response", "logs", "toxicity_score", "bias_score", "random_column"]
        
        detected = detect_columns(data_columns)
        
        assert "throughput" in detected
        assert "toxicity" in detected
        assert "bias" in detected
        
        # Check throughput detection
        assert detected["throughput"]["primary_found"] == False  # "events" not in data_columns
        assert "logs" in detected["throughput"]["alternatives_found"]
        
        # Check toxicity detection
        assert detected["toxicity"]["primary_found"] == False  # "output" not in data_columns
        assert "response" in detected["toxicity"]["alternatives_found"]
        assert "toxicity_score" in detected["toxicity"]["fallbacks_found"]
        
        # Check bias detection
        assert detected["bias"]["primary_found"] == False  # "output" not in data_columns
        assert "response" in detected["bias"]["alternatives_found"]
        assert "bias_score" in detected["bias"]["fallbacks_found"]
    
    def test_suggest_columns(self):
        """Test column suggestion functionality."""
        data_columns = ["response", "logs", "toxicity_score", "bias_score", "random_column"]
        
        suggestions = suggest_columns(data_columns)
        
        assert "throughput" in suggestions
        assert "toxicity" in suggestions
        assert "bias" in suggestions
        
        # Check that suggestions are provided
        assert len(suggestions["throughput"]) > 0
        assert len(suggestions["toxicity"]) > 0
        assert len(suggestions["bias"]) > 0
    
    def test_get_evaluation_kwargs(self):
        """Test getting evaluation kwargs with column configuration."""
        kwargs = get_evaluation_kwargs("throughput")
        
        assert "log_column" in kwargs
        assert "alternative_columns" in kwargs
        assert "fallback_to_other_metrics" in kwargs
        
        # Test with custom config
        custom_config = {
            "primary_column": "my_events",
            "alternative_columns": ["custom_logs"]
        }
        
        kwargs = get_evaluation_kwargs("throughput", custom_config)
        assert kwargs["log_column"] == "my_events"
        assert kwargs["alternative_columns"] == ["custom_logs"]
    
    def test_error_messages_include_helpful_information(self):
        """Test that error messages include helpful information."""
        data = pd.DataFrame({
            "random_column": ["value1", "value2"],
            "another_column": [1, 2]
        })
        
        result = evaluate_throughput(data)
        
        assert "error" in result
        assert "available_columns" in result
        assert "suggested_alternatives" in result
        assert "random_column" in result["available_columns"]
        assert "another_column" in result["available_columns"]
        
        # Check that the error message is informative
        error_msg = result["details"]
        assert "Required column 'events' not found" in error_msg
        assert "Available columns:" in error_msg
        assert "Tried alternatives:" in error_msg
    
    def test_throughput_with_multiple_fallback_metrics(self):
        """Test throughput evaluation with multiple fallback metrics available."""
        data = pd.DataFrame({
            "tokens_per_second": [100, 120, 110],
            "throughput_rate": [150, 180, 160],
            "processing_speed": [200, 220, 210],
            "model_name": ["test-model"]
        })
        
        result = evaluate_throughput(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["method"] == "fallback_analysis"
        assert "available_metrics" in result["raw_data"]
        assert len(result["raw_data"]["available_metrics"]) > 1
    
    def test_toxicity_with_multiple_fallback_metrics(self):
        """Test toxicity evaluation with multiple fallback metrics available."""
        data = pd.DataFrame({
            "toxicity_score": [0.1, 0.8, 0.3],
            "harm_score": [0.2, 0.7, 0.4],
            "safety_score": [0.9, 0.2, 0.6],
            "model_name": ["test-model"]
        })
        
        result = evaluate_toxicity(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["method"] == "fallback_analysis"
        assert "available_metrics" in result["raw_data"]
        assert len(result["raw_data"]["available_metrics"]) > 1
    
    def test_bias_with_multiple_fallback_metrics(self):
        """Test bias evaluation with multiple fallback metrics available."""
        data = pd.DataFrame({
            "bias_score": [0.2, 0.7, 0.3],
            "fairness_score": [0.8, 0.3, 0.7],
            "demographic_bias": [0.1, 0.6, 0.2],
            "model_name": ["test-model"]
        })
        
        result = evaluate_bias(data)
        
        assert result["score"] > 0
        assert result["num_samples"] == 3
        assert result["method"] == "fallback_analysis"
        assert "available_metrics" in result["raw_data"]
        assert len(result["raw_data"]["available_metrics"]) > 1


if __name__ == "__main__":
    pytest.main([__file__])