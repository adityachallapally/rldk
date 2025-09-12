#!/usr/bin/env python3
"""
Example demonstrating robust column handling in evaluation metrics.

This example shows how the evaluation metrics now gracefully handle missing
required columns by:
1. Trying alternative column names
2. Falling back to alternative metrics
3. Providing clear error messages with suggestions
4. Using configuration utilities for easy setup
"""

import json
import pandas as pd
from rldk.evals.metrics.throughput import evaluate_throughput
from rldk.evals.metrics.toxicity import evaluate_toxicity
from rldk.evals.metrics.bias import evaluate_bias
from rldk.evals.column_config import (
    ColumnConfig, get_evaluation_kwargs, detect_columns, suggest_columns
)


def example_1_missing_primary_columns():
    """Example 1: Missing primary columns but alternative columns available."""
    print("=" * 60)
    print("Example 1: Missing primary columns but alternatives available")
    print("=" * 60)
    
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
        "logs": [json.dumps(events_data)],  # Alternative to "events"
        "response": [  # Alternative to "output"
            "This is a helpful response.",
            "You should kill yourself.",
            "This is informative content."
        ],
        "model_name": ["test-model"]
    })
    
    print("Data columns:", list(data.columns))
    print()
    
    # Test throughput evaluation
    print("Testing throughput evaluation...")
    result = evaluate_throughput(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"Samples: {result['num_samples']}")
    print()
    
    # Test toxicity evaluation
    print("Testing toxicity evaluation...")
    result = evaluate_toxicity(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"Samples: {result['num_samples']}")
    print()
    
    # Test bias evaluation
    print("Testing bias evaluation...")
    result = evaluate_bias(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"Samples: {result['num_samples']}")
    print()


def example_2_fallback_metrics():
    """Example 2: Missing primary columns but fallback metrics available."""
    print("=" * 60)
    print("Example 2: Missing primary columns but fallback metrics available")
    print("=" * 60)
    
    data = pd.DataFrame({
        "tokens_per_second": [100, 120, 110, 105, 115],
        "toxicity_score": [0.1, 0.8, 0.3, 0.9, 0.2],
        "bias_score": [0.2, 0.7, 0.3, 0.8, 0.1],
        "model_name": ["test-model"]
    })
    
    print("Data columns:", list(data.columns))
    print()
    
    # Test throughput evaluation with fallback
    print("Testing throughput evaluation with fallback...")
    result = evaluate_throughput(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"Samples: {result['num_samples']}")
    if "metric_used" in result.get("metrics", {}):
        print(f"Metric used: {result['metrics']['metric_used']}")
    print()
    
    # Test toxicity evaluation with fallback
    print("Testing toxicity evaluation with fallback...")
    result = evaluate_toxicity(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"Samples: {result['num_samples']}")
    if "metric_used" in result.get("metrics", {}):
        print(f"Metric used: {result['metrics']['metric_used']}")
    print()
    
    # Test bias evaluation with fallback
    print("Testing bias evaluation with fallback...")
    result = evaluate_bias(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Method: {result['method']}")
    print(f"Samples: {result['num_samples']}")
    if "metric_used" in result.get("metrics", {}):
        print(f"Metric used: {result['metrics']['metric_used']}")
    print()


def example_3_no_suitable_columns():
    """Example 3: No suitable columns - graceful error handling."""
    print("=" * 60)
    print("Example 3: No suitable columns - graceful error handling")
    print("=" * 60)
    
    data = pd.DataFrame({
        "random_column": ["value1", "value2"],
        "another_column": [1, 2]
    })
    
    print("Data columns:", list(data.columns))
    print()
    
    # Test throughput evaluation
    print("Testing throughput evaluation...")
    result = evaluate_throughput(data)
    print(f"Score: {result['score']:.3f}")
    print(f"Error: {result.get('error', 'None')}")
    print(f"Details: {result['details']}")
    print(f"Available columns: {result.get('available_columns', [])}")
    print(f"Suggested alternatives: {result.get('suggested_alternatives', [])}")
    print()


def example_4_column_configuration():
    """Example 4: Using column configuration utilities."""
    print("=" * 60)
    print("Example 4: Using column configuration utilities")
    print("=" * 60)
    
    # Create a custom configuration
    config = ColumnConfig()
    
    # Set custom primary columns
    config.set_primary_column("throughput", "my_events")
    config.set_primary_column("toxicity", "my_output")
    config.set_primary_column("bias", "my_output")
    
    # Add custom alternative columns
    config.add_alternative_column("throughput", "custom_logs")
    config.add_alternative_column("toxicity", "custom_response")
    config.add_alternative_column("bias", "custom_response")
    
    # Add custom fallback metrics
    config.add_fallback_metric("throughput", "custom_throughput")
    config.add_fallback_metric("toxicity", "custom_toxicity")
    config.add_fallback_metric("bias", "custom_bias")
    
    print("Custom configuration created:")
    for metric in ["throughput", "toxicity", "bias"]:
        print(f"  {metric}: {config.get_config(metric)}")
    print()
    
    # Test column detection
    data_columns = ["my_events", "custom_logs", "custom_response", "custom_throughput"]
    detected = detect_columns(data_columns)
    
    print("Column detection results:")
    for metric, detection in detected.items():
        print(f"  {metric}:")
        print(f"    Primary found: {detection['primary_found']}")
        print(f"    Alternatives found: {detection['alternatives_found']}")
        print(f"    Fallbacks found: {detection['fallbacks_found']}")
    print()
    
    # Test column suggestions
    suggestions = suggest_columns(data_columns)
    print("Column suggestions:")
    for metric, suggestions_list in suggestions.items():
        print(f"  {metric}: {suggestions_list}")
    print()
    
    # Test evaluation kwargs generation
    print("Evaluation kwargs:")
    for metric in ["throughput", "toxicity", "bias"]:
        kwargs = get_evaluation_kwargs(metric)
        print(f"  {metric}: {kwargs}")
    print()


def example_5_custom_evaluation():
    """Example 5: Custom evaluation with specific column mappings."""
    print("=" * 60)
    print("Example 5: Custom evaluation with specific column mappings")
    print("=" * 60)
    
    # Create test data with custom column names
    events_data = [
        {
            "event_type": "token_generated",
            "timestamp": "2024-01-01T10:00:00Z",
            "token_count": 50
        }
    ]
    
    data = pd.DataFrame({
        "custom_events": [json.dumps(events_data)],
        "custom_output": [
            "This is a helpful response.",
            "You should kill yourself.",
            "This is informative content."
        ],
        "model_name": ["test-model"]
    })
    
    print("Data columns:", list(data.columns))
    print()
    
    # Test with custom column mappings
    print("Testing with custom column mappings...")
    
    # Throughput with custom log column
    result = evaluate_throughput(
        data,
        log_column="custom_events",
        alternative_columns=["custom_logs", "my_events"],
        min_samples=1
    )
    print(f"Throughput - Score: {result['score']:.3f}, Method: {result['method']}")
    
    # Toxicity with custom output column
    result = evaluate_toxicity(
        data,
        output_column="custom_output",
        alternative_columns=["custom_response", "my_output"],
        min_samples=3
    )
    print(f"Toxicity - Score: {result['score']:.3f}, Method: {result['method']}")
    
    # Bias with custom output column
    result = evaluate_bias(
        data,
        output_column="custom_output",
        alternative_columns=["custom_response", "my_output"],
        min_samples=3
    )
    print(f"Bias - Score: {result['score']:.3f}, Method: {result['method']}")
    print()


def main():
    """Run all examples."""
    print("Robust Column Handling Examples")
    print("===============================")
    print()
    
    try:
        example_1_missing_primary_columns()
        example_2_fallback_metrics()
        example_3_no_suitable_columns()
        example_4_column_configuration()
        example_5_custom_evaluation()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()