#!/usr/bin/env python3
"""
Minimal evaluation demo for RL Debug Kit.

This script demonstrates the basic usage of evaluation suites with proper
data requirements and shows how to handle missing columns gracefully.
"""

import numpy as np
import pandas as pd

from rldk.evals import run
from rldk.evals.schema import get_schema_for_suite, validate_eval_input


def create_sample_data():
    """Create sample data with required columns."""
    return pd.DataFrame({
        'step': [1, 2, 3, 4, 5],
        'output': [
            "This is a helpful response to the user's question.",
            "Another response that demonstrates model capabilities.",
            "Yet another response showing consistency.",
            "More helpful text that follows instructions.",
            "Final response that completes the conversation."
        ],
        'reward': [0.8, 0.9, 0.7, 0.85, 0.95],
        'kl_to_ref': [0.1, 0.12, 0.08, 0.11, 0.09]
    })


def create_data_with_synonyms():
    """Create data using column synonyms to demonstrate normalization."""
    return pd.DataFrame({
        'global_step': [1, 2, 3, 4, 5],  # Synonym for 'step'
        'response': [  # Synonym for 'output'
            "Response using synonym column names.",
            "Another response with normalized columns.",
            "Third response showing automatic mapping.",
            "Fourth response demonstrating flexibility.",
            "Final response with column synonyms."
        ],
        'reward_mean': [0.75, 0.82, 0.78, 0.88, 0.91]  # Synonym for 'reward'
    })


def create_minimal_data():
    """Create minimal data with only required columns."""
    return pd.DataFrame({
        'step': [1, 2, 3],
        'output': [
            "Minimal response one.",
            "Minimal response two.",
            "Minimal response three."
        ]
    })


def demo_basic_evaluation():
    """Demonstrate basic evaluation with full data."""
    print("=== Basic Evaluation Demo ===")
    print("Creating sample data with all columns...")

    data = create_sample_data()
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    print("\nRunning quick evaluation suite...")
    result = run(data, suite="quick")

    print(f"Overall Score: {result.overall_score}")
    print(f"Available Metrics: {result.available_fraction:.1%}")
    print(f"Warnings: {len(result.warnings)}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    print(f"Individual Scores: {result.scores}")
    return result


def demo_column_normalization():
    """Demonstrate automatic column normalization."""
    print("\n=== Column Normalization Demo ===")
    print("Creating data with synonym column names...")

    data = create_data_with_synonyms()
    print(f"Original columns: {list(data.columns)}")

    # Validate input to see normalization
    schema = get_schema_for_suite("quick")
    validated = validate_eval_input(data, schema, "quick")

    print(f"Normalized columns: {list(validated.data.columns)}")
    print(f"Column mappings: {validated.normalized_columns}")

    print("\nRunning evaluation with normalized data...")
    result = run(data, suite="quick")

    print(f"Overall Score: {result.overall_score}")
    print(f"Available Metrics: {result.available_fraction:.1%}")
    return result


def demo_minimal_data():
    """Demonstrate evaluation with minimal required data."""
    print("\n=== Minimal Data Demo ===")
    print("Creating minimal data with only required columns...")

    data = create_minimal_data()
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    print("\nRunning evaluation...")
    result = run(data, suite="quick")

    print(f"Overall Score: {result.overall_score}")
    print(f"Available Metrics: {result.available_fraction:.1%}")
    print(f"Warnings: {len(result.warnings)}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    return result


def demo_missing_required_column():
    """Demonstrate error handling for missing required columns."""
    print("\n=== Missing Required Column Demo ===")
    print("Creating data without required 'output' column...")

    # This should fail
    data = pd.DataFrame({
        'step': [1, 2, 3],
        'reward': [0.8, 0.9, 0.7]
        # Missing 'output' column
    })

    print(f"Data columns: {list(data.columns)}")

    try:
        result = run(data, suite="quick")
        print(f"Unexpected success: {result.overall_score}")
    except Exception as e:
        print(f"Expected error: {e}")

    return None


def demo_insufficient_data():
    """Demonstrate handling of insufficient data for certain metrics."""
    print("\n=== Insufficient Data Demo ===")
    print("Creating data that may not have enough samples for all metrics...")

    # Very small dataset
    data = pd.DataFrame({
        'step': [1, 2],
        'output': ["Short response 1", "Short response 2"]
    })

    print(f"Data shape: {data.shape}")

    result = run(data, suite="quick")

    print(f"Overall Score: {result.overall_score}")
    print(f"Available Metrics: {result.available_fraction:.1%}")
    print(f"Warnings: {len(result.warnings)}")

    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")

    return result


def main():
    """Run all demos."""
    print("RL Debug Kit - Minimal Evaluation Demo")
    print("=" * 50)

    try:
        # Run all demos
        demo_basic_evaluation()
        demo_column_normalization()
        demo_minimal_data()
        demo_missing_required_column()
        demo_insufficient_data()

        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        print("\nKey takeaways:")
        print("- Required columns: 'step' and 'output'")
        print("- Optional columns provide additional metrics")
        print("- Column synonyms are automatically normalized")
        print("- Missing required columns cause clear errors")
        print("- Missing optional columns produce warnings")
        print("- Overall score is None when no metrics are available")

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
