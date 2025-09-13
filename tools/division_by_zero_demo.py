#!/usr/bin/env python3
"""Demo script showing robust division by zero handling in RLDK."""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import RLDK components
from src.rldk.utils.math_utils import try_divide, safe_percentage, safe_rate, nan_aware_mean
from src.rldk.evals.metrics.throughput import evaluate_throughput
from src.rldk.evals.suites import evaluate_consistency, evaluate_robustness, evaluate_efficiency


def create_test_data_with_zeros():
    """Create test data with various zero denominator scenarios."""
    # Create event logs with some problematic data
    events = []
    base_time = datetime.now()
    
    # Normal events
    for i in range(10):
        events.append({
            "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
            "token_count": 10 + i * 2
        })
    
    # Events with zero time intervals (same timestamp)
    events.append({
        "timestamp": (base_time + timedelta(seconds=10)).isoformat(),
        "token_count": 30
    })
    events.append({
        "timestamp": (base_time + timedelta(seconds=10)).isoformat(),  # Same timestamp
        "token_count": 35
    })
    
    # Events with zero processing time
    events.append({
        "timestamp": (base_time + timedelta(seconds=11)).isoformat(),
        "batch_size": 50,
        "processing_time": 0.0  # Zero processing time
    })
    
    # Events with negative time intervals
    events.append({
        "timestamp": (base_time + timedelta(seconds=9)).isoformat(),  # Earlier timestamp
        "token_count": 40
    })
    
    return events


def create_test_dataframe_with_zeros():
    """Create a test DataFrame with zero values in denominators."""
    n_samples = 50
    
    # Create data with some zeros and NaN values
    data = {
        "reward_mean": np.random.normal(0.5, 0.2, n_samples),
        "step": range(n_samples),
        "training_time": np.random.uniform(0.1, 1.0, n_samples),
        "memory_usage": np.random.uniform(1.0, 8.0, n_samples),
    }
    
    # Inject some zeros and NaN values
    data["reward_mean"][5:10] = 0.0  # Zero rewards
    data["training_time"][15:20] = 0.0  # Zero training time
    data["reward_mean"][25] = float('nan')  # NaN reward
    data["training_time"][30] = float('nan')  # NaN training time
    
    # Create event logs
    events = create_test_data_with_zeros()
    data["events"] = [json.dumps(events) for _ in range(n_samples)]
    
    return pd.DataFrame(data)


def demo_math_utils():
    """Demonstrate the math utilities with various scenarios."""
    print("🔢 Math Utilities Demo")
    print("=" * 50)
    
    # Test try_divide with different scenarios
    print("\n1. try_divide function:")
    print(f"  10 / 2 = {try_divide(10, 2)}")
    print(f"  10 / 0 (skip) = {try_divide(10, 0, on_zero='skip')}")
    print(f"  10 / 0 (zero) = {try_divide(10, 0, on_zero='zero')}")
    print(f"  10 / 0 (nan) = {try_divide(10, 0, on_zero='nan')}")
    print(f"  10 / -1 (skip) = {try_divide(10, -1, on_zero='skip')}")
    
    # Test safe_percentage
    print("\n2. safe_percentage function:")
    print(f"  25% of 100 = {safe_percentage(25, 100)}")
    print(f"  25% of 0 (skip) = {safe_percentage(25, 0, on_zero='skip')}")
    print(f"  25% of 0 (zero) = {safe_percentage(25, 0, on_zero='zero')}")
    
    # Test safe_rate
    print("\n3. safe_rate function:")
    print(f"  100 tokens in 10 seconds = {safe_rate(100, 10)}")
    print(f"  100 tokens in 0 seconds (skip) = {safe_rate(100, 0, on_zero='skip')}")
    print(f"  100 tokens in 0 seconds (zero) = {safe_rate(100, 0, on_zero='zero')}")
    
    # Test nan_aware functions
    print("\n4. NaN-aware functions:")
    values_with_nan = [1.0, 2.0, float('nan'), 4.0, 5.0]
    print(f"  Values: {values_with_nan}")
    print(f"  nan_aware_mean: {nan_aware_mean(values_with_nan)}")
    print(f"  Regular mean would be: {np.mean(values_with_nan)}")


def demo_throughput_metrics():
    """Demonstrate throughput metrics with counters."""
    print("\n\n📊 Throughput Metrics Demo")
    print("=" * 50)
    
    # Create test data
    data = create_test_dataframe_with_zeros()
    
    print(f"\nTest data shape: {data.shape}")
    print(f"Zero training times: {(data['training_time'] == 0).sum()}")
    print(f"NaN rewards: {data['reward_mean'].isna().sum()}")
    
    # Evaluate throughput
    print("\n1. Evaluating throughput with robust division:")
    result = evaluate_throughput(data)
    
    print(f"  Score: {result['score']:.3f}")
    print(f"  Details: {result['details']}")
    print(f"  Samples used: {result['counters']['samples_used']}")
    print(f"  Samples seen: {result['counters']['samples_seen']}")
    print(f"  Zero denominator skipped: {result['counters']['zero_denominator_skipped']}")
    print(f"  Non-positive time skipped: {result['counters']['non_positive_time_skipped']}")
    print(f"  Other skip reasons: {result['counters']['other_skip_reasons']}")
    
    # Show metrics
    if 'metrics' in result:
        print(f"\n  Mean tokens/sec: {result['metrics']['mean_tokens_per_sec']:.2f}")
        print(f"  Std tokens/sec: {result['metrics']['std_tokens_per_sec']:.2f}")
        print(f"  Total tokens: {result['metrics']['total_tokens']}")


def demo_suite_evaluations():
    """Demonstrate suite evaluations with robust division."""
    print("\n\n🎯 Suite Evaluations Demo")
    print("=" * 50)
    
    # Create test data
    data = create_test_dataframe_with_zeros()
    
    print(f"\nTest data with zeros and NaN values:")
    print(f"  Zero rewards: {(data['reward_mean'] == 0).sum()}")
    print(f"  Zero training times: {(data['training_time'] == 0).sum()}")
    print(f"  NaN rewards: {data['reward_mean'].isna().sum()}")
    
    # Test consistency evaluation
    print("\n1. Consistency evaluation:")
    consistency_result = evaluate_consistency(data)
    print(f"  Score: {consistency_result['score']:.3f}")
    print(f"  Details: {consistency_result['details']}")
    
    # Test robustness evaluation
    print("\n2. Robustness evaluation:")
    robustness_result = evaluate_robustness(data)
    print(f"  Score: {robustness_result['score']:.3f}")
    print(f"  Details: {robustness_result['details']}")
    
    # Test efficiency evaluation
    print("\n3. Efficiency evaluation:")
    efficiency_result = evaluate_efficiency(data)
    print(f"  Score: {efficiency_result['score']:.3f}")
    print(f"  Details: {efficiency_result['details']}")


def demo_nan_behavior():
    """Demonstrate NaN behavior when forced."""
    print("\n\n🧪 NaN Behavior Demo")
    print("=" * 50)
    
    # Create data that will force NaN results
    data = pd.DataFrame({
        "reward_mean": [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
        "step": [1, 2, 3, 4, 5],
        "training_time": [0.0, 0.0, 0.0, 0.0, 0.0],  # All zeros
        "events": [json.dumps([]) for _ in range(5)]  # Empty events
    })
    
    print("\n1. Forcing NaN with all zero denominators:")
    print(f"  All rewards are zero: {data['reward_mean'].unique()}")
    print(f"  All training times are zero: {data['training_time'].unique()}")
    
    # Test throughput with empty events
    print("\n2. Throughput with empty events:")
    result = evaluate_throughput(data)
    print(f"  Score: {result['score']}")
    print(f"  Samples used: {result['counters']['samples_used']}")
    print(f"  Samples seen: {result['counters']['samples_seen']}")
    
    # Test suite evaluations with all zeros
    print("\n3. Suite evaluations with all zeros:")
    consistency_result = evaluate_consistency(data)
    robustness_result = evaluate_robustness(data)
    efficiency_result = evaluate_efficiency(data)
    
    print(f"  Consistency score: {consistency_result['score']}")
    print(f"  Robustness score: {robustness_result['score']}")
    print(f"  Efficiency score: {efficiency_result['score']}")
    
    # Verify no crashes occurred
    print("\n✅ All evaluations completed without crashes!")


def main():
    """Run the complete demo."""
    print("🚀 RLDK Division by Zero Handling Demo")
    print("=" * 60)
    
    try:
        # Demo math utilities
        demo_math_utils()
        
        # Demo throughput metrics
        demo_throughput_metrics()
        
        # Demo suite evaluations
        demo_suite_evaluations()
        
        # Demo NaN behavior
        demo_nan_behavior()
        
        print("\n\n🎉 Demo completed successfully!")
        print("\nKey benefits of robust division handling:")
        print("  ✅ No crashes from division by zero")
        print("  ✅ Clear provenance tracking with counters")
        print("  ✅ Configurable behavior (skip, zero, NaN)")
        print("  ✅ NaN-aware aggregation functions")
        print("  ✅ Detailed skip reason reporting")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()